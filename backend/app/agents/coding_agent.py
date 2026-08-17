"""Coding Agent - can read/write files in the workspace, run tests through
the Sandbox, and use git (branch/commit/diff). It CANNOT touch production
code directly: all edits happen in the workspace copy, and applying them to
production requires an owner-approved change via the Approval System.
"""
from __future__ import annotations

import difflib
import subprocess
from pathlib import Path

from ..config import WORKSPACE_DIR
from ..database import db
from ..sandbox.runner import sandbox
from ..security import audit

SOURCE_ROOT = Path(__file__).resolve().parent.parent  # the running backend 'app' package


class CodingAgent:
    def __init__(self):
        self._ensure_repo()

    def _ensure_repo(self) -> None:
        if not (WORKSPACE_DIR / ".git").exists():
            subprocess.run(["git", "init", "-q"], cwd=WORKSPACE_DIR, check=False)
            subprocess.run(["git", "config", "user.email", "agent@omniai.local"], cwd=WORKSPACE_DIR)
            subprocess.run(["git", "config", "user.name", "OmniAI Coding Agent"], cwd=WORKSPACE_DIR)

    # ---- capabilities ------------------------------------------------------
    def read_source_file(self, rel_path: str) -> str:
        """Read a file from the running backend source (read-only view)."""
        path = (SOURCE_ROOT / rel_path).resolve()
        if not str(path).startswith(str(SOURCE_ROOT)):
            raise ValueError("path escape rejected")
        return path.read_text()

    def list_source_files(self) -> list[str]:
        return sorted(str(p.relative_to(SOURCE_ROOT))
                      for p in SOURCE_ROOT.rglob("*.py"))

    def stage_copy_to_workspace(self, rel_path: str) -> Path:
        """Copy a production file into the git workspace for experimentation."""
        dest = WORKSPACE_DIR / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(self.read_source_file(rel_path))
        return dest

    def write_workspace_file(self, rel_path: str, content: str) -> Path:
        dest = (WORKSPACE_DIR / rel_path).resolve()
        if not str(dest).startswith(str(WORKSPACE_DIR.resolve())):
            raise ValueError("path escape rejected")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
        audit.log("coding_agent", "workspace_write", {"file": rel_path})
        return dest

    def run_tests_in_sandbox(self, files: dict[str, str]) -> dict:
        return sandbox.run_pytest(files)

    def git(self, *args: str) -> str:
        out = subprocess.run(["git", *args], cwd=WORKSPACE_DIR,
                             capture_output=True, text=True, check=False)
        return (out.stdout + out.stderr).strip()

    def create_branch(self, name: str) -> str:
        return self.git("checkout", "-B", name)

    def commit(self, message: str) -> str:
        self.git("add", "-A")
        return self.git("commit", "-m", message)

    def diff_patch(self, rel_path: str, new_content: str) -> str:
        old = self.read_source_file(rel_path)
        return "".join(difflib.unified_diff(
            old.splitlines(keepends=True), new_content.splitlines(keepends=True),
            fromfile=f"a/{rel_path}", tofile=f"b/{rel_path}"))

    def apply_approved_patch(self, patch_text: str, affected_files: list[str]) -> dict:
        """Apply an owner-approved unified patch to production, with backup and rollback on test failure."""
        import tempfile, shutil
        from ..config import BACKUPS_DIR
        if not patch_text.strip(): raise ValueError("empty patch")
        safe_files=[]
        for rel in affected_files:
            p=(SOURCE_ROOT/rel).resolve()
            if not str(p).startswith(str(SOURCE_ROOT.resolve())): raise ValueError("path escape rejected")
            safe_files.append(p)
        backup_dir=BACKUPS_DIR/db.new_id(); backup_dir.mkdir(parents=True, exist_ok=True)
        for p in safe_files:
            if p.exists():
                dest=backup_dir/p.relative_to(SOURCE_ROOT); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p,dest)
        patch_file=backup_dir/'change.patch'; patch_file.write_text(patch_text)
        check=subprocess.run(['patch','-p1','--dry-run','--forward'],cwd=SOURCE_ROOT,input=patch_text,text=True,capture_output=True)
        if check.returncode!=0: raise ValueError(f"patch validation failed: {check.stderr or check.stdout}")
        applied=subprocess.run(['patch','-p1','--forward'],cwd=SOURCE_ROOT,input=patch_text,text=True,capture_output=True)
        if applied.returncode!=0: raise ValueError(f"patch apply failed: {applied.stderr or applied.stdout}")
        testproc=subprocess.run(['python','-m','pytest','-q'],cwd=SOURCE_ROOT.parent,capture_output=True,text=True); test={'exit_code':testproc.returncode,'stdout':testproc.stdout[-12000:],'stderr':testproc.stderr[-4000:],'timed_out':False}
        if test['exit_code']!=0:
            for p in safe_files:
                b=backup_dir/p.relative_to(SOURCE_ROOT)
                if b.exists(): shutil.copy2(b,p)
            audit.log('owner','code_change_rolled_back',{'backup':str(backup_dir),'test':test})
            return {'applied':False,'rolled_back':True,'backup':str(backup_dir),'tests':test}
        branch='approved/'+db.new_id(); self.create_branch(branch)
        for rel in affected_files:
            self.stage_copy_to_workspace(rel)
        self.git('add','-A'); commit=self.git('commit','-m','Owner-approved code change')
        audit.log('owner','code_change_applied',{'files':affected_files,'backup':str(backup_dir),'commit':commit,'branch':branch})
        return {'applied':True,'rolled_back':False,'backup':str(backup_dir),'tests':test,'commit':commit,'branch':branch}


coding_agent = CodingAgent()
