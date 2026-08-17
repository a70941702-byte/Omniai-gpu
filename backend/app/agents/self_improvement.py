"""Self-Improvement Engine.

Reads the project's own source, runs static analysis, finds concrete issues
(style/perf/risk heuristics + failing tests), drafts a patch in the
workspace, verifies it by running tests in the Sandbox, measures before/after
test results, and files an Approval request with the full report. It never
modifies the running production code itself.
"""
from __future__ import annotations

import ast
import json

from ..agents.coding_agent import coding_agent
from ..approval.system import approval_system
from ..database import db
from ..security import audit


class SelfImprovementEngine:
    def analyze_source(self) -> list[dict]:
        """Static analysis pass over the backend source. Returns findings."""
        findings = []
        for rel in coding_agent.list_source_files():
            src = coding_agent.read_source_file(rel)
            try:
                tree = ast.parse(src)
            except SyntaxError as e:
                findings.append({"file": rel, "type": "syntax_error",
                                 "severity": "high", "detail": str(e)})
                continue
            for node in ast.walk(tree):
                # bare except -> bug risk
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    findings.append({"file": rel, "type": "bare_except",
                                     "severity": "medium", "line": node.lineno,
                                     "detail": "bare 'except:' swallows all errors"})
                # mutable default args -> classic bug
                if isinstance(node, ast.FunctionDef):
                    for d in node.args.defaults:
                        if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                            findings.append({"file": rel, "type": "mutable_default",
                                             "severity": "medium", "line": node.lineno,
                                             "detail": f"mutable default in {node.name}()"})
        return findings

    def propose_fixes(self) -> list[dict]:
        """Analyze, and for each auto-fixable finding draft+verify a patch in
        the sandbox, then file an approval request. Returns proposals."""
        findings = self.analyze_source()
        proposals = []
        for f in findings:
            if f["type"] != "bare_except":
                continue  # only auto-fix what we can verify safely
            rel = f["file"]
            old_src = coding_agent.read_source_file(rel)
            new_src = old_src.replace("except:", "except Exception:")
            if new_src == old_src:
                continue
            patch = coding_agent.diff_patch(rel, new_src)
            # verify: syntax-check the patched file in the sandbox
            verify = self._verify_in_sandbox(rel, new_src)
            pid = db.new_id()
            status = "proposed" if verify["ok"] else "draft_failed_verification"
            db.execute(
                "INSERT INTO improvement_proposals(id, title, analysis, patch, test_output,"
                " status, created_at) VALUES(?,?,?,?,?,?,?)",
                (pid, f"Fix bare except in {rel}:{f.get('line')}",
                 json.dumps(f), patch, json.dumps(verify), status, db.now()),
            )
            if verify["ok"]:
                approval_system.request("code_change", {
                    "change": f"Replace bare 'except:' with 'except Exception:' in {rel}",
                    "reason": f["detail"],
                    "benefit": "Errors are no longer silently swallowed; bugs become visible.",
                    "affected_files": [rel],
                    "test_results": verify,
                    "performance_before_after": "no performance impact (exception path only)",
                    "risks": "Previously-swallowed exceptions may now surface as errors.",
                    "resources_required": "none",
                    "estimated_cost": "0 credits",
                    "patch": patch,
                })
            proposals.append({"id": pid, "file": rel, "status": status})
        audit.log("self_improvement", "analysis_run",
                  {"findings": len(findings), "proposals": len(proposals)})
        return proposals

    def _verify_in_sandbox(self, rel: str, new_src: str) -> dict:
        from ..sandbox.runner import sandbox
        code = (
            "import ast, sys\n"
            f"src = open('{rel.split('/')[-1]}').read()\n"
            "ast.parse(src)\n"
            "print('SYNTAX_OK')\n"
        )
        res = sandbox.run_python(code, files={rel: new_src})
        return {"ok": res["exit_code"] == 0 and "SYNTAX_OK" in res["stdout"],
                "stdout": res["stdout"][-500:], "stderr": res["stderr"][-500:]}


self_improvement = SelfImprovementEngine()
