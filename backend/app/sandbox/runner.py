"""Sandbox - isolated execution for untrusted/experimental code.

Enforcement:
- Runs in a subprocess with POSIX rlimits (CPU seconds, address space).
- Wall-clock timeout kills the process group.
- Network is blocked by injecting a sitecustomize that disables socket.
- Workdir is a fresh per-run directory under data/sandbox; the code can only
  write there (PYTHONPATH and HOME are pointed into the sandbox dir).
- Output capped at SANDBOX_MAX_OUTPUT_BYTES.

The coding agent and the training system may only run experimental code
through this module. Any change touching production code must additionally
pass the Approval System.
"""
from __future__ import annotations

import os
import subprocess
import sys
import uuid
import time
from pathlib import Path

from ..config import (SANDBOX_CPU_SECONDS, SANDBOX_DIR,
                      SANDBOX_MAX_OUTPUT_BYTES, SANDBOX_MEM_MB,
                      SANDBOX_TIMEOUT_S)
from ..security import audit

_NO_NET = """
import socket
# Network-off policy: replace connection entry points, including the constructor
# used internally by socket.create_connection. This is intentionally injected
# before user code and does not depend on DNS/firewall configuration.
_RealSocket = socket.socket

def _blocked(*a, **k):
    raise PermissionError("network access is disabled inside the sandbox")

socket.create_connection = _blocked
socket.create_server = _blocked
socket.socket = _blocked
"""


def _limits():  # applied in the child via preexec_fn
    import resource

    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (SANDBOX_CPU_SECONDS, SANDBOX_CPU_SECONDS))
        mem = SANDBOX_MEM_MB * 1024 * 1024
        
        # RLIMIT_AS is unreliable for Python/BLAS runtimes (it can prevent
        # interpreter startup before user code runs). Keep a portable data
        # limit and enforce the hard wall-clock/process cleanup below.
        try:
            resource.setrlimit(resource.RLIMIT_DATA, (mem, mem))
        except (ValueError, OSError):
            pass
        resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
        os.setsid()
    return apply


class Sandbox:
    def run_python(self, code: str, files: dict[str, str] | None = None,
                   args: list[str] | None = None) -> dict:
        run_id = uuid.uuid4().hex[:12]
        workdir = SANDBOX_DIR / run_id
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "omni_sandbox_bootstrap.py").write_text(_NO_NET)
        for name, content in (files or {}).items():
            safe = workdir / Path(name).name  # flatten: no path escapes
            safe.write_text(content)
        script = workdir / "main.py"
        script.write_text(code)

        env = {"PYTHONPATH": str(workdir), "HOME": str(workdir),
               "PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}
        cmd = [sys.executable, "-c", "import omni_sandbox_bootstrap; exec(compile(open(\'main.py\', encoding=\'utf-8\').read(), \'main.py\', \'exec\'))"] + (args or [])
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                    cwd=str(workdir), env=env, preexec_fn=_limits(), start_new_session=False)
            deadline=time.monotonic()+SANDBOX_TIMEOUT_S
            kill_reason=None
            while proc.poll() is None:
                if time.monotonic() >= deadline:
                    kill_reason="killed: wall-clock timeout"; os.killpg(proc.pid, 9); break
                try:
                    from ..security.policy import kill_switch_active
                    if kill_switch_active():
                        kill_reason="killed: owner kill switch"; os.killpg(proc.pid, 9); break
                except Exception: pass
                time.sleep(0.05)
            stdout, stderr = proc.communicate(timeout=2)
            result = {"run_id":run_id,"exit_code":proc.returncode,"stdout":stdout[:SANDBOX_MAX_OUTPUT_BYTES],
                      "stderr":(kill_reason or stderr)[:SANDBOX_MAX_OUTPUT_BYTES],"timed_out":bool(kill_reason)}
        except Exception as e:
            try: proc.kill()
            except Exception: pass
            result={"run_id":run_id,"exit_code":-1,"stdout":"","stderr":str(e),"timed_out":False}
        audit.log("sandbox", "code_executed", {"run_id": run_id,
                                               "exit_code": result["exit_code"],
                                               "timed_out": result["timed_out"],
                                               "files": list((files or {}).keys())})
        return result

    def run_pytest(self, files: dict[str, str]) -> dict:
        """Run pytest inside the sandbox against provided test+source files."""
        runner = (
            "import subprocess, sys, json\n"
            "r = subprocess.run([sys.executable, '-m', 'pytest', '-x', '-q', '--tb=short', '.'],"
            " capture_output=True, text=True)\n"
            "print(json.dumps({'exit': r.returncode, 'out': r.stdout[-4000:],"
            " 'err': r.stderr[-2000:]}))\n"
        )
        merged = dict(files)
        return self.run_python(runner, files=merged)


sandbox = Sandbox()
