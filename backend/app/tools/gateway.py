"""Single choke point for every tool call."""
from __future__ import annotations
import ast, math, operator, subprocess, time, urllib.request
from ..database import db
from ..security import audit
from ..security.policy import allowed, domain_allowed, kill_switch_active

class ToolGateway:
    def __init__(self): self.tools = {}
    def register(self, name, handler, capability=None, timeout=15, approval="LOW"):
        self.tools[name] = (handler, capability or name, timeout, approval)
    def call(self, name: str, **kwargs):
        if kill_switch_active(): raise PermissionError("kill switch active")
        if name not in self.tools: raise ValueError(f"unknown tool: {name}")
        handler, capability, timeout, approval = self.tools[name]
        if not allowed(capability): raise PermissionError(f"tool disabled: {name}")
        # Agent calls cannot self-approve HIGH/CRITICAL operations. Owner APIs may
        # execute approved changes through ApprovalSystem instead.
        if approval in ("HIGH", "CRITICAL") and not db.get_controls().get("agent_approved_tools", False):
            raise PermissionError(f"approval required: {approval}")
        start=time.time(); aid=db.new_id()
        db.execute("INSERT INTO tool_audit(id,tool,actor,started_at,ok,detail) VALUES(?,?,?,?,?,?)",(aid,name,"agent",start,0,"started"))
        try:
            out = handler(**kwargs)
            elapsed=time.time()-start
            db.execute("UPDATE tool_audit SET finished_at=?,ok=1,detail=? WHERE id=?",(time.time(),str(out)[:2000],aid))
            audit.log("agent", "tool_call", {"tool":name,"ok":True,"elapsed_s":round(elapsed,4)})
            return out
        except Exception as e:
            db.execute("UPDATE tool_audit SET finished_at=?,ok=0,detail=? WHERE id=?",(time.time(),str(e),aid))
            audit.log("agent", "tool_call", {"tool":name,"ok":False,"error":str(e)})
            raise

def calculator(expression: str):
    tree=ast.parse(expression, mode="eval")
    ops={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.Pow:operator.pow,
         ast.Mod:operator.mod,ast.USub:operator.neg,ast.UAdd:operator.pos}
    def ev(n):
        if isinstance(n,ast.Constant) and isinstance(n.value,(int,float)): return n.value
        if isinstance(n,ast.UnaryOp) and type(n.op) in ops: return ops[type(n.op)](ev(n.operand))
        if isinstance(n,ast.BinOp) and type(n.op) in ops: return ops[type(n.op)](ev(n.left),ev(n.right))
        raise ValueError("unsupported expression")
    return ev(tree.body)

gateway=ToolGateway()
gateway.register("calculator", calculator, "python")

# Built-in adapters. All calls still pass through the same policy choke point.
def _python(code, files=None):
    from ..sandbox.runner import sandbox
    return sandbox.run_python(code, files or {})
def _pytest(files):
    from ..sandbox.runner import sandbox
    return sandbox.run_pytest(files)
def _terminal(command):
    import shlex, subprocess
    if isinstance(command,str): command=shlex.split(command)
    if not command or any(x in command for x in ("sudo","/etc","/root")): raise PermissionError("unsafe terminal command")
    p=subprocess.run(command,capture_output=True,text=True,timeout=15,cwd=str(__import__('pathlib').Path(__file__).resolve().parents[2]))
    return {"exit_code":p.returncode,"stdout":p.stdout[-10000:],"stderr":p.stderr[-5000:]}
def _files_read(path):
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]/"data"/"workspace"
    p=(root/path).resolve()
    if not str(p).startswith(str(root.resolve())): raise ValueError("path escape rejected")
    return p.read_text() if p.exists() else None
def _git(command):
    from ..agents.coding_agent import coding_agent
    return coding_agent.git(*command.split()) if isinstance(command,str) else coding_agent.git(*command)
def _web(url, timeout=10):
    from ..security.network import get
    return get(url, timeout)

gateway.register("python", _python, "python")
gateway.register("pytest", _pytest, "python")
gateway.register("terminal", _terminal, "terminal", approval="HIGH")
gateway.register("files", _files_read, "files")
gateway.register("git", _git, "code_edit", approval="HIGH")
gateway.register("web", _web, "web", approval="CRITICAL")
