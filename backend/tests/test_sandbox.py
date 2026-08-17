"""Sandbox isolation: resource limits, network block, timeout, path safety."""


def test_sandbox_runs_code_and_returns_output():
    from app.sandbox.runner import sandbox
    res = sandbox.run_python("print('hello from sandbox')")
    assert res["exit_code"] == 0
    assert "hello from sandbox" in res["stdout"]


def test_sandbox_blocks_network():
    from app.sandbox.runner import sandbox
    code = ("import socket\n"
            "try:\n"
            "    socket.create_connection(('8.8.8.8', 53), timeout=2)\n"
            "    print('NET_OK')\n"
            "except PermissionError:\n"
            "    print('NET_BLOCKED')\n")
    res = sandbox.run_python(code)
    assert "NET_BLOCKED" in res["stdout"]
    assert "NET_OK" not in res["stdout"]


def test_sandbox_kills_infinite_loop():
    from app.sandbox.runner import sandbox
    res = sandbox.run_python("while True: pass")
    assert res["timed_out"] or res["exit_code"] != 0


def test_sandbox_memory_limit():
    from app.sandbox.runner import sandbox
    # try to allocate far more than the 256MB limit
    code = ("x = []\n"
            "try:\n"
            "    for _ in range(1000):\n"
            "        x.append(bytearray(10 * 1024 * 1024))\n"
            "    print('MEM_OK')\n"
            "except MemoryError:\n"
            "    print('MEM_BLOCKED')\n")
    res = sandbox.run_python(code)
    assert "MEM_OK" not in res["stdout"]


def test_sandbox_files_are_flattened_no_escape():
    from app.sandbox.runner import sandbox
    res = sandbox.run_python(
        "import os; print(sorted(os.listdir('.')))",
        files={"../../etc/evil.py": "print('should not escape')", "ok.py": "x=1"})
    assert "evil.py" in res["stdout"]          # written, but flattened into workdir
    assert "etc" not in res["stdout"]          # no directory traversal happened


def test_sandbox_pytest_runner():
    from app.sandbox.runner import sandbox
    files = {
        "calc.py": "def add(a, b):\n    return a + b\n",
        "test_calc.py": "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
    }
    res = sandbox.run_pytest(files)
    assert res["exit_code"] == 0
    assert "'exit': 0" in res["stdout"] or '"exit": 0' in res["stdout"]
