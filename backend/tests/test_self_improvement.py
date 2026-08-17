"""Self-improvement engine and coding agent behavior."""

H = {"Authorization": "Bearer test-owner-token"}


def test_static_analysis_finds_issues():
    from app.agents.self_improvement import self_improvement
    findings = self_improvement.analyze_source()
    # engine should at least run across the whole source tree without crashing
    assert isinstance(findings, list)


def test_propose_fixes_never_touches_production():
    from app.agents.coding_agent import coding_agent
    from app.agents.self_improvement import self_improvement
    # snapshot a source file before running the engine
    files = coding_agent.list_source_files()
    assert files
    before = {f: coding_agent.read_source_file(f) for f in files}
    self_improvement.propose_fixes()
    after = {f: coding_agent.read_source_file(f) for f in files}
    assert before == after  # production code untouched by the engine itself


def test_coding_agent_workspace_and_git():
    from app.agents.coding_agent import coding_agent
    dest = coding_agent.write_workspace_file("experiments/hello.py", "print('hi')\n")
    assert dest.exists()
    coding_agent.create_branch("experiment/test-branch")
    log = coding_agent.commit("test commit from coding agent")
    branch = coding_agent.git("rev-parse", "--abbrev-ref", "HEAD")
    assert "experiment/test-branch" in branch


def test_coding_agent_rejects_path_escape():
    import pytest
    from app.agents.coding_agent import coding_agent
    with pytest.raises(ValueError):
        coding_agent.write_workspace_file("../../etc/passwd", "x")
    with pytest.raises(ValueError):
        coding_agent.read_source_file("../../etc/passwd")


def test_diff_patch_format():
    from app.agents.coding_agent import coding_agent
    src = coding_agent.read_source_file("config.py")
    patch = coding_agent.diff_patch("config.py", src + "\n# comment\n")
    assert patch.startswith("--- a/config.py") and "+++ b/config.py" in patch
    assert "+# comment" in patch


def test_improve_api_endpoints(client):
    r = client.post("/api/v1/improve/analyze", headers=H)
    assert r.status_code == 200
    r = client.post("/api/v1/improve/propose", headers=H)
    assert r.status_code == 200
    proposals = client.get("/api/v1/improve/proposals", headers=H).json()
    assert isinstance(proposals, list)
