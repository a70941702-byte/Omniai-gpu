import threading, time

def test_session_login_and_revoke(client):
    r=client.post('/api/v1/auth/login',json={'token':'test-owner-token'}); assert r.status_code==200
    token=r.json()['token']; assert token != 'test-owner-token'
    assert client.get('/api/v1/status',headers={'Authorization':'Bearer '+token}).status_code==200
    client.post('/api/v1/auth/logout',headers={'Authorization':'Bearer '+token})
    assert client.get('/api/v1/status',headers={'Authorization':'Bearer '+token}).status_code==403

def test_owner_controls_are_enforced(client):
    H={'Authorization':'Bearer test-owner-token'}
    client.post('/api/v1/controls',json={'values':{'python_enabled':False,'training_enabled':False,'web_enabled':False}},headers=H)
    assert client.post('/api/v1/sandbox/run',json={'code':'print(1)'},headers=H).status_code==403
    assert client.post('/api/v1/training/start',headers=H).status_code==403
    client.post('/api/v1/controls',json={'values':{'python_enabled':True,'training_enabled':True}},headers=H)

def test_tool_gateway_never_bypasses_policy():
    from app.tools.gateway import gateway
    from app.database import db
    db.set_control('python_enabled',False)
    try:
        try: gateway.call('python',code='print(1)')
        except PermissionError: pass
        else: assert False
    finally: db.set_control('python_enabled',True)

def test_llm_engine_is_separate_from_neural_core():
    from app.llm.engine import llm_engine
    from app.models_core.neural_core import NeuralCore
    assert hasattr(llm_engine,'load') and hasattr(llm_engine,'stream')
    assert isinstance(NeuralCore(num_outputs=2), NeuralCore)

def test_kill_switch_stops_running_sandbox():
    from app.sandbox.runner import sandbox
    from app.database import db
    db.set_control('kill_switch',False)
    out={}
    def run(): out.update(sandbox.run_python('import time; time.sleep(30)'))
    t=threading.Thread(target=run); t.start(); time.sleep(.3); db.set_control('kill_switch',True); t.join(5); db.set_control('kill_switch',False)
    assert out.get('exit_code') != 0 and 'kill switch' in out.get('stderr','').lower()
