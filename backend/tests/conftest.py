"""Test fixtures. Each test session runs against a FRESH temporary database
and data directories, so tests never touch the real data dir."""
import importlib
import os
import sys
import tempfile
from pathlib import Path

import pytest

TMP = Path(tempfile.mkdtemp(prefix="omniai_test_"))
os.environ["OMNI_OWNER_TOKEN"] = "test-owner-token"

# Point config at a temp data dir BEFORE importing app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.config as config  # noqa: E402

config.DATA_DIR = TMP / "data"
config.DATASETS_DIR = config.DATA_DIR / "datasets"
config.CHECKPOINTS_DIR = config.DATA_DIR / "checkpoints"
config.BACKUPS_DIR = config.DATA_DIR / "backups"
config.WORKSPACE_DIR = config.DATA_DIR / "workspace"
config.SANDBOX_DIR = config.DATA_DIR / "sandbox"
config.DB_PATH = config.DATA_DIR / "omniai.db"
config.OWNER_TOKEN_FILE = config.DATA_DIR / "owner_token.txt"
for d in (config.DATA_DIR, config.DATASETS_DIR, config.CHECKPOINTS_DIR,
          config.BACKUPS_DIR, config.WORKSPACE_DIR, config.SANDBOX_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Reload modules that captured config paths at import time
for mod in ["app.database", "app.security.audit", "app.security.auth",
            "app.memory.store", "app.knowledge.base",
            "app.models_core.registry", "app.training.anti_forgetting",
            "app.training.trainer", "app.sandbox.runner",
            "app.agents.coding_agent", "app.agents.self_improvement",
            "app.approval.system", "app.backup.manager",
            "app.evaluation.engine", "app.orchestrator.core",
            "app.api.routes", "app.main"]:
    if mod in sys.modules:
        importlib.reload(sys.modules[mod])


@pytest.fixture()
def owner_headers():
    return {"Authorization": "Bearer test-owner-token"}


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)
