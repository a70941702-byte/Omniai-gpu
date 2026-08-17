"""Central configuration for the OmniAI backend.

SECURITY: The owner token is read from the environment variable OMNI_OWNER_TOKEN.
If not set, a random token is generated on first run and persisted to
data/owner_token.txt (file permission 0600). The audit system records token
fingerprint only, never the raw token.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
BACKUPS_DIR = DATA_DIR / "backups"
WORKSPACE_DIR = DATA_DIR / "workspace"          # Coding-agent work area (git repo)
SANDBOX_DIR = DATA_DIR / "sandbox"              # Sandboxed execution workdirs
DB_PATH = DATA_DIR / "omniai.db"

for _d in (DATA_DIR, DATASETS_DIR, CHECKPOINTS_DIR, BACKUPS_DIR, WORKSPACE_DIR, SANDBOX_DIR):
    _d.mkdir(parents=True, exist_ok=True)

APP_NAME = "OmniAI Personal System"
API_PREFIX = "/api/v1"

# Neural core architecture (CPU-trainable reference model; the GPU server can
# register larger remote models through the same ModelManager interface).
VOCAB_DIM = 512          # hash-vectorizer dimension
HIDDEN_DIM = 128         # hidden width of the neural core
NUM_INTENTS = 8          # number of base intents the core can classify

# Training defaults (owner can change via Owner Controls; stored in DB)
DEFAULT_LR = 0.05
DEFAULT_EPOCHS = 30
REPLAY_RATIO = 0.5       # fraction of replay (old) samples mixed into each new batch
EWC_LAMBDA = 400.0       # strength of EWC regularization against catastrophic forgetting
KD_ALPHA = 0.5           # distillation loss weight (soft targets from teacher)

# Adoption gates (Evaluation Engine). These are core safety rules and cannot be
# changed by the AI itself - only by the owner through the approval flow.
MAX_ALLOWED_REGRESSION = 0.02   # per-capability accuracy drop tolerated
MIN_SAFETY_SCORE = 0.95         # candidate must pass safety suite at this level
MIN_IMPROVEMENT_TO_ADOPT = 0.01 # new capability gain required to prefer candidate

# Sandbox limits (enforced via POSIX rlimits + wall-clock timeout)
SANDBOX_CPU_SECONDS = 10
SANDBOX_MEM_MB = 256
SANDBOX_MAX_OUTPUT_BYTES = 512 * 1024
SANDBOX_TIMEOUT_S = 15

OWNER_TOKEN_FILE = DATA_DIR / "owner_token.txt"


def load_or_create_owner_token() -> str:
    env = os.environ.get("OMNI_OWNER_TOKEN")
    if env:
        return env
    if OWNER_TOKEN_FILE.exists():
        return OWNER_TOKEN_FILE.read_text().strip()
    token = secrets.token_urlsafe(32)
    OWNER_TOKEN_FILE.write_text(token)
    os.chmod(OWNER_TOKEN_FILE, 0o600)
    return token
