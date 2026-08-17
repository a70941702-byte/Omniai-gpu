# OmniAI GPU Template - RunPod/Vast.ai ready
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/hf-cache

WORKDIR /workspace

# متطلبات النظام
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && rm -rf /var/lib/apt/lists/*

# متطلبات بايثون (الخادم + LLM)
COPY backend/requirements.txt backend/requirements.txt
COPY backend/requirements-llm.txt backend/requirements-llm.txt
RUN pip install --no-cache-dir -r backend/requirements.txt \
    && pip install --no-cache-dir "transformers>=4.45" "peft>=0.13" "datasets>=2.20" "accelerate>=0.34"

# كود المشروع
COPY backend backend

# سكربت التشغيل
COPY start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

EXPOSE 8000
CMD ["/workspace/start.sh"]
