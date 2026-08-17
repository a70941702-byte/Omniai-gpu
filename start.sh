#!/usr/bin/env bash
# OmniAI GPU — تشغيل تلقائي كامل
set -e
cd /workspace/backend

echo "=============================================="
echo "  OmniAI GPU Server"
echo "  Model: ${OMNI_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
echo "=============================================="

# عرض كشف الـ GPU
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY')" || true

# رمز المالك: من متغير البيئة أو توليد تلقائي
if [ -z "${OMNI_OWNER_TOKEN:-}" ]; then
  echo "⚠️  OMNI_OWNER_TOKEN غير محدد — سيتم توليد رمز تلقائي في data/owner_token.txt"
else
  echo "✅ رمز المالك: محدد من متغير البيئة"
fi

# تشغيل الخادم
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

# انتظار جاهزية الخادم
for i in $(seq 1 30); do
  curl -s -o /dev/null http://127.0.0.1:8000/ && break
  sleep 1
done

# تحميل النموذج تلقائياً لو مطلوب
if [ "${OMNI_AUTOLOAD_MODEL:-true}" = "true" ]; then
  TOKEN="${OMNI_OWNER_TOKEN:-$(cat data/owner_token.txt 2>/dev/null)}"
  echo "⏳ تحميل النموذج: ${OMNI_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
  curl -s -m 900 -X POST "http://127.0.0.1:8000/api/v1/models/llm/load?model_ref=${OMNI_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}" \
       -H "Authorization: Bearer $TOKEN" && echo "" && echo "✅ النموذج محمّل وجاهز" || echo "⚠️ فشل تحميل النموذج — حمّله يدوياً عبر API"
fi

echo "=============================================="
echo "  🟢 OmniAI شغال على المنفذ 8000"
echo "  🔑 رمز المالك: ${OMNI_OWNER_TOKEN:-<شوف data/owner_token.txt>}"
echo "=============================================="

wait $SERVER_PID
