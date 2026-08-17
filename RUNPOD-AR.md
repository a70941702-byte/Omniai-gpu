# تشغيل OmniAI على RunPod GPU — خطوات من المتصفح

## أ) رفع القالب على GitHub
1. أنشئ مستودع جديد github.com/new باسم: omniai-gpu (Public)
2. ارفع محتويات هذا المجلد كاملة (Dockerfile + start.sh + مجلد backend)
   أو ارفع omniai-gpu-template.zip واستخدم workflow لفك الضغط

## ب) إنشاء الحساب والقالب على RunPod
1. افتح runpod.io وسجّل حساب (بإيميلك أو GitHub)
2. اشحن رصيد: Billing ← Add Funds (5$ تكفيك أيام من التجربة بالساعة)
3. من القائمة: Templates ← New Template:
   - Name: omniai-gpu
   - Type: إن كان عندك خيار "Import from GitHub/Dockerfile" اختاره ووجّهه لمستودع omniai-gpu
   - أو اختار أسهل طريق: Container Image مبني مسبقاً (شوف الخطوة ج)
   - Container Disk: 30 GB على الأقل
   - Expose HTTP Ports: 8000
   - Environment Variables:
       OMNI_OWNER_TOKEN = رمزك السري (20+ حرف — احفظه)
       OMNI_MODEL = Qwen/Qwen2.5-1.5B-Instruct
       OMNI_AUTOLOAD_MODEL = true

## ج) الطريق الأسهل — ابنِ الصورة على GitHub Actions (مجاناً)
أضف .github/workflows/docker.yml (موجود جاهز في القالب):
- سيبني صورة Docker ويرفعها على GitHub Container Registry تلقائياً
- بعدها في RunPod اختار "Custom Container" وحط رابط الصورة:
  ghcr.io/USERNAME/omniai-gpu:latest

## د) التشغيل
1. Deploy ← اختار كرت GPU:
   - RTX 4090 (24GB) — الأفضل قيمة (~$0.24-0.44/ساعة)
   - RTX 3090 (24GB) — أرخص شوية ويكفي لنماذج حتى 14B بـ4-bit
2. اضغط Deploy On-Demand واستنى ~5 دقائق (أول مرة يبني الصورة ويحمّل النموذج)
3. من صفحة الـ Pod: اضغط Connect ← HTTP Port 8000 ← هيديك رابط عام زي:
   https://xxxx-8000.proxy.runpod.net

## هـ) الربط في تطبيق OmniAI على هاتفك
- Server URL: https://xxxx-8000.proxy.runpod.net/api/v1
- Owner Token: نفس OMNI_OWNER_TOKEN اللي حطيته
- Connect ✅ وابدأ المحادثة بالنموذج الكبير

## ⚠️ مهم جداً — التوفير
- لو مش بتستخدمه: اضغط Stop من صفحة الـ Pod فوراً — الساعة بتحسب!
- التدريب: شغّل الـPod، اعمل دورة التدريب، اقفله — تدفع دقايق بس
- النموذج والبيانات بتتحفظ في /workspace (الـ Network Volume لو ضفته — أنصح بيه 50GB)

## نماذج مقترحة حسب الكرت
| الكرت | VRAM | النموذج المقترح (OMNI_MODEL) |
|---|---|---|
| RTX 3090/4090 | 24GB | Qwen/Qwen2.5-7B-Instruct أو 14B (4-bit) |
| RTX 3060 | 12GB | Qwen/Qwen2.5-3B-Instruct |
| أي كرت صغير | 8GB | Qwen/Qwen2.5-1.5B-Instruct (الافتراضي) |
