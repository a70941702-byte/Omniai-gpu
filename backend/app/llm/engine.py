"""Optional real LLM runtime. NeuralCore is intentionally not used for generation."""
from __future__ import annotations
import gc, threading
from dataclasses import dataclass
from typing import Iterator, Optional

@dataclass
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512

class LLMEngine:
    _lock = threading.RLock()
    _loaded_key: Optional[str] = None
    _model = None
    _tokenizer = None
    _backend = None

    def __init__(self):
        self.device = self._detect_device()

    @staticmethod
    def _detect_device():
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def load(self, model_ref: str, quantization: str | None = None):
        """Load one HF model (or a GGUF via llama_cpp when installed)."""
        with self._lock:
            key = f"{model_ref}|{quantization or ''}|{self.device}"
            if self._loaded_key == key and self._model is not None:
                return {"loaded": True, "reused": True, "backend": self._backend, "device": self.device}
            self.unload()
            if model_ref.lower().endswith(".gguf"):
                try:
                    from llama_cpp import Llama
                except ImportError as e:
                    raise RuntimeError("llama-cpp-python is required for GGUF models") from e
                self._model = Llama(model_path=model_ref, n_gpu_layers=-1 if self.device == "cuda" else 0,
                                     verbose=False)
                self._backend = "llama.cpp"
            else:
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                except ImportError as e:
                    raise RuntimeError("transformers is required for Hugging Face models") from e
                kwargs = {}
                if self.device == "cuda":
                    kwargs["torch_dtype"] = "auto"
                    kwargs["device_map"] = "auto"
                self._tokenizer = AutoTokenizer.from_pretrained(model_ref)
                self._model = AutoModelForCausalLM.from_pretrained(model_ref, **kwargs)
                if self.device == "cpu":
                    self._model.to("cpu")
                self._backend = "transformers"
            self._loaded_key = key
            return {"loaded": True, "reused": False, "backend": self._backend, "device": self.device}

    def unload(self):
        with self._lock:
            self._model = None
            self._tokenizer = None
            self._loaded_key = None
            self._backend = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            except Exception:
                pass

    @property
    def loaded(self): return self._model is not None

    def generate(self, messages: list[dict], config: GenerationConfig | None = None) -> str:
        cfg = config or GenerationConfig()
        with self._lock:
            if self._model is None: raise RuntimeError("no LLM loaded")
            if self._backend == "llama.cpp":
                out = self._model.create_chat_completion(messages=messages, temperature=cfg.temperature,
                    top_p=cfg.top_p, max_tokens=cfg.max_tokens)
                return out["choices"][0]["message"]["content"]
            prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._tokenizer(prompt, return_tensors="pt")
            try: inputs = {k: v.to(self._model.device) for k,v in inputs.items()}
            except Exception: pass
            ids = self._model.generate(**inputs, do_sample=cfg.temperature > 0,
                temperature=max(cfg.temperature, 1e-5), top_p=cfg.top_p, max_new_tokens=cfg.max_tokens,
                pad_token_id=self._tokenizer.eos_token_id)
            return self._tokenizer.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def stream(self, messages: list[dict], config: GenerationConfig | None = None) -> Iterator[str]:
        # llama.cpp has native streaming; transformers uses TextIteratorStreamer.
        cfg = config or GenerationConfig()
        if self._backend == "llama.cpp":
            out = self._model.create_chat_completion(messages=messages, temperature=cfg.temperature,
                top_p=cfg.top_p, max_tokens=cfg.max_tokens, stream=True)
            for item in out:
                token = item["choices"][0].get("delta", {}).get("content")
                if token: yield token
            return
        try:
            from transformers import TextIteratorStreamer
            from threading import Thread
            prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._tokenizer(prompt, return_tensors="pt")
            streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
            Thread(target=self._model.generate, kwargs={**inputs, "streamer": streamer,
                "do_sample": cfg.temperature > 0, "temperature": max(cfg.temperature,1e-5),
                "top_p": cfg.top_p, "max_new_tokens": cfg.max_tokens,
                "pad_token_id": self._tokenizer.eos_token_id}, daemon=True).start()
            yield from streamer
        except ImportError:
            yield self.generate(messages, cfg)

llm_engine = LLMEngine()
