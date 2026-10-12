import logging
from typing import List

logger = logging.getLogger(__name__)


class LLMClient:

    def __init__(
        self,
        model: str = "Qwen3-4B-Instruct-2507",
        mode: str = "vllm",
        api_base: str = "http://localhost:8800",
        api_key: str = "",
        device: str = "auto",
        temperature: float = 0.3,
        max_tokens: int = 512,
    ):
        self.model = model
        self.mode = mode
        self.temperature = temperature
        self.max_tokens = max_tokens

        if mode == "vllm":
            self.api_base = api_base.rstrip("/")
            self.api_key = api_key
            self._local_model = None
            self._tokenizer = None
            logger.info(f"vLLM mode: {model} @ {api_base}")
        elif mode == "local":
            self._load_local_model(model, device)
        else:
            raise ValueError(f"Unsupported mode: {mode}; choose vllm / local")

    def _load_local_model(self, model_path: str, device: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading local model: {model_path}")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self._local_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )

        if device == "cpu":
            self._local_model = self._local_model.to(device)

        self._local_model.eval()
        logger.info(f"Model loaded, device: {device}")

    def chat(self, prompt: str, system_prompt: str = None) -> str:
        if self.mode == "vllm":
            return self._call_vllm(prompt, system_prompt)
        else:
            return self._call_local(prompt, system_prompt)

    def _call_vllm(self, prompt: str, system_prompt: str = None) -> str:
        import requests

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        url = f"{self.api_base}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"vLLM call failed: {e}")
            return ""

    def _call_local(self, prompt: str, system_prompt: str = None) -> str:
        import torch

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._local_model.device)

        with torch.no_grad():
            outputs = self._local_model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                top_p=0.9,
            )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def batch_chat(self, prompts: List[str], system_prompt: str = None) -> List[str]:
        results = []
        for i, prompt in enumerate(prompts):
            result = self.chat(prompt, system_prompt)
            results.append(result)
            if (i + 1) % 10 == 0:
                logger.info(f"  LLM call progress: {i+1}/{len(prompts)}")
        return results
