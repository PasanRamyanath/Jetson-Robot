"""Cross-encoder rerank of the retrieval top-24 (§11.6): BAAI/bge-reranker-v2-m3 FP16 on GPU1.

Loads in a background thread; until it is ready (or if it fails) rows pass through in fused-score order. The
relevance (sigmoid of the logit) is added to each row's fused score with the dense weight, so recency and
importance still count.
"""
import logging
import math
import threading

log = logging.getLogger("rerank")
W = 1.5


class Reranker:
    def __init__(self, repo="BAAI/bge-reranker-v2-m3", device="cuda"):
        self.repo, self.device = repo, device
        self.model = self.tok = None
        self.lock = threading.Lock()
        threading.Thread(target=self._load, daemon=True, name="rerank-load").start()

    def _load(self):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.repo)
            model = AutoModelForSequenceClassification.from_pretrained(self.repo, torch_dtype=torch.float16)
            self.torch, self.tok, self.model = torch, tok, model.to(self.device).eval()
            log.info("reranker %s ready", self.repo)
        except Exception:
            log.exception("reranker failed to load; retrieval keeps the fused order")

    def scores(self, query, texts):
        with self.lock, self.torch.inference_mode():
            x = self.tok([[query, t] for t in texts], padding=True, truncation=True, max_length=384,
                         return_tensors="pt").to(self.device)
            return self.model(**x).logits.view(-1).float().tolist()

    def __call__(self, query, rows):
        if self.model is None or not rows:
            return rows
        try:
            logits = self.scores(query, [r["text"] for r in rows])
        except Exception:
            log.exception("rerank failed")
            return rows
        for r, z in zip(rows, logits):
            r["_rerank"] = 1.0 / (1.0 + math.exp(-z))
            r["_score"] = r.get("_score", 0.0) + W * r["_rerank"]
        return sorted(rows, key=lambda r: r["_score"], reverse=True)
