"""bge-small-en-v1.5 text embeddings on the Jetson CPU (onnxruntime + HF tokenizers), 384-d, L2-normalised.

Same model as the brain's episode embeddings (§11.1.5), so vectors are comparable across devices.
~15-30 ms per short sentence on one A57 core with the int8 export.
Files: $BENI_MODELS/bge-small/{model.onnx | model_quantized.onnx, tokenizer.json}
"""
import os
import threading

import numpy as np

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, d, max_len=128, threads=1):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        model = next((os.path.join(d, f) for f in ("model_quantized.onnx", "model_int8.onnx", "model.onnx")
                      if os.path.exists(os.path.join(d, f))), None)
        if model is None:
            raise FileNotFoundError("no bge-small onnx in %s" % d)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
        self.tok.enable_truncation(max_len)
        self.tok.enable_padding()
        self.dim = 384
        self.lock = threading.Lock()

    def encode(self, texts, query=False):
        if isinstance(texts, str):
            texts = [texts]
        if query:
            texts = [QUERY_PREFIX + t for t in texts]
        enc = self.tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], np.int64)
        feed = {"input_ids": ids, "attention_mask": np.array([e.attention_mask for e in enc], np.int64)}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        with self.lock:
            out = self.sess.run(None, feed)[0]
        v = out[:, 0] if out.ndim == 3 else out           # CLS pooling (bge)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    def __call__(self, text):
        """Single query vector (used by Retriever)."""
        return self.encode(text, query=True)[0]

    def doc(self, text):
        return self.encode(text)[0]


class HashEmbedder:
    """Deterministic bag-of-words fallback (tests / model not downloaded yet). Not semantically meaningful."""
    dim = 384

    def encode(self, texts, query=False):
        import re
        import zlib
        if isinstance(texts, str):
            texts = [texts]
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                out[i, zlib.crc32(w.encode()) % self.dim] += 1.0
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)

    def __call__(self, text):
        return self.encode(text)[0]

    def doc(self, text):
        return self.encode(text)[0]


def load(models, threads=1):
    try:
        return Embedder(os.path.join(models, "bge-small"), threads=threads)
    except Exception:
        return HashEmbedder()
