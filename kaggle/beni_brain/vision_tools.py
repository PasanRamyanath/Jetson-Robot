"""Specialised vision on GPU1 (§10.2, §10.7): Florence-2 caption / open-vocabulary detection / phrase grounding.

The VLM stays the primary grounder; this is the fast fallback the `locate` tool uses, plus dense captions for sleep
replay (§11.9). Boxes are pixel [x1, y1, x2, y2] in the snapshot; `bearing` turns one into head angles.
"""
import io
import logging
import math
import threading

log = logging.getLogger("vision")

TASKS = {"caption": "<MORE_DETAILED_CAPTION>", "od": "<OD>", "detect": "<OPEN_VOCABULARY_DETECTION>",
         "ground": "<CAPTION_TO_PHRASE_GROUNDING>"}
HFOV_DEG = (62.2, 62.2)                              # IMX219 V2 / NoIR, same as the bridge's hfov_deg (§5.1)


def bearing(box, w, h, cam=0):
    """Pixel box -> (dpan, dtilt) degrees from the optical axis; left and up are positive (bridge convention)."""
    fx = 0.5 * w / math.tan(math.radians(HFOV_DEG[min(cam, len(HFOV_DEG) - 1)]) / 2)
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    return -math.degrees(math.atan((cx - w / 2.0) / fx)), -math.degrees(math.atan((cy - h / 2.0) / fx))


def side(dpan):
    return "straight ahead" if abs(dpan) < 8 else ("to the left" if dpan > 0 else "to the right")


def boxes(res):
    """Florence post-processed dict -> [(label, [x1, y1, x2, y2])], largest first."""
    labels = res.get("labels") or res.get("bboxes_labels") or []
    out = [(str(lb).strip() or "object", [float(v) for v in b]) for b, lb in zip(res.get("bboxes") or [], labels)]
    return sorted(out, key=lambda o: -(o[1][2] - o[1][0]) * (o[1][3] - o[1][1]))


class Florence:
    """microsoft/Florence-2-large (0.77B) FP16, greedy decoding. ~80-200 ms per call on a T4."""

    def __init__(self, repo="microsoft/Florence-2-large", device="cuda"):
        import torch
        from unittest import mock
        from transformers import AutoModelForCausalLM, AutoProcessor
        from transformers.dynamic_module_utils import get_imports

        def no_flash(path):                          # the remote code imports flash_attn, which T4 can't use
            return [i for i in get_imports(path) if i != "flash_attn"]

        self.torch, self.device = torch, device
        with mock.patch("transformers.dynamic_module_utils.get_imports", no_flash):
            self.model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, torch_dtype=torch.float16,
                                                              attn_implementation="eager").to(device).eval()
        self.proc = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
        self.lock = threading.Lock()

    def run(self, jpeg, task, text=""):
        from PIL import Image
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        tag = TASKS[task]
        with self.lock, self.torch.inference_mode():
            x = self.proc(text=tag + text, images=img, return_tensors="pt").to(self.device, self.torch.float16)
            ids = self.model.generate(input_ids=x["input_ids"], pixel_values=x["pixel_values"], max_new_tokens=256,
                                      num_beams=1, do_sample=False)
        raw = self.proc.batch_decode(ids, skip_special_tokens=False)[0]
        return self.proc.post_process_generation(raw, task=tag, image_size=img.size)[tag], img.size

    def caption(self, jpeg):
        return str(self.run(jpeg, "caption")[0]).strip()

    def find(self, jpeg, phrase):
        """-> ([(label, box)], (w, h)); open-vocabulary detection, falling back to phrase grounding."""
        res, size = self.run(jpeg, "detect", phrase)
        found = boxes(res)
        if not found:
            res, size = self.run(jpeg, "ground", phrase)
            found = boxes(res)
        return found, size


class StubVision:
    """No model: nothing is ever found (keeps the tool path testable without a GPU)."""

    def caption(self, jpeg):
        return "an indoor room"

    def find(self, jpeg, phrase):
        return [], (1280, 720)


class Lazy:
    """Loads the real model in a background thread so the gateway comes up at once; calls wait up to `wait` s."""

    def __init__(self, factory, wait=60.0):
        self.impl, self.err, self.wait = None, None, wait
        self.ready = threading.Event()
        threading.Thread(target=self._load, args=(factory,), daemon=True, name="vision-load").start()

    def _load(self, factory):
        try:
            self.impl = factory()
        except Exception as e:                       # vision is optional: the VLM still sees snapshots
            log.exception("vision model failed to load")
            self.err = e
        self.ready.set()

    def _get(self):
        if not self.ready.wait(self.wait) or self.impl is None:
            raise RuntimeError("vision model unavailable (%s)" % (self.err or "still loading"))
        return self.impl

    def caption(self, jpeg):
        return self._get().caption(jpeg)

    def find(self, jpeg, phrase):
        return self._get().find(jpeg, phrase)


def load(cfg):
    if cfg.stub or cfg.vision == "none":
        return StubVision()
    return Lazy(lambda: Florence(cfg.vision))
