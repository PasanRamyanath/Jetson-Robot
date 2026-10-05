"""Brain configuration from the environment (Kaggle Secrets are exported to env by brain_notebook.py)."""
import os
from dataclasses import dataclass, field


def _env(k, d=None):
    v = os.environ.get(k)
    return d if v in (None, "") else v


@dataclass
class Config:
    token: str = ""
    host: str = "0.0.0.0"
    port: int = 8765
    run_dir: str = "/kaggle/working/beni"
    db: str = "/tmp/beni/memory.db"                  # ephemeral mirror; the Jetson owns the system of record
    llm_url: str = "http://127.0.0.1:8000/v1"
    llm_model: str = "beni-llm"
    stt_model: str = ""                              # faster-whisper dir (large-v3-turbo); "" -> stub
    stt_device: str = "cuda"
    tts: str = "kokoro"                              # kokoro | cosyvoice | stub
    tts_voice: str = "af_heart"
    tts_prompt_wav: str = ""                         # CosyVoice2: Beni's reference voice (3-10 s, 16 kHz+ wav)
    cosyvoice_dir: str = ""                          # CosyVoice repo checkout (has cosyvoice/ and third_party/)
    cosyvoice_model: str = ""                        # CosyVoice2-0.5B snapshot dir
    vision: str = "microsoft/Florence-2-large"      # HF repo or local dir; "none" disables the locate tool
    embed_dir: str = ""                              # bge-small onnx dir (same export as the Jetson)
    reranker: str = "BAAI/bge-reranker-v2-m3"       # §11.6 cross-encoder over the top-24; "none" disables
    home_city: str = "Colombo"
    persona: str = "curious, warm, playful, a little cheeky; loves learning about the family"
    stub: bool = False                               # tests / dry runs: stub STT, TTS and LLM
    max_tool_rounds: int = 3
    speculate: bool = True                           # §8.6 #1: start the LLM on a stable STT partial
    action_timeout_s: float = 20.0
    history_turns: int = 8
    idle_consolidate_s: float = 600.0
    models: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls, **over):
        stub = _env("BENI_STUB", "0") == "1"
        c = cls(
            token=_env("BENI_TOKEN", ""),
            host=_env("BENI_HOST", "0.0.0.0"),
            port=int(_env("BENI_PORT", "8765")),
            run_dir=_env("BENI_RUN_DIR", "/kaggle/working/beni"),
            db=_env("BENI_DB", "/tmp/beni/memory.db"),
            llm_url=_env("BENI_LLM_URL", "http://127.0.0.1:8000/v1"),
            llm_model=_env("BENI_LLM_MODEL", "beni-llm"),
            stt_model="" if stub else _env("BENI_STT_MODEL", ""),
            stt_device=_env("BENI_STT_DEVICE", "cuda"),
            tts="stub" if stub else _env("BENI_TTS", "kokoro"),
            tts_voice=_env("BENI_TTS_VOICE", "af_heart"),
            tts_prompt_wav=_env("BENI_TTS_PROMPT_WAV", ""),
            cosyvoice_dir=_env("BENI_COSYVOICE_DIR", ""),
            cosyvoice_model=_env("BENI_COSYVOICE_MODEL", ""),
            vision=_env("BENI_VISION", "microsoft/Florence-2-large"),
            embed_dir=_env("BENI_EMBED_DIR", ""),
            reranker="none" if stub else _env("BENI_RERANKER", "BAAI/bge-reranker-v2-m3"),
            home_city=_env("BENI_HOME_CITY", "Colombo"),
            persona=_env("BENI_PERSONA", cls.persona),
            stub=stub,
        )
        for k, v in over.items():
            setattr(c, k, v)
        return c
