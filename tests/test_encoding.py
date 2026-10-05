"""Every source file must be UTF-8: Python 3 on the Nano refuses anything else at import."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXT = {".py", ".sh", ".cpp", ".hpp", ".h", ".ino", ".md", ".yml", ".yaml", ".service", ".txt", ".json", ".toml"}


def test_sources_are_utf8():
    bad = []
    for p in ROOT.rglob("*"):
        if p.suffix in EXT and not {".git", "build", "__pycache__", ".pytest_cache"} & set(p.parts):
            try:
                p.read_bytes().decode("utf-8")
            except UnicodeDecodeError:
                bad.append(str(p.relative_to(ROOT)))
    assert not bad
