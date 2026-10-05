"""shared/schemas.hpp (C++ msgpack) <-> beni_common.schemas (Python msgpack) round trip. Skips without g++."""
import os
import shutil
import subprocess

import pytest

from beni_common import schemas as S

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GXX = shutil.which("g++")


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    if not GXX:
        pytest.skip("g++ not available")
    out = str(tmp_path_factory.mktemp("cpp") / "rt.exe")
    src = os.path.join(ROOT, "tests", "contract", "cpp", "schemas_roundtrip.cpp")
    subprocess.check_call([GXX, "-std=c++14", "-O1", "-Wall", "-Werror", "-I", os.path.join(ROOT, "shared"),
                           src, "-o", out])
    return out


def test_cpp_writer_decodes_in_python(exe):
    msg = S.validate(S.unpack(subprocess.check_output([exe, "write"])), "det")
    assert msg["v"] == S.V and msg["src"] == "vision" and isinstance(msg["t"], float)
    fr = msg["frames"][0]
    assert (fr["cam"], fr["fn"], fr["ts"]) == (1, 123456, -5000000000)
    o = fr["objs"][0]
    assert (o["tid"], o["cls"], o["gie"], o["conf"]) == (70000, 0, 1, 0.877)
    assert o["bbox"] == [10.3, -3.0, 300.0, 0.5] and o["emb"] == b"\x00\x01\x02\xff"


def test_python_writer_decodes_in_cpp(exe):
    buf = S.pack({"op": "bitrate", "n": 2.5, "neg": -40000, "big": 2 ** 40, "arr": list(range(20)),
                  "long": "x" * 300, "flag": True})
    out = subprocess.run([exe], input=buf, stdout=subprocess.PIPE, check=True).stdout.decode()
    assert out.strip() == "op=bitrate n=2.500 neg=-40000 big=1099511627776 arr=20 s=300 ok=1"


def test_cpp_rejects_truncated(exe):
    buf = S.pack({"op": "x" * 40, "arr": [1, 2, 3]})
    assert subprocess.run([exe], input=buf[:-2], stdout=subprocess.PIPE).returncode == 2
