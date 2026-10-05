"""§6.4 detector selection: YOLO26n primary, YOLOv8n fallback, same names across configs, build and services."""
import os
import re
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CFG = os.path.join(ROOT, "jetson", "vision", "configs")


def _vision(monkeypatch):
    """Import beni_vision without DeepStream: stub gi and zmq just for the import."""
    gi = types.ModuleType("gi")
    gi.require_version = lambda *a: None
    repo = types.ModuleType("gi.repository")
    repo.GLib = repo.Gst = repo.GstVideo = object()
    gi.repository = repo
    for name, mod in (("gi", gi), ("gi.repository", repo), ("zmq", types.ModuleType("zmq"))):
        monkeypatch.setitem(sys.modules, name, mod)
    import importlib.util
    path = os.path.join(ROOT, "jetson", "vision", "beni_vision.py")
    spec = importlib.util.spec_from_file_location("beni_vision", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _prop(path, key):
    with open(path) as f:
        for ln in f:
            if ln.startswith(key + "="):
                return ln.split("=", 1)[1].strip()
    return None


def test_pgie_configs_follow_5_4_1():
    y26, v8 = os.path.join(CFG, "pgie_yolo26n.txt"), os.path.join(CFG, "pgie_yolov8n.txt")
    assert _prop(y26, "cluster-mode") == "4" and _prop(v8, "cluster-mode") == "2"     # NMS-free vs NMS
    with open(y26) as f:
        assert "nms-iou-threshold" not in f.read()
    assert _prop(v8, "nms-iou-threshold") == "0.45"
    for key in ("batch-size", "network-mode", "interval", "workspace-size", "filter-out-class-ids",
                "pre-cluster-threshold", "custom-lib-path", "parse-bbox-func-name"):
        assert _prop(y26, key) == _prop(v8, key), key
    assert _prop(y26, "workspace-size") == "1024"


def test_engine_names_match_build_all():
    with open(os.path.join(ROOT, "jetson", "engines", "build_all.sh")) as f:
        built = dict(re.findall(r"^build (\S+)\s+(\S+)", f.read(), re.M))
    for conf in ("pgie_yolo26n.txt", "pgie_yolov8n.txt"):
        onnx = os.path.basename(_prop(os.path.join(CFG, conf), "onnx-file"))
        eng = os.path.basename(_prop(os.path.join(CFG, conf), "model-engine-file"))
        assert built.get(onnx) == eng, conf
    with open(os.path.join(ROOT, "jetson", "vision_core", "src", "main.cpp")) as f:
        src = f.read()
    for eng in ('"yolo26n_512x288_b2_fp16.engine"', '"yolov8n_512x288_b2_fp16.engine"'):
        assert eng in src and eng.strip('"') in built.values()


def test_fallback_to_the_model_that_exists(monkeypatch, tmp_path):
    V = _vision(monkeypatch)
    for d in V.DETECTORS:
        with open(str(tmp_path / ("pgie_%s.txt" % d)), "w") as f:
            f.write("[property]\nonnx-file=%s/%s.onnx\nmodel-engine-file=%s/%s.engine  # c\n"
                    % (tmp_path, d, tmp_path, d))
    assert V.model_files(str(tmp_path / "pgie_yolo26n.txt")) == ["%s/yolo26n.onnx" % tmp_path,
                                                                "%s/yolo26n.engine" % tmp_path]
    assert V.pgie_config(str(tmp_path), "yolo26n").endswith("pgie_yolo26n.txt")      # nothing built yet
    open(str(tmp_path / "yolov8n.engine"), "w").close()
    assert V.pgie_config(str(tmp_path), "yolo26n").endswith("pgie_yolov8n.txt")
    open(str(tmp_path / "yolo26n.onnx"), "w").close()
    assert V.pgie_config(str(tmp_path), "yolo26n").endswith("pgie_yolo26n.txt")
    assert V.pgie_config(str(tmp_path), "yolov8n").endswith("pgie_yolov8n.txt")
