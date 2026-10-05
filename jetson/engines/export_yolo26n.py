"""§6.4.2 YOLO26n -> ONNX for vision_core / trtexec. Run on a PC or Kaggle (ultralytics, py>=3.8), never the Nano.

    python export_yolo26n.py [weights.pt]            ->  yolo26n_512x288_b2.onnx  (copy to /ssd/beni/models, then
    python export_yolo26n.py [weights.pt] 224 416    ->  yolo26n_416x224_b2.onnx   make engines; 416 = light mode)
"""
import os
import sys

BAD_OPS = {"NonZero", "ScatterND", "LayerNormalization", "If", "Loop"}  # red flags for the TRT 8.2 parser


def main():
    from ultralytics import YOLO
    import onnx

    weights = sys.argv[1] if len(sys.argv) > 1 else "yolo26n.pt"
    h, w = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (288, 512)
    opset = int(os.environ.get("OPSET", "12"))                               # 13 if 12 fails; never > 13 (TRT 8.2)
    out = YOLO(weights).export(format="onnx", opset=opset, imgsz=(h, w), batch=2, dynamic=False, simplify=True,
                               half=False, nms=False)
    m = onnx.load(out)
    ops = {n.op_type for n in m.graph.node}
    shape = [d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim]
    print("opset", [o.version for o in m.opset_import], "output", m.graph.output[0].name, shape)
    if ops & BAD_OPS:
        print("WARNING: ops the TRT 8.2 parser may reject:", sorted(ops & BAD_OPS), "(try opset=13 or §6.4.3)")
    if len(shape) != 3 or shape[0] != 2 or shape[2] != 6:
        sys.exit("unexpected output %s: vision_core wants [2, 300, 6] (x1 y1 x2 y2 score class)" % shape)
    dst = "yolo26n_%dx%d_b2.onnx" % (w, h)
    os.replace(out, dst)
    print("ok ->", dst)


if __name__ == "__main__":
    main()
