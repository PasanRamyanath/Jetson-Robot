"""§4 item 39 TRT-Pose ResNet18 -> ONNX for vision_core / trtexec. Run on a PC or Kaggle (torch + trt_pose).

    pip install git+https://github.com/NVIDIA-AI-IOT/trt_pose   (the models only; no torch2trt needed)
    python export_trtpose.py resnet18_baseline_att_224x224_A_epoch_249.pth   ->  trtpose_r18_224_b4.onnx

Static batch 4 (vision_core poses at most 4 people per camera). Outputs `cmap` [4,18,56,56] and `paf` [4,42,56,56];
vision_core reads only `cmap`. Copy the .onnx to /ssd/beni/models, then run build_all.sh.
"""
import sys

BAD_OPS = {"NonZero", "ScatterND", "LayerNormalization", "If", "Loop"}


def main():
    import onnx
    import torch
    import trt_pose.models

    weights = sys.argv[1] if len(sys.argv) > 1 else "resnet18_baseline_att_224x224_A_epoch_249.pth"
    model = trt_pose.models.resnet18_baseline_att(18, 2 * 21).eval()   # human_pose.json: 18 parts, 21 links
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    dst = "trtpose_r18_224_b4.onnx"
    torch.onnx.export(model, torch.zeros(4, 3, 224, 224), dst, opset_version=11, input_names=["input"],
                      output_names=["cmap", "paf"], do_constant_folding=True)
    m = onnx.load(dst)
    ops = {n.op_type for n in m.graph.node}
    shapes = {o.name: [d.dim_value for d in o.type.tensor_type.shape.dim] for o in m.graph.output}
    print("outputs", shapes)
    if ops & BAD_OPS:
        print("WARNING: ops the TRT 8.2 parser may reject:", sorted(ops & BAD_OPS))
    if shapes.get("cmap") != [4, 18, 56, 56]:
        sys.exit("unexpected cmap shape: vision_core wants [4, 18, H/4, W/4]")
    print("ok ->", dst)


if __name__ == "__main__":
    main()
