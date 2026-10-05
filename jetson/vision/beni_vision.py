#!/usr/bin/env python3
"""beni_vision: DeepStream 6.0.1 bring-up/fallback vision process (§5.4). Host Python 3.6 + pyds 1.1.1.

No dataclasses / walrus / f-string '=' here (py3.6). Publishes the same ZMQ schema as the C++ vision_core:
  PUB  vision.sock       b"det"  {v,t,src,frames:[{cam,fn,ts,objs:[{tid,cls,gie,conf,bbox,emb?}]}]}
                         b"jpeg" {v,t,src,cam,req,jpeg,objs}
  REP  vision_ctrl.sock  {op: snapshot|bitrate|idr|interval|ping}

Pipeline (both cameras 720p60, NVMM end to end, CPU never touches pixels):
  cam0 -> tee -> mux(512x288,b2) -> YOLO26n(PGIE) -> NvDCF -> face det(SGIE) -> face emb(SGIE) -> probe -> fakesink
       -> videorate 30 -> H.265 -> splitmuxsink         (recording, before the mux so it keeps 720p)
       -> valve -> nvjpegenc -> appsink                  (one-shot snapshot for the VLM)
       -> 960x540 H.264 -> RTP/MPEG-TS udp:5000          (teleop, MediaMTX re-serves as WebRTC)
  cam1 -> tee -> mux / recording
"""
import argparse
import ctypes
import os
import sys
import threading
import time

import numpy as np
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import GLib, Gst, GstVideo  # noqa: E402
import zmq  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))
from beni_common import schemas as S  # noqa: E402

DS_LIB = "/opt/nvidia/deepstream/deepstream-6.0/lib"
TRACKER_CFG = "/opt/nvidia/deepstream/deepstream-6.0/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"
FACE_DET_GIE, FACE_EMB_GIE = 2, 3
DETECTORS = ("yolo26n", "yolov8n")        # §6.4: YOLO26n primary, YOLOv8n fallback


def model_files(conf):
    """pgie config -> the onnx-file / model-engine-file paths it names."""
    out = []
    with open(conf) as f:
        for ln in f:
            k, _, v = ln.partition("=")
            if k.strip() in ("onnx-file", "model-engine-file"):
                out.append(v.split("#")[0].strip())
    return out


def pgie_config(cfg_dir, detector):
    """§6.4.3: the chosen detector's config, or the next one whose ONNX/engine exists (DeepStream needs one)."""
    order = [detector] + [d for d in DETECTORS if d != detector]
    for d in order:
        conf = os.path.join(cfg_dir, "pgie_%s.txt" % d)
        if os.path.exists(conf) and any(os.path.exists(p) for p in model_files(conf)):
            if d != detector:
                sys.stderr.write("[vision] no %s model, falling back to %s\n" % (detector, d))
            return conf
    return os.path.join(cfg_dir, "pgie_%s.txt" % detector)     # let DeepStream report what is missing


def build_pipeline(a):
    cfg = a.config
    cams = []
    for i in range(a.cams):
        c = ("nvarguscamerasrc sensor-id={i} bufapi-version=1 aeantibanding=2 tnr-mode=1 ee-mode=0 ! "
             "video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1,format=NV12 ! tee name=t{i}\n"
             "t{i}. ! queue max-size-buffers=4 leaky=2 ! mux.sink_{i}\n").format(i=i)
        if a.rec:
            c += ("t{i}. ! queue max-size-buffers=8 ! videorate drop-only=true ! "
                  "video/x-raw(memory:NVMM),framerate=30/1 ! nvv4l2h265enc bitrate={br} iframeinterval=60 "
                  "insert-sps-pps=1 maxperf-enable=1 ! h265parse ! splitmuxsink "
                  "location={d}/cam{i}_%05d.mkv muxer=matroskamux max-size-time=300000000000\n").format(
                      i=i, br=4000000, d=a.rec_dir)
        cams.append(c)
    cams.append(
        "t0. ! queue max-size-buffers=2 leaky=2 ! valve name=snapvalve drop=true ! nvvideoconvert ! "
        "video/x-raw(memory:NVMM),format=I420,width=1024,height=576 ! nvjpegenc quality=85 ! "
        "appsink name=snapsink emit-signals=true max-buffers=1 drop=true sync=false\n")
    if a.teleop:
        cams.append(
            "t0. ! queue max-size-buffers=2 leaky=2 ! videorate drop-only=true ! "
            "video/x-raw(memory:NVMM),framerate=30/1 ! nvvideoconvert ! "
            "video/x-raw(memory:NVMM),format=I420,width=960,height=540 ! nvv4l2h264enc name=teleopenc control-rate=1 "
            "bitrate=2000000 preset-level=1 profile=0 insert-sps-pps=1 idrinterval=30 maxperf-enable=1 poc-type=2 ! "
            "h264parse config-interval=-1 ! mpegtsmux alignment=7 ! udpsink host=127.0.0.1 port=5000 sync=false "
            "async=false\n")
    head = ("nvstreammux name=mux batch-size={n} width=512 height=288 live-source=1 batched-push-timeout=33000 "
            "enable-padding=1 ! ").format(n=a.cams)
    if a.infer:
        head += ("nvinfer name=pgie config-file-path={pgie} batch-size={n} ! "
                 "nvtracker ll-lib-file={lib}/libnvds_nvmultiobjecttracker.so ll-config-file={trk} "
                 "tracker-width=512 tracker-height=288 enable-batch-process=1 ! "
                 "nvinfer config-file-path={c}/sgie_face_det.txt ! "
                 "nvinfer config-file-path={c}/sgie_face_embed.txt ! ").format(
                     c=cfg, n=a.cams, lib=DS_LIB, trk=TRACKER_CFG, pgie=pgie_config(cfg, a.detector))
    head += "identity name=probe_here ! fakesink sync=false async=false\n"
    return head + "".join(cams)


def embedding(obj_meta, pyds):
    l_user = obj_meta.obj_user_meta_list
    while l_user is not None:
        um = pyds.NvDsUserMeta.cast(l_user.data)
        if um.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
            tm = pyds.NvDsInferTensorMeta.cast(um.user_meta_data)
            layer = pyds.get_nvds_LayerInfo(tm, 0)
            n = layer.inferDims.numElements
            ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
            v = np.ctypeslib.as_array(ptr, shape=(n,))
            nrm = float(np.linalg.norm(v))
            return (v / nrm).astype(np.float16).tobytes() if nrm > 0 else None
        l_user = l_user.next
    return None


class Vision(object):
    def __init__(self, a):
        self.a = a
        ctx = zmq.Context.instance()
        os.makedirs(S.SOCK_DIR, exist_ok=True)
        self.pub = ctx.socket(zmq.PUB)
        self.pub.setsockopt(zmq.SNDHWM, 20)
        self.pub.bind(S.EP["vision"])
        self.rep = ctx.socket(zmq.REP)
        self.rep.bind(S.EP["vision_ctrl"])
        self.last_objs = {}           # cam -> last objs (attached to snapshots for grounding)
        self.snap_req = None
        self.pipeline = None

    def publish(self, topic, msg):
        try:
            self.pub.send_multipart([topic, S.pack(msg)], flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    # ------------------------------------------------------------------ probes / callbacks
    def det_probe(self, pad, info, _):
        import pyds
        buf = info.get_buffer()
        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        frames = []
        l_frame = batch.frame_meta_list
        while l_frame is not None:
            fm = pyds.NvDsFrameMeta.cast(l_frame.data)
            objs = []
            l_obj = fm.obj_meta_list
            while l_obj is not None:
                om = pyds.NvDsObjectMeta.cast(l_obj.data)
                r = om.rect_params
                d = {"tid": int(om.object_id), "cls": int(om.class_id), "gie": int(om.unique_component_id),
                     "conf": round(float(om.confidence), 3),
                     "bbox": [round(r.left, 1), round(r.top, 1), round(r.width, 1), round(r.height, 1)]}
                if om.unique_component_id == FACE_DET_GIE:
                    d["parent"] = int(om.parent.object_id) if om.parent is not None else -1
                    e = embedding(om, pyds)
                    if e is not None:
                        d["emb"] = e
                objs.append(d)
                l_obj = l_obj.next
            cam = int(fm.source_id)
            self.last_objs[cam] = objs
            frames.append(S.det_frame(cam, int(fm.frame_num), int(fm.buf_pts), objs))
            l_frame = l_frame.next
        if any(f["objs"] for f in frames):   # empty scenes publish nothing: the agent treats silence as "nobody"
            self.publish(S.T_DET, S.envelope("vision", frames=frames))
        return Gst.PadProbeReturn.OK

    def on_jpeg(self, sink):
        sample = sink.emit("pull-sample")
        self.valve.set_property("drop", True)             # one frame per request
        buf = sample.get_buffer()
        ok, mi = buf.map(Gst.MapFlags.READ)
        if ok:
            try:
                self.publish(S.T_JPEG, S.envelope("vision", cam=0, req=self.snap_req, jpeg=bytes(mi.data),
                                                  objs=[{k: o[k] for k in ("tid", "cls", "conf", "bbox")}
                                                        for o in self.last_objs.get(0, ())]))
            finally:
                buf.unmap(mi)
        return Gst.FlowReturn.OK

    # ------------------------------------------------------------------ control
    def ctrl_loop(self):
        while True:
            try:
                msg = S.unpack(self.rep.recv())
                op = msg.get("op")
                if op == "snapshot":
                    self.snap_req = msg.get("req")
                    GLib.idle_add(self.valve.set_property, "drop", False)
                    res = {"ok": True}
                elif op == "bitrate" and self.teleop_enc is not None:
                    GLib.idle_add(self.teleop_enc.set_property, "bitrate", int(msg["bps"]))
                    res = {"ok": True}
                elif op == "idr" and self.teleop_enc is not None:   # §5.3 viewer joined: keyframe now
                    GLib.idle_add(self.force_idr)
                    res = {"ok": True}
                elif op == "interval" and self.pgie is not None:   # power modes: detector every N+1 frames
                    GLib.idle_add(self.pgie.set_property, "interval", int(msg["n"]))
                    res = {"ok": True}
                elif op == "ping":
                    res = {"ok": True, "infer": self.a.infer}
                else:
                    res = {"ok": False, "err": "unknown op"}
            except Exception as e:   # never leave REP in a bad state
                res = {"ok": False, "err": str(e)}
            self.rep.send(S.pack(res))

    def force_idr(self):
        ev = GstVideo.video_event_new_upstream_force_key_unit(Gst.CLOCK_TIME_NONE, True, 0)
        self.teleop_enc.get_static_pad("src").send_event(ev)
        return False

    def run(self):
        Gst.init(None)
        self.pipeline = Gst.parse_launch(build_pipeline(self.a))
        self.valve = self.pipeline.get_by_name("snapvalve")
        self.teleop_enc = self.pipeline.get_by_name("teleopenc")
        self.pgie = self.pipeline.get_by_name("pgie")
        self.pipeline.get_by_name("snapsink").connect("new-sample", self.on_jpeg)
        if self.a.infer:
            self.pipeline.get_by_name("probe_here").get_static_pad("src").add_probe(
                Gst.PadProbeType.BUFFER, self.det_probe, 0)
        t = threading.Thread(target=self.ctrl_loop)
        t.daemon = True
        t.start()
        loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()

        def on_err(_b, m):
            err, dbg = m.parse_error()
            sys.stderr.write("GST ERROR: %s\n%s\n" % (err, dbg))
            loop.quit()
        bus.connect("message::error", on_err)
        bus.connect("message::eos", lambda *_: loop.quit())
        self.pipeline.set_state(Gst.State.PLAYING)
        rc = 0
        try:
            loop.run()
            rc = 1     # the pipeline only stops on error; systemd restarts us
        except KeyboardInterrupt:
            pass
        finally:
            self.pipeline.send_event(Gst.Event.new_eos())   # finalise the open recording segment
            time.sleep(1)
            self.pipeline.set_state(Gst.State.NULL)
        return rc


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=os.path.join(here, "configs"))
    p.add_argument("--detector", choices=DETECTORS, default=os.environ.get("BENI_DETECTOR", "yolo26n").split("_")[0],
                   help="PGIE model (§6.4); falls back to the other if its ONNX/engine is missing")
    p.add_argument("--cams", type=int, default=2, choices=(1, 2))
    p.add_argument("--no-infer", dest="infer", action="store_false", help="cameras/record/teleop only")
    p.add_argument("--no-rec", dest="rec", action="store_false")
    p.add_argument("--no-teleop", dest="teleop", action="store_false")
    p.add_argument("--rec-dir", default="/ssd/beni/rec")
    p.add_argument("--print", action="store_true", help="print the pipeline and exit")
    a = p.parse_args(argv)
    if a.print:
        print(build_pipeline(a))
        return 0
    if a.rec:
        os.makedirs(a.rec_dir, exist_ok=True)
    return Vision(a).run()


if __name__ == "__main__":
    sys.exit(main())
