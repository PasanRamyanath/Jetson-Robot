// §7.4 beni_zmq_bridge: the only door between ROS and the rest of Beni.
//   ZMQ PUB robot_state (10 Hz)  <- TF map->base_link, /odometry/filtered, /battery, /base/flags, joint_states
//   ZMQ REP robot_cmd            -> Nav2 NavigateToPose, /cmd_vel_behaviour, /head/cmd, base/estop
//   ZMQ SUB vision "det"         -> tracks; each box is projected to 2D with the median lidar range along its ray
// Motion ops reply at once ({"ok":true,"result":"started","id":n}); progress/result ride on state.task.
// Everything runs on the node's one executor thread (ZMQ is polled by a 100 Hz timer), so there are no locks.
#include <sys/stat.h>
#include <zmq.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "schemas.hpp"
#include "sensor_msgs/msg/battery_state.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "std_msgs/msg/u_int8.hpp"
#include "slam_toolbox/srv/serialize_pose_graph.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Transform.h"
#include "tf2/utils.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

namespace beni {

namespace {
const char* const kCoco[80] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"};
constexpr int kPerson = 0, kObjGie = 1;
constexpr uint8_t kStEstop = 1, kStBumpL = 2, kStBumpR = 4, kStCliff = 8, kStCharging = 16, kLinkDown = 128;

double mono() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
double wrap(double a) { return std::atan2(std::sin(a), std::cos(a)); }

bool has_word(const std::string& s, const char* w) {  // whole-word match: "up" is not in "cup"
  size_t n = std::strlen(w);
  for (size_t p = s.find(w); p != std::string::npos; p = s.find(w, p + 1))
    if ((p == 0 || !std::isalpha(uint8_t(s[p - 1]))) && (p + n == s.size() || !std::isalpha(uint8_t(s[p + n]))))
      return true;
  return false;
}

// "my phone" -> 67, "teddy bear" -> 77 (longest name wins), "mugs" -> 41. -1: not a class the detector knows.
int class_of(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  int best = -1;
  for (int i = 0; i < 80; ++i)
    if (has_word(s, kCoco[i]) && (best < 0 || std::strlen(kCoco[i]) > std::strlen(kCoco[best]))) best = i;
  if (best >= 0) return best;
  static const std::pair<const char*, int> alias[] = {
      {"people", 0}, {"someone", 0}, {"man", 0}, {"woman", 0}, {"child", 0}, {"kid", 0}, {"phone", 67},
      {"mobile", 67}, {"mug", 41}, {"glass", 40}, {"sofa", 57}, {"television", 62}, {"table", 60},
      {"plant", 58}, {"bag", 26}, {"ball", 32}, {"teddy", 77}, {"puppy", 16}, {"kitten", 15}, {"bike", 1}};
  for (auto& a : alias)
    if (has_word(s, a.first)) return a.second;
  if (s.size() > 3 && s.back() == 's') return class_of(s.substr(0, s.size() - 1));
  return -1;
}
}  // namespace

class Bridge : public rclcpp::Node {
  using Twist = geometry_msgs::msg::Twist;
  using NavTo = nav2_msgs::action::NavigateToPose;
  using NavGH = rclcpp_action::ClientGoalHandle<NavTo>;

 public:
  explicit Bridge(const rclcpp::NodeOptions& opt) : Node("zmq_bridge", opt) {
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
    map_id_ = declare_parameter<std::string>("map_id", "home");
    cam_frames_ = declare_parameter<std::vector<std::string>>(
        "cam_frames", std::vector<std::string>{"cam0_optical_frame", "cam1_optical_frame"});
    auto hfov = declare_parameter<std::vector<double>>("hfov_deg", std::vector<double>{62.2, 62.2});
    det_w_ = declare_parameter("det_w", 512.0);
    det_h_ = declare_parameter("det_h", 288.0);
    for (double h : hfov) fx_.push_back(0.5 * det_w_ / std::tan(0.5 * h * M_PI / 180.0));
    v_max_ = declare_parameter("v_max", 0.4);
    w_max_ = declare_parameter("w_max", 1.2);
    kv_ = declare_parameter("follow.kv", 0.8);
    kw_ = declare_parameter("follow.kw", 1.8);
    follow_dist_ = declare_parameter("follow.distance", 1.0);
    lost_nav_s_ = declare_parameter("follow.lost_nav_s", 1.0);
    lost_give_up_s_ = declare_parameter("follow.lost_give_up_s", 15.0);
    follow_max_s_ = declare_parameter("follow.max_s", 600.0);
    stop_dist_ = declare_parameter("front_stop", 0.3);
    search_w_ = declare_parameter("search.w", 0.6);
    pan_max_ = declare_parameter("head.pan_max", 1.4);
    look_hold_s_ = declare_parameter("look.hold_s", 8.0);
    dock_ = {declare_parameter("dock.x", std::nan("")), declare_parameter("dock.y", std::nan("")),
             declare_parameter("dock.yaw", 0.0)};
    dock_speed_ = declare_parameter("dock.speed", 0.06);
    dock_travel_ = declare_parameter("dock.travel", 0.45);
    map_path_ = declare_parameter<std::string>("map_path", "/maps/home");
    map_autosave_s_ = declare_parameter("map_autosave_s", 600.0);

    tf_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tfl_ = std::make_shared<tf2_ros::TransformListener>(*tf_, this, false);
    auto sensor = rclcpp::SensorDataQoS();
    auto latched = rclcpp::QoS(1).transient_local();
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "odometry/filtered", sensor, [this](nav_msgs::msg::Odometry::ConstSharedPtr m) {
          v_ = m->twist.twist.linear.x, w_ = m->twist.twist.angular.z;
          double yaw = tf2::getYaw(m->pose.pose.orientation);
          if (!std::isnan(odom_yaw_)) yaw_travel_ += wrap(yaw - odom_yaw_);
          odom_yaw_ = yaw, odom_x_ = m->pose.pose.position.x, odom_y_ = m->pose.pose.position.y;
          odom_t_ = mono();
        });
    batt_sub_ = create_subscription<sensor_msgs::msg::BatteryState>(
        "battery", latched, [this](sensor_msgs::msg::BatteryState::ConstSharedPtr m) {
          batt_pct_ = std::isnan(m->percentage) ? -1 : int(std::lround(m->percentage * 100));
          batt_v_ = m->voltage;
        });
    flags_sub_ = create_subscription<std_msgs::msg::UInt8>(
        "base/flags", latched, [this](std_msgs::msg::UInt8::ConstSharedPtr m) { flags_ = m->data; });
    js_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        "joint_states", rclcpp::QoS(5), [this](sensor_msgs::msg::JointState::ConstSharedPtr m) {
          for (size_t i = 0; i < m->name.size() && i < m->position.size(); ++i) {
            if (m->name[i] == "head_pan_joint") pan_ = m->position[i];
            else if (m->name[i] == "head_tilt_joint") tilt_ = m->position[i];
          }
        });
    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
        "scan", sensor, [this](sensor_msgs::msg::LaserScan::ConstSharedPtr m) { on_scan(*m); });
    cmd_pub_ = create_publisher<Twist>("cmd_vel_behaviour", rclcpp::QoS(1));
    teleop_pub_ = create_publisher<Twist>("cmd_vel_teleop", rclcpp::QoS(1));
    head_pub_ = create_publisher<geometry_msgs::msg::Vector3>("head/cmd", rclcpp::QoS(1));
    objs_pub_ = create_publisher<geometry_msgs::msg::PoseArray>("beni/objects", rclcpp::QoS(1));
    nav_ = rclcpp_action::create_client<NavTo>(this, "navigate_to_pose");
    estop_ = create_client<std_srvs::srv::SetBool>("base/estop");
    save_map_ = create_client<slam_toolbox::srv::SerializePoseGraph>("slam_toolbox/serialize_map");

    open_zmq();
    zmq_timer_ = create_wall_timer(std::chrono::milliseconds(10), [this] { poll_zmq(); });
    ctl_timer_ = create_wall_timer(std::chrono::milliseconds(50), [this] { control(); });
    state_timer_ = create_wall_timer(std::chrono::milliseconds(100), [this] { publish_state(); });
    map_timer_ = create_wall_timer(std::chrono::seconds(5), [this] { housekeeping(); });
  }

  ~Bridge() override {
    for (void* s : {pub_, rep_, sub_})
      if (s) zmq_close(s);
    if (ctx_) zmq_ctx_term(ctx_);
  }

 private:
  // ------------------------------------------------------------------ model
  struct Track {
    int cam = 0, tid = 0, cls = 0, gie = 1, parent = -1;
    double conf = 0, u = 0, v = 0, w = 0, h = 0, t = 0;
    double bearing = 0, range = -1, mx = 0, my = 0, xy_t = -1;  // base-frame bearing; map (or odom) xy
  };
  struct Task {
    int id = 0;
    std::string op = "none", status = "idle", detail;
    double t0 = 0, t_end = 0;
    int phase = 0, attempt = 0, recoveries = 0;
    std::vector<std::pair<double, double>> stuck;
    // targets
    int cam = -1, tid = -1, cls = -1;
    double gx = 0, gy = 0, gyaw = 0, ax = 0, ay = 0, t1 = 0, lost_t = -1, lx = NAN, ly = NAN, mark = 0, near_t = -1;
    bool nav_sent = false;
    std::vector<int> seen;
  };
  struct Pose {
    double x = NAN, y = NAN, yaw = 0;
    bool ok() const { return !std::isnan(x); }
  };
  static int64_t key(int cam, int tid) { return (int64_t(cam) << 32) | uint32_t(tid); }

  // ------------------------------------------------------------------ zmq
  void open_zmq() {
    ctx_ = zmq_ctx_new();
    zmq_ctx_set(ctx_, ZMQ_IO_THREADS, 1);
    int linger = 0, hwm = 10, rhwm = 4;
    pub_ = zmq_socket(ctx_, ZMQ_PUB);
    rep_ = zmq_socket(ctx_, ZMQ_REP);
    sub_ = zmq_socket(ctx_, ZMQ_SUB);
    for (void* s : {pub_, rep_, sub_}) zmq_setsockopt(s, ZMQ_LINGER, &linger, sizeof linger);
    zmq_setsockopt(pub_, ZMQ_SNDHWM, &hwm, sizeof hwm);
    zmq_setsockopt(sub_, ZMQ_RCVHWM, &rhwm, sizeof rhwm);
    zmq_setsockopt(sub_, ZMQ_SUBSCRIBE, topic::kDet, std::strlen(topic::kDet));
    bind(pub_, ep::kRobotState);
    bind(rep_, ep::kRobotCmd);
    if (zmq_connect(sub_, ep::kVision) != 0) RCLCPP_ERROR(get_logger(), "connect %s: %s", ep::kVision, zmq_strerror(errno));
  }
  void bind(void* s, const char* ep) {
    if (zmq_bind(s, ep) != 0) {
      RCLCPP_FATAL(get_logger(), "bind %s: %s", ep, zmq_strerror(errno));
      throw std::runtime_error("zmq bind failed");
    }
    if (!std::strncmp(ep, "ipc://", 6)) chmod(ep + 6, 0666);  // the agent runs as a normal user, we run as root
  }

  static bool recv_frame(void* s, std::string& out, bool& more) {
    zmq_msg_t m;
    zmq_msg_init(&m);
    if (zmq_msg_recv(&m, s, ZMQ_DONTWAIT) < 0) return zmq_msg_close(&m), false;
    out.assign(static_cast<const char*>(zmq_msg_data(&m)), zmq_msg_size(&m));
    more = zmq_msg_more(&m);
    zmq_msg_close(&m);
    return true;
  }

  void poll_zmq() {
    std::string a, b;
    bool more = false;
    for (int n = 0; n < 64 && recv_frame(sub_, a, more); ++n) {  // [topic, msgpack]
      if (!more) continue;
      recv_frame(sub_, b, more);
      while (more && recv_frame(sub_, a, more)) {
      }
      Value msg;
      if (unpack(b.data(), b.size(), msg)) on_det(msg);
    }
    for (int n = 0; n < 8 && recv_frame(rep_, a, more); ++n) {
      while (more && recv_frame(rep_, b, more)) {
      }
      Value req;
      std::string rep;
      try {
        rep = unpack(a.data(), a.size(), req) && req.type == Value::MAP ? handle(req) : reply_err("bad request");
      } catch (const std::exception& e) {  // a REP socket must always answer
        rep = reply_err(e.what());
      }
      zmq_send(rep_, rep.data(), rep.size(), 0);
    }
  }

  // ------------------------------------------------------------------ perception
  void on_scan(const sensor_msgs::msg::LaserScan& s) {
    tf2::Transform T;
    if (!lookup(base_frame_, s.header.frame_id, T)) return;
    pts_.clear();
    front_ = INFINITY;
    for (size_t i = 0; i < s.ranges.size(); ++i) {
      float r = s.ranges[i];
      if (!std::isfinite(r) || r < s.range_min || r > s.range_max) continue;
      double a = s.angle_min + i * s.angle_increment;
      tf2::Vector3 p = T * tf2::Vector3(r * std::cos(a), r * std::sin(a), 0);
      pts_.push_back({p.x(), p.y()});
      if (p.x() > 0 && std::fabs(p.y()) < 0.17) front_ = std::min(front_, p.x());
    }
    scan_t_ = mono();
  }

  // bbox centre -> base-frame bearing, and the median lidar distance of beams inside the box's central angle.
  void project(Track& tr, const tf2::Transform& T_bc, const Pose& world) {
    double fx = fx_[std::min<size_t>(tr.cam, fx_.size() - 1)];
    tf2::Vector3 d = T_bc.getBasis() * tf2::Vector3((tr.u - 0.5 * det_w_) / fx, (tr.v - 0.5 * det_h_) / fx, 1.0);
    double n = std::hypot(d.x(), d.y());
    if (n < 1e-3) return;
    double dx = d.x() / n, dy = d.y() / n, ox = T_bc.getOrigin().x(), oy = T_bc.getOrigin().y();
    tr.bearing = std::atan2(dy, dx);
    if (mono() - scan_t_ > 0.5) return;
    double tol = std::tan(std::clamp(0.25 * tr.w / fx, 0.017, 0.035));
    hits_.clear();
    for (auto& p : pts_) {
      double vx = p.first - ox, vy = p.second - oy, along = vx * dx + vy * dy;
      if (along > 0.15 && std::fabs(dx * vy - dy * vx) < tol * along) hits_.push_back(along);
    }
    if (hits_.size() < 2) return;
    auto mid = hits_.begin() + hits_.size() / 2;
    std::nth_element(hits_.begin(), mid, hits_.end());
    double bx = ox + *mid * dx, by = oy + *mid * dy;
    tr.range = std::hypot(bx, by), tr.bearing = std::atan2(by, bx);
    if (world.ok()) {
      double c = std::cos(world.yaw), s = std::sin(world.yaw);
      tr.mx = world.x + c * bx - s * by, tr.my = world.y + s * bx + c * by, tr.xy_t = tr.t;
    }
  }

  void on_det(const Value& msg) {
    const Value* frames = msg.get("frames");
    if (!frames || frames->type != Value::ARR) return;
    double now = mono();
    Pose world = pose();
    for (const Value& f : frames->a) {
      int cam = int(f.num("cam"));
      tf2::Transform T_bc;
      bool geo = cam >= 0 && size_t(cam) < cam_frames_.size() && lookup(base_frame_, cam_frames_[cam], T_bc);
      const Value* objs = f.get("objs");
      if (!objs || objs->type != Value::ARR) continue;
      for (const Value& o : objs->a) {
        const Value* bb = o.get("bbox");
        if (!bb || bb->type != Value::ARR || bb->a.size() < 4) continue;
        auto num = [](const Value& x) { return x.type == Value::FLOAT ? x.f : double(x.i); };
        Track& tr = tracks_[key(cam, int(o.num("tid")))];
        tr.cam = cam, tr.tid = int(o.num("tid")), tr.cls = int(o.num("cls")), tr.gie = int(o.num("gie", 1));
        tr.parent = int(o.num("parent", -1)), tr.conf = o.num("conf"), tr.t = now, tr.range = -1;
        tr.w = num(bb->a[2]), tr.h = num(bb->a[3]), tr.u = num(bb->a[0]) + 0.5 * tr.w, tr.v = num(bb->a[1]) + 0.5 * tr.h;
        if (geo) project(tr, T_bc, world);
        if (task_.op == "search" && task_.status == "running" && tr.gie == kObjGie && tr.conf >= 0.4 &&
            std::find(task_.seen.begin(), task_.seen.end(), tr.cls) == task_.seen.end())
          task_.seen.push_back(tr.cls);
      }
    }
    for (auto it = tracks_.begin(); it != tracks_.end();) it = now - it->second.t > 3.0 ? tracks_.erase(it) : ++it;
  }

  // The agent keys people by (cam, parent person tid) and falls back to the face tid: both resolve here.
  Track* find(int cam, int tid, double max_age = 0.6) {
    auto it = tracks_.find(key(cam, tid));
    if (it == tracks_.end() || mono() - it->second.t > max_age) return nullptr;
    return &it->second;
  }
  Track* nearest(int cls, double max_age = 0.6) {
    Track* best = nullptr;
    double now = mono();
    for (auto& kv : tracks_) {
      Track& t = kv.second;
      if (t.gie != kObjGie || t.cls != cls || now - t.t > max_age) continue;
      double score = t.range > 0 ? t.range : 50 - t.h;  // unranged: taller box = closer
      if (!best || score < (best->range > 0 ? best->range : 50 - best->h)) best = &t;
    }
    return best;
  }

  bool lookup(const std::string& target, const std::string& source, tf2::Transform& out) {
    try {
      tf2::fromMsg(tf_->lookupTransform(target, source, tf2::TimePointZero).transform, out);
      return true;
    } catch (const tf2::TransformException&) {
      return false;
    }
  }
  Pose pose() {
    Pose p;
    tf2::Transform T;
    frame_ = map_frame_;
    if (!lookup(map_frame_, base_frame_, T)) {
      frame_ = odom_frame_;
      if (!lookup(odom_frame_, base_frame_, T)) return p;
    }
    p.x = T.getOrigin().x(), p.y = T.getOrigin().y(), p.yaw = tf2::getYaw(T.getRotation());
    return p;
  }

  // ------------------------------------------------------------------ commands
  std::string handle(const Value& r) {
    const std::string op = r.text("op");
    if (op == "ping") return reply_ok();
    if (op == "drive") {  // teleop: resend at >= 5 Hz, the base mux drops it after 0.5 s
      Twist t;
      t.linear.x = std::clamp(r.num("v"), -v_max_, v_max_), t.angular.z = std::clamp(r.num("w"), -w_max_, w_max_);
      teleop_pub_->publish(t);
      return reply_ok();
    }
    if (op == "save_map") return save_map() ? started("saving map") : reply_err("SLAM is not running");
    if (op == "stop") {
      finish("canceled", "stopped");
      head(0, 0, 0, true);
      return started("stopped");
    }
    if (op == "estop") {
      auto q = std::make_shared<std_srvs::srv::SetBool::Request>();
      q->data = r.num("on", 1) != 0;
      if (!estop_->service_is_ready()) return reply_err("base driver not running");
      estop_->async_send_request(q);
      if (q->data) finish("canceled", "e-stop");
      return started(q->data ? "e-stop engaged" : "e-stop released");
    }
    if (op == "save_place") {
      Pose p = pose();
      if (!p.ok()) return reply_err("I don't know where I am");
      MsgWriter w(96);
      w.map(2).key("ok").b(true).key("result").map(5);
      w.key("x").fr(p.x, 3).key("y").fr(p.y, 3).key("yaw").fr(p.yaw, 3).key("map_id").str(map_id_);
      w.key("frame").str(frame_);
      return w.buf;
    }
    if (op == "look_at") return look_at(r);
    // everything below moves the base
    if (flags_ & kStEstop) return reply_err("e-stop is engaged");
    if (flags_ & kLinkDown) return reply_err("my wheels are not responding");
    if (op == "goto") {
      if (!r.has("x") || !r.has("y")) return reply_err("unknown place " + r.text("place", "?"));
      begin(op, r.text("place"));
      task_.gx = r.num("x"), task_.gy = r.num("y"), task_.gyaw = r.num("yaw");
      if (!send_nav(task_.gx, task_.gy, task_.gyaw)) return fail_now("navigation is not ready");
      return started("going");
    }
    if (op == "dock") {
      double x = r.num("x", dock_[0]), y = r.num("y", dock_[1]), yaw = r.num("yaw", dock_[2]);
      if (std::isnan(x) || std::isnan(y)) return reply_err("I don't know where my dock is");
      if (flags_ & kStCharging) return reply_ok();
      dock_[0] = x, dock_[1] = y, dock_[2] = yaw;
      begin(op, "dock");
      task_.gx = x, task_.gy = y, task_.gyaw = yaw;
      if (!send_nav(x, y, yaw)) return fail_now("navigation is not ready");
      return started("going to the dock");
    }
    if (op == "follow" || op == "come_here") {
      Track* t = r.has("tid") ? find(int(r.num("cam")), int(r.num("tid")), 2.0) : nearest(kPerson, 1.0);
      begin(op, r.text("name"));
      task_.cls = kPerson;
      if (t) {
        task_.cam = t->cam, task_.tid = t->tid;
      } else if (op == "follow") {
        return fail_now("I can't see them");
      } else {
        task_.phase = 9, task_.mark = yaw_travel_;  // come_here: turn around looking for someone first
      }
      return started(op == "follow" ? "following" : "coming");
    }
    if (op == "search") {
      std::string what = r.text("object", r.text("target"));
      begin(op, what);
      task_.cls = what.empty() ? -2 : class_of(what);
      task_.mark = yaw_travel_;
      head(0, -0.1, 0, false);
      return started(task_.cls == -1 ? "looking around (I can't recognise '" + what + "' directly)" : "searching");
    }
    return reply_err("unknown op " + op);
  }

  std::string look_at(const Value& r) {
    std::string t = r.text("target");
    std::transform(t.begin(), t.end(), t.begin(), ::tolower);
    double pan = pan_, tilt = tilt_, turn = 0;
    if (r.has("dpan") || r.has("dtilt")) {  // degrees relative to the head camera's axis (brain grounding)
      pan = pan_ + r.num("dpan", 0) * M_PI / 180, tilt = tilt_ + r.num("dtilt", 0) * M_PI / 180;
    } else if (r.has("pan") || r.has("tilt")) {
      pan = r.num("pan", pan_ * 180 / M_PI) * M_PI / 180, tilt = r.num("tilt", tilt_ * 180 / M_PI) * M_PI / 180;
    } else if (r.has("tid")) {
      Track* tr = find(int(r.num("cam")), int(r.num("tid")), 2.0);
      if (!tr) return reply_err("I can't see them right now");
      begin("look_at", r.text("name"));
      task_.cam = tr->cam, task_.tid = tr->tid, task_.cls = tr->cls;
      return started("looking");
    } else if (has_word(t, "me") || has_word(t, "you") || has_word(t, "us")) {
      Track* p = nearest(kPerson);
      if (!p) return reply_err("I can't see anyone");
      begin("look_at", "you");
      task_.cam = p->cam, task_.tid = p->tid, task_.cls = kPerson;
      return started("looking");
    } else if (has_word(t, "left")) {
      pan = 1.0;
    } else if (has_word(t, "right")) {
      pan = -1.0;
    } else if (has_word(t, "up")) {
      tilt = 0.45;
    } else if (has_word(t, "down")) {
      tilt = -0.3;
    } else if (has_word(t, "behind") || has_word(t, "around") || has_word(t, "back")) {
      pan = 0, tilt = 0, turn = M_PI;
    } else if (t.empty() || has_word(t, "forward") || has_word(t, "ahead") || has_word(t, "straight")) {
      pan = 0, tilt = 0;
    } else {
      int cls = class_of(t);
      Track* tr = cls >= 0 ? nearest(cls, 2.0) : nullptr;
      if (!tr) return reply_err(cls < 0 ? "I can't recognise '" + t + "'" : "I don't see a " + std::string(kCoco[cls]));
      begin("look_at", kCoco[cls]);
      task_.cam = tr->cam, task_.tid = tr->tid, task_.cls = cls;
      return started("looking");
    }
    bool turning = std::fabs(pan) > pan_max_ || turn != 0;
    if (turning) {  // beyond the neck: turn the body for the rest
      if (flags_ & (kStEstop | kLinkDown)) return reply_err("I can't turn right now");
      turn += pan - std::copysign(std::min(std::fabs(pan), pan_max_ * 0.5), pan);
      pan = std::copysign(std::min(std::fabs(pan), pan_max_ * 0.5), pan);
      begin("look_at", t);
      task_.phase = 1, task_.gyaw = turn, task_.mark = yaw_travel_;
    }
    head(pan, tilt, 0, false);
    return turning ? started("turning") : reply_ok();
  }

  std::string started(const std::string& what) {
    MsgWriter w(48);
    w.map(3).key("ok").b(true).key("result").str(what).key("id").i(task_.id);
    return w.buf;
  }
  std::string fail_now(const std::string& why) {
    finish("failed", why);
    return reply_err(why);
  }

  void begin(const std::string& op, const std::string& detail) {
    if (task_.status == "running") finish("canceled", "replaced by " + op);
    int id = task_.id + 1;
    task_ = Task();
    task_.id = id, task_.op = op, task_.status = "running", task_.detail = detail, task_.t0 = mono();
  }

  void finish(const std::string& status, const std::string& detail) {
    if (task_.status != "running") return;
    task_.status = status, task_.detail = detail, task_.t_end = mono();
    cancel_nav();
    if (moving_cmd_) drive(0, 0), moving_cmd_ = false;
    RCLCPP_INFO(get_logger(), "task %d %s: %s (%s)", task_.id, task_.op.c_str(), status.c_str(), detail.c_str());
  }

  // ------------------------------------------------------------------ actuators
  void drive(double v, double w) {
    Twist t;
    t.linear.x = std::clamp(v, -v_max_, v_max_);
    if (t.linear.x > 0 && front_ < stop_dist_ && mono() - scan_t_ < 0.5) t.linear.x = 0;
    t.angular.z = std::clamp(w, -w_max_, w_max_);
    cmd_pub_->publish(t);
    moving_cmd_ = v != 0 || w != 0;
  }
  void head(double pan, double tilt, double speed, bool force) {
    double now = mono();
    if (!force && now - head_t_ < 0.08) return;
    head_t_ = now;
    geometry_msgs::msg::Vector3 m;
    m.x = std::clamp(pan, -pan_max_, pan_max_), m.y = tilt, m.z = speed;
    head_pub_->publish(m);
  }

  bool send_nav(double x, double y, double yaw) {
    if (!nav_->action_server_is_ready()) return false;
    NavTo::Goal g;
    g.pose.header.frame_id = map_frame_;
    g.pose.pose.position.x = x, g.pose.pose.position.y = y;
    tf2::Quaternion q;
    q.setRPY(0, 0, yaw);
    g.pose.pose.orientation.x = q.x(), g.pose.pose.orientation.y = q.y();
    g.pose.pose.orientation.z = q.z(), g.pose.pose.orientation.w = q.w();
    int seq = ++nav_seq_;
    nav_state_ = 1;
    rclcpp_action::Client<NavTo>::SendGoalOptions o;
    o.goal_response_callback = [this, seq](NavGH::SharedPtr gh) {
      if (seq != nav_seq_) {
        if (gh) nav_->async_cancel_goal(gh);
        return;
      }
      if (!gh) nav_state_ = 3;
      nav_gh_ = gh;
    };
    o.feedback_callback = [this, seq](NavGH::SharedPtr, const std::shared_ptr<const NavTo::Feedback> f) {
      if (seq != nav_seq_ || f->number_of_recoveries <= task_.recoveries) return;
      task_.recoveries = f->number_of_recoveries;
      Pose p = pose();
      if (p.ok()) task_.stuck.push_back({p.x, p.y});
    };
    o.result_callback = [this, seq](const NavGH::WrappedResult& res) {
      if (seq != nav_seq_) return;
      nav_gh_.reset();
      nav_state_ = res.code == rclcpp_action::ResultCode::SUCCEEDED ? 2 : res.code == rclcpp_action::ResultCode::CANCELED ? 4 : 3;
    };
    nav_->async_send_goal(g, o);
    return true;
  }
  void cancel_nav() {
    if (nav_state_ != 1) return;
    ++nav_seq_;  // late callbacks of the old goal are ignored; a late accept cancels itself
    if (nav_gh_) nav_->async_cancel_goal(nav_gh_);
    nav_gh_.reset();
    nav_state_ = 0;
  }
  int take_nav() {  // 0 idle, 1 running, 2 succeeded, 3 failed, 4 canceled (consumed on read when done)
    int s = nav_state_;
    if (s >= 2) nav_state_ = 0;
    return s;
  }

  // ------------------------------------------------------------------ control (20 Hz)
  void control() {
    if (task_.status != "running") return;
    double now = mono(), age = now - task_.t0;
    if (flags_ & kStEstop) return finish("failed", "e-stop");
    const std::string& op = task_.op;
    if (op == "goto") {
      int s = take_nav();
      if (s == 2) finish("succeeded", task_.detail);
      else if (s == 3) finish("failed", "could not reach " + (task_.detail.empty() ? "the goal" : task_.detail));
      else if (s == 4) finish("canceled", "navigation canceled");
    } else if (op == "dock") {
      dock_step(now);
    } else if (op == "follow" || op == "come_here") {
      if (task_.phase == 9) return seek_person();
      follow_step(now, op == "come_here");
      if (age > follow_max_s_ && task_.status == "running") finish("succeeded", "followed long enough");
    } else if (op == "search") {
      search_step();
    } else if (op == "look_at") {
      look_step(now, age);
    }
  }

  bool turn_by(double target) {  // true while turning; odom yaw travel since task_.mark
    double err = target - (yaw_travel_ - task_.mark);
    if (std::fabs(err) < 0.05) {
      drive(0, 0);
      return false;
    }
    drive(0, std::copysign(std::clamp(2.0 * std::fabs(err), 0.3, 0.9), err));
    return true;
  }

  void look_step(double now, double age) {
    if (task_.phase == 1) {
      if (!turn_by(task_.gyaw)) finish("succeeded", "turned");
      return;
    }
    Track* tr = find(task_.cam, task_.tid);
    if (!tr) {
      if (now - std::max(task_.lost_t, task_.t0) > 2.0) finish("succeeded", "lost sight");
      if (task_.lost_t < 0) task_.lost_t = now;
      return drive(0, 0);
    }
    task_.lost_t = -1;
    gaze(*tr);
    // neck near its limit: turn the body under the head
    if (std::fabs(tr->bearing) > pan_max_ * 0.8) drive(0, kw_ * tr->bearing);
    else if (moving_cmd_) drive(0, 0);
    if (age > look_hold_s_) finish("succeeded", "looked");
  }

  void gaze(const Track& tr) {  // keep a track centred: head camera closes the loop in the image, body camera by bearing
    if (tr.cam == 0) {
      double fx = fx_[0];
      head(pan_ - 0.6 * std::atan((tr.u - 0.5 * det_w_) / fx), tilt_ - 0.4 * std::atan((tr.v - 0.4 * det_h_) / fx), 0,
           false);
    } else {
      head(tr.bearing, tilt_, 0, false);
    }
  }

  void seek_person() {  // come_here with nobody in view: turn in place until someone shows up
    Track* p = nearest(kPerson);
    if (p) {
      task_.cam = p->cam, task_.tid = p->tid, task_.phase = 0;
      return drive(0, 0);
    }
    if (std::fabs(yaw_travel_ - task_.mark) > 2 * M_PI) return finish("failed", "I couldn't find anyone");
    drive(0, search_w_);
  }

  void follow_step(double now, bool approach) {
    Track* tr = find(task_.cam, task_.tid, 0.5);
    if (!tr && task_.lost_t > 0) tr = reacquire(now);
    if (tr) {
      if (task_.nav_sent) cancel_nav(), task_.nav_sent = false;
      task_.lost_t = -1;
      if (tr->xy_t > 0) task_.lx = tr->mx, task_.ly = tr->my;
      gaze(*tr);
      double b = tr->bearing, v = 0, w = kw_ * b;
      if (tr->range > 0) {
        v = std::max(0.0, kv_ * (tr->range - follow_dist_)) * std::clamp(1.0 - std::fabs(b) / 0.8, 0.0, 1.0);
        if (approach) {
          if (tr->range < follow_dist_ + 0.15) {
            if (task_.near_t < 0) task_.near_t = now;
            if (now - task_.near_t > 0.5) return finish("succeeded", "here");
          } else {
            task_.near_t = -1;
          }
        }
      }
      return drive(v, w);
    }
    if (task_.lost_t < 0) task_.lost_t = now;
    double lost = now - task_.lost_t;
    if (lost > lost_give_up_s_) return finish("lost", "lost them");
    Pose p;
    if (lost > lost_nav_s_ && !task_.nav_sent && !std::isnan(task_.lx) && (p = pose()).ok()) {
      double dx = task_.lx - p.x, dy = task_.ly - p.y, d = std::hypot(dx, dy);
      double k = std::max(0.0, d - 0.6) / std::max(d, 1e-3);  // head for the last known spot, stopping short of it
      task_.nav_sent = send_nav(p.x + k * dx, p.y + k * dy, std::atan2(dy, dx));
      if (task_.nav_sent) moving_cmd_ = false;
    } else if (nav_state_ != 1) {
      take_nav();
      drive(0, 0);
    }
  }

  Track* reacquire(double now) {  // same class, near where they vanished, seen since
    if (std::isnan(task_.lx)) return nullptr;
    double lx = task_.lx, ly = task_.ly, lost = now - task_.lost_t, radius = std::min(1.0 + 0.3 * lost, 2.5);
    Track* best = nullptr;
    double bd = radius;
    for (auto& kv : tracks_) {
      Track& t = kv.second;
      if (t.gie != kObjGie || t.cls != task_.cls || t.t < task_.lost_t || now - t.t > 0.5 || t.xy_t < task_.lost_t)
        continue;
      double d = std::hypot(t.mx - lx, t.my - ly);
      if (d < bd) bd = d, best = &t;
    }
    if (best) task_.cam = best->cam, task_.tid = best->tid;
    return best;
  }

  void search_step() {
    if (task_.cls >= 0) {
      for (auto& kv : tracks_) {
        Track& t = kv.second;
        if (t.gie == kObjGie && t.cls == task_.cls && t.conf >= 0.4 && t.t >= task_.t0 && mono() - t.t < 0.3) {
          drive(0, 0);
          head(t.bearing, 0, 0, true);
          char buf[96];
          if (t.xy_t > 0) std::snprintf(buf, sizeof buf, "found %s at %.2f,%.2f", kCoco[t.cls], t.mx, t.my);
          else std::snprintf(buf, sizeof buf, "found %s", kCoco[t.cls]);
          return finish("succeeded", buf);
        }
      }
    }
    if (std::fabs(yaw_travel_ - task_.mark) < 2 * M_PI) return drive(0, search_w_);
    std::string seen;
    for (int c : task_.seen) seen += (seen.empty() ? "" : ", ") + std::string(kCoco[c]);
    finish(task_.cls >= 0 ? "failed" : "succeeded", (task_.cls >= 0 ? "not found; saw: " : "saw: ") +
                                                     (seen.empty() ? std::string("nothing I recognise") : seen));
  }

  void dock_step(double now) {
    bool charging = flags_ & kStCharging;
    switch (task_.phase) {
      case 0: {  // Nav2 to the pre-dock pose (facing the contacts)
        int s = take_nav();
        if (s == 3 || s == 4) return finish("failed", "could not reach the dock");
        if (s == 2) task_.phase = 1, task_.ax = odom_x_, task_.ay = odom_y_, task_.t1 = now;
        return;
      }
      case 1: {  // creep forward on odometry until the contacts close
        double d = std::hypot(odom_x_ - task_.ax, odom_y_ - task_.ay);
        if (charging) {
          if (task_.near_t < 0) task_.near_t = now;
          drive(0.02, 0);
          if (now - task_.near_t > 0.4) return finish("succeeded", "docked");
          return;
        }
        if (d < dock_travel_ && now - task_.t1 < dock_travel_ / dock_speed_ + 3) return drive(dock_speed_, 0);
        if (++task_.attempt > 2) return finish("failed", "could not make contact with the dock");
        task_.phase = 2, task_.ax = odom_x_, task_.ay = odom_y_;
        return;
      }
      case 2: {  // back off and try again from the pre-dock pose
        if (std::hypot(odom_x_ - task_.ax, odom_y_ - task_.ay) < 0.25) {
          Twist t;
          t.linear.x = -dock_speed_;
          cmd_pub_->publish(t);
          moving_cmd_ = true;
          return;
        }
        drive(0, 0);
        if (!send_nav(task_.gx, task_.gy, task_.gyaw)) return finish("failed", "navigation is not ready");
        task_.phase = 0;
        return;
      }
    }
  }

  // ------------------------------------------------------------------ state (10 Hz)
  void publish_state() {
    Pose p = pose();
    double now = mono();
    bool objs = ++state_n_ % 5 == 0;
    bool charging = flags_ & kStCharging;
    bool docked = charging && (std::isnan(dock_[0]) || !p.ok() || std::hypot(p.x - dock_[0], p.y - dock_[1]) < 0.8);
    bool running = task_.status == "running";
    w_state_.clear();
    w_state_.envelope("ros", 16 + (objs ? 1 : 0));
    w_state_.key("pose");
    if (p.ok()) w_state_.map(3).key("x").fr(p.x, 3).key("y").fr(p.y, 3).key("yaw").fr(p.yaw, 3);
    else w_state_.nil();
    w_state_.key("frame").str(frame_);
    w_state_.key("v").fr(now - odom_t_ < 1 ? v_ : 0, 3).key("w").fr(now - odom_t_ < 1 ? w_ : 0, 3);
    w_state_.key("battery");
    if (batt_pct_ >= 0) w_state_.i(batt_pct_);
    else w_state_.nil();
    w_state_.key("batt_v").fr(batt_v_, 2).key("charging").b(charging).key("docked").b(docked);
    w_state_.key("moving").b(std::fabs(v_) > 0.02 || std::fabs(w_) > 0.05);
    w_state_.key("nav").str(running ? task_.op : "idle");
    w_state_.key("bumpers").i(((flags_ & kStBumpL) ? 1 : 0) | ((flags_ & kStBumpR) ? 2 : 0));
    w_state_.key("cliff").b(flags_ & kStCliff).key("estop").b(flags_ & kStEstop).key("link").b(!(flags_ & kLinkDown));
    w_state_.key("head").map(2).key("pan").fr(pan_, 3).key("tilt").fr(tilt_, 3);
    w_state_.key("task").map(7).key("id").i(task_.id).key("op").str(task_.op).key("status").str(task_.status);
    w_state_.key("detail").str(task_.detail);
    w_state_.key("dur").fr((running ? now : task_.t_end) - task_.t0, 1).key("recov").i(task_.recoveries);
    w_state_.key("stuck").arr(task_.stuck.size());
    for (auto& s : task_.stuck) w_state_.arr(2).fr(s.first, 2).fr(s.second, 2);
    if (objs) write_objs(now);
    zmq_send(pub_, topic::kState, std::strlen(topic::kState), ZMQ_SNDMORE | ZMQ_DONTWAIT);
    zmq_send(pub_, w_state_.buf.data(), w_state_.buf.size(), ZMQ_DONTWAIT);
  }

  void write_objs(double now) {  // 2 Hz: projected detections for spatial memory (§11.10) and rviz
    std::vector<const Track*> out;
    for (auto& kv : tracks_)
      if (kv.second.gie == kObjGie && kv.second.xy_t > 0 && now - kv.second.xy_t < 0.5) out.push_back(&kv.second);
    w_state_.key("objs").arr(out.size());
    geometry_msgs::msg::PoseArray pa;
    bool rviz = objs_pub_->get_subscription_count() > 0;
    for (const Track* t : out) {
      w_state_.map(8).key("cam").i(t->cam).key("tid").i(t->tid).key("cls").i(t->cls).key("conf").fr(t->conf, 2);
      w_state_.key("x").fr(t->mx, 2).key("y").fr(t->my, 2).key("r").fr(t->range, 2).key("b").fr(t->bearing, 3);
      if (rviz) {
        geometry_msgs::msg::Pose ps;
        ps.position.x = t->mx, ps.position.y = t->my, ps.orientation.w = 1;
        pa.poses.push_back(ps);
      }
    }
    if (rviz) {
      pa.header.frame_id = frame_, pa.header.stamp = now_ros();
      objs_pub_->publish(pa);
    }
  }
  rclcpp::Time now_ros() { return get_clock()->now(); }

  // ------------------------------------------------------------------ persistence (5 s)
  bool save_map() {
    if (!save_map_->service_is_ready()) return false;
    auto q = std::make_shared<slam_toolbox::srv::SerializePoseGraph::Request>();
    q->filename = map_path_;
    save_map_->async_send_request(q);
    map_saved_t_ = mono(), moved_since_save_ = 0;
    RCLCPP_INFO(get_logger(), "serializing map to %s", map_path_.c_str());
    return true;
  }
  // The last map pose goes to <map_path>.pose so the next boot starts SLAM where the robot was switched off; the pose
  // graph is saved after the robot has moved and gone idle again (and at most every map_autosave_s).
  void housekeeping() {
    Pose p = pose();
    if (!p.ok() || frame_ != map_frame_) return;
    double d = std::isnan(saved_x_) ? 1 : std::hypot(p.x - saved_x_, p.y - saved_y_) + 0.3 * std::fabs(wrap(p.yaw - saved_yaw_));
    if (d > 0.05) {
      moved_since_save_ += d;
      saved_x_ = p.x, saved_y_ = p.y, saved_yaw_ = p.yaw;
      std::ofstream f(map_path_ + ".pose.tmp");
      f << p.x << ' ' << p.y << ' ' << p.yaw << '\n';
      f.close();
      if (!f.fail()) std::rename((map_path_ + ".pose.tmp").c_str(), (map_path_ + ".pose").c_str());
    }
    bool idle = task_.status != "running" && std::fabs(v_) < 0.01 && std::fabs(w_) < 0.02;
    if (idle && moved_since_save_ > 1.0 && mono() - map_saved_t_ > map_autosave_s_) save_map();
  }

  // ------------------------------------------------------------------ members
  std::string base_frame_, map_frame_, odom_frame_, map_id_, frame_ = "map";
  std::vector<std::string> cam_frames_;
  std::vector<double> fx_;
  double det_w_, det_h_, v_max_, w_max_, kv_, kw_, follow_dist_, lost_nav_s_, lost_give_up_s_, follow_max_s_;
  double stop_dist_, search_w_, pan_max_, look_hold_s_, dock_[3], dock_speed_, dock_travel_, map_autosave_s_;
  std::string map_path_;
  double map_saved_t_ = -1e9, moved_since_save_ = 0, saved_x_ = NAN, saved_y_ = NAN, saved_yaw_ = 0;

  void *ctx_ = nullptr, *pub_ = nullptr, *rep_ = nullptr, *sub_ = nullptr;
  MsgWriter w_state_{1024};

  std::unordered_map<int64_t, Track> tracks_;
  std::vector<std::pair<double, double>> pts_;
  std::vector<double> hits_;
  double scan_t_ = -1, front_ = INFINITY;
  double v_ = 0, w_ = 0, odom_yaw_ = NAN, odom_x_ = 0, odom_y_ = 0, odom_t_ = -1, yaw_travel_ = 0;
  double pan_ = 0, tilt_ = 0, batt_v_ = 0, head_t_ = 0;
  int batt_pct_ = -1;
  uint8_t flags_ = kLinkDown;
  unsigned state_n_ = 0;
  bool moving_cmd_ = false;

  Task task_;
  int nav_seq_ = 0, nav_state_ = 0;
  NavGH::SharedPtr nav_gh_;

  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::shared_ptr<tf2_ros::TransformListener> tfl_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::BatteryState>::SharedPtr batt_sub_;
  rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr flags_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr js_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Publisher<Twist>::SharedPtr cmd_pub_, teleop_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr head_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr objs_pub_;
  rclcpp_action::Client<NavTo>::SharedPtr nav_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr estop_;
  rclcpp::Client<slam_toolbox::srv::SerializePoseGraph>::SharedPtr save_map_;
  rclcpp::TimerBase::SharedPtr zmq_timer_, ctl_timer_, state_timer_, map_timer_;
};

}  // namespace beni

RCLCPP_COMPONENTS_REGISTER_NODE(beni::Bridge)
