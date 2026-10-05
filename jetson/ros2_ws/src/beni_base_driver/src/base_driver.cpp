// §7.5 host side of the ESP32 base link: termios UART + COBS/CRC frames (shared/proto/base_proto.h).
// Publishes /odom/wheel, /imu/data_raw, /battery, /range/*, /base/flags; sends MSG_CMD_VEL at cmd_rate (the MCU
// watchdog keep-alive). Also the velocity mux (§7.4 twist_mux: teleop > safety > nav > behaviour) so there is no extra
// process or DDS hop between Nav2 and the wheels.
#include <fcntl.h>
#include <poll.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "base_proto.h"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/battery_state.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/range.hpp"
#include "std_msgs/msg/u_int8.hpp"
#include "std_srvs/srv/set_bool.hpp"

namespace beni {

using geometry_msgs::msg::Twist;
constexpr uint8_t kLinkDown = 1 << 7;  // /base/flags bit set by the host when no MSG_STATE for 0.5 s

// 3S Li-ion open-circuit curve (per cell V -> %), good enough for "go charge" decisions.
static double cell_pct(double v) {
  static const double V[] = {3.00, 3.30, 3.50, 3.60, 3.70, 3.80, 3.90, 4.00, 4.10, 4.20};
  static const double P[] = {0, 5, 15, 30, 50, 65, 78, 88, 95, 100};
  if (v <= V[0]) return 0;
  for (int i = 1; i < 10; ++i)
    if (v < V[i]) return P[i - 1] + (P[i] - P[i - 1]) * (v - V[i - 1]) / (V[i] - V[i - 1]);
  return 100;
}

class BaseDriver : public rclcpp::Node {
 public:
  explicit BaseDriver(const rclcpp::NodeOptions& opt) : Node("base_driver", opt) {
    port_ = declare_parameter<std::string>("port", "/dev/ttyTHS1");
    baud_ = declare_parameter("baud", 921600);
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
    imu_frame_ = declare_parameter<std::string>("imu_frame", "imu_link");
    v_max_ = declare_parameter("v_max", 0.5);
    w_max_ = declare_parameter("w_max", 2.5);
    cells_ = declare_parameter("battery_cells", 3);
    cfg_ = {float(declare_parameter("kp", 0.9)), float(declare_parameter("ki", 6.0)),
            float(declare_parameter("kd", 0.0)), float(v_max_), float(w_max_)};
    auto names = declare_parameter("mux.names", std::vector<std::string>{"teleop", "safety", "nav", "behaviour"});
    auto prio = declare_parameter("mux.priorities", std::vector<int64_t>{100, 90, 50, 30});
    auto tmo = declare_parameter("mux.timeouts", std::vector<double>{0.5, 0.3, 0.5, 0.5});
    double rate = declare_parameter("cmd_rate", 50.0);

    auto qos = rclcpp::SensorDataQoS().keep_last(5);
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("odom/wheel", qos);
    imu_pub_ = create_publisher<sensor_msgs::msg::Imu>("imu/data_raw", qos);
    batt_pub_ = create_publisher<sensor_msgs::msg::BatteryState>("battery", rclcpp::QoS(1).transient_local());
    flags_pub_ = create_publisher<std_msgs::msg::UInt8>("base/flags", rclcpp::QoS(1).transient_local());
    const char* rn[4] = {"cliff_l", "cliff_r", "us_l", "us_r"};
    for (int i = 0; i < 4; ++i)
      range_pub_[i] = create_publisher<sensor_msgs::msg::Range>(std::string("range/") + rn[i], qos);

    srcs_.resize(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
      auto& s = srcs_[i];
      s.name = names[i];
      s.prio = i < prio.size() ? int(prio[i]) : 0;
      s.timeout = i < tmo.size() ? tmo[i] : 0.5;
      s.sub = create_subscription<Twist>("cmd_vel_" + s.name, rclcpp::QoS(1).best_effort(),
                                         [this, i](Twist::ConstSharedPtr m) {
                                           std::lock_guard<std::mutex> lk(mux_mtx_);
                                           srcs_[i].last = *m;
                                           srcs_[i].stamp = now();
                                         });
    }
    estop_srv_ = create_service<std_srvs::srv::SetBool>(
        "base/estop", [this](const std::shared_ptr<std_srvs::srv::SetBool::Request> rq,
               std::shared_ptr<std_srvs::srv::SetBool::Response> rs) {
          estop_want_ = rq->data ? 1 : 0;
          estop_pending_ = true;
          rs->success = true;
          rs->message = link_up() ? "sent" : "link down; will send on reconnect";
        });
    param_cb_ = add_on_set_parameters_callback([this](const std::vector<rclcpp::Parameter>& ps) {
      for (const auto& p : ps) {
        const auto& n = p.get_name();
        if (n == "kp") cfg_.kp = float(p.as_double());
        else if (n == "ki") cfg_.ki = float(p.as_double());
        else if (n == "kd") cfg_.kd = float(p.as_double());
        else if (n == "v_max") cfg_.v_max = float(v_max_ = p.as_double());
        else if (n == "w_max") cfg_.w_max = float(w_max_ = p.as_double());
        else continue;
        cfg_pending_ = true;
      }
      rcl_interfaces::msg::SetParametersResult r;
      r.successful = true;
      return r;
    });
    tick_ = create_wall_timer(std::chrono::duration<double>(1.0 / rate), [this] { tick(); });
    reader_ = std::thread([this] { reader(); });
  }

  ~BaseDriver() override {
    run_ = false;
    if (reader_.joinable()) reader_.join();
    if (fd_ >= 0) close(fd_);
  }

 private:
  struct Source {
    std::string name;
    int prio = 0;
    double timeout = 0.5;
    rclcpp::Subscription<Twist>::SharedPtr sub;
    Twist last;
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
  };

  // ------------------------------------------------------------------ serial
  bool open_port() {
    int fd = open(port_.c_str(), O_RDWR | O_NOCTTY | O_CLOEXEC);
    if (fd < 0) return false;
    termios t{};
    tcgetattr(fd, &t);
    cfmakeraw(&t);
    t.c_cflag |= CLOCAL | CREAD;
    t.c_cflag &= ~(CSTOPB | CRTSCTS);
    t.c_cc[VMIN] = 0;
    t.c_cc[VTIME] = 0;
    speed_t sp = baud_ == 921600 ? B921600 : baud_ == 460800 ? B460800 : baud_ == 230400 ? B230400 : B115200;
    cfsetispeed(&t, sp);
    cfsetospeed(&t, sp);
    if (tcsetattr(fd, TCSANOW, &t) != 0) {
      close(fd);
      return false;
    }
    tcflush(fd, TCIOFLUSH);
    fd_ = fd;
    cfg_pending_ = true;  // (re)connect: push gains and any latched e-stop request
    RCLCPP_INFO(get_logger(), "opened %s @ %d", port_.c_str(), baud_);
    return true;
  }

  void send(uint8_t type, const void* payload, uint8_t len) {
    uint8_t buf[BASE_MAX_FRAME];
    std::lock_guard<std::mutex> lk(tx_mtx_);
    if (fd_ < 0) return;
    size_t n = base_frame(type, tx_seq_++, payload, len, buf);
    if (write(fd_, buf, n) != ssize_t(n)) RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "uart write short");
  }

  void reader() {
    std::vector<uint8_t> acc;
    acc.reserve(BASE_MAX_FRAME);
    uint8_t buf[512], dec[BASE_MAX_FRAME];
    while (run_) {
      if (fd_ < 0) {
        if (!open_port()) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000, "cannot open %s", port_.c_str());
          std::this_thread::sleep_for(std::chrono::seconds(1));
          continue;
        }
      }
      pollfd p{fd_, POLLIN, 0};
      int r = poll(&p, 1, 100);
      if (r < 0 || (p.revents & (POLLERR | POLLHUP | POLLNVAL))) {
        reopen();
        continue;
      }
      if (r == 0) continue;
      ssize_t n = read(fd_, buf, sizeof(buf));
      if (n <= 0) {
        if (n < 0) reopen();
        continue;
      }
      for (ssize_t i = 0; i < n; ++i) {
        if (buf[i] != 0) {
          if (acc.size() < BASE_MAX_FRAME) acc.push_back(buf[i]);
          else acc.clear(), ++bad_;  // runaway frame: resync at the next 0x00
          continue;
        }
        size_t m = acc.empty() ? 0 : base_cobs_decode(acc.data(), acc.size(), dec);
        acc.clear();
        if (m < 5 || size_t(dec[2]) + 5 != m) {
          ++bad_;
          continue;
        }
        uint16_t crc = uint16_t(dec[m - 2] | (dec[m - 1] << 8));
        if (crc != base_crc16(dec, m - 2)) {
          ++bad_;
          continue;
        }
        on_frame(dec[0], dec + 3, dec[2]);
      }
    }
  }

  void reopen() {
    std::lock_guard<std::mutex> lk(tx_mtx_);
    if (fd_ >= 0) close(fd_);
    fd_ = -1;
    RCLCPP_WARN(get_logger(), "uart error; reopening");
  }

  bool link_up() const { return fd_ >= 0 && last_rx_ns_ != 0 && now_ns() - last_rx_ns_ < 500000000LL; }
  static int64_t now_ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
  }

  // ------------------------------------------------------------------ rx
  void on_frame(uint8_t type, const uint8_t* p, uint8_t len) {
    last_rx_ns_ = now_ns();
    switch (type) {
      case MSG_STATE:
        if (len == sizeof(BaseStateMsg)) {
          BaseStateMsg s;
          std::memcpy(&s, p, sizeof(s));
          on_state(s);
        }
        break;
      case MSG_ACK:
        if (len >= 3 && p[2]) {
          if (p[0] == MSG_CONFIG) cfg_pending_ = false;
          if (p[0] == MSG_ESTOP) estop_pending_ = false;
        } else if (len >= 3) {
          RCLCPP_WARN(get_logger(), "mcu rejected msg 0x%02x", p[0]);
          if (p[0] == MSG_CONFIG) cfg_pending_ = false;  // invalid gains: don't spin
        }
        break;
      case MSG_EVENT:
        if (len >= 2) on_event(p[0], p[1]);
        break;
      case MSG_LOG:
        RCLCPP_INFO(get_logger(), "mcu: %.*s", int(len), reinterpret_cast<const char*>(p));
        break;
      default:
        break;
    }
  }

  void on_event(uint8_t kind, uint8_t arg) {
    switch (kind) {
      case EV_BOOT:
        RCLCPP_WARN(get_logger(), "mcu boot (reset reason %u)", arg);
        cfg_pending_ = true;
        if (estop_want_) estop_pending_ = true;
        break;
      case EV_BUMP: RCLCPP_WARN(get_logger(), "bump L%d R%d", arg & 1, (arg >> 1) & 1); break;
      case EV_CLIFF: RCLCPP_WARN(get_logger(), "cliff L%d R%d", arg & 1, (arg >> 1) & 1); break;
      case EV_ESTOP: RCLCPP_WARN(get_logger(), "estop %s", arg ? "engaged" : "cleared"); break;
      case EV_LOW_BATT: RCLCPP_WARN(get_logger(), "low battery %u%%", arg); break;
      case EV_WATCHDOG: RCLCPP_DEBUG(get_logger(), "mcu watchdog %u", arg); break;
      case EV_SENSOR_FAULT: RCLCPP_ERROR(get_logger(), "sensor fault imu=%d tof_l=%d tof_r=%d", arg & 1, (arg >> 1) & 1,
                                         (arg >> 2) & 1); break;
      default: break;
    }
  }

  // MCU ms -> ROS time: min-latency offset tracker (a frame can arrive late, never early), allowing 100 ppm drift.
  rclcpp::Time stamp_of(uint32_t t_ms) {
    double host = now().seconds(), mcu = t_ms * 1e-3;
    if (t_ms < last_t_ms_ || off_ == 0) off_ = host - mcu, last_mcu_ = mcu;  // first frame or MCU reboot
    off_ = std::min(off_ + 1e-4 * (mcu - last_mcu_), host - mcu);
    last_t_ms_ = t_ms;
    last_mcu_ = mcu;
    return rclcpp::Time(int64_t((mcu + off_) * 1e9), RCL_ROS_TIME);
  }

  void on_state(const BaseStateMsg& s) {
    auto st = stamp_of(s.t_ms);
    auto o = std::make_unique<nav_msgs::msg::Odometry>();
    o->header.stamp = st;
    o->header.frame_id = odom_frame_;
    o->child_frame_id = base_frame_;
    o->pose.pose.position.x = s.x_m;
    o->pose.pose.position.y = s.y_m;
    o->pose.pose.orientation.z = std::sin(s.yaw_rad * 0.5);
    o->pose.pose.orientation.w = std::cos(s.yaw_rad * 0.5);
    o->twist.twist.linear.x = s.v_mm_s * 1e-3;
    o->twist.twist.angular.z = s.w_mrad_s * 1e-3;
    // The EKF fuses only vx, vyaw (odom0 differential=false, pose unused). Diff drive: vy ~ 0 with high confidence.
    const double pc[6] = {0.05, 0.05, 1e3, 1e3, 1e3, 0.1}, tc[6] = {0.004, 1e-4, 1e3, 1e3, 1e3, 0.02};
    for (int i = 0; i < 6; ++i) o->pose.covariance[i * 7] = pc[i], o->twist.covariance[i * 7] = tc[i];
    odom_pub_->publish(std::move(o));

    auto m = std::make_unique<sensor_msgs::msg::Imu>();
    m->header.stamp = st;
    m->header.frame_id = imu_frame_;
    m->orientation_covariance[0] = -1;  // no orientation estimate
    m->angular_velocity.z = s.gyro_z_mdps * 1e-3 * M_PI / 180.0;
    m->angular_velocity_covariance = {4e-4, 0, 0, 0, 4e-4, 0, 0, 0, 4e-4};
    m->linear_acceleration.x = s.acc_x_mg * 9.80665e-3;
    m->linear_acceleration.y = s.acc_y_mg * 9.80665e-3;
    m->linear_acceleration.z = s.acc_z_mg * 9.80665e-3;
    m->linear_acceleration_covariance = {0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01};
    imu_pub_->publish(std::move(m));

    uint8_t flags = s.flags;
    if (flags != last_flags_) {
      last_flags_ = flags;
      std_msgs::msg::UInt8 f;
      f.data = flags;
      flags_pub_->publish(f);
    }
    if (++n_state_ % 5 == 0) publish_ranges(s, st);  // 10 Hz
    if (n_state_ % 50 == 0) publish_battery(s, st);  // 1 Hz
  }

  void publish_ranges(const BaseStateMsg& s, const rclcpp::Time& st) {
    static const char* frames[4] = {"cliff_l_link", "cliff_r_link", "us_l_link", "us_r_link"};
    for (int i = 0; i < 4; ++i) {
      auto r = std::make_unique<sensor_msgs::msg::Range>();
      r->header.stamp = st;
      r->header.frame_id = frames[i];
      bool us = i >= 2;
      r->radiation_type = us ? sensor_msgs::msg::Range::ULTRASOUND : sensor_msgs::msg::Range::INFRARED;
      r->field_of_view = us ? 0.26f : 0.44f;
      r->min_range = us ? 0.02f : 0.0f;
      r->max_range = 2.55f;  // uint8 cm on the wire
      r->range = (us && s.range_cm[i] == 0) ? INFINITY : s.range_cm[i] * 0.01f;  // REP 117: +inf = no echo
      range_pub_[i]->publish(std::move(r));
    }
  }

  void publish_battery(const BaseStateMsg& s, const rclcpp::Time& st) {
    auto b = std::make_unique<sensor_msgs::msg::BatteryState>();
    b->header.stamp = st;
    b->voltage = s.batt_mv * 1e-3f;
    b->current = s.batt_ma ? -s.batt_ma * 1e-3f : NAN;  // ROS: negative while discharging
    b->percentage = float(cell_pct(b->voltage / cells_) / 100.0);
    using B = sensor_msgs::msg::BatteryState;
    b->power_supply_technology = B::POWER_SUPPLY_TECHNOLOGY_LION;
    b->power_supply_status = !(s.flags & ST_CHARGING) ? B::POWER_SUPPLY_STATUS_DISCHARGING
                             : b->percentage > 0.98f  ? B::POWER_SUPPLY_STATUS_FULL
                                                      : B::POWER_SUPPLY_STATUS_CHARGING;
    b->present = true;
    b->charge = b->capacity = b->design_capacity = NAN;
    batt_pub_->publish(std::move(b));
  }

  // ------------------------------------------------------------------ tx (executor thread)
  void tick() {
    if (!link_up() && !(last_flags_ & kLinkDown) && last_rx_ns_) {  // before the fd check: a port that won't reopen
      last_flags_ |= kLinkDown;                                         // is a lost link too
      std_msgs::msg::UInt8 f;
      f.data = last_flags_;
      flags_pub_->publish(f);
      RCLCPP_ERROR(get_logger(), "base link lost (bad frames so far: %u)", unsigned(bad_));
    }
    if (fd_ < 0) return;
    if (cfg_pending_ && ++cfg_retry_ % 10 == 1) send(MSG_CONFIG, &cfg_, sizeof(cfg_));  // retry at 5 Hz until acked
    if (estop_pending_ && ++estop_retry_ % 5 == 1) {
      uint8_t e = estop_want_;
      send(MSG_ESTOP, &e, 1);
    }
    Twist cmd;
    const Source* best = nullptr;
    {
      std::lock_guard<std::mutex> lk(mux_mtx_);
      auto t = now();
      for (const auto& s : srcs_)
        if (s.stamp.nanoseconds() && (t - s.stamp).seconds() < s.timeout && (!best || s.prio > best->prio))
          best = &s;
      if (best) cmd = best->last;
      const std::string& name = best ? best->name : kNone;
      if (name != active_) {
        RCLCPP_INFO(get_logger(), "cmd source: %s", name.c_str());
        active_ = name;
      }
    }
    CmdVelMsg c;
    c.v_mm_s = int16_t(std::lround(std::clamp(cmd.linear.x, -v_max_, v_max_) * 1000.0));
    c.w_mrad_s = int16_t(std::lround(std::clamp(cmd.angular.z, -w_max_, w_max_) * 1000.0));
    send(MSG_CMD_VEL, &c, sizeof(c));  // zeros too: the keep-alive that holds off the MCU watchdog
  }

  inline static const std::string kNone = "none";
  std::string port_, base_frame_, odom_frame_, imu_frame_, active_ = kNone;
  int baud_, cells_;
  double v_max_, w_max_;
  ConfigMsg cfg_;
  std::atomic<int> fd_{-1};
  std::atomic<bool> run_{true}, cfg_pending_{true}, estop_pending_{false};
  std::atomic<uint8_t> estop_want_{0}, last_flags_{0xFF};
  std::atomic<int64_t> last_rx_ns_{0};
  std::atomic<uint32_t> bad_{0};
  uint32_t cfg_retry_ = 0, estop_retry_ = 0, n_state_ = 0, last_t_ms_ = 0;
  double off_ = 0, last_mcu_ = 0;
  uint8_t tx_seq_ = 0;
  std::mutex tx_mtx_, mux_mtx_;
  std::vector<Source> srcs_;
  std::thread reader_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
  rclcpp::Publisher<sensor_msgs::msg::BatteryState>::SharedPtr batt_pub_;
  rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr flags_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Range>::SharedPtr range_pub_[4];
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr estop_srv_;
  rclcpp::TimerBase::SharedPtr tick_;
  OnSetParametersCallbackHandle::SharedPtr param_cb_;
};

}  // namespace beni

RCLCPP_COMPONENTS_REGISTER_NODE(beni::BaseDriver)
