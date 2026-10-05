// §7.5 head pan/tilt: PCA9685 on /dev/i2c-1 (header pins 3/5). The chip makes the PWM, so Linux jitter is harmless;
// this node only plans smooth (accel-limited) motion at 50 Hz and publishes joint_states for the head-camera TF.
// /head/cmd geometry_msgs/Vector3: x = pan rad (+left), y = tilt rad (+up), z = max speed rad/s (0 = default).
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "geometry_msgs/msg/vector3.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

namespace beni {

class Head : public rclcpp::Node {
 public:
  explicit Head(const rclcpp::NodeOptions& opt) : Node("head", opt) {
    dev_ = declare_parameter<std::string>("i2c_dev", "/dev/i2c-1");
    addr_ = declare_parameter("address", 0x40);
    rate_ = declare_parameter("rate", 50.0);
    speed_ = declare_parameter("max_speed", 3.0);
    accel_ = declare_parameter("max_accel", 10.0);
    relax_after_ = declare_parameter("relax_after", 5.0);
    const char* names[2] = {"pan", "tilt"};
    const double lo[2] = {-1.4, -0.35}, hi[2] = {1.4, 0.6};
    for (int i = 0; i < 2; ++i) {
      std::string n = names[i];
      Axis& a = ax_[i];
      a.channel = declare_parameter(n + ".channel", i);
      a.lo = declare_parameter(n + ".min", lo[i]);
      a.hi = declare_parameter(n + ".max", hi[i]);
      a.center_us = declare_parameter(n + ".center_us", 1500.0);
      a.us_per_rad = declare_parameter(n + ".us_per_rad", 2000.0 / M_PI) * declare_parameter(n + ".dir", 1.0);
      a.joint = declare_parameter<std::string>(n + ".joint", "head_" + n + "_joint");
    }
    js_pub_ = create_publisher<sensor_msgs::msg::JointState>("joint_states", rclcpp::QoS(5));
    cmd_sub_ = create_subscription<geometry_msgs::msg::Vector3>(
        "head/cmd", rclcpp::QoS(5), [this](geometry_msgs::msg::Vector3::ConstSharedPtr m) {
          ax_[0].target = std::clamp(m->x, ax_[0].lo, ax_[0].hi);
          ax_[1].target = std::clamp(m->y, ax_[1].lo, ax_[1].hi);
          cmd_speed_ = m->z > 0 ? std::min(m->z, speed_) : speed_;
          last_cmd_ = now();
          if (relaxed_) relaxed_ = false, dirty_ = true;
        });
    last_cmd_ = now();
    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / rate_), [this] { step(); });
  }

  ~Head() override {
    if (fd_ >= 0) {
      relax();
      close(fd_);
    }
  }

 private:
  struct Axis {
    int channel = 0;
    double lo = 0, hi = 0, center_us = 1500, us_per_rad = 636.6, pos = 0, vel = 0, target = 0;
    std::string joint;
    int last_ticks = -1;
  };

  bool wr(const uint8_t* b, size_t n) { return fd_ >= 0 && write(fd_, b, n) == ssize_t(n); }
  bool reg(uint8_t r, uint8_t v) {
    uint8_t b[2] = {r, v};
    return wr(b, 2);
  }

  bool open_chip() {
    fd_ = open(dev_.c_str(), O_RDWR | O_CLOEXEC);
    if (fd_ < 0 || ioctl(fd_, I2C_SLAVE, addr_) < 0) return fail();
    // 25 MHz internal osc; prescale 121 -> 50.03 Hz. MODE1: sleep to set prescale, then auto-increment + restart.
    if (!reg(0x00, 0x10) || !reg(0xFE, 121) || !reg(0x00, 0x20) || !reg(0x01, 0x04)) return fail();
    usleep(600);
    if (!reg(0x00, 0xA0)) return fail();
    period_us_ = 1e6 / (25e6 / (4096.0 * 122));
    for (auto& a : ax_) a.last_ticks = -1;
    RCLCPP_INFO(get_logger(), "PCA9685 @0x%02x on %s", addr_, dev_.c_str());
    return true;
  }
  bool fail() {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000, "PCA9685 not reachable on %s", dev_.c_str());
    if (fd_ >= 0) close(fd_);
    fd_ = -1;
    return false;
  }

  bool set_ticks(Axis& a, int off) {  // off = -1: full off (servo limp, no hum, no current)
    uint8_t b[5] = {uint8_t(0x06 + 4 * a.channel), 0, 0, uint8_t(off < 0 ? 0 : off & 0xFF),
                    uint8_t(off < 0 ? 0x10 : (off >> 8) & 0x0F)};
    if (!wr(b, 5)) return false;
    a.last_ticks = off;
    return true;
  }

  void relax() {
    for (auto& a : ax_) set_ticks(a, -1);
    relaxed_ = true;
  }

  void step() {
    if (fd_ < 0 && ++retry_ % int(rate_) == 1 && !open_chip()) return;
    const double dt = 1.0 / rate_;
    bool moving = false;
    for (auto& a : ax_) {  // accel-limited approach: v -> sign(e) * min(speed, sqrt(2 a |e|))
      double e = a.target - a.pos;
      double v_want = std::copysign(std::min(cmd_speed_, std::sqrt(2.0 * accel_ * std::fabs(e))), e);
      a.vel += std::clamp(v_want - a.vel, -accel_ * dt, accel_ * dt);
      if (std::fabs(e) < 1e-3 && std::fabs(a.vel) < 0.05) a.pos = a.target, a.vel = 0;
      else a.pos += a.vel * dt, moving = true;
    }
    if (fd_ >= 0) {
      if (!moving && relax_after_ > 0 && !relaxed_ && (now() - last_cmd_).seconds() > relax_after_) {
        relax();
      } else if (!relaxed_ || dirty_) {
        dirty_ = false;
        for (auto& a : ax_) {
          int t = int(std::lround((a.center_us + a.pos * a.us_per_rad) * 4096.0 / period_us_));
          if (t != a.last_ticks && !set_ticks(a, t)) {
            fail();
            break;
          }
        }
      }
    }
    auto js = std::make_unique<sensor_msgs::msg::JointState>();
    js->header.stamp = now();
    js->name = {ax_[0].joint, ax_[1].joint};
    js->position = {ax_[0].pos, ax_[1].pos};
    js->velocity = {ax_[0].vel, ax_[1].vel};
    js_pub_->publish(std::move(js));
  }

  std::string dev_;
  int addr_, fd_ = -1;
  unsigned retry_ = 0;
  double rate_, speed_, accel_, relax_after_, cmd_speed_ = 3.0, period_us_ = 20000;
  bool relaxed_ = false, dirty_ = false;
  Axis ax_[2];
  rclcpp::Time last_cmd_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr js_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr cmd_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace beni

RCLCPP_COMPONENTS_REGISTER_NODE(beni::Head)
