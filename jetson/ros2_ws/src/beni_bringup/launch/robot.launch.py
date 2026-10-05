"""§7.4 Beni ROS graph: one isolated component container (drivers, SLAM, Nav2, bridge) + sllidar + EKF processes.

Every param file is handed to the container process, so sub-nodes (the costmaps inside controller/planner) get
their parameters too. If the container dies, launch shuts down and systemd restarts the docker container.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

MAPS = "/maps"


def _on(ctx, name):
    return LaunchConfiguration(name).perform(ctx).lower() in ("1", "true", "yes")


def _slam_params(map_path):
    """Continue the saved pose graph from where the robot was switched off (<map>.pose), else from the dock."""
    if not os.path.exists(map_path + ".posegraph"):
        return {}
    p = {"map_file_name": map_path}
    try:
        with open(map_path + ".pose") as f:
            p["map_start_pose"] = [float(v) for v in f.read().split()[:3]]
    except (OSError, ValueError):
        p["map_start_at_dock"] = True
    return p


def _setup(ctx):
    share = get_package_share_directory("beni_bringup")
    cfg = lambda n: os.path.join(share, "config", n)  # noqa: E731
    with open(os.path.join(get_package_share_directory("beni_description"), "urdf", "beni.urdf")) as f:
        urdf = f.read()
    nav, slam, lidar = _on(ctx, "nav"), _on(ctx, "slam"), _on(ctx, "lidar")
    keepout = nav and os.path.exists(os.path.join(MAPS, "keepout.yaml"))
    files = [cfg("beni.yaml"), cfg("nav2.yaml"), cfg("slam.yaml")] + ([cfg("keepout.yaml")] if keepout else [])

    def comp(pkg, plugin, name, params=(), remap=()):
        return ComposableNode(package=pkg, plugin=plugin, name=name, parameters=list(params), remappings=list(remap))

    nodes = [
        comp("robot_state_publisher", "robot_state_publisher::RobotStatePublisher", "robot_state_publisher",
             [{"robot_description": urdf}]),
        comp("beni_base_driver", "beni::BaseDriver", "base_driver"),
        comp("beni_head", "beni::Head", "head"),
        comp("beni_zmq_bridge", "beni::Bridge", "zmq_bridge"),
    ]
    if slam:
        nodes.append(comp("slam_toolbox", "slam_toolbox::AsynchronousSlamToolbox", "slam_toolbox",
                          [_slam_params(os.path.join(MAPS, "home"))]))
    if nav:
        ctrl = [("cmd_vel", "cmd_vel_ctrl")]
        nodes += [
            comp("nav2_controller", "nav2_controller::ControllerServer", "controller_server", remap=ctrl),
            comp("nav2_planner", "nav2_planner::PlannerServer", "planner_server"),
            comp("nav2_behaviors", "behavior_server::BehaviorServer", "behavior_server", remap=ctrl),
            comp("nav2_bt_navigator", "nav2_bt_navigator::BtNavigator", "bt_navigator"),
            comp("nav2_velocity_smoother", "nav2_velocity_smoother::VelocitySmoother", "velocity_smoother",
                 remap=ctrl),
            comp("nav2_collision_monitor", "nav2_collision_monitor::CollisionMonitor", "collision_monitor"),
            comp("nav2_lifecycle_manager", "nav2_lifecycle_manager::LifecycleManager",
                 "lifecycle_manager_navigation"),
        ]
    if keepout:
        nodes += [
            comp("nav2_map_server", "nav2_map_server::MapServer", "filter_mask_server"),
            comp("nav2_map_server", "nav2_map_server::CostmapFilterInfoServer", "costmap_filter_info_server"),
            comp("nav2_lifecycle_manager", "nav2_lifecycle_manager::LifecycleManager", "lifecycle_manager_filters"),
        ]

    out = [ComposableNodeContainer(
        name="beni_container", namespace="", package="rclcpp_components", executable="component_container_isolated",
        composable_node_descriptions=nodes, parameters=files, output="screen",
        on_exit=Shutdown(reason="component container exited"))]
    out.append(Node(package="robot_localization", executable="ekf_node", name="ekf_filter_node",
                    parameters=[cfg("ekf.yaml")], output="screen", respawn=True, respawn_delay=2.0))
    if lidar:
        out.append(Node(package="sllidar_ros2", executable="sllidar_node", name="sllidar_node", output="screen",
                        respawn=True, respawn_delay=2.0,
                        parameters=[{"channel_type": "serial", "serial_port": "/dev/rplidar",
                                     "serial_baudrate": 115200, "frame_id": "laser", "inverted": False,
                                     "angle_compensate": True}]))
    return out


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("nav", default_value="true", description="start Nav2"),
        DeclareLaunchArgument("slam", default_value="true", description="start slam_toolbox"),
        DeclareLaunchArgument("lidar", default_value="true", description="start the RPLIDAR driver"),
        OpaqueFunction(function=_setup),
    ])
