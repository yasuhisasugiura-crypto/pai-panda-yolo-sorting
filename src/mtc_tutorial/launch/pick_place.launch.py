"""Panda sorting application launch, v0.4.3.
Start the matching MoveIt Task Constructor backend separately.
Defaults: offline display and fixed objects. Camera mode uses a newly authored
sensor-only Gazebo world; target_frame is restricted to world."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder


def validate_modes(context):
    def enabled(name):
        return LaunchConfiguration(name).perform(context).strip().lower() in ("true", "1", "yes")
    if (enabled("yolo_drives_robot") and enabled("enable_mtc") and
            not enabled("allow_experimental_yolo_control")):
        raise RuntimeError("YOLO motion is blocked by default: the supplied Gazebo world is "
                           "sensor-only and has no object-motion synchronization. "
                           "Use enable_mtc:=false for observation-only verification.")
    if enabled("yolo_drives_robot"):
        if (not enabled("use_camera") or not enabled("enable_perception") or
                LaunchConfiguration("perception_mode").perform(context) != "live3d" or
                LaunchConfiguration("source_mode").perform(context) != "detections"):
            raise RuntimeError("YOLO control requires use_camera=true, enable_perception=true, "
                               "perception_mode=live3d and source_mode=detections")
    return []


def generate_launch_description():
    def value(name, kind):
        return ParameterValue(LaunchConfiguration(name), value_type=kind)

    config = MoveItConfigsBuilder(
        "moveit_resources_panda", package_name="moveit_resources_panda_moveit_config"
    ).to_moveit_configs().to_dict()

    world_sdf = PathJoinSubstitution([FindPackageShare("mtc_tutorial"), "worlds", "sorting.sdf"])
    bridge_yaml = PathJoinSubstitution([FindPackageShare("mtc_tutorial"), "config", "ros_gz_bridge.yaml"])

    args = [
        # --- shared / baseline ---
        DeclareLaunchArgument("object_classes", default_value="bottle,cup,wine glass,banana,apple,orange"),
        DeclareLaunchArgument("image_dir", default_value="~/pai_samples"),
        DeclareLaunchArgument("model_path", default_value="yolov8n.pt"),
        DeclareLaunchArgument("enable_perception", default_value="true"),
        DeclareLaunchArgument("allow_model_download", default_value="false"),
        DeclareLaunchArgument("strict_perception", default_value="false"),
        DeclareLaunchArgument("reset_demo_scene", default_value="false"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("velocity_scaling", default_value="0.2"),
        DeclareLaunchArgument("acceleration_scaling", default_value="0.2"),
        DeclareLaunchArgument("max_solutions", default_value="5"),
        DeclareLaunchArgument("cycle_pause_sec", default_value="1.0"),
        DeclareLaunchArgument("startup_timeout_sec", default_value="30.0"),
        # --- Stage A/B/C switches (defaults = baseline) ---
        DeclareLaunchArgument("use_camera", default_value="false"),
        DeclareLaunchArgument("perception_mode", default_value="offline"),   # offline|live|live3d
        DeclareLaunchArgument("source_mode", default_value="fixed"),         # fixed|detections
        DeclareLaunchArgument("autonomous", default_value="false"),
        DeclareLaunchArgument("retry_max", default_value="3"),
        DeclareLaunchArgument("detect_wait_sec", default_value="60.0"),  # time to start object_spawner
        DeclareLaunchArgument("camera_topic", default_value="/rgbd_camera/image"),
        DeclareLaunchArgument("depth_topic", default_value="/rgbd_camera/depth_image"),
        DeclareLaunchArgument("camera_info_topic", default_value="/rgbd_camera/camera_info"),
        DeclareLaunchArgument("target_frame", default_value="world", choices=["world"]),
        # true: YOLO+depth 3D detections drive the robot (also run object_spawner
        # with publish_ground_truth:=false). false (default): overlay only.
        DeclareLaunchArgument("yolo_drives_robot", default_value="false"),
        DeclareLaunchArgument("enable_mtc", default_value="true"),
        DeclareLaunchArgument("allow_experimental_yolo_control", default_value="false"),
        DeclareLaunchArgument("grasp_offset_mode", default_value="baseline", choices=["baseline", "height"]),
    ]

    # Stage A/B: launch Gazebo + ros_gz_bridge only when use_camera:=true.
    gz = ExecuteProcess(
        cmd=["gz", "sim", "-r", world_sdf], output="screen",
        condition=IfCondition(LaunchConfiguration("use_camera")))
    bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge", name="ros_gz_bridge",
        output="screen", parameters=[{"config_file": bridge_yaml}],
        condition=IfCondition(LaunchConfiguration("use_camera")))

    # Camera pose in sorting.sdf: <pose>0.42 0 0.90 0 1.5708 0</pose> (looking down).
    # ROS optical frame (x right, y down, z forward) expressed in world:
    #   x_opt = -Y_world, y_opt = -X_world, z_opt = -Z_world
    #   -> rotation of 180 deg about (1,-1,0)/sqrt(2): q = (0.7071068, -0.7071068, 0, 0)
    # Keep these values in sync with worlds/sorting.sdf.
    camera_tf = Node(
        package="tf2_ros", executable="static_transform_publisher", name="camera_static_tf",
        output="screen",
        arguments=["--x", "0.42", "--y", "0.0", "--z", "0.90",
                   "--qx", "0.7071068", "--qy", "-0.7071068", "--qz", "0.0", "--qw", "0.0",
                   "--frame-id", "world", "--child-frame-id", "rgbd_camera"],
        condition=IfCondition(LaunchConfiguration("use_camera")))

    common = {"object_classes": value("object_classes", str),
              "use_sim_time": value("use_sim_time", bool)}

    perception = Node(
        package="pick_ai", executable="perception_node", name="perception_node",
        output="screen", condition=IfCondition(LaunchConfiguration("enable_perception")),
        parameters=[dict(common,
                         perception_mode=value("perception_mode", str),
                         image_dir=value("image_dir", str),
                         model_path=value("model_path", str),
                         allow_model_download=value("allow_model_download", bool),
                         strict_perception=value("strict_perception", bool),
                         camera_topic=value("camera_topic", str),
                         depth_topic=value("depth_topic", str),
                         camera_info_topic=value("camera_info_topic", str),
                         target_frame=value("target_frame", str),
                         camera_optical_frame="rgbd_camera",
                         publish_3d=value("yolo_drives_robot", bool),
                         publish_period_sec=0.2, thumb=240)],
    )

    mtc = Node(
        package="mtc_tutorial", executable="mtc_node", name="mtc_node", output="screen",
        condition=IfCondition(LaunchConfiguration("enable_mtc")),
        parameters=[config, dict(common,
            source_mode=value("source_mode", str),
            grasp_offset_mode=value("grasp_offset_mode", str),
            detections_topic=ParameterValue(PythonExpression([
                "'/perception/detections_3d' if '", LaunchConfiguration("yolo_drives_robot"),
                "'.lower() in ('true','1','yes') else '/ground_truth/detections_3d'"
            ]), value_type=str),
            autonomous=value("autonomous", bool),
            retry_max=value("retry_max", int),
            detect_wait_sec=value("detect_wait_sec", float),
            velocity_scaling=value("velocity_scaling", float),
            acceleration_scaling=value("acceleration_scaling", float),
            max_solutions=value("max_solutions", int),
            cycle_pause_sec=value("cycle_pause_sec", float),
            startup_timeout_sec=value("startup_timeout_sec", float),
            reset_demo_scene=value("reset_demo_scene", bool))],
    )
    return LaunchDescription(args + [OpaqueFunction(function=validate_modes), gz, bridge, camera_tf, perception, mtc])

