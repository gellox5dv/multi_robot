from launch import LaunchDescription
from launch.actions import GroupAction, IncludeLaunchDescription
from launch_ros.actions import Node, PushRosNamespace
from launch.substitutions import Command
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def spawn_robot(robot_name, x, y):
    pkg_path = get_package_share_directory('multi_robot')
    xacro_file = os.path.join(pkg_path, 'description', 'robot.urdf.xacro')

    robot_description = Command(['xacro ', xacro_file])

    return GroupAction([
        PushRosNamespace(robot_name),

        # Robot state publisher (TF)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[
                {'robot_description': robot_description},
                {'frame_prefix': f'{robot_name}/'}
            ],
            output='screen'
        ),

        # Spawn in Gazebo
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-topic', 'robot_description',
                '-entity', robot_name,
                '-robot_namespace', f'/{robot_name}',
                '-x', str(x),
                '-y', str(y),
                '-z', '0.01'
            ],
            output='screen'
        ),
    ])


def generate_launch_description():
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch',
                'gazebo.launch.py',
            )
        )
    )

    return LaunchDescription([
        gazebo_launch,
        spawn_robot('r1', 0.0, 0.0),
        spawn_robot('r2', 2.0, 0.0),  # Agent 2
    ])
