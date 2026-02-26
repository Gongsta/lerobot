#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotAction,
    RobotProcessorPipeline,
)
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.robot_kinematic_processor import (
    ForwardKinematicsJointsToEE,
)

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_koch_leader")
@dataclass
class BiKochLeaderConfig(TeleoperatorConfig):
    left_arm_port: str
    right_arm_port: str

    # Enable intervention mode (press SPACE to toggle between policy and teleop during recording)
    intervention_enabled: bool = False

    # When True, leader motors track the follower during policy execution (smooth handoff).
    # Set to False if the leader arms are USB-powered only (no wall adapter) to avoid
    # drawing torque current through the USB port.
    inverse_follow: bool = True


def make_bimanual_koch_teleop_processors(
    teleop, display_data: bool
) -> RobotProcessorPipeline[RobotAction, RobotAction]:
    urdf_path = "/home/steven/research/lerobot/assets/koch_follower.urdf"
    left_teleop_kinematics_solver = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name="ee_frame",
        joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5"],
    )
    right_teleop_kinematics_solver = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name="ee_frame",
        joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5"],
    )

    teleop_motor_names = list(teleop.left_arm.bus.motors.keys())
    left_teleop_motor_names = ["left_" + motor for motor in teleop_motor_names]
    right_teleop_motor_names = ["right_" + motor for motor in teleop_motor_names]

    teleop_to_ee = RobotProcessorPipeline[RobotAction, RobotAction](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=left_teleop_kinematics_solver,
                motor_names=left_teleop_motor_names,
                gripper_name="left_gripper",
                display_data=display_data,
                entity_path_prefix="left_leader",
                offset=0.4,
            ),
            ForwardKinematicsJointsToEE(
                kinematics=right_teleop_kinematics_solver,
                motor_names=right_teleop_motor_names,
                gripper_name="right_gripper",
                display_data=display_data,
                entity_path_prefix="right_leader",
                offset=0.6,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return teleop_to_ee
