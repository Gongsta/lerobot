# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import numbers
import os
from typing import Any

import numpy as np
import rerun as rr

from .constants import OBS_PREFIX, OBS_STR


def init_rerun(session_name: str = "lerobot_control_loop") -> None:
    """Initializes the Rerun SDK for visualizing the control loop."""
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    rr.spawn(memory_limit=memory_limit)

    rr.log_file_from_path("/home/steven/research/low_cost_robot/simulation/low_cost_robot_6dof/low-cost-arm.urdf", static=True)


def _is_scalar(x):
    return (
        isinstance(x, float)
        or isinstance(x, numbers.Real)
        or isinstance(x, (np.integer, np.floating))
        or (isinstance(x, np.ndarray) and x.ndim == 0)
    )


def log_rerun_data(
    observation: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
) -> None:
    """
    Logs observation and action data to Rerun for real-time visualization.

    This function iterates through the provided observation and action dictionaries and sends their contents
    to the Rerun viewer. It handles different data types appropriately:
    - Scalar values (floats, ints) are logged as `rr.Scalars`.
    - 3D NumPy arrays that resemble images (e.g., with 1, 3, or 4 channels first) are transposed
      from CHW to HWC format and logged as `rr.Image`.
    - 1D NumPy arrays are logged as a series of individual scalars, with each element indexed.
    - Other multi-dimensional arrays are flattened and logged as individual scalars.

    Keys are automatically namespaced with "observation." or "action." if not already present.

    Args:
        observation: An optional dictionary containing observation data to log.
        action: An optional dictionary containing action data to log.
    """
    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                arr = v
                # Convert CHW -> HWC when needed
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 1:
                    for i, vi in enumerate(arr):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    rr.log(key, rr.Image(arr), static=True)

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith("action.") else f"action.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    for i, vi in enumerate(v):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    # Fall back to flattening higher-dimensional arrays
                    flat = v.flatten()
                    for i, vi in enumerate(flat):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))


import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
import rerun as rr

def build_maps_skip_fixed(urdf_path: str, root_prefix: str = "robot"):
    urdf = ET.parse(urdf_path).getroot()

    # Parse links & joints
    all_links = set(l.attrib["name"] for l in urdf.findall("link"))
    joints = []  # list of dicts for convenience
    for j in urdf.findall("joint"):
        jname  = j.attrib["name"]
        jtype  = j.attrib["type"]
        parent = j.find("parent").attrib["link"]
        child  = j.find("child").attrib["link"]
        axis_e = j.find("axis")
        axis   = np.array([0.0, 0.0, 1.0]) if axis_e is None else np.array(list(map(float, axis_e.attrib["xyz"].split())))
        n = np.linalg.norm(axis); axis = axis / (n if n > 1e-12 else 1.0)
        joints.append(dict(name=jname, type=jtype, parent=parent, child=child, axis=axis))
        all_links.update((parent, child))

    # Quick lookups
    # parent_link -> [(joint_name, joint_type, child_link)]
    children = defaultdict(list)
    for jd in joints:
        children[jd["parent"]].append((jd["name"], jd["type"], jd["child"]))

    # child_link -> (joint_name, joint_type, parent_link)
    joint_of_child = {jd["child"]: (jd["name"], jd["type"], jd["parent"]) for jd in joints}

    # Find base link
    base_candidates = [L for L in all_links if L not in joint_of_child]
    if len(base_candidates) != 1:
        raise RuntimeError(f"Expected exactly one base link, got {base_candidates}")
    base = base_candidates[0]

    # Build exact Rerun paths (same as loader)
    link_path  = {base: f"{root_prefix}/{base}"}
    joint_path = {}
    stack = [base]
    while stack:
        parent = stack.pop()
        parent_path = link_path[parent]
        for jname, jtype, child in children[parent]:
            joint_path[jname] = f"{parent_path}/{jname}"  # log motion HERE
            link_path[child]  = f"{parent_path}/{jname}/{jname}_jointbody/{jname}_offset/{child}"
            stack.append(child)

    # Resolve each *revolute/prismatic* joint’s REAL child link by skipping fixed joints
    def resolve_real_child(start_link: str) -> str:
        cur = start_link
        # follow chains of exactly-one-outgoing FIXED joints until you hit a non-fixed or leaf
        while True:
            outs = children.get(cur, [])
            if len(outs) != 1:
                return cur
            jname, jtype, child = outs[0]
            if jtype != "fixed":
                return cur  # cur is the real child link for our actuated joint
            cur = child

    joint_axis = {jd["name"]: jd["axis"] for jd in joints}
    joint_type = {jd["name"]: jd["type"] for jd in joints}
    joint_parent_link = {jd["name"]: jd["parent"] for jd in joints}
    # For actuated joints use the resolved real child
    joint_child_real = {}
    for jd in joints:
        if jd["type"] in ("revolute", "continuous", "prismatic"):
            joint_child_real[jd["name"]] = resolve_real_child(jd["child"])

    return {
        "base": base,
        "link_path": link_path,
        "joint_path": joint_path,
        "joint_axis": joint_axis,
        "joint_type": joint_type,
        "joint_parent_link": joint_parent_link,
        "joint_child_real": joint_child_real,
    }

def _inv(T):
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4)
    Ti[:3,:3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def _axis_angle_from_R(R):
    tr = float(np.trace(R))
    cos_th = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
    theta = float(np.arccos(cos_th))
    if theta < 1e-8:
        return np.array([0,0,1], float), 0.0
    rv = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]], float) * 0.5
    n = np.linalg.norm(rv)
    axis = rv / (n if n > 1e-12 else 1.0)
    return axis, theta

def log_joints_from_world_fk(step: int, robot, maps):
    rr.set_time_sequence("frame", step)

    base       = maps["base"]
    link_path  = maps["link_path"]
    joint_path = maps["joint_path"]
    j_axis     = maps["joint_axis"]
    j_type     = maps["joint_type"]
    j_parent   = maps["joint_parent_link"]
    j_childR   = maps["joint_child_real"]

    # Gather world FK for real links (the ones your solver knows)
    world_T = {}
    for link in link_path.keys():
        # we only care about *real* links: those that appear as keys in link_path AND in your FK
        try:
            Tw = robot.get_T_world_frame(link)
        except Exception:
            continue
        world_T[link] = Tw

    # Place the base (optional)
    if base in world_T:
        Tb = world_T[base]
        rr.log(link_path[base], rr.Transform3D(
            translation=Tb[:3,3].tolist(),
            mat3x3=Tb[:3,:3],
        ))

    # For each actuated joint, compute parent->real_child relative and log ON THE JOINT
    for jname, jpath in joint_path.items():
        if j_type.get(jname) not in ("revolute", "continuous", "prismatic"):
            continue
        parent = j_parent[jname]
        child  = j_childR.get(jname)  # resolved real child (e.g., link1, link2, …)
        if parent not in world_T or child not in world_T:
            # name mismatch between URDF links and your FK — print once if needed
            # print(f"[skip] {jname}: missing FK for {parent} or {child}")
            continue

        T_rel = _inv(world_T[parent]) @ world_T[child]
        R_rel = T_rel[:3,:3]
        t_rel = T_rel[:3, 3]
        axis  = j_axis[jname]

        if j_type[jname] in ("revolute", "continuous"):
            axR, theta = _axis_angle_from_R(R_rel)
            signed_theta = float(theta * np.sign(np.dot(axis, axR)))
            rr.log(jpath, rr.Transform3D(
                rotation=rr.RotationAxisAngle(axis=axis.tolist(), angle=signed_theta)
            ))
        else:  # prismatic
            disp = float(np.dot(axis, t_rel))
            rr.log(jpath, rr.Transform3D(
                translation=(axis * disp).tolist()
            ))

URDF = "/home/steven/research/low_cost_robot/simulation/low_cost_robot_6dof/low-cost-arm.urdf"

# Log the URDF once (static) under "robot/..."
rr.log_file_from_path(URDF, entity_path_prefix="robot", static=True)

# Build maps that skip the fixed “jointbody/offset” chain
maps = build_maps_skip_fixed(URDF, root_prefix="robot")

for j, child in maps["joint_child_real"].items():
    print(j, "parent=", maps["joint_parent_link"][j], "child_real=", child)

def visualize_robot(robot, step: int = 0):
    log_joints_from_world_fk(step, robot, maps)
