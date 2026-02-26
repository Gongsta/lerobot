# Bimanual Koch — DAgger Data Collection

DAgger (Dataset Aggregation) lets you run a trained policy on the robot and intervene in real-time whenever it makes a mistake. Your corrections are recorded as a separate dataset. Retraining on both the original and correction data produces a policy that handles failure states it actually encounters.

---

## Hardware Setup

### Finding Ports

```bash
lerobot-find-port
```

### Environment Variables

Add these to your `~/.bashrc` (or `~/.zshrc`):

```bash
# macOS (USB modem paths)
export FOLLOWER_LEFT_PORT=/dev/tty.usbmodem14201
export FOLLOWER_RIGHT_PORT=/dev/tty.usbmodem1301
export LEADER_LEFT_PORT=/dev/tty.usbmodem1201
export LEADER_RIGHT_PORT=/dev/tty.usbmodem1101
export TOP_CAMERA_INDEX_OR_PATH=0
export FRONT_CAMERA_INDEX_OR_PATH=1

# Linux (udev rule paths — see setup instructions for udev config)
export FOLLOWER_LEFT_PORT=/dev/follower-left
export FOLLOWER_RIGHT_PORT=/dev/follower-right
export LEADER_LEFT_PORT=/dev/leader-left
export LEADER_RIGHT_PORT=/dev/leader-right
export TOP_CAMERA_INDEX_OR_PATH=/dev/camera-top
export FRONT_CAMERA_INDEX_OR_PATH=/dev/camera-front
```

### Calibration

```bash
lerobot-calibrate \
    --robot.type=bi_koch_follower \
    --robot.left_arm_port=$FOLLOWER_LEFT_PORT \
    --robot.right_arm_port=$FOLLOWER_RIGHT_PORT \
    --robot.id=bimanual_follower

lerobot-calibrate \
    --teleop.type=bi_koch_leader \
    --teleop.left_arm_port=$LEADER_LEFT_PORT \
    --teleop.right_arm_port=$LEADER_RIGHT_PORT \
    --teleop.id=bimanual_leader
```

---

## Teleoperation (no policy)

```bash
lerobot-teleoperate \
    --robot.type=bi_koch_follower \
    --robot.left_arm_port=$FOLLOWER_LEFT_PORT \
    --robot.right_arm_port=$FOLLOWER_RIGHT_PORT \
    --robot.id=bimanual_follower \
    --robot.cameras="{ top: {type: opencv, index_or_path: $TOP_CAMERA_INDEX_OR_PATH, width: 640, height: 480, fps: 30} }" \
    --teleop.type=bi_koch_leader \
    --teleop.left_arm_port=$LEADER_LEFT_PORT \
    --teleop.right_arm_port=$LEADER_RIGHT_PORT \
    --teleop.id=bimanual_leader \
    --display_data=true
```

Keyboard controls during recording:
- `ESC` — stop data collection
- `→` (right arrow) — terminate episode early
- `←` (left arrow) — re-record episode

---

## Policy Inference (no correction data)

Use this to evaluate a trained policy before running DAgger.

```bash
export HF_USER=Gongsta

lerobot-record \
    --robot.type=bi_koch_follower \
    --robot.left_arm_port=$FOLLOWER_LEFT_PORT \
    --robot.right_arm_port=$FOLLOWER_RIGHT_PORT \
    --robot.id=bimanual_follower \
    --robot.cameras="{ top: {type: opencv, index_or_path: $TOP_CAMERA_INDEX_OR_PATH, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: $LEFT_WRIST_CAMERA_INDEX_OR_PATH, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: $RIGHT_WRIST_CAMERA_INDEX_OR_PATH, width: 640, height: 480, fps: 30} }" \
    --dataset.repo_id=${HF_USER}/eval_$(date +%Y-%m-%d_%H-%M-%S) \
    --dataset.single_task="Fold the t-shirt and put it in the bin" \
    --display_data=true \
    --policy.path=${HF_USER}/act_policy_koch-tshirt-folding-v2 \
    --policy.device=mps \
    --dataset.push_to_hub=false
```

### Available Trained Policies

| Policy | Repo ID |
|--------|---------|
| ACT (t-shirt folding v2) | `Gongsta/act_policy_koch-tshirt-folding-v2` |
| ACT (t-shirt shirt) | `Gongsta/act_koch-shirt` |
| SmolVLA | `Gongsta/smolvla-hf-policy-koch-baby-tshirt-folding` |
| pi0 | `Gongsta/pi0-hf-policy-koch-baby-tshirt-folding` |

---

## DAgger Collection

DAgger runs the policy and lets you intervene by pressing **SPACE** on the keyboard. This requires `--teleop.intervention_enabled=true` and an `--intervention_repo_id` for the correction dataset.

### How it works

1. Policy runs on both follower arms. By default, leader arms mirror the follower (inverse-follow) so they are already at the right position when you grab them.
2. Press **SPACE** — leader arm torque disables, you take over both arms simultaneously.
3. Press **SPACE** again — leader arm torque re-engages, policy resumes. Your correction is saved as a separate episode in the corrections dataset.
4. At the end of each main episode, all correction fragments are saved in order.

### USB-powered leader arms (no wall adapter)

If your leader arms are powered only through USB (not connected to a wall adapter), enabling torque could draw too much current through the USB port. Add `--teleop.inverse_follow=false` to disable all torque writes to the leader arms. The SPACE-key intervention toggle still works — the only trade-off is that the leader arms won't track the follower during policy execution, so there may be a position jump when you grab them to intervene.

### Command

```bash
export HF_USER=Gongsta

lerobot-record \
    --robot.type=bi_koch_follower \
    --robot.left_arm_port=$FOLLOWER_LEFT_PORT \
    --robot.right_arm_port=$FOLLOWER_RIGHT_PORT \
    --robot.id=bimanual_follower \
    --robot.cameras="{ top: {type: opencv, index_or_path: $TOP_CAMERA_INDEX_OR_PATH, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: $LEFT_WRIST_CAMERA_INDEX_OR_PATH, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: $RIGHT_WRIST_CAMERA_INDEX_OR_PATH, width: 640, height: 480, fps: 30} }" \
    --teleop.type=bi_koch_leader \
    --teleop.left_arm_port=$LEADER_LEFT_PORT \
    --teleop.right_arm_port=$LEADER_RIGHT_PORT \
    --teleop.id=bimanual_leader \
    --teleop.intervention_enabled=true \
    --dataset.repo_id=${HF_USER}/koch-tshirt-dagger-main \
    --dataset.single_task="Fold the t-shirt and put it in the bin" \
    --dataset.num_episodes=10 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=15 \
    --intervention_repo_id=${HF_USER}/koch-tshirt-dagger-corrections \
    --policy.path=${HF_USER}/act_policy_koch-tshirt-folding-v2 \
    --display_data=true --teleop.inverse_follow=false
```

To resume a previous DAgger session (adds episodes to existing datasets):

```bash
# Add --resume=true to the command above
    --resume=true \
```

### Outputs

After collection you will have two datasets:

| Dataset | Contents |
|---------|----------|
| `koch-tshirt-dagger-main` | Full-length episodes (policy behavior, with gaps where you intervened) |
| `koch-tshirt-dagger-corrections` | Short correction episodes — `(observation, human_action)` pairs at failure points |

---

## Re-training After DAgger

Combine both datasets using LeRobot's dataset merge tools, then retrain. Example with ACT:

```bash
export HF_USER=Gongsta

# Merge datasets (original demo data + DAgger corrections)
lerobot-merge-datasets \
    --repo_ids ${HF_USER}/koch-baby-tshirt-folding ${HF_USER}/koch-tshirt-dagger-corrections \
    --output_repo_id ${HF_USER}/koch-tshirt-dagger-merged

# Retrain ACT on merged data
lerobot-train \
    --dataset.repo_id=${HF_USER}/koch-tshirt-dagger-merged \
    --dataset.video_backend=pyav \
    --policy.type=act \
    --output_dir=outputs/act_koch-shirt-dagger \
    --job_name=act_koch_dagger \
    --policy.device=cuda \
    --wandb.enable=true \
    --save_freq=2000 \
    --steps=20000 \
    --policy.repo_id=${HF_USER}/act_koch-shirt-dagger \
    --num_workers=12
```

Then evaluate the new policy using the inference command above, and repeat the DAgger loop as needed.
