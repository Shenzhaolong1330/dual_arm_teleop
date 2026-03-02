def main():
    print("""
==================================================
 Dobot Nova5 Dual-Arm Teleoperation - Command Reference
==================================================

Core Commands:
  dobot-record           Record teleoperation dataset
  dobot-replay           Replay a recorded dataset
  dobot-visualize        Visualize recorded dataset
  dobot-reset            Reset the robot to initial state
  dobot-train            Train a policy on the recorded dataset

Utility Commands:
  utils-joint-offsets    Compute joint offsets for teleoperation

Tool Commands:
  tools-check-dataset    Check local dataset information
  tools-check-rs         Retrieve connected RealSense camera serial numbers
  tools-check-robotiq    Check Robotiq gripper serial ports

Shell Tools:
  check_robotiq_ports.sh  Get Robotiq gripper serial ports

Test Commands:
  test-gripper-ctrl      Run gripper control command (operate the gripper)

--------------------------------------------------
 Tip: Use 'dobot-help' anytime to see this summary.
==================================================
""")
