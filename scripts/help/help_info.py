def main():
    print("""
==================================================
Dual Arm Teleoperation, Data collection and policy training
Command Reference
==================================================

Core Commands:
  robot-record           Record teleoperation dataset
  robot-replay           Replay a recorded dataset
  robot-visualize        Visualize recorded dataset
  robot-reset            Reset the robot to initial state
  robot-train            Train a policy on the recorded dataset
  robot-dagger           Run DAgger rounds for policy improvement
  robot-dagger-export    Export DAgger data from raw run_mix logs

Tool Commands:
  tools-check-dataset    Check local dataset information
  tools-check-dagger-dataset
                         Audit an exported ACT DAgger dataset before training
  tools-check-rs         Retrieve connected RealSense camera serial numbers
  tools-preprocess-dataset
                         Preprocess a LeRobot dataset for ACT training
  tools-split-label-dataset
                         Split long LeRobot episodes and label sub-episodes
  tools-merge-datasets
                         Merge multiple local LeRobot datasets into one

Shell Tools:
  check_robotiq_ports.sh  Get Robotiq gripper serial ports
  map_gripper.sh          Helper for gripper device mapping

--------------------------------------------------
 Tip: Use 'robot-help' anytime to see this summary.
==================================================
""")
