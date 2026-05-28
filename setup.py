from setuptools import setup, find_packages

setup(
    name="dual_arm_teleop",
    version="0.1.0",
    description="Dual Arm Teleoperation, Data collection and policy training",
    python_requires=">=3.10",
    packages=find_packages(
        where=".",
        include=[
            "scripts*",
            "scripts.*",
            "robots*",
            "robots.*",
            "teleoperators*",
            "teleoperators.*",
            "lerobot_robot_agilex_nero*",
            "lerobot_teleoperator_oculus*",
        ],
    ),
    include_package_data=True,
    install_requires=[
        "send2trash",
        "pyrealsense2",
        "scipy",
        "zerorpc",
        "numpy",
        "easyhid",
        "OneEuroFilter>=0.2.1",
    ],
    entry_points={
        "console_scripts": [
            # core commands
            "robot-record = scripts.core.run_record:main",
            "robot-replay = scripts.core.run_replay:main",
            "robot-visualize = scripts.core.run_visualize:main",
            "robot-reset = scripts.core.reset_robot:main",
            "robot-train = scripts.core.run_train:main",
            "robot-dagger = scripts.core.run_dagger_rounds:main",
            "robot-dagger-export = scripts.core.run_dagger_export:main",

            # tools commands (helper tools)
            "tools-check-dataset = scripts.tools.check_dataset_info:main",
            "tools-check-dagger-dataset = scripts.tools.check_dagger_dataset:main",
            "tools-check-rs = scripts.tools.rs_devices:main",
            "tools-preprocess-dataset = scripts.tools.preprocess_dataset:main",
            "tools-split-label-dataset = scripts.tools.split_label_dataset:main",

            # unified help command
            "robot-help = scripts.help.help_info:main",
        ]
    },
)
