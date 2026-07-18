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
        "zarr>=2.12,<3",
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
            "tools-check-rgbd-sidecar = scripts.check_rgbd_sidecar_dataset:main",
            "tools-export-rgbd-preview = scripts.tools.export_rgbd_sidecar_preview:main",
            "tools-export-ffs-pair = scripts.tools.export_ffs_stereo_pair:main",
            "tools-benchmark-rgbd-sidecar = scripts.tools.benchmark_rgbd_zarr_sidecar:main",
            "tools-export-realsense-calibration = scripts.tools.export_realsense_calibration:main",
            "tools-check-rs = scripts.tools.rs_devices:main",
            "tools-reset-rs = scripts.tools.reset_realsense_usb:main",
            "tools-preprocess-dataset = scripts.tools.preprocess_dataset:main",
            "tools-split-label-dataset = scripts.tools.split_label_dataset:main",
            "tools-merge-datasets = scripts.tools.merge_lerobot_datasets:main",
            "tools-monitor-flexiv-force = scripts.tools.monitor_flexiv_force:main",

            # unified help command
            "robot-help = scripts.help.help_info:main",
        ]
    },
)
