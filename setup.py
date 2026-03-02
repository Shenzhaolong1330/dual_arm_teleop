from setuptools import setup, find_packages
from pathlib import Path

# ====== Project root ======
ROOT = Path(__file__).parent.resolve()

setup(
    name="lerobot_dobot_dual_arm_teleop",
    version="0.1.0",
    description="Dobot Nova5 dual-arm teleoperation and dataset collection utilities",
    python_requires=">=3.10",
    packages=find_packages(where=".", include=["scripts*", "scripts.*"]),
    include_package_data=True,
    install_requires=[
        "send2trash",
        f"lerobot_robot @ file:///{ROOT}/lerobot_robot",
        f"lerobot_teleoperator @ file:///{ROOT}/lerobot_teleoperator"
    ],
    scripts=[
        "scripts/tools/check_master_port.sh",
    ],
    entry_points={
        "console_scripts": [
            # core commands
            "dobot-record = scripts.core.run_record:main",
            "dobot-replay = scripts.core.run_replay:main",
            "dobot-visualize = scripts.core.run_visualize:main",
            "dobot-reset = scripts.core.reset_robot:main",
            "dobot-train = scripts.core.run_train:main",

            # tools commands (helper tools)
            "tools-check-dataset = scripts.tools.check_dataset_info:main",
            "tools-check-rs = scripts.tools.rs_devices:main",

            # unified help command
            "dobot-help = scripts.help.help_info:main",
        ]
    },
)
