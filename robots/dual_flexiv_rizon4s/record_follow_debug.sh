#!/usr/bin/env bash
# Generate temporary diagnostic configs; leave the normal recording config intact.
set -euo pipefail
source /home/ziang/szl_ws/flexiv-local/env.sh
task_repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
task_debug_dir="$(mktemp -d /tmp/flexiv-follow-debug.XXXXXX)"
python - "$task_repo_root" "$task_debug_dir" <<'PY'
import sys
from pathlib import Path
import yaml
root, dest = map(Path, sys.argv[1:])
record = yaml.safe_load((root / "scripts/config/record_cfg.yaml").read_text())
hardware = yaml.safe_load((root / "scripts/config/robots/flexiv_config.yaml").read_text())
if record["record"]["robot_type"] != "flexiv_dual_arm":
    raise ValueError("This diagnostic is only for the Flexiv workstation")
hardware["robot"].update(action_debug=True, action_debug_every_n=15,
                         timing_debug=True, timing_debug_every_n=30)
hardware["teleop"]["oculus_config"].update(timing_debug=True, timing_debug_every_n=30)
record["record"]["das_config_path"] = str(dest / "hardware.yaml")
(dest / "hardware.yaml").write_text(yaml.safe_dump(hardware, sort_keys=False))
(dest / "record.yaml").write_text(yaml.safe_dump(record, sort_keys=False))
print(f"Diagnostic configuration: {dest / 'record.yaml'}", flush=True)
PY
if [[ "${1:-}" == "--prepare-only" ]]; then
    exit 0
fi
cd "$task_repo_root"
task_log_path="/home/ziang/szl_ws/flexiv-local/logs/follow-debug-$(date +%Y%m%d-%H%M%S).log"
echo "Recording diagnostic log: $task_log_path"
python -u scripts/core/run_record.py --config "$task_debug_dir/record.yaml" 2>&1 | tee "$task_log_path"
