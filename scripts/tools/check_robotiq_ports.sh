#!/bin/bash
# Identify Robotiq gripper serial devices and print their /dev nodes and attributes
# Usage: check_robotiq_ports.sh [--json] [--verbose] [--help]

set -euo pipefail

SCRIPT_NAME=$(basename "$0")

print_usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [options]

Options:
  --json       Output results as JSON array
  --verbose    Print more device properties
  --help       Show this help message

Description:
  Scan /dev/serial/by-id and system udev properties to find serial devices
  that are likely Robotiq grippers (by matching name/vendor/model case-insensitively).
  If none are found, the script will list all serial devices and their properties to
  help identification.
EOF
}

OUTPUT_JSON=false
VERBOSE=false

while [ "$#" -gt 0 ]; do
    case "$1" in
        --json) OUTPUT_JSON=true; shift ;;
        --verbose) VERBOSE=true; shift ;;
        --help) print_usage; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; print_usage; exit 2 ;;
    esac
done

BYID_DIR=/dev/serial/by-id
matches=()

collect_info() {
    local byid_path="$1"
    local name
    name=$(basename -- "$byid_path")
    local devpath
    devpath=$(readlink -f -- "$byid_path" 2>/dev/null || true)
    if [ -z "$devpath" ]; then
        return
    fi
    # get udev properties if possible
    local props
    props=$(udevadm info --query=property --name="$devpath" 2>/dev/null || true)
    local vendor=$(echo "$props" | awk -F= '/ID_VENDOR=/ {print $2}')
    local model=$(echo "$props" | awk -F= '/ID_MODEL=/ {print $2}')
    local vid=$(echo "$props" | awk -F= '/ID_VENDOR_ID=/ {print $2}')
    local pid=$(echo "$props" | awk -F= '/ID_MODEL_ID=/ {print $2}')

    printf '%s\t%s\t%s\t%s\t%s\n' "$byid_path" "$devpath" "$vendor" "$model" "$vid:$pid"
}

is_robotiq_like() {
    # Accept if by-id name or vendor/model contains 'robotiq' (case-insensitive)
    local byid_path="$1"
    local info_line
    info_line=$(collect_info "$byid_path") || return 1
    # fields: byid, devpath, vendor, model, vid:pid
    local byid_name dev vendor model ids
    byid_name=$(echo "$info_line" | awk -F"\t" '{print $1}')
    dev=$(echo "$info_line" | awk -F"\t" '{print $2}')
    vendor=$(echo "$info_line" | awk -F"\t" '{print $3}')
    model=$(echo "$info_line" | awk -F"\t" '{print $4}')
    ids=$(echo "$info_line" | awk -F"\t" '{print $5}')

    local lower_name=$(echo "$byid_name $vendor $model" | tr '[:upper:]' '[:lower:]')
    if echo "$lower_name" | grep -qi 'robotiq'; then
        printf '%s\t%s\t%s\t%s\t%s' "$byid_name" "$dev" "$vendor" "$model" "$ids"
        return 0
    fi

    # Fallback: some Robotiq devices use FTDI/other USB-serial adapters. If vendor/model
    # contain 'ftdi' or known adapter names but the by-id name contains 'Robotiq' (covered above),
    # we avoid guessing too broadly. For now, only match explicit 'robotiq' strings.
    return 1
}

if [ ! -d "$BYID_DIR" ]; then
    echo "No $BYID_DIR directory; no serial-by-id entries available." >&2
    exit 1
fi

shopt -s nullglob
all_byid=("$BYID_DIR"/*)
shopt -u nullglob

if [ ${#all_byid[@]} -eq 0 ]; then
    echo "No devices under $BYID_DIR" >&2
    exit 1
fi

results=()

for p in "${all_byid[@]}"; do
    if info=$(is_robotiq_like "$p"); then
        # info fields: name, dev, vendor, model, ids
        name=$(echo "$info" | awk -F"\t" '{print $1}')
        dev=$(echo "$info" | awk -F"\t" '{print $2}')
        vendor=$(echo "$info" | awk -F"\t" '{print $3}')
        model=$(echo "$info" | awk -F"\t" '{print $4}')
        ids=$(echo "$info" | awk -F"\t" '{print $5}')
        results+=("$name|$dev|$vendor|$model|$ids")
    fi
done

if [ ${#results[@]} -gt 0 ]; then
    if [ "$OUTPUT_JSON" = true ]; then
        # build JSON array
        printf '['
        first=true
        for r in "${results[@]}"; do
            IFS='|' read -r name dev vendor model ids <<< "$r"
            # escape strings minimally
            name_js=$(printf '%s' "$name" | sed 's/"/\\"/g')
            vendor_js=$(printf '%s' "$vendor" | sed 's/"/\\"/g')
            model_js=$(printf '%s' "$model" | sed 's/"/\\"/g')
            dev_js=$(printf '%s' "$dev" | sed 's/"/\\"/g')
            ids_js=$(printf '%s' "$ids" | sed 's/"/\\"/g')
            if [ "$first" = true ]; then
                first=false
            else
                printf ', '
            fi
            printf '{"by-id":"%s","dev":"%s","vendor":"%s","model":"%s","ids":"%s"}' "$name_js" "$dev_js" "$vendor_js" "$model_js" "$ids_js"
        done
        printf '\n]\n'
    else
        echo "Found ${#results[@]} Robotiq-like serial device(s):"
        for r in "${results[@]}"; do
            IFS='|' read -r name dev vendor model ids <<< "$r"
            echo "- by-id: $name"
            echo "  dev:    $dev"
            echo "  vendor: ${vendor:-N/A}"
            echo "  model:  ${model:-N/A}"
            echo "  ids:    ${ids:-N/A}"
            if [ "$VERBOSE" = true ]; then
                echo "  udev info:"
                udevadm info --query=property --name="$dev" | sed 's/^/    /'
            fi
        done
    fi
    exit 0
else
    echo "No Robotiq-like devices detected under $BYID_DIR. Listing all serial devices to help identification:"
    for p in "${all_byid[@]}"; do
        info=$(collect_info "$p") || continue
        byid_name=$(echo "$info" | awk -F"\t" '{print $1}')
        dev=$(echo "$info" | awk -F"\t" '{print $2}')
        vendor=$(echo "$info" | awk -F"\t" '{print $3}')
        model=$(echo "$info" | awk -F"\t" '{print $4}')
        ids=$(echo "$info" | awk -F"\t" '{print $5}')
        echo "- by-id: $byid_name"
        echo "  dev:    $dev"
        echo "  vendor: ${vendor:-N/A}"
        echo "  model:  ${model:-N/A}"
        echo "  ids:    ${ids:-N/A}"
        if [ "$VERBOSE" = true ]; then
            echo "  udev info:"
            udevadm info --query=property --name="$dev" | sed 's/^/    /'
        fi
    done
    echo "\nIf you expect Robotiq devices but none were detected, plug them in and run this script, or check dmesg/udevadm to find identifying strings (manufacturer/model)."
    exit 1
fi
