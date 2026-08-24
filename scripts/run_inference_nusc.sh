CONFIG=${1:-src/drivingdepth/configs/inference_nusc_4f1s1i.yaml}
NUM_PROCESSES=${NUM_PROCESSES:-8}

accelerate launch \
    --main_process_port 25901 \
    --num_processes="${NUM_PROCESSES}" \
    src/drivingdepth/inference/__main__.py \
    --config "${CONFIG}"

# Print the aggregated per-view metrics from summary.json
SAVE_DIR=$(python - "$CONFIG" <<'EOF'
import sys
from omegaconf import OmegaConf
print(OmegaConf.load(sys.argv[1]).inference.save_dir or "")
EOF
)

if [ -n "$SAVE_DIR" ]; then
    python scripts/show_summary.py "${SAVE_DIR}/nuscenes/summary.json"
fi
