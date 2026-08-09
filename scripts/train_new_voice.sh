#!/usr/bin/env bash
# AI Voice Changer - Voice Model Training Pipeline Launcher

set -e

DATASET_DIR="${1:-./dataset}"
MODEL_NAME="${2:-custom_voice_model}"
EPOCHS="${3:-100}"
BATCH_SIZE="${4:-16}"

echo "==================================================="
echo "  AI Voice Changer - Model Training Pipeline"
echo "==================================================="
echo "Dataset Path : ${DATASET_DIR}"
echo "Model Name   : ${MODEL_NAME}"
echo "Epochs       : ${EPOCHS}"
echo "Batch Size   : ${BATCH_SIZE}"
echo "==================================================="

# Validate Dataset Directory
if [ ! -d "$DATASET_DIR" ]; then
    echo "[ERROR] Dataset directory '${DATASET_DIR}' does not exist!"
    echo "Usage: $0 <dataset_directory> [model_name] [epochs] [batch_size]"
    exit 1
fi

# Count audio files in dataset
AUDIO_COUNT=$(find "$DATASET_DIR" -type f \( -name "*.wav" -o -name "*.flac" -o -name "*.mp3" \) | wc -l)
echo "[INFO] Found ${AUDIO_COUNT} target audio sample files in dataset."

if [ "$AUDIO_COUNT" -eq 0 ]; then
    echo "[ERROR] No .wav, .flac, or .mp3 files found in '${DATASET_DIR}'."
    exit 1
fi

# Activate Virtual Environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check CUDA device
if command -v nvidia-smi &> /dev/null; then
    echo "[OK] NVIDIA GPU detected for training."
else
    echo "[WARNING] No GPU detected! Training on CPU will be extremely slow."
fi

# Execute training script
echo "[INFO] Initiating feature extraction and pitch processing..."
python3 -m app.training.train \
    --dataset_path "$DATASET_DIR" \
    --model_name "$MODEL_NAME" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --sample_rate 40000 \
    --f0_method rmvpe

echo "==================================================="
echo "[SUCCESS] Voice model training finished! Model saved to models/${MODEL_NAME}.pth"
echo "==================================================="
