# Custom Voice Model Training Guide

This document provides a comprehensive guide for recording, preprocessing, training, evaluating, and exporting custom Retrieval-based Voice Conversion (RVC) models for use with the Real-Time AI Voice Changer.

---

## 1. Dataset Requirements & Quality Guidelines

The quality of your trained AI voice model directly depends on the cleanliness, consistency, and fidelity of your training audio dataset.

### Dataset Checklist
- **Target Audio Duration**:
  - *Minimum*: **10 minutes** of clean, continuous speech.
  - *Optimal*: **20 to 45 minutes** of speech across varying emotional dynamics.
  - *Diminishing Returns*: Datasets longer than 2 hours without quality control increase training time without significant fidelity gains.
- **Sample Rate & Format**: Uncompressed WAV format (`16-bit` or `24-bit` PCM), **48kHz sample rate**, mono or stereo (auto-converted to mono during preprocessing).
- **Acoustic Environment**:
  - Dry room with minimal acoustic reverb / room reflection.
  - Zero background noise (no fans, AC hum, mechanical keyboard clicks, or crosstalk).
  - No background music or overlapping speakers.
- **Vocal Delivery**:
  - Clear pronunciation and natural speaking cadence.
  - Include pitch variation (conversational, energetic, calm tone) representative of target stream usage.
  - Avoid vocal fry, heavy whispering, or excessive clipping/distortion.

---

## 2. Dataset Preparation & Preprocessing

### Step 1: Directory Structure Setup
Place raw WAV audio recordings into a designated dataset directory:

```text
ai-voice-changer/
└── data/
    └── target_speaker/
        ├── raw_recording_01.wav
        ├── raw_recording_02.wav
        └── raw_recording_03.wav
```

---

### Step 2: Audio Slicing & Voice Activity Detection (VAD)
Raw recordings must be sliced into small, manageable chunks (typically 2 to 10 seconds in length) and loudness-normalized to -23 LUFS.

Run the automated preprocessing command:

```bash
python src/training/preprocess.py \
  --input-dir ./data/target_speaker/ \
  --output-dir ./logs/target_speaker_preprocessed/ \
  --sample-rate 48000 \
  --min-interval-ms 300 \
  --max-duration-sec 10.0 \
  --normalize-lufs -23.0
```

#### What Preprocessing Executes:
1. **Denoising**: WebRTC VAD and high-pass filtering (> 50Hz) to strip rumble.
2. **Audio Slicing**: Splits long files on silences greater than 300ms.
3. **Resampling**: Standardizes all audio buffers to 40kHz or 48kHz.
4. **Loudness Normalization**: Normalizes dynamic peak levels across all slices.

---

### Step 3: Feature Extraction (ContentVec & Pitch Tracking)

After slicing, extract semantic feature representations and pitch contours ($F_0$) for every slice.

```bash
python src/training/extract_features.py \
  --dataset-dir ./logs/target_speaker_preprocessed/ \
  --pitch-algorithm rmvpe \
  --embedder-model hubert_base \
  --device cuda:0
```

- **Feature Embedder**: Uses `ContentVec` / `HuBERT` base model to convert wave slices into 256-dimensional phonetic vector files (`.npy`).
- **Pitch Trackers**:
  - **`rmvpe`** (Recommended): Fast, robust against noise, high pitch accuracy.
  - **`harvest`**: Maximum accuracy for singing or wide pitch ranges.
  - **`pm`**: Fastest extraction speed (CPU friendly).

---

## 3. Model Training Execution

Once feature extraction completes, initialize the neural training pipeline.

```bash
python src/training/train.py \
  --model-name target_speaker \
  --sample-rate 48000 \
  --batch-size 8 \
  --epochs 300 \
  --save-frequency 50 \
  --learning-rate 0.0001 \
  --gpu-id 0
```

### VRAM & Hardware Guidelines

| Batch Size | VRAM Needed | GPU Examples | Recommended Epoch Count |
| :--- | :--- | :--- | :--- |
| **4** | 6 GB VRAM | GTX 1660 Super, RTX 2060 | 300 - 400 epochs |
| **8** | 8 GB VRAM | RTX 3060, RTX 4060 | 250 - 300 epochs |
| **16** | 12+ GB VRAM | RTX 3080, RTX 4070 | 200 - 250 epochs |
| **32** | 24 GB VRAM | RTX 3090, RTX 4090 | 150 - 200 epochs |

---

### TensorBoard Monitoring

Track generator loss, discriminator loss, pitch accuracy, and spectrogram reconstruction in real time:

```bash
tensorboard --logdir ./logs/target_speaker/
```

Open `http://localhost:6006` in your browser.

#### Key Metrics to Watch:
- **`g/loss_g`** (Generator Loss): Should smoothly decrease and plateau around 1.5 - 2.5.
- **`d/loss_d`** (Discriminator Loss): Stabilizes around 0.5 - 1.0.
- **Overfitting Indicator**: If discriminator loss drops close to 0 while generator loss spikes, reduce total epochs or increase dataset diversity.

---

## 4. Feature Index Creation (FAISS)

To enhance fine timbral accuracy and accent matching, generate a FAISS (Facebook AI Similarity Search) index file.

```bash
python src/training/create_index.py \
  --dataset-dir ./logs/target_speaker_preprocessed/ \
  --output-path ./models/weights/target_speaker.index
```

The output `.index` file stores feature vectors used during real-time inference for nearest-neighbor retrieval.

---

## 5. Model Export & Optimization

### Exporting Checkpoint Weights

Convert the heavy training state (`.pth` containing optimizer states) into a lean runtime weights file:

```bash
python src/training/export.py \
  --checkpoint ./logs/target_speaker/G_300.pth \
  --model-name target_speaker \
  --output-dir ./models/weights/
```

This generates:
- `models/weights/target_speaker.pth` (Small inference model, ~50MB - 60MB).
- `models/weights/target_speaker.index` (Feature index file, ~10MB - 30MB).

---

### Exporting to ONNX / TensorRT for Low Latency

For maximum performance in real-time mode, compile the `.pth` weights into an optimized ONNX or TensorRT engine:

```bash
# Export to ONNX FP16
python src/training/export_onnx.py \
  --model-path ./models/weights/target_speaker.pth \
  --output-path ./models/weights/target_speaker.onnx \
  --fp16

# Build TensorRT Engine (NVIDIA GPUs only)
trtexec --onnx=./models/weights/target_speaker.onnx \
        --saveEngine=./models/weights/target_speaker.engine \
        --fp16 \
        --minShapes=speech:1x1x256 \
        --optShapes=speech:1x1x512 \
        --maxShapes=speech:1x1x2048
```

---

## 6. Model Evaluation & Testing

Validate your newly trained model before deploying it live:

```bash
# Test offline conversion on a test audio file
python src/inference/eval.py \
  --model-path ./models/weights/target_speaker.pth \
  --index-path ./models/weights/target_speaker.index \
  --input-audio ./tests/fixtures/sample_voice.wav \
  --output-audio ./output_test.wav \
  --pitch-shift 0 \
  --index-rate 0.6
```

### Listening Checklist
- **Timbre Match**: Does the voice sound authentically like the target speaker?
- **Artifacts**: Listen for metallic metallic reverberation, bubbly pitch artifacts, or breath cut-offs.
- **Pitch Shift Test**: Test shifting pitch $+4$ or $-4$ semitones to ensure stability across ranges.
- **Index Rate Adjustment**:
  - `0.0`: Fast inference, relies purely on neural model (less memory).
  - `0.6 - 0.7`: Recommended balance between feature retrieval and model generalization.
  - `1.0`: Maximum feature matching (may introduce stutter if dataset contains noise).
