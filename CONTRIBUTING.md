# Contributing Guidelines

Thank you for your interest in contributing to the **Real-Time AI Voice Changer** project! We welcome contributions, bug reports, feature proposals, and performance optimizations.

To maintain code quality, security, and ultra-low latency standards, please review the following guidelines before submitting issues or Pull Requests.

---

## 1. Development Environment Setup

### Prerequisites
- Python 3.10 or higher
- Git LFS (Large File Storage for ML models and test audio binaries)
- CUDA Toolkit 11.8 or 12.1 (if developing GPU feature acceleration)
- Node.js 18+ and `pnpm` / `npm` (if working on the web dashboard)

### Setting Up Your Local Repository

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR-USERNAME/ai-voice-changer.git
cd ai-voice-changer

# 2. Initialize Git LFS
git lfs install
git lfs pull

# 3. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# 4. Install development & testing dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 5. Install pre-commit hooks
pre-commit install
```

---

## 2. Code Style Guidelines

To keep the codebase maintainable and readable across real-time DSP, deep learning inference, and API server modules, we strictly enforce linting and formatting.

### Python Guidelines
- **Formatting**: We use `black` with a line length of **88 characters**.
- **Import Sorting**: We use `isort` configured for `black` profile.
- **Linting**: We use `ruff` or `flake8` to catch common syntax and code quality errors.
- **Type Hinting**: All new Python functions must include static type annotations validated by `mypy`.

```bash
# Run formatters and linters
black src tests
isort src tests
ruff check src tests
mypy src
```

#### Code Snippet Format Example
```python
from typing import Optional, Tuple
import numpy as np

def apply_gain_staging(
    audio_buffer: np.ndarray,
    target_gain_db: float = 0.0,
    clip_protection: bool = True
) -> Tuple[np.ndarray, float]:
    """Applies clean linear gain adjustment with optional hard clipping guard.

    Args:
        audio_buffer: Input PCM float32 array in range [-1.0, 1.0].
        target_gain_db: Gain value in decibels.
        clip_protection: If True, limits peak amplitude to 0.99.

    Returns:
        A tuple of (processed_buffer, peak_amplitude).
    """
    gain_linear = 10.0 ** (target_gain_db / 20.0)
    output_buffer = audio_buffer * gain_linear
    
    if clip_protection:
        output_buffer = np.clip(output_buffer, -0.99, 0.99)
        
    peak = float(np.max(np.abs(output_buffer)))
    return output_buffer, peak
```

### TypeScript / Frontend Guidelines
- Use **TypeScript** strict mode.
- Use **ESLint** and **Prettier** for formatting.
- Components should be functional and follow React best practices.

---

## 3. Pull Request Process

### Branch Naming Strategy
Use the following prefix conventions when creating branches:
- `feature/` - New features or capabilities (e.g., `feature/tensorrt-support`)
- `fix/` - Bug fixes or stability patches (e.g., `fix/buffer-underrun-mac`)
- `perf/` - Latency or memory optimization (e.g., `perf/reduce-crepe-latency`)
- `docs/` - Documentation additions or corrections (e.g., `docs/obs-guide`)
- `refactor/` - Code structure improvements without functional change

### Commit Messages
Follow the Conventional Commits format:
```text
feat(inference): add INT8 quantization support for RVC models
fix(audio): eliminate ring buffer deadlock under high CPU load
docs(readme): add troubleshooting section for BlackHole audio driver
test(dsp): add unit tests for formant pitch shifter filter
```

### Pull Request Submission Checklist
1. **Target Branch**: Ensure PRs target `main` (or `develop` if active).
2. **Tests**: Ensure all existing tests pass (`pytest`) and add new test coverage for new functionality.
3. **Documentation**: Update docstrings and markdown files in `docs/` as necessary.
4. **Latency Verification**: If changing `src/audio` or `src/inference`, verify that end-to-end processing latency remains under `< 50ms`.
5. **No Secrets/Weights**: Do **NOT** commit trained `.pth`, `.onnx`, or `.engine` binary weight files into Git directly. Use release artifacts or LFS test fixtures.

---

## 4. Testing Guidelines

We use `pytest` for Python testing. Tests are divided into unit tests, integration tests, and latency benchmarks.

### Running Tests

```bash
# Run unit tests
pytest tests/test_audio.py tests/test_inference.py

# Run full test suite with coverage
pytest --cov=src tests/

# Run benchmark tests (measures CPU/GPU latency per block)
pytest tests/test_latency.py -s
```

### Writing Tests
- **Deterministic Inputs**: Use synthesized sine waves or calibrated WAV fixtures located in `tests/fixtures/`.
- **Latency Assertions**: Latency benchmarks must log execution times and verify threshold constraints.

#### Latency Benchmark Test Example
```python
import time
import torch
import pytest
from src.inference.engine import VoiceConversionEngine

@pytest.mark.benchmark
def test_inference_chunk_latency(sample_audio_chunk):
    engine = VoiceConversionEngine(model_path="tests/fixtures/dummy_voice.pth")
    
    start_time = time.perf_counter()
    output_audio = engine.process_chunk(sample_audio_chunk)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    
    assert output_audio is not None
    assert output_audio.shape == sample_audio_chunk.shape
    assert elapsed_ms < 25.0, f"Inference took {elapsed_ms:.2f}ms, exceeding 25ms budget!"
```

---

## 5. Security & Ethics Policy

Any PR that attempts to bypass security checks, disable audio watermarking, or obscure synthetic audio identifiers without explicit justification will be immediately rejected.

Thank you for contributing to making real-time AI voice technology fast, accessible, and ethical!
