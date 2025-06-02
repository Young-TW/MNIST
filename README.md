# MNIST

This is a practical implementation of the MNIST dataset using PyTorch. The MNIST dataset consists of handwritten digits from 0 to 9, and is commonly used for training various image processing systems.

## Dependencies

The following command will install the required dependencies:

- AMD ROCm

```bash
uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.3
```

- Apple Metal

```zsh
uv pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```
