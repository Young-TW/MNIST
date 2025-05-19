import torch

def detect_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        return f"Using GPU: {torch.cuda.get_device_name(device)}"
    return "No GPU available, using CPU."
