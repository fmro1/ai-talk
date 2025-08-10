import torch

print("PyTorch version:", torch.__version__)
print("ROCm version:", torch.version.hip)
print("Is ROCm build:", torch.version.hip is not None)
print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))
