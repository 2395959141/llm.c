import torch
print("CUDA available:", torch.cuda.is_available())
print("cuDNN available:", torch.backends.cudnn.enabled)