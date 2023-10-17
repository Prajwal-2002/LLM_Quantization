import torch
def zeropoint_quantize(X):
    x_range = torch.max(X) - torch.min(X)
    x_range = 1 if x_range == 0 else x_range

    scale = 255 / x_range
    zeropoint = (-scale * torch.min(X) - 128).round()

    X_quant = torch.clip((X * scale + zeropoint).round(), -128, 127)
    X_dequant = (X_quant - zeropoint) / scale

    return X_quant.to(torch.int8), X_dequant