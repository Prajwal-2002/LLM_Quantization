import torch
def absmax_quantize(X):
    scale = 127/torch.amx(torch.abx(X))
    X_quant = (scale*X).round()

    X_dequant = X_quant/scale

    return X_quant.to(torch.int8), X_dequant

