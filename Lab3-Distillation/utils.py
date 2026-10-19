from PIL import Image
import numpy as np


def torch_to_pil(tensor):
    if tensor.dim() == 4:
        _b, *_ = tensor.shape
        if _b == 1:
            tensor = tensor.squeeze(0)
        else:
            tensor = tensor[0, ...]
    
    tensor = tensor.permute(1, 2, 0)
    np_tensor = tensor.detach().cpu().numpy()
    np_tensor = (np_tensor * 255.0).astype(np.uint8)
    pil_tensor = Image.fromarray(np_tensor)
    return pil_tensor