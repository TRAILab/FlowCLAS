#* Utility functions for SAM 2 model


import numpy as np


def rle_to_mask(rles):
    h, w = rles[0]['size']
    mask = np.zeros(h * w, dtype=np.uint8)
    for i, rle in enumerate(rles):
        current_pos = 0
        is_zero = True
        for count in rle['counts']:
            if not is_zero:
                mask[current_pos:current_pos + count] = i + 1
            current_pos += count
            is_zero = not is_zero
    return mask.reshape(w, h).T #* (H, W)