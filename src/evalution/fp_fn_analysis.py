import numpy as np

def get_fp_fn_indices(y_true, y_pred, threshold=0.5):
    pred_label = (y_pred > threshold).astype(int)
    fp = np.where((y_true == 0) & (pred_label == 1))[0]
    fn = np.where((y_true == 1) & (pred_label == 0))[0]
    return fp, fn
