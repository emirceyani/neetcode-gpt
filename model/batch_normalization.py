import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        
        m = momentum
        if training:
            mu_batch,var_batch = np.mean(np.array(x),axis =0), np.var(np.array(x), axis = 0)
            xhat = (np.array(x) - mu_batch)/ np.sqrt(var_batch + eps)
            running_mean = np.round((1-m) * np.array(running_mean) + m * mu_batch, decimals=4).tolist()
            running_var = np.round((1-m) * np.array(running_var) + m * var_batch, decimals=4).tolist()
        else:
            xhat = (np.array(x) - np.array(running_mean))/ np.sqrt(np.array(running_var) +eps)

        y = np.array(gamma) * xhat + np.array(beta); y= np.round(y,decimals=4).tolist()
        
        return (y, running_mean, running_var)
