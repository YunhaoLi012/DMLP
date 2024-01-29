import torch
from typing import Dict

def ddpm_schedule(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1

    beta_t = beta_t*0 + 0.008
    # beta_t = 0.008
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    sqrta = torch.sqrt(alpha_t)
    oneover_sqrta = 1 / sqrta
    mab = 1 - alphabar_t
    sqrtmab = torch.sqrt(mab)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab
    loss_weight = 0.5 * beta_t / (alpha_t * mab)
    sigma = sqrt_beta_t
    sqrtmab_over_sqrtab = sqrtmab / sqrtab
    sigma_diff = sqrtmab_over_sqrtab[1:] - sqrtmab_over_sqrtab[:-1]
    return {
        "beta_t": beta_t,
        "alpha_t": alpha_t,  # \alpha_t
        "sqrta": sqrta,
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "mab": mab,
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
        "sigma" : sigma,
        'loss_weight':loss_weight,
        'diff_sigma': sigma_diff
    }