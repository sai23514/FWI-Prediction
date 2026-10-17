"""Canadian FWI System equation for the final index (Van Wagner, 1987).

FWI is a closed-form function of ISI and BUI. We use it as a physics
reference to sanity-check the learned models.
"""
import numpy as np


def fwi_from_isi_bui(isi, bui):
    isi = np.asarray(isi, dtype=float)
    bui = np.asarray(bui, dtype=float)
    fd = np.where(bui <= 80, 0.626 * bui ** 0.809 + 2.0,
                  1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui)))
    b = 0.1 * isi * fd
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(b > 1, np.exp(2.72 * (0.434 * np.log(np.maximum(b, 1e-9))) ** 0.647), b)
    return s
