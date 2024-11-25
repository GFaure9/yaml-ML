import numpy as np
from typing import List
from src.logger_cfg import logger


def clean(x, mtd: str, var_type: str) -> List[int]:
    ids = []

    if mtd == "remove_outliers":
        if var_type != "cont":
            err_msg = "To use 'remove_outliers' method, `var_type` must be 'cont'"
            logger.error(err_msg)
            raise ValueError(err_msg)
        else:
            logger.info("Using IQR method to compute outliers")
            x = np.asarray(x)
            q1, q3 = np.nanpercentile(x, 25), np.nanpercentile(x, 75)
            iqr = q3 - q1
            l, u = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            ids = np.argwhere(~((x >= l) & (x <= u))).flatten().tolist()

    if mtd == "remove_nans":
        def safe_convert(val):
            try:
                return float(val)
            except ValueError:
                return 0.
        x_np = np.asarray(x, dtype=object)
        nan_mask = np.isnan(np.vectorize(safe_convert)(x_np))
        ids = np.where(nan_mask)[0].tolist()

    return ids


if __name__ == "__main__":
    x1 = np.array([[1, 2, 4., 10., 0.1, 1, 2, 3, 2, 0.01, 0.0001, 1000]]).transpose()
    x2 = np.array([[1, np.nan, 4., 10., 0.1, np.nan, 2, np.nan, 2, 0.01, 0.0001, 1000]]).transpose()
    x3 = [["chat"], [np.nan], ["chien"], ["chat"], ["ours"], [np.nan]]

    print(clean(x1, "remove_outliers", "cont"))
    print(clean(x2, "remove_nans", "cont"))
    print(clean(x3, "remove_nans", "cat"))
