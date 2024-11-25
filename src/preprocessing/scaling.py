from sklearn.base import TransformerMixin
from src.types import ArrayLike
from src.logger_cfg import logger
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler


def scale(x, mtd: str, var_type: str) -> ArrayLike:
    try:
        scaler: TransformerMixin = SCALERs[var_type][mtd]()
    except Exception as e:
        err_msg = f"Could not initialize scaler due to: {e}"
        logger.error(err_msg)
        logger.debug(f"Hint: check if your {mtd} and {var_type} are compatible with:\n{SCALERs}")
        raise ValueError(err_msg)

    return scaler.fit_transform(x)


SCALERs = {
    "cont": {
        "min_max": MinMaxScaler,
        "abs_max": MaxAbsScaler,
        "standard": StandardScaler,
        "robust": RobustScaler,
    },
    "cat": None,
}


if __name__ == "__main__":
    import numpy as np
    x1 = np.array([[1, 2, 0.01, 2, 2, 1, 0.1]]).transpose()
    x2 = [["yes"], ["yes"], ["no"], ["yes"], ["no"], ["no"], ["no"]]

    print(scale(x1, "min_max", "cont"))
    print(scale(x1, "abs_max", "cont"))
    print(scale(x1, "standard", "cont"))
    print(scale(x1, "robust", "cont"))
    print(scale(x2, "robust", "cat"))