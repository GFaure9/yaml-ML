from yaml_ml.special_types import ArrayLike
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.base import TransformerMixin
from yaml_ml.logger_cfg import logger


def encode(x, mtd: str, var_type: str) -> ArrayLike:
    try:
        kwargs = {"drop": "first", "sparse_output": False} if mtd == "one_hot" else {}
        encoder: TransformerMixin = ENCODERs[var_type][mtd](**kwargs)
    except Exception as e:
        err_msg = f"Could not initialize encoder due to: {e}"
        logger.error(err_msg)
        logger.debug(f"Hint: check if your {mtd} and {var_type} are compatible with:\n{ENCODERs}")
        raise ValueError(err_msg)

    return encoder.fit_transform(x)


ENCODERs = {
    "cont": None,
    "cat": {
        "binary": OrdinalEncoder,
        "one_hot": OneHotEncoder,
        "ordinal": OrdinalEncoder,
    }
}



if __name__ == "__main__":
    import numpy as np

    x1 = np.array([[1, 2, 0.01, 2, 2, 1, 0.1]]).transpose()
    x2 = [["yes"], ["yes"], ["no"], ["yes"], ["no"], ["no"], ["no"]]
    x3 = [["chat"], ["chien"], ["chat"], ["ours"], ["ours"]]

    print(encode(x2, "binary", "cat"))
    print(encode(x3, "one_hot", "cat"))
    print(encode(x3, "ordinal", "cat"))
    print(encode(x1, "binary", "cont"))
