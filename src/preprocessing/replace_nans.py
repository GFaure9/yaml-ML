import numpy as np
from sklearn.impute import SimpleImputer
from typing import Tuple, Union, Any
from src.logger_cfg import logger
from src.types import ArrayLike


def replace_nans(x, mtd: Union[str, Tuple[str, Any]], var_type: str) -> ArrayLike:
    mtd = (mtd,) if not isinstance(mtd, tuple) else mtd
    kwargs = {"strategy": STRATEGIES[var_type][mtd[0]]}
    if mtd[0] == "value":
        if not isinstance(mtd, tuple):
            err_msg = "You must provide a tuple of the form ('value', YOUR_VALUE) for `mtd`"
            logger.error(err_msg)
            raise ValueError(err_msg)
        kwargs["fill_value"] = mtd[1]
    imp = SimpleImputer(missing_values=np.nan, **kwargs)
    return imp.fit_transform(x)


STRATEGIES = {
    "cont": {
        "mean": "mean",
        "median": "median",
        "most_frequent": "most_frequent",
        "value": "constant",
    },
    "cat": {
        "most_frequent": "most_frequent",
        "value": "constant",
    }
}


if __name__ == "__main__":
    x1 = np.array([[1, 2, np.nan, 2, 2, 1, 0.1]]).transpose()
    x2 = [["east"], ["north"], ["east"], [np.nan], [np.nan], ["south"], ["north"], ["east"]]

    print(replace_nans(x1, "mean", "cont"))
    print(replace_nans(x1, "median", "cont"))
    print(replace_nans(x1, "most_frequent", "cont"))
    print(replace_nans(x1, ("value", 42), "cont"))
    print(replace_nans(x2, ("value", "west"), "cat"))
    print(replace_nans(x2, "most_frequent", "cat"))
