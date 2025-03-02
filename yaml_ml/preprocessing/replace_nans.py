import numpy as np
from sklearn.impute import SimpleImputer
from typing import Dict, Union, Any
from yaml_ml.logger_cfg import logger
from yaml_ml.special_types import ArrayLike


def replace_nans(x, mtd: Union[str, Dict[str, Any]], var_type: str) -> ArrayLike:
    mtd_name = mtd if not isinstance(mtd, dict) else next(iter(mtd))
    kwargs = {"strategy": STRATEGIES[var_type][mtd_name]}
    if isinstance(mtd, dict):
        if mtd_name != "value":
            err_msg = "You must provide a dict of the form {'value': YOUR_VALUE} for `mtd`"
            logger.error(err_msg)
            raise ValueError(err_msg)
        kwargs["fill_value"] = mtd[mtd_name]
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
