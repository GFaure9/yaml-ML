from typing import Dict, NamedTuple, Optional
import numpy as np
from src.logger_cfg import logger
from sklearn.model_selection import train_test_split


class Xy(NamedTuple):
    X: np.ndarray
    y: np.ndarray

class DataSplit(NamedTuple):
    train: Xy
    test: Xy
    val: Xy
    val: Optional[Xy] = None


class SplitterConfig:
    def __init__(self, train_size: int, test_size: int, val_size: int = None, stratify: bool = False):
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.stratify = stratify

    @classmethod
    def from_dict(cls, d: Dict):
        if "train" in d.keys():
            train_size = d["train"]
            test_size = 100 - d["train"]
            if "test" in d.keys():
                if d["test"] != test_size:
                    wrn_msg = "".join([
                        f"Found test size ({d['test']}) is different from (100 - train size = {test_size})",
                        ": readjusting..."
                        ])
                    logger.warning(wrn_msg)
        elif "test" in d.keys():
            test_size = d["test"]
            train_size = 100 - test_size
        else:
            train_size = 75
            test_size = 25
            wrn_msg = "No train and test size were given: choosing a 75/25 train/test split"
            logger.warning(wrn_msg)

        if "stratified" in d.keys():
            stratify = d["stratified"]
            logger.info("Stratification was activated in splitting configuration")
        else:
            stratify = False

        if "val" in d.keys():
            val_size = d["val"]
        else:
            val_size = None

        return cls(train_size=train_size, test_size=test_size, val_size=val_size, stratify=stratify)

    def __repr__(self):
        txt = " | ".join([
            f"train_size: {self.train_size} (%)",
            f"test_size: {self.test_size} (%)",
            f"val_size: {self.val_size} (%)",
            f"stratify: {self.stratify}",
        ])
        return txt


class Splitter:
    def __init__(self, cfg: SplitterConfig):
        self.cfg = cfg

    def run(self, X: np.ndarray, y: np.ndarray) -> DataSplit:
        ts, val = self.cfg.test_size, self.cfg.val_size

        info_msg = "Creating train/test data without stratification"
        stratification = None
        if self.cfg.stratify:
            info_msg = "Creating train/test data with stratification"
            stratification = y.copy()

        logger.info(info_msg)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=ts, stratify=stratification, random_state=42,
        )

        if val:
            logger.info(f"Allocating {val}% of training data to future validation")
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val, random_state=42,
            )

        res = DataSplit(
            Xy(X_train, y_train),
            Xy(X_test, y_test),
            Xy(X_val, y_val) if val else None,
        )

        return res


