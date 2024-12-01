from typing import Dict, List, Tuple
from src.types import ArrayLike
from src.logger_cfg import logger
import numpy as np
import copy


class DatasetConfig:
    def __init__(self, folder_path: str, dataset_name: str, extension: str, separator: str):
        self.folder_path = folder_path
        self.dataset_name = dataset_name
        self.extension = extension
        self.separator = separator

    @classmethod
    def from_dict(cls, d: Dict):
        cfg = cls(
            folder_path=d["folder"],
            dataset_name=d["name"],
            extension=d["format"],
            separator=d["separator"],
        )
        return cfg

    def __repr__(self):
        txt = " | ".join([
            f"folder_path: '{self.folder_path}'",
            f"dataset_name: '{self.dataset_name}'",
            f"extension: '{self.extension}'",
            f"separator: '{self.separator}'",
        ])
        return txt


class Dataset:
    def __init__(
            self,
            arrays: List[ArrayLike] = None,  # same order than `arrays_names`
            arrays_names: List[str] = None,
    ):
        self.arrays = arrays
        self.arrays_names = arrays_names

    def __copy__(self):
        cls = self.__class__
        new_instance = cls.__new__(cls)
        new_instance.arrays = copy.copy(self.arrays)
        new_instance.arrays_names = copy.copy(self.arrays_names)
        return new_instance

    def remove_rows(self, ids: List[int]):
        # inplace
        for arr in self.arrays:
            for idx in sorted(ids, reverse=True):
                if idx < len(arr):
                    arr.pop(idx)

    def get_id_name(self, name: str) -> int:
        np_names = np.array(self.arrays_names, dtype=object)
        return int(np.argwhere(np_names == name)[0][0])

    @classmethod
    def from_config(cls, cfg: DatasetConfig):
        valid_extensions = [
            "csv",
            "txt",
        ]
        if cfg.extension not in valid_extensions:
            err_msg = f"{cfg.extension} not handled yet. Please convert your file to {' or '.join(valid_extensions)}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        elif cfg.extension in ["csv", "txt"]:
            data = np.genfromtxt(
                f"{cfg.folder_path}/{cfg.dataset_name}.{cfg.extension}",
                delimiter=cfg.separator,
                dtype=None,
                encoding='utf-8',
                names=True
            )
            arrays_names = data.dtype.names
            arrays = [data[name].tolist() for name in arrays_names]

            return cls(arrays=arrays, arrays_names=arrays_names)

    @property
    def as_array(self):
        return np.asarray(self.arrays, dtype=object).T

    def get_X_y(self, trg_var_name: str) -> Tuple[np.ndarray, np.ndarray]:
        id_trg = self.get_id_name(trg_var_name)
        # X, y = np.delete(self.as_array, id_trg, axis=1), self.as_array[:, id_trg].reshape(-1, 1)
        X, y = np.delete(self.as_array, id_trg, axis=1), self.as_array[:, id_trg]
        return X, y


if __name__ == "__main__":
    config = DatasetConfig(
        folder_path="../../datasets",
        dataset_name="insurance_charges",
        extension="csv",
        separator=",",
    )
    ds = Dataset.from_config(config)
    breakpoint()
    print(ds.arrays)
    print(ds.as_array)
