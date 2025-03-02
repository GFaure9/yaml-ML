from typing import Dict, List
from yaml_ml import modules, Dataset
from yaml_ml.logger_cfg import logger
from .cleaning import clean
from .replace_nans import replace_nans
from .encoding import encode
from .scaling import scale
import copy
import numpy as np


class PreProcessorConfig:
    valid_preprocess = {
        "cont": modules["preprocessing"]["cont"],
        "cat": modules["preprocessing"]["cat"],
    }
    valid_var_types = list(valid_preprocess.keys())

    def __init__(self, variables_types: Dict[str, str], tasks: Dict[str, Dict]):
        for var, tp in variables_types.items():
            tsk = tasks[var]
            if tp not in self.valid_var_types:
                err_msg = f"Error in preprocessing config for '{var}'."
                err_msg += f"'{tp}' is not a valid variable type.\nValid types are: {self.valid_var_types}"
                logger.error(err_msg)
                ValueError(err_msg)

            tasks_types = set(tsk.keys())
            if not tasks_types.issubset(self.valid_preprocess[tp].keys()):
                err_msg = f"Error in preprocessing config for '{var}'."
                err_msg += f"Invalid tasks types {tasks_types}.\n Valid types are: {list(self.valid_preprocess[tp].keys())}"
                logger.error(err_msg)
                raise ValueError(err_msg)
            else:
                for task_type, task_name in tsk.items():
                    if isinstance(task_name, str):
                        task_name = {task_name}
                    if isinstance(task_name, list):
                        task_name = set(task_name)
                    if isinstance(task_name, dict):
                        assert len(task_name) == 1  # debug
                        task_name = {next(iter(task_name))}
                    if not task_name.issubset(self.valid_preprocess[tp][task_type]):
                        err_msg = f"Error in preprocessing config for '{var}' with task type {task_type}."
                        err_msg += f"{task_name} is not a valid task. Valid tasks are: {self.valid_preprocess[tp][task_type]}"
                        logger.error(err_msg)
                        raise ValueError(err_msg)

        self.variables_types = variables_types
        self.tasks = tasks

    @classmethod
    def from_dict(cls, d: Dict):
        variables_types = {}
        for name, info in d.items():
            try:
                variables_types[name] = info["type"]
            except Exception as e:
                variables_types[name] = "cat"
                wrn_msg = "".join([
                    f"No type was given for variable '{name}' due to: {e}",
                    f"\nDefault 'cat' type was given. ",
                    f"If you do not plan to remove this variable, please choose the right type in {cls.valid_var_types}"
                ])
                logger.warning(wrn_msg)
        tasks = {name: {k: info[k] for k in info.keys() if k != "type"} for name, info in d.items()}
        return cls(variables_types=variables_types, tasks=tasks)


class PreProcessor:
    def __init__(self, cfg: PreProcessorConfig):
        self.cfg = cfg

    def run(self, dataset: Dataset) -> Dataset:
        new_dataset = copy.copy(dataset)
        variables_types, tasks = self.cfg.variables_types, self.cfg.tasks

        # start by cleaning tasks if any
        vars_to_remove = []
        for var, tp in variables_types.items():
            var_tasks = tasks[var]
            if "cleaning" in var_tasks.keys():
                mtd = var_tasks["cleaning"]
                if mtd == "remove_col":
                    cols_np = np.array(new_dataset.arrays_names)
                    ids_to_remove = np.argwhere(cols_np == var).flatten()
                    new_dataset.arrays_names = np.delete(cols_np, ids_to_remove).tolist()
                    new_dataset.arrays = np.delete(new_dataset.as_array, ids_to_remove, axis=1).transpose().tolist()
                    vars_to_remove.append(var)
                    logger.info(f"Removed column '{var}' in the dataset")
        for var in vars_to_remove:
            del variables_types[var]
            del tasks[var]

        cleaned = False
        rows_to_remove = set()
        for var, tp in variables_types.items():
            var_tasks = tasks[var]

            if "cleaning" in var_tasks.keys():
                cleaned = True

                mtd = var_tasks["cleaning"]

                if mtd != "remove_col":
                    mtd = [mtd] if isinstance(mtd, str) else mtd

                    logger.info(f"Cleaning '{var}' data with: {mtd}")
                    i = new_dataset.arrays_names.index(var)
                    x = new_dataset.arrays[i]

                    for m in mtd:
                        ids = clean(x, mtd=m, var_type=tp)
                        rows_to_remove = rows_to_remove.union(set(ids))

        new_dataset.remove_rows(ids=list(rows_to_remove))
        if cleaned:
            logger.info("Successfully cleaned dataset")

        # then perform other preprocessing tasks
        for var, tp in variables_types.items():
            var_tasks = tasks[var]
            idx = new_dataset.get_id_name(name=var)
            new_x = new_dataset.as_array[:, idx].reshape(-1, 1)
            new_name = [var]

            for tsk, mtd in var_tasks.items():
                if tsk == "replace_nans":
                    new_x = replace_nans(new_x, mtd=mtd, var_type=tp)
                    logger.info(f"Replacing NaNs in '{var}' data with: '{mtd}'")

                if tsk == "scaling":
                    new_x = scale(new_x, mtd=mtd, var_type=tp)
                    logger.info(f"Scaling '{var}' data with: '{mtd}'")

                if tsk == "encoding":
                    new_x = encode(new_x, mtd=mtd, var_type=tp)
                    n_cols = new_x.shape[1]
                    if n_cols > 1:
                        new_name = [f"{var}_{k}" for k in range(n_cols)]
                    logger.info(f"Encoding '{var}' data with: '{mtd}'")

            arrays, names = new_dataset.arrays, new_dataset.arrays_names
            inserted_arrays = [new_x[:, i].flatten().tolist() for i in range(len(new_name))]
            new_dataset.arrays = arrays[: idx] + inserted_arrays + arrays[idx + 1:]
            new_dataset.arrays_names = tuple(list(names[: idx]) + new_name + list(names[idx + 1:]))

        return new_dataset
