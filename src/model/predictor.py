from typing import Dict, Union
from src import modules
from src.logger_cfg import logger, with_spinner
import numpy as np
from .regression import REGRESSOR, RegModel
from .classification import CLASSIFIER, ClasModel


class PredictorConfig:
    def __init__(self, prediction_type: str, model_name: str, h_params: Dict):
        valid_prediction_types = list(modules["models"].keys())
        if prediction_type not in valid_prediction_types:
            err_msg = " ".join([
                f"`prediction_type` must be in {valid_prediction_types}",
                f"(but was given '{prediction_type}')",
            ])
            logger.error(err_msg)
            raise ValueError(err_msg)

        valid_model_names = modules["models"][prediction_type]
        if model_name not in valid_model_names:
            err_msg = " ".join([
                f"`model_name` must be in {valid_model_names}",
                f"(but was given '{model_name}')",
            ])
            logger.error(err_msg)
            raise ValueError(err_msg)

        self.prediction_type = prediction_type
        self.model_name = model_name
        self.h_params = h_params

    @classmethod
    def from_dict(cls, d: Dict):
        prediction_type = next(iter(d))
        model_name = next(iter(d[prediction_type]))
        h_params = d[prediction_type][model_name]
        return cls(prediction_type=prediction_type, model_name=model_name, h_params=h_params)

    def __repr__(self):
        txt = " | ".join([
            f"prediction_type: '{self.prediction_type}'",
            f"model_name: '{self.model_name}'",
            f"h_params (model's hyperparameters): {self.h_params}",
        ])
        return txt


class Predictor:
    models_dicts = {
        "regression": REGRESSOR,
        "classification": CLASSIFIER,
    }
    def __init__(self, cfg: PredictorConfig, model: Union[RegModel, ClasModel] = None):
        self.cfg = cfg
        self.model = model
        self.predictor = None

    def initialize(self):
        d, c = self.models_dicts, self.cfg
        self.model: RegModel = d[c.prediction_type][c.model_name]
        if self.model.fmt == "sklearn":
            self.predictor = self.model.mdl(**c.h_params)
        logger.info(f"Initialized '{c.model_name}' {c.prediction_type} model with hyperparameters: {c.h_params}")

    @with_spinner(style="moon")
    def train(self, X: np.ndarray, y: np.ndarray):
        if self.model.fmt == "sklearn":
            self.predictor.fit(X, y)

    def infer(self, x):
        if self.model.fmt == "sklearn":
            return self.predictor.predict(x)


# todo: add method to save the model
# todo: add method to compute scores (in regression/classification files put the methods)
# todo: add method to save scores
# todo: add all these to the pipeline
