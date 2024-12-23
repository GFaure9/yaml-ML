from typing import Dict, Union, List
from yaml_ml import modules
from yaml_ml.logger_cfg import logger, with_spinner
import numpy as np
from .regression import REGRESSOR, RegModel, REGRESSION_SCORE
from .classification import CLASSIFIER, ClasModel, CLASSIFICATION_SCORE
from .utils import format_table
from sklearn.preprocessing import LabelEncoder
import pickle


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
    scores_dicts = {
        "regression": REGRESSION_SCORE,
        "classification": CLASSIFICATION_SCORE,
    }
    def __init__(self, cfg: PredictorConfig, model: Union[RegModel, ClasModel] = None):
        self.cfg = cfg
        self.model = model
        self.save_path = None
        self.predictor = None
        self.score = None

    def initialize(self):
        d, c = self.models_dicts, self.cfg
        self.model: RegModel = d[c.prediction_type][c.model_name]
        if self.model.fmt == "sklearn":
            self.predictor = self.model.mdl(**c.h_params)
        logger.info(f"Initialized '{c.model_name}' {c.prediction_type} model with hyperparameters: {c.h_params}")

    @with_spinner(style="bouncing")  # "moon"
    def train(self, X: np.ndarray, y: np.ndarray):
        if self.model.fmt == "sklearn":
            self.predictor.fit(X, y)

    def infer(self, x: np.ndarray):
        if self.model.fmt == "sklearn":
            if self.cfg.prediction_type == "classification":
                return self.predictor.predict(x), self.predictor.predict_proba(x)
            if self.cfg.prediction_type == "regression":
                return self.predictor.predict(x), None

    def compute_score(self, X: np.ndarray, y: np.ndarray, score: Union[str, List[str]]) -> Dict[str, float]:
        if isinstance(score, str):
            score = [score]

        predict_type = self.cfg.prediction_type
        valid_scores = modules["scores"][predict_type]
        if not set(score).issubset(valid_scores):
            err_msg = f"{score} is contains invalid score functions names. Please choose scores among: {valid_scores}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        score_vals = {}
        for s in score:
            score_fun = self.scores_dicts[predict_type][s]
            y_pred, y_pred_proba = self.infer(X)
            logger.info(f"Computing {s} score")
            if predict_type == "classification":
                if self.model.fmt == "sklearn":
                    # label_enc = LabelEncoder()
                    # score_vals[s] = score_fun(label_enc.fit_transform(y), y_pred, average="weighted")
                    # score_vals[s] = score_fun(y, y_pred, average="weighted")
                    if s in ["f1", "recall", "precision"]:
                        score_vals[s] = score_fun(y, y_pred, average="weighted")
                    elif s in ["auc", "cross_entropy"]:
                        score_vals[s] = score_fun(y, y_pred_proba)
                    else:
                        score_vals[s] = score_fun(y, y_pred)
            if predict_type == "regression":
                score_vals[s] = score_fun(y, y_pred)
        self.score = score_vals
        return score_vals

    def save(self, out_folder: str, out_name: str):
        if self.predictor:
            save_path = f"{out_folder}/{out_name}.pkl"
            with open(save_path, "wb") as f:
                pickle.dump(self.predictor, f)
            logger.info(f"Saved the trained model in a binary format at: {save_path}")
            self.save_path = save_path

            info_path = f"{out_folder}/{out_name}__info.txt"
            with open(info_path, "w", encoding="utf-8") as f:
                f.write(self.__repr__(), )
            logger.info(f"Information file (model, status, scores...) was saved at: {info_path}")

        else:
            wrn_msg = "No model was saved since no predictor was built. Please run `initialize()` and `train()` method first"
            logger.warning(wrn_msg)

    def __repr__(self):
        mdl_txt = f"{'MODEL':<20} | "
        mdl_txt += f"{self.cfg.prediction_type.capitalize()}[model={self.cfg.model_name.upper()}, format={self.model.fmt}]" if self.model else "NONE"

        h_params_txt = f"\n{'HYPERPARAMETERS':<20}\n{format_table(self.cfg.h_params)}"

        trained = "yes" if self.predictor else "no"
        train_txt = f"{'TRAINED':<20} | {trained}"

        sv_pth = self.save_path if self.save_path else "None"
        sv_txt = f"{'PATH':<20} | {sv_pth}"

        scr_txt = f"{'SCORE':<20} | "
        scr_txt += "".join([f"{s.upper()} [{v:.3f}]    " for s, v in self.score.items()]) if self.score else "NONE"

        n = max([len(mdl_txt), len(scr_txt), len(sv_txt), len(train_txt)])

        txt = "\n".join([
            "",
            n * "_",
            mdl_txt,
            h_params_txt,
            train_txt,
            sv_txt,
            n * "=",
            scr_txt,
            n * "_",
        ])
        return txt
