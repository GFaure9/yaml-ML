import os.path

from yaml_ml.config_loader import load_yaml_config, LoadedYAMLType
from yaml_ml.logger_cfg import logger, FORMAT
from yaml_ml import (
Dataset, DatasetConfig,
PreProcessor, PreProcessorConfig,
Splitter, SplitterConfig,
Predictor, PredictorConfig
)


class Pipeline:
    def __init__(self, cfg_pth: str = None):
        self.cfg_pth = cfg_pth

    def load_cfg(self, pth: str) -> LoadedYAMLType:
        if pth is not None:
            self.cfg_pth = pth
        elif self.cfg_pth is None:
            err_msg = "You must provide a path for YAML config file `pth` or `self.cfg_pth`"
            logger.error(err_msg)
            raise ValueError(err_msg)

        return load_yaml_config(self.cfg_pth)

    def run(self, pth: str = None):
        # Load YAML config
        logs, trg_var, loading_dic, prepro_dic, split_dic, model_dic, score_name, out_folder, name = self.load_cfg(pth)

        # Configure logger
        if logs:
            debug_filename = f"debug__{name}.log"
            logger.add(debug_filename, level="DEBUG", colorize=False, format=FORMAT)
            logger.info(f"Enabling writing logs at: './{debug_filename}'\n")
        else:
            logger.info("Debug logs writing is disabled\n")
        k_dash = 20

        # Create the output folder (if necessary)
        if not os.path.isdir(out_folder):
            os.mkdir(out_folder)
            logger.info(f"Created output folder: {out_folder}\n")
        else:
            logger.warning(f"Output folder already exists at: {out_folder}\n")

        # Load the dataset
        logger.info(k_dash * "-" + " Starting dataset loading...")

        loading_cfg = DatasetConfig.from_dict(loading_dic)
        logger.info(f"Loaded DatasetConfig:\n{loading_cfg}")

        dataset = Dataset.from_config(cfg=loading_cfg)
        logger.info(
            f"Loaded Dataset using previous config:\n{dataset.as_array}\nVariables: {dataset.arrays_names}"
        )

        logger.info(k_dash * "-" + " Success!\n")

        # 1. Preprocessing
        if prepro_dic:
            logger.info(k_dash * "-" + " Starting preprocessing...")

            prepro_config = PreProcessorConfig.from_dict(prepro_dic)
            logger.info(f"Loaded PreProcessorConfig")

            # dataset = PreProcessor(cfg=prepro_config).run(dataset=dataset)
            preprocessor = PreProcessor(cfg=prepro_config)
            dataset = preprocessor.run(dataset=dataset)
            logger.info(
                f"Preprocessed Dataset using previous config:\n{dataset.as_array}\nVariables: {dataset.arrays_names}"
            )

            logger.info(k_dash * "-" + " Success!\n")
        else:
            logger.warning("No preprocessing instructions were given. Trying to use data as it is...")

        X, y = dataset.get_X_y(trg_var_name=trg_var)

        # 2. Create training and test datasets (splitting)
        logger.info(k_dash * "-" + " Starting train/test/(val) splitting...")

        if split_dic:
            split_config = SplitterConfig.from_dict(d=split_dic)
        else:
            tr, ts = 75, 25
            split_config = SplitterConfig.from_dict(d={"train": tr, "test": ts})
            logger.warning(f"Since no split information was given, preforming a classical {tr}/{ts} train/test split")

        logger.info(f"Loaded SplitterConfig:\n{split_config}")

        if split_config.stratify:
            trg_var_type = None
            try:
                trg_var_type = prepro_config.variables_types[trg_var]
            except Exception as e:
                wrn_msg = "\n".join([
                    f"Could not retrieve target variable type due to: {e}",
                    "Will try to perform stratified split anyways..."
                ])
                logger.warning(wrn_msg)
                pass
            if trg_var_type:
                if trg_var_type != "cat":
                    err_msg = " ".join([
                        f"Target variable type is not categorical ('cat') but '{trg_var_type}'.",
                        "Consider setting `stratified` parameter to `no` in the input YAML file"
                    ])
                    logger.error(err_msg)
                    raise ValueError(err_msg)

        splitter = Splitter(cfg=split_config)
        data_split = splitter.run(X, y)

        logger.info(k_dash * "-" + " Success!\n")

        # 3. Train the model / compute its score / save the model
        if model_dic:
            logger.info(k_dash * "-" + " Starting building model...")

            model_config = PredictorConfig.from_dict(model_dic)
            logger.info(f"Loaded PredictorConfig:\n{model_config}")

            model = Predictor(cfg=model_config)

            model.initialize()

            logger.info(f"Starting training model on data")
            model.train(data_split.train.X, data_split.train.y)
            logger.info("Model trained")

            logger.info(k_dash * "-" + " Success!")

            logger.info(k_dash * "-" + " Computing score...")
            scores = model.compute_score(data_split.test.X, data_split.test.y, score_name)
            logger.info(k_dash * "-" + " Success!")

            logger.info(k_dash * "-" + " Saving trained model...")
            model.save(out_folder=out_folder, out_name=name)
            logger.info(k_dash * "-" + " Success!\n")

            logger.info(f"{model}")

            return {name: scores}


if __name__ == "__main__":
    # pipeline = Pipeline(cfg_pth="../../examples/example_regression_cfg.yaml")
    pipeline = Pipeline(cfg_pth="../../examples/example_classification_cfg.yaml")
    pipeline.run()
