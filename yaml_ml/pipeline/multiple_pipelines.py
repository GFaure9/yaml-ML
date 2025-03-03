from .pipeline import Pipeline
from .utils import plot_scores
from yaml_ml.logger_cfg import logger
from yaml_ml.config_loader import load_yaml_config
from multiprocessing import Pool
from tqdm import tqdm
import os


def run_pipeline(config_filepath: str):
    return Pipeline(cfg_pth=config_filepath).run()


class MultiPipelines:
    def __init__(self, cfg_folder_path, n_processes: int = None):
        self.cfg_folder_path = cfg_folder_path
        self.n_processes = n_processes

    def run(self):
        # Get all YAML configuration files inside the given folder at `self.cfg_folder_path`
        configs_filepaths =  [
            os.path.join(self.cfg_folder_path, f)
            for f in os.listdir(self.cfg_folder_path)
            if f.endswith(".yaml") or f.endswith(".yml")
        ]
        n_configs = len(configs_filepaths)

        logger.info(f"Found {n_configs} YAML configuration files at: {self.cfg_folder_path}\n")

        # Use multiprocessing Pool for parallel computing on number of given `self.n_processes` if not None
        scores = []

        if self.n_processes:
            logger.info(
                f"Running config files pipelines in parallel with batch size {self.n_processes} using same number of worker processes...\n"
            )

            k = n_configs // self.n_processes

            for i in tqdm(range(k + 1)):
                batch = configs_filepaths[self.n_processes * i: self.n_processes * (i + 1)]
                with Pool(self.n_processes) as p:
                    batch_scores = p.map(run_pipeline, batch)
                    scores.extend(batch_scores)

        # Else computations are done sequentially
        else:
            logger.info(f"Running config files pipelines sequentially...\n")
            for config_filepath in tqdm(configs_filepaths):
                model_scores = run_pipeline(config_filepath)
                scores.append(model_scores)

        logger.info("Successfully run all config files pipelines!\n")

        # Plot scores of all for all configs + order by cumulative score
        output_folder_path = load_yaml_config(configs_filepaths[0])[-2] # taking 1st config file's output folder

        configs_scores_plot_fpath = f"{output_folder_path}/configs_scores.png"
        plot_scores(scores, save_path=configs_scores_plot_fpath)

        logger.info(f"Saved scores histograms for all config files at: {configs_scores_plot_fpath}")
