"""
To run the module:
1. Open a terminal in the yaml-ML folder
2. Activate your virtual environment with installed yaml-ML required packages
3. Run the following command

>>> python -m MODULE_NAME --cfg path/to/your/config/yaml/file

or

>>> python -m MODULE_NAME --cfg path/to/your/config/files/folder --cpu NUM_CPU

N.B: For now `MODULE_NAME` will be `yaml_ml`.
"""


from yaml_ml import Pipeline, MultiPipelines
import argparse
import os


def main(
        config_path: str,
        cpu: int = None,
):
    if os.path.isfile(config_path):
        Pipeline(cfg_pth=config_path).run()

    elif os.path.isdir(config_path):
        MultiPipelines(cfg_folder_path=config_path, cpu=cpu).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run your ML pipeline(s) with the parameters specified in given YAML file(s)"
    )
    cfg_help_text = "Enter the full path to your pipeline configuration YAML file OR to a folder containing multiple configuration files"
    cpu_help_text = "[OPTIONAL] If you provided a path to a folder of configuration files, indicate the number of CPU cores to use for parallel computing. Default is `None`"
    parser.add_argument("--cfg", required=True, type=str, help=cfg_help_text)
    parser.add_argument("--cpu", required=False, type=int, help=cpu_help_text)
    args = parser.parse_args()
    cfg_pth = args.cfg

    main(cfg_pth)
