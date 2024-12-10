"""
To run the module:
1. Open a terminal in the yaml-ML folder
2. Activate your virtual environment with installed yaml-ML required packages
3. Run the following command

>>> python -m MODULE_NAME --cfg path/to/your/config/yaml/file

N.B: For now `MODULE_NAME` will be `yaml_ml`.
"""


from yaml_ml import Pipeline
import argparse


def main(config_path: str):
    Pipeline(cfg_pth=config_path).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run your ML pipeline with the parameters specified in a given YAML file"
    )
    help_text = "Enter the full path to your pipeline configuration YAML file"
    parser.add_argument("--cfg", required=True, type=str, help=help_text)
    args = parser.parse_args()
    cfg_pth = args.cfg

    main(cfg_pth)
