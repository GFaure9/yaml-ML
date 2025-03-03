# yaml-ML

*`yaml_ml` streamlines machine learning workflows by letting you
define data preprocessing, model training, 
and evaluation in one YAML file. Automate your ML pipeline with minimal code.*

Have a look at the GitHub repo for detailed description and usage examples:
[yaml-ML](https://github.com/GFaure9/yaml-ML).

## 1. Installation

Create a virtual environment (e.g. with `conda`), activate it and upgrade `pip`:

```commandline
conda create --name yaml_ml_env
conda activate yaml_ml_env
pip install --upgrade pip
```

Then install the package:

```commandline
pip install yaml-ml
```

## 2. Usage

#### <u>With one configuration file</u>

First, create a YAML configuration file: see [docs](#3-docs).
Then, after having activated the environment where `yaml_ml` is installed, run the command:

```commandline
python -m yaml_ml --cfg path/to/your/config/yaml/file
```

#### <u>With multiple configuration files</u>

In case you want to test different configurations, create corresponding YAML files
and put them in a unique folder. 
To launch all the corresponding pipelines in parallel using multiprocessing with `N` worker processes, run the command:

```commandline
python -m yaml_ml --cfg path/to/your/configs/folder --n_processes N 
```

**N.B**:
Without providing the `--n_processes` argument, pipelines will be launched sequentially.