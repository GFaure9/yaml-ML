<p align="center">
  <img src="./logo.png" width="400" />
</p>

---

### *Your whole ML pipeline in one YAML file!*

![GitHub Repo stars](https://img.shields.io/github/stars/GFaure9/yaml-ML?style=social)
![GitHub All Releases](https://img.shields.io/github/downloads/GFaure9/yaml-ML/total)
![Development Stage](https://img.shields.io/badge/stage-Beta-yellow)

*`yaml_ml` streamlines machine learning workflows by letting you
define data preprocessing, model training, 
and evaluation in one YAML file. Automate your ML pipeline with minimal code.*

> [!IMPORTANT]
> Disclaimer: this is the very first version of the package. It is still under development.
> Use it at your own risk.

# Table of Contents 

[Quickstart](#-quickstart)
1. [Installation](#1-installation)
2. [Usage](#2-usage)
3. [Docs](#3-docs)

[Usage Example](#-usage-example-step-by-step)

[Dependencies](#-dependencies)

[Tests](#-tests)

[About the framework](#-about-the-yaml_ml-framework)

# ⏩ Quickstart

## 1. Installation

Create a virtual environment (e.g. with `conda`), activate it and upgrade `pip`:

```commandline
conda create --name yaml_ml_env
conda activate yaml_ml_env
pip install --upgrade pip
```

Then install the package:

```commandline
pip install yaml_ml
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

>[!NOTE]
> Without providing the `--n_processes` argument, pipelines will be launched sequentially.

## 3. Docs

Some guidelines about how to define a configuration file are given in the [Configuration File Documentation](https://gfaure9.github.io/yaml-ML/).

All available options are consolidated in the [Modules File](https://github.com/GFaure9/yaml-ML/tree/main/yaml_ml/modules.yaml).

You can also find examples of `yaml_ml` configuration files in the [Examples Folder](https://github.com/GFaure9/yaml-ML/tree/main/examples)
and a template file [template_cfg.yaml](https://github.com/GFaure9/yaml-ML/tree/main/template_cfg.yaml).

# 📖 Usage Example: Step-by-Step

Check out explanations of a complete usage example [here](https://github.com/GFaure9/yaml-ML/tree/main/examples/usage_example).

---

## 🔗 Dependencies

`yaml_ml` is mainly based on __Scikit-learn tools__: https://scikit-learn.org/stable/.

By default, installing `yaml_ml` will also install:
- `lightgbm` (see https://lightgbm.readthedocs.io/en/stable/) to allow for training light gradient boosting models
- `catboost` (see https://catboost.ai/) to allow for training CatBoost models

If you do not want to use them, you can install `yaml_ml` from sources after 
commenting `requirements.txt` lines corresponding to these libraries. To do so, first clone the repo:

```commandline
git clone https://github.com/GFaure9/yaml-ML.git
```

Then comment unwanted packages in the requirements file and run in your virtual environment:

```commandline
cd ./yaml-ML
pip install -e .
```

## ✅ Tests

If you cloned the repo and installed the package from sources (`pip install -e .`), 
you can make sure everything works fine before using it by running:

```commandline
cd ./tests
python test_yaml_ml.py
```

At the end, you should get something like:

```
Ran 4 tests in 120.840s

OK
```

## 🧩 About the `yaml_ml` framework...

`yaml_ml` was designed with a modular architecture, with the aim of facilitating the
integration of new models and data preprocessing techniques as needed. 
So do not hesitate to fork the project and extend the list of available ML models 
or preprocessing methods by "plugging" your favorite ones following the package's architecture.

---

### Latest Release

- v0.0.1