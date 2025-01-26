[//]: # (<p align="center">)

[//]: # (  <img src="./logo/logo2_colored.png" width="400" />)

[//]: # (</p>)

[//]: # (<p align="center">)

[//]: # (  <img src="./logo/logo_drawing.png" width="400" />)

[//]: # (</p>)

[//]: # (<p align="center">)

[//]: # (  <img src="./logo/logo_sentence.png" width="400" />)

[//]: # (</p>)
<p align="center">
  <img src="./logo/logo3.png" width="400" />
</p>

---

### *Your whole ML pipeline in one YAML file!*

![GitHub Repo stars](https://img.shields.io/github/stars/GFaure9/yaml-ML?style=social)
![GitHub All Releases](https://img.shields.io/github/downloads/GFaure9/yaml-ML/total)
![Development Stage](https://img.shields.io/badge/stage-Beta-yellow)

[//]: # (*`yaml_ml` is a Python package that facilitates the creation and running of pipelines to preprocess data )

[//]: # (and train basic machine learning models in a supervised manner, )

[//]: # (by providing the whole instructions through a bunch of keywords in a YAML file.*)

[//]: # (*`yaml_ml` makes defining and running machine learning pipelines as)

[//]: # (simple as editing a YAML file. From data preprocessing to training and evaluation,)

[//]: # (the entire workflow is just a configuration away.*)

*`yaml_ml` streamlines machine learning workflows by letting you
define data preprocessing, model training, 
and evaluation in one YAML file. Automate your ML pipeline with minimal code.*

> [!IMPORTANT]
> Disclaimer: this is the very first version of the package, that is still under development.
> Do not use in production, or at your own risk.

# Table of Contents 

[Quickstart](#-quickstart)
1. [Installation](#1-installation)
2. [Usage](#2-usage)
3. [Docs](#3-docs)

[Usage Example](#-usage-example-step-by-step)

[Dependencies](#-dependencies)

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
To launch all the corresponding pipelines using multiprocessing over `N` CPU cores, run the command:

```commandline
python -m yaml_ml --cfg path/to/your/configs/folder --cpu N 
```

>[!NOTE]
> Without providing the `--cpu` argument, pipelines will be launched sequentially.

## 3. Docs

Some guidelines about how to define a configuration file are given in the [Configuration File Documentation](https://gfaure9.github.io/yaml-ML/).

All available options are consolidated in the [Modules File](https://github.com/GFaure9/yaml-ML/tree/main/yaml_ml/modules.yaml).

You can also find examples of `yaml_ml` configuration files in the [Examples Folder](https://github.com/GFaure9/yaml-ML/tree/main/examples).

# 📖 Usage Example: Step-by-Step

Check out explanations of a complete usage example [here](https://github.com/GFaure9/yaml-ML/tree/main/examples/usage_example).

---

## 🔗 Dependencies

`yaml_ml` is mainly based on __Scikit-learn tools__: https://scikit-learn.org/stable/.

By default, installing `yaml_ml` will also notably install:
- `lightgbm` (see https://lightgbm.readthedocs.io/en/stable/) to allow for training light gradient boosting models
- `catboost` (see https://catboost.ai/) to allow for training CatBoost models

If you do not want to use them, you can also install `yaml_ml` from source after 
commenting `requirements.txt` lines corresponding to these libraries.

```commandline
git clone https://github.com/GFaure9/yaml-ML.git
```

Then comment unwanted packages in the requirements file and run in your virtual environment:

```commandline
cd ./yaml-ML.git
pip install -r requirements.txt
```

## 🧩 About the `yaml_ml` framework...

`yaml_ml` was designed with a modular architecture, with the aim of facilitating the
integration of new models and data preprocessing techniques as needed. 
So do not hesitate to fork the project and extend the list of available ML models 
or preprocessing methods by "plugging" your favorite ones following the package's architecture.


