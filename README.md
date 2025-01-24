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

*`yaml_ml` is a Python package that facilitates the creation and running of pipelines to preprocess data 
and train basic machine learning models in a supervised manner, 
by providing the whole instructions through a bunch of keywords in a YAML file.*

> [!IMPORTANT]
> Disclaimer: this is the very first version of the package, that is still under development.
> Do not use in production, or at your own risk.

# Table of Contents 

[Quickstart](#-quickstart)
1. [Installation](#1-installation)
2. [Usage](#2-usage)
3. [Docs](#3-docs)

[Usage Example](#-usage-example-step-by-step)

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

Now that the package is installed, let's have a look to a practical example!

Let's say that we have a dataset
[`datasets/customers.csv`](https://github.com/GFaure9/yaml-ML/tree/main/yaml_ml/datasets/customers.csv)
containing information about a shop's customers (demographic, socioeconomic, behavioral):

```
CustomerID,Gender,Age,AnnualIncome,SpendingScore,Profession,WorkExperience,FamilySize
1,Male,19,15000,39,Healthcare,1,4
2,Male,21,35000,81,Engineer,3,3
3,Female,20,86000,6,Engineer,1,1
4,Female,23,59000,77,Lawyer,0,2
...
1998,Male,87,90961,14,Healthcare,9,2
1999,Male,77,182109,4,Executive,7,2
2000,Male,90,110610,52,Entertainment,5,2
```

And that **we want to build a model to predict how much a new
customer will spend based on its profile, and evaluate the performance of this model**.\
Here, `SpendingScore` will thus be our *target variable*.

To achieve our goal, we need to define:
1) How to preprocess the features values
2) What should be the proportions of our train and test datasets
3) The type of model and its hyperparameters
4) The evaluation metrics to use

### Step 1
Let's say that we have already performed some exploratory data analysis that led
us to want the following preprocessing:
- remove `CustomerID`
- on `Gender`: remove rows with null values, encode it as binary
- on `Age`: remove rows with null values and perform a robust scaling
- on `AnnualIncome`: remove rows with outliers, replace null values by the median and standardize data
- on `Profession`: remove rows with null values and perform a one-hot encoding
- on `WorkExperience`: remove rows with null values and perform a maximum absolute scaling
- on `FamilySize`: remove rows with null values and outliers, and perform a min-max normalization

Then, we will have to write the following lines in our configuration file:

```yaml
preprocessing:
  
  CustomerID:
    type: "cont"
    cleaning: 'remove_col'
    
  Age:
    type: "cat"
    cleaning: 'remove_nans'
    scaling: 'robust'
  
  AnnualIncome:
    type: "cont"
    cleaning: 'remove_outliers'
    replace_nans: 'median'
    scaling: 'standard'

  Profession:
    type: "cat"
    cleaning: 'remove_nans'
    encoding: 'one_hot'

  WorkExperience:
    type: "cont"
    cleaning: 'remove_nans'
    scaling: 'abs_max'

  FamilySize:
    type: "cont"
    cleaning: ['remove_nans', 'remove_outliers']
    scaling: 'min_max'
```

Note that we must also specify the type of the variable (either `"cont"` for continuous or `"cat"`
for categorical).

### Step 2
???

### Step 3
???

### Step 4
???

[//]: # (# todo: continue + at the end say about csv loading + logs + name)
[//]: # (https://www.kaggle.com/datasets/harisrehmanhh/customer-data?resource=download)