# 📖 Usage Example: Step-by-Step

Let's have a look to a practical example.

Let's say that we have a dataset
[`datasets/customers.csv`](https://github.com/GFaure9/yaml-ML/tree/main/yaml_ml/datasets/customers.csv)
containing information about a shop's customers (demographic, socioeconomic, behavioral)[^1]:

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

We will create a blank `customers_pipeline.yaml` config file and fill it step-by-step.

### Preliminary steps
Before going through preprocessing and model definition steps, we need to specify:
- a name for our pipeline (that will be used in outputs files names)
- a path to a folder where to store the outputs (that will be created if not existing already)
- whether we want logs to be written to a text file during the execution of the pipeline
- instructions to load the data (folder path, file name and format, separator type)
- the target variable

This is done in our case by writing the following lines in our configuration file:

```yaml
name: "CustomersPipeline"

output_folder: "./CustomersPipeline_outputs"

logs: yes

loading:
  folder: "./datasets"
  name: "customers"
  format: 'csv'
  separator: ','

target_var: 'SpendingScore'
```

### Step 1
Let's say that we have already performed some exploratory data analysis that led
us to want the following preprocessing:
- remove `CustomerID`
- on `SpendingScore`: remove rows with null values
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
    
  SpendingScore:
    type: "cont"
    cleaning: 'remove_nans'
    
  Gender:
    type: "cat"
    cleaning: 'remove_nans'
    encoding: 'binary'
    
  Age:
    type: "cont"
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

Note that we must also specify the type of each variable (either `"cont"` for continuous or `"cat"`
for categorical).

You can refer to the [docs](#3-docs) to see what are the available options
for dataset preprocessing.

### Step 2
We will then specify in which proportions to split the dataset between training and
testing subsets. As classically done, we will impose a 80%-20% split. For this, we will have
to add the following lines to our configuration file.

```yaml
dataset:
  
  split:
    train: 80
```

One can indifferently set the train size or the test size (for instance `test: 20` instead
of `train: 80`).

> [!NOTE]
> One can also set `stratified: yes` (indented like `train`)
> to preserve the target variable's distribution across train and test datasets.

### Step 3
At this stage, we need to indicate what is the type of ML model we want to fit to
the data and with what hyperparameters. You can refer to the [docs](#3-docs) to
see what are the available options for the model.

Here, for illustration, we choose a ridge regression model, optimizing 
$\min_w ||y - Xw||_2^2 + \alpha ||w||_2^2$ with $\alpha = 0.1$ and including 
the intercept in the training process.

For this, we will have to write the following lines in our configuration file:

```yaml
model:
  
  regression:
    
    lasso:
      alpha: 0.1
      fit_intercept: yes
```

Note that you must specify whether you want to train a model for a regression task 
(keyword: `regression`) or a classification task (keyword: `classification`). 
This is particularly important for methods that can be used for both, such as 
Decision Trees, Random Forests, and K-Nearest Neighbors.

### Step 4
Finally, we want to indicate which evaluation score(s) to compute on the test dataset with
the trained model.

In our case, assuming that we want to check the coefficient of determination R2, 
the Root Mean Square Error (RMSE) and the Mean Absolute Error (MAE), 
we will have to add the following lines in our configuration file:

```yaml
score: ['r2', 'rmse', 'mae']
```

You can also specify a single score as a string instead of using a one-item list.

Please refer to the [docs](#3-docs) to see what are the available options for the scores.
Note that score types are specific to regression and classification tasks.

### Running the pipeline
To launch the pipeline as defined in the configuration file:
- open a terminal in the `customers_pipeline.yaml` folder
- activate the environment where `yaml_ml` is installed
- run the command

```commandline
python -m yaml_ml --cfg customers_pipeline.yaml
```

The trained model in PKL format,
along with a summary file of the computed evaluation scores,
will be saved in the output folder.

>[!NOTE]
> Running the command for multiple configuration files (i.e. giving the folder path were
> these configurations are stored) will also generate plots showing histograms of
> the resulting scores for tested pipelines.

[^1]: Dataset adapted from 
[_harisrehmanhh_ __customer data__ Kaggle dataset](https://www.kaggle.com/datasets/harisrehmanhh/customer-data?resource=download).