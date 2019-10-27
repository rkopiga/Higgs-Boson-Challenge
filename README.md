# Higgs Boson challenge

This project is the first required project of the EPFL course Machine Learning. Known worldwide, the Machine Learning **Higgs Boson Challenge** is a classifying problem that consists of finding the particle collision events corresponding to Higgs Boson particles.

We were given a training set of 250 000 data points (one for each collision) with 30 features each.

The only allowed library was numpy.

**Team**: Yann Meier, Kopiga Rasiah, and Rayan Daod Nathoo

**Results**: 

- 8th team over TODO on AIcrowd
- Categorical accuracy: 0.837
- F1-Score: 0.755 


**Deadline**: October 28th, 2019

## Getting Started

To run our Machine Learning algorithm with the best parameters we found:

- Clone this project
- Download the datasets on AIcrowd
- Put it in a folder called "data" inside the repository
- Run the run.py file with the following command:

`python run.py`

(Default value parameters are the ones giving the best performance we found)

## Prerequisites

Python, Numpy


## Parameters
If you want to tweak the parameters, param.py file will be your best friend since it is the control tower of this project.

##### *Preprocessing*

`SHUFFLE_DATA` (boolean):

`REMOVE_PHIS`:

`GROUP_1`:

`GROUP_2`:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`LESS_GROUPS`:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`ADDITIONAL_SPLITTING`:

`REMOVE_INV_FEATURES`:

`REPLACE_UNWANTED_VALUE`:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`VALUE`:

`STD`:

`REPLACE_OUTLIER`:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`OUTLIER_VALUE`:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`THRESHOLD`:

`REMOVE_DUPLICATE_FEATURES`:

##### *Feature engineering*

`FEATURE_EXPANSION`:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`DEGREE`:

`FEATURE_MULTIPLICATION`:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`TRIPLE_MULTIPLICATION`:

`ADD_COS`:

`ADD_SIN`:

`ADD_TAN`:

`ADD_EXP`:

`ADD_LOG`:

`ADD_SQRT`:

`ADD_COS2`:

`ADD_SIN2`:

`ONE_COLUMN`:

##### *Local prediction parameters*

`LOCAL_PREDICTION`:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`RATIO`:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`CROSS_VALIDATION`:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`K`:

##### *Implementation parameters*

`IMPLEMENTATION`:

Ridge regression parameter:

`RIDGE_LAMBDA`:

Logistic regression parameters:

`MAX_ITERS`:

`GAMMA`:

`DECREASING_GAMMA`

`r`:

`LOG_LAMBDA`:

## Preprocessing

All the preprocessing functions are in preprocessing.py.

## Feature engineering

All the feature engineering functions are in feature_engineering.py.

## Local prediction

All the prediction functions are in local_prediction.py.


## Ackowledgements

Thank you to Martin Jaggi, Rüdiger Urbanke, and all the EPFL Machine Learning team for this vers interesting project!