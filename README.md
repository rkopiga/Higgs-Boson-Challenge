# Higgs Boson challenge

The Machine Learning Higgs Boson Challenge is a classifying problem that consists of finding the particle collision events corresponding to Higgs Boson particles.

We were given a training set of 250'000 data points (one for each collision) with 30 features each, and were asked to find the best prediction for a ~568'000 data points test set which labels were stored online on AIcrowd.

The only allowed library was numpy.

**Team**: Yann Meier, Kopiga Rasiah, and Rayan Daod Nathoo

**Results**: 

- 8th team over 226 on AIcrowd
- Categorical accuracy: 0.837
- F1-Score: 0.754


**Deadline**: October 28th, 2019

## Getting Started

To run our Machine Learning algorithm with the best parameters we found:

- Clone this project
- Download the datasets on AIcrowd
- Put it in a folder called "data" inside the repository
- Run the run.py file with the following command:

`python run.py`

- Once the program is done, the prediction output is saved in the data folder

(Default value parameters are the ones giving the best performance we found)

## Prerequisites

Python, Numpy

## Implementations

- `gamma_function(step, r=params.r)`: Gamma function returning a gamma decreasing with the step number, according to the parameter `r` in `params.py`.

- `mean_square_error(y, tx, w)`: Mean Squared Error (MSE)

- `mean_absolute_error(y, tx, w)`: Mean Absolute Error (MAE)

- `least_squares_GD(y, tx, initial_w, max_iters, gamma)`: Least-squares Gradient Descent

- `least_squares_SGD(y, tx, initial_w, max_iters, gamma)`: Least-squares Stochastic Gradient Descent

- `compute_stoch_gradient(y, tx, w)`: Randomly select a data point and compute its gradient.

- `least_squares(y, tx)`: Least-squares with direct resolution (Normal Equations)

- `ridge_regression(y, tx, lambda_)`: Ridge regression with direct resolution (Normal Equations)

- `logistic_function(z)`: Compute the logistic function.

- `logistic_loss(y, tx, w, lambda_)`: Compute the logistic loss.

- `logistic_regression(y, tx, initial_w, max_iters, gamma, decreasing_gamma=False)`: 

- `reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, decreasing_gamma=False)`: Regularized logistic regression Stochastic Gradient Descent

- `batch_iter(y, tx, batch_size, num_batches=1, shuffle=True)`: Generate a minibatch iterator for a dataset with `batch_size` data points.

## Parameters
If you want to tweak the parameters, param.py file will be your best friend since it is the control tower of this project.

### *Preprocessing*

- **`SHUFFLE_DATA - boolean`**: Set as True if you want the training set to be shuffled before a local test, False otherwise. This value should be set as True if **`CROSS_VALIDATION`** is, except there is a specific reason.

- **`REMOVE_PHIS - boolean`**: Set as True if you want to remove the _phi features from the datasets, False otherwise.

- **`GROUP_1 - boolean`**: Set as True if you want to group the data points according to the appearance of `UNWANTED_VALUE` in the dataset, False otherwise.

- **`GROUP_2 - boolean`**: Set as True if you want to group the data points according to the value of the feature `PRI_jet_num` (taking values between 0 and 3, which will then result in 4 groups), False otherwise.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **`LESS_GROUPS - boolean`**: Set as True if you want to merge the last 2 groups, False otherwise.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **`ADDITIONAL_SPLITTING - boolean`**: Set as True if you want to split each of the 3 or 4 groups into 2 additional groups based on the appearance of **`UNWANTED_VALUE`**, False otherwise.

- **`REMOVE_INV_FEATURES - boolean`**: Set as True if you want to remove all the invariable features, i.e the features having the same value for every data points, False otherwise.

- **`REPLACE_UNWANTED_VALUE - boolean`**: Set as True if you want to replace each occurence of the **`UNWANTED_VALUE`** by the mean or the median of the remaining values (according to **`VALUE`**) in each feature, False otherwise.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **`VALUE - string`**: Set to `mean` or `median` if you want the **`UNWANTED_VALUE`** occurences to be replaced by the mean or the median of the other values of that feature.

- **`STD - boolean`**: Set as True if you want to standardize each feature, i.e subtract the mean of the feature to every value, and divide them by the standard deviation of that feature, False otherwise.

- **`REPLACE_OUTLIER - boolean`**: Set as True if you want to replace the outliers by a more coherent value, False otherwise.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **`OUTLIER_VALUE - string`**: Set to `clip`, `mean`, or `upper_lower_mean` if you want the outliers to be replaced by the thresholds (= ± **`THRESHOLD`*** σ<sub>feature</sub>), the mean, or the upper-lower means (mean of all the other values above the global mean, and same for the values below) of the other values of that feature.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **`THRESHOLD - float`**: (alpha symbol in the project report) Set to a positive value that will determine the outliers, i.e values outside the range [-**`THRESHOLD`*** σ<sub>feature</sub> , **`THRESHOLD`*** σ<sub>feature</sub>].

- **`REMOVE_DUPLICATE_FEATURES - boolean`**: Set as True if you only want to keep the unique features, i.e remove the duplicates (recommended), False otherwise.

### *Feature engineering*

- **`FEATURE_EXPANSION - boolean`**: Set as True if you want to add new features that are the current features with exponent up to **`DEGREE`**, False otherwise.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **`DEGREE - integer`**: Set to a positive value controlling up to what degree we want the polynomial expansion to produce new features. For example, if **`FEATURE_EXPANSION`** is set to True and **`DEGREE`** = 3, the algorithm will add all the features with exponent 2, and the same features with exponent 3.

- **`FEATURE_MULTIPLICATION - boolean`**: Set as True if you want to add all the pair-multiplication combinations between the current features, False otherwise. **`FEATURE_MULTIPLICATION`** and **`FEATURE_EXPANSION`** should not be set to True at the same time.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **`TRIPLE_MULTIPLICATION - boolean`**: Set as True if you want to add all the 3-tuple-multiplication combinations between the current features, False otherwise.

- **`ADD_COS - boolean`**: Set as True if you want to add the cosine of the current features as new features, False otherwise.

- **`ADD_SIN - boolean`**: Set as True if you want to add the sine of the current features as new features, False otherwise.

- **`ADD_TAN - boolean`**: Set as True if you want to add the tangent of the current features as new features, False otherwise.

- **`ADD_EXP - boolean`**: Set as True if you want to add the exponential of the current features as new features, False otherwise.

- **`ADD_LOG - boolean`**: Set as True if you want to add the logarithm of the current features as new features, False otherwise.

- **`ADD_SQRT - boolean`**: Set as True if you want to add the square root of the current features as new features, False otherwise.

- **`ADD_COS2 - boolean`**: Set as True if you want to add the squared cosine of the current features as new features, False otherwise.

- **`ADD_SIN2 - boolean`**: Set as True if you want to add the squared sine of the current features as new features, False otherwise.

- **`ONE_COLUMN - boolean`**: Set as True if you want to add a one column as a new feature, False otherwise.

### *Local prediction parameters*

- **`LOCAL_PREDICTION - boolean`**: Set as True if you want to run the algorithm locally by splitting the training set into a training set and a test set, False otherwise.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **`RATIO - float`**: Set to a positive value between 0 and 1 to decide the proportion of the new training set after splitting the original training set.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **`CROSS_VALIDATION - boolean`**: Set as True if you want to do a K-fold cross validation, False otherwise.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **`K - integer`**: Set to a positive integer indicating the number of folds for the cross-validation.

### *Implementation parameters*

**`IMPLEMENTATION - integer`**: Set to 0 (Least-Squares), 1 (Ridge regression), 2 (Regularized logistic regression), to choose the implementation to apply at the end.

- Ridge regression parameter:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **`RIDGE_LAMBDA - float`**: Set to a positive value (recommended between 1e-10 and 1e-3) to choose the lambda parameter of the ridge regression.

- Regularized logistic regression parameters:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **`MAX_ITERS - integer`**: Set to an integer positive value (recommended above 20) to choose the number of iterations for the regularized logistic regression.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **`GAMMA - float`**: Set to an integer positive value between 0 and 1 to choose the gamma parameter for the regularized logistic regression.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **`DECREASING_GAMMA - boolean`**: Set as True if you want to call the `gamma_function` in `implementations.py`which decreases the gamma in terms of the step number, False otherwise.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **`r - float`**: Set to a positive value between 0.5 and 1 to control the decreasing speed of gamma.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **`LOG_LAMBDA - float`**: Set to a positive value between 0 and 1 to choose the lambda parameter for the regularized logistic regression.

## Preprocessing

All the preprocessing functions are in `preprocessing.py`.

Refer to the pdf report for further details.

## Feature engineering

All the feature engineering functions are in `feature_engineering.py`.

Refer to the pdf report for further details.

## Local prediction

All the prediction functions are in `local_prediction.py`.

Refer to the pdf report for further details.


## Further improvements
- One could try to standardize the features of the test set with the same means and standard-deviations than the ones used in the training set.
- One could try different methods of normalization/standardization for the features, e.g min-max-normalization etc.
- One could also try to do the feature expansion with an array of degrees, e.g `[2, 4, 5]` to find the best combination of polynomial expansions without overfitting.


## Ackowledgements

Thank you to **Martin Jaggi**, **Rüdiger Urbanke**, and all the EPFL Machine Learning team for this vers interesting project!