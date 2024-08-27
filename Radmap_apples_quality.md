# Apple Quality - EDA and Classification Models

This project is a part of the kaggle.com [dataset](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality), which goal was to predict whether the apple quality is good or bad.

### Part 1 - EDA

- The distribution of the features to find out that most of them are approximately normal.
- The distribution of the target variable was almost equal, so we faced a balanced dataset.
- There was none to little correlation between variables.

### Part 2 - Cleaning

- Changing *Acidity* variable dtype to numeric.
- Droping ID column.
- Droping duplicates if any exist.
- Outlier removal based on *z-score*, as we are facing approximately normally distributed variables.

### Part 3 - Preprocessing

- Scaling the data using *Standard Scaler*, as we are facing approximately normally distributed variables.

### Part 4 - Classifiers

Choosing the best model was based on an *accuracy* metric:
$$accuracy = \frac{TP+TN}{TP+TN+FP+FN}$$
with use of custom class *CustomClassifier*, that utilized *StratifiedKFold* to make sure we test our model on every part of training data.

#### Thanks to the simple grid search parameters, we got top $3$ classifiers:

1. Multilayer Perceptron $\to$ $92\%$ *accuracy*
2. Support Vector Machine $\to$ $91\%$ *accuracy*
3. ex aequo XGBoost and k-Nearest Neighbors $\to$ $90\%$ *accuracy*