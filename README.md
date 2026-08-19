## Linear Regression from Scratch

A linear regression implementation from scratch in Python using a custom gradient-descent optimization algorithm.

The project does not use an existing linear regression solver. Instead, the gradient, parameter updates, feature normalization, and weight restoration are implemented manually.

It also uses "mat_opr_python", a custom array backend implemented separately from scratch in Python.

## Gradient Descent

For linear regression, the prediction is:

y_hat = Xw

The error is:

error = Xw - y

The gradient of the mean squared error with respect to the weights is:

gradient = Xᵀ(Xw - y) / m

The weights are then updated iteratively using the calculated gradient.

## Adaptive Learning Rate

Instead of using one fixed learning rate, this implementation maintains a separate learning rate for each weight.

The learning rates are gradually increased while the optimization continues in the same direction.

When the sign of a gradient changes, the corresponding parameter is considered to have overshot its minimum. Its learning rate is reduced and its acceleration is stopped.

This allows different weights to use different learning rates.

## Feature and Target Normalization

Before optimization, each feature is centered and divided by its standard deviation:

X_normalized = (X - mean) / std

The target is normalized in the same way.

Normalization puts the variables on comparable scales and makes the gradient-based optimization more stable.

After convergence, the weights are converted back to the original feature scale and the intercept is restored.

## L2 Regularization

The implementation also supports optional L2 regularization through the "lambda_" parameter.

model = LinearRegression(lambda_=0.1)

The regularization term added to the gradient is:

lambda * w / m

With the default value:

lambda_ = 0

the model performs ordinary linear regression without regularization.

## Custom Array Backend

The model uses "mat_opr_python" as its array backend.

"mat_opr_python" is a separate project implemented from scratch in Python. It provides the array operations required by this implementation, including:

- Matrix multiplication
- Broadcasting
- Element-wise operations
- Transpose views
- Reshaping
- Slicing and indexing
- Boolean masking
- Mean and standard deviation
- "where"
- Strided array access

The backend uses Python lists for underlying storage and implements the array operations itself rather than relying on NumPy's linear algebra routines.

## Comparison with Scikit-learn

The example compares the custom implementation with "sklearn.linear_model.LinearRegression" using the same dataset.

sk_model = skl()
sk_model.fit(x_train, y_train)

custom_model = LinearRegression()
custom_model.fit(array(x_train), array(y_train))

The predictions from both models can then be compared to verify the implementation.

## Dataset

The repository contains "train-test.npz", a synthetic regression dataset used for testing.

It contains:

x_train: (700, 1)
y_train: (700,)
x_test:  (300, 1)
y_test:  (300,)

The dataset also contains several outliers to test how the model behaves with less ideal data.

## Requirements

- Python
- NumPy
- scikit-learn

NumPy is used for loading the dataset and for comparison with scikit-learn. The actual model can use the custom "mat_opr_python" backend for its array operations.

The mat_opr_python backend is available as a separate repository on my GitHub profile.
