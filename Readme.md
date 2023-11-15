# Linear Regression Scripts
## Using Framework
### Overview
This script demonstrates linear regression using the scikit-learn framework. It includes data loading, visualization, model training, evaluation, and prediction. The dataset is loaded from a CSV file, and the linear regression model is trained using the scikit-learn library.

#### Dependencies
- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn

#### Setup
- Ensure Python and required libraries are installed.
- Place the dataset (dataset.csv) in the ./data/ directory.

#### Instructions
- Run the script.
- Input the investment value when prompted for prediction.

#### Execution
The script generates visualizations and provides metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R2 Score.

## Without Framework
### Overview
This script implements linear regression without using any external frameworks. It calculates coefficients, performs model training, makes predictions, and evaluates the model's performance. The dataset is loaded from a CSV file, and the linear regression model is implemented manually.

#### Dependencies
- Python

#### Setup
- Place the dataset (dataset.csv) in the root directory.

#### Instructions
- Run the script.
- Input the investment value when prompted for prediction.

#### Execution
The script prints the coefficients, model error, and predicted return for the provided investment value.

Note: Ensure that the required dataset (dataset.csv) is available in the specified location before running the scripts. The scripts can be executed individually, and each provides a demonstration of linear regression, one using scikit-learn and the other without any external frameworks.