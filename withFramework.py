import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


# Load dataset
df = pd.read_csv('./data/dataset.csv')
print('\nData loaded succefully!')
print('Shape: ', df.shape)
print(df.head())

# Viewing data
df.plot(x = 'Investment', y = 'Return', style = 'o')
plt.title('Investment X Return')
plt.xlabel('Investment')
plt.ylabel('Return')
plt.savefig('images/part1-chart1.png')
# plt.show()

# Preparing data
X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

# Split data into training and test (70/30)
X_training, X_test, y_training, y_test = train_test_split(X, y, test_size = .3, random_state = 0)

# Adjusts the format and type of training data
X_training = X_training.reshape(-1, 1).astype(np.float32)

# Model building
model = LinearRegression()

# Training model
model.fit(X_training, y_training)
print('\nModel trained succesfully!')

# Showing intercept B0 and coef B1
print('B0 (intercept_): ', model.intercept_)
print('B1 (coef_): ', model.coef_)

# Plotting the regression line
regression_line = model.coef_ * X + model.intercept_
plt.scatter(X, y)
plt.title('Investment X Return')
plt.xlabel('Investment')
plt.ylabel('Expected Return')
plt.plot(X, regression_line, color = 'red')
plt.savefig('images/part1-regressionLine.png')
# plt.show()

# Data test predicting
y_pred = model.predict(X_test)

# Real X Predicted
df_values = pd.DataFrame({'Real Value': y_test, 'Predicted Value': y_pred})
# print(df_values)

# Plot
fig, ax = plt.subplots()
index = np.arange(len(X_test))
bar_width = .35
actual = plt.bar(index, df_values['Real Value'], bar_width, label = 'Real Value')
predicted = plt.bar(index + bar_width, df_values['Predicted Value'], bar_width, label = 'Predicted Value')
plt.xlabel('Investment')
plt.ylabel('Expected Return')
plt.title('Real X Predicted')
plt.xticks(index + bar_width, X_test)
plt.legend()
plt.savefig('images/part1-actualvspredicted.png')
# plt.show()

# Evaluating the model
print('\n')
print('MAE (Mean Absolute Error):', mean_absolute_error(y_test, y_pred))
print('MSE (Mean Squared Error):', mean_squared_error(y_test, y_pred))
print('RMSE (Root Mean Squared Error):', math.sqrt(mean_squared_error(y_test, y_pred)))
print('R2 Score:', r2_score(y_test, y_pred))

# Predicting return over investment with new data.
print('\n')
input_investment = float(input('\nDigit the investment value: '))
investment = np.array([input_investment])
investment = investment.reshape(-1, 1)

# Predicts
pred_score = model.predict(investment)
print('\n')
print('Investment made = ', input_investment)
print('Predicted return = {:.4}'.format(pred_score[0]))
print('\n')