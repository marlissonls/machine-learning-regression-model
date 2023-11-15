from os import getcwd
from math import sqrt
from csv import reader

# Get the root path
path = getcwd()

# Calculates the mean
def mean(values):
    return sum(values) / float(len(values))

# Calculates the covariance
def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar

# Calculates the variance
def variance(list, mean):
    return sum([(x - mean)**2 for x in list])

# Calculates the coefficients
def coefficient(covar, var, mean_x, mean_y):
    b1 = covar / var
    b0 = mean_y - (b1 * mean_x)
    return b1, b0

# Load the dataset
def load_data(dataset):
    init = 0
    x = list()
    y = list()
    with open(dataset) as file:
        content = reader(file)
        for row in content:
            if init == 0:
                init = 1
            else:
                x.append(row[0])
                y.append(row[1])
    return x, y

# Splits the data into training and test
def split_dataset(x, y):
    x_training = list()
    y_training = list()
    x_test = list()
    y_test = list()

    training_size = int(0.8 * len(x))

    x_training, x_test = x[0:training_size], x[training_size::]
    y_training, y_test = y[0:training_size], y[training_size::]

    return x_training, y_training, x_test, y_test

# Calculates y = B1 * x + B0
def predict(b0, b1, test_x):
    predicted_y = list()
    for i in test_x:
        predicted_y.append(b0 + b1 * i)
    return predicted_y

# Calculates RMSE
def rmse(predicted_y, test_y):
    sum_error = 0.0
    for i in range(len(predicted_y)):
        sum_error += (predicted_y[i] - test_y[i]) ** 2
    return sqrt(sum_error / float(len(test_y)))


def main():
    try:
        # Load dataset
        dataset = path + '/data/dataset.csv'
        x, y = load_data(dataset)

        # Prepare the data
        x = [float(i) for i in x]
        y = [float(i) for i in y]

        # Splits the data into x and y
        x_training, y_training, x_test, y_test = split_dataset(x, y)

        # Calculates the x and y mean values, covariance and variance
        mean_x = mean(x_training)
        mean_y = mean(y_training)
        covar = covariance(x_training, mean_x, y_training, mean_y)
        var = variance(x_training, mean_x)

        # Calculates the coefficients b1 ans b0 (model training)
        b1, b0 = coefficient(covar, var, mean_x, mean_y)

        # Printing the coefficients
        print('\nCoefficients')
        print('B1:', b1)
        print('B0:', b0)

        # Predictions
        predicted_y = predict(b0, b1, x_test)

        # Calculates and prints the model error
        root_mean = rmse(predicted_y, y_test)
        print('\nLinear Regression Model without using Frameworks')
        print(f'Mean model error {root_mean}\n')

        # Predicting with new values
        new_x = int(input("Digit the investment value: "))
        new_y = b0 + b1 * new_x
        print(f'\nPredicted return: {new_y}\n')

    except Exception as e:
        print(e)

main()