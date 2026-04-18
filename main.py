import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

def load_dataset(filename):

    df = pd.read_excel(filename, sheet_name='Subscription_Data')

    
    x = df.drop(columns=['Purchased_Premium']).to_numpy().tolist()
    y = df['Purchased_Premium'].to_numpy().tolist()
    return x, y


filename = "logistic_regression_training_data.xlsx"
train_x, train_y = load_dataset(filename)

def sigmoid(z):

    return 1 / (1 + math.exp(-z))

def hypothesis(features, weights, bias):

    total = 0
    n = len(features)
    for i in range(n):
        total += features[i] * weights[i]
    prediction = total + bias
    return sigmoid(prediction)

def compute_regularization(weights, error, delta, m):
    cost = 0
    n = len(weights)
    for j in range(n):
        cost += weights[j] ** 2
    reg_cost = (delta / (2 * m)) * cost
    return error + reg_cost

def compute_cost(x, y, weights, bias, delta):

    m = len(x)
    error = 0

    epsilon = 1e-15
    for i in range(m):

        probability = hypothesis(x[i], weights, bias)
        prediction = max(epsilon, min(1 - epsilon, probability))
        loss = -y[i] * math.log(prediction) - (1 - y[i]) * math.log(1 - prediction)
        error += loss
    
    error = error / m
    cost = compute_regularization(weights, error, delta, m)
    return cost

def compute_cost_derivative(x, y, weights, bias, delta):

    m = len(x)
    feature_number = len(x[0])
    dj_dw = [0] * feature_number
    dj_db = 0

    for i in range(m):
        prediction = hypothesis(x[i], weights, bias)
        loss = prediction - y[i]
        
        for j in range(feature_number):
            dj_dw[j] += loss * x[i][j]
        
        dj_db += loss

    dj_dw = [dj_dw_i / m for dj_dw_i in dj_dw]
    dj_db = dj_db / m

    for j in range(feature_number):
        reg_cost = (delta / m) * weights[j]
        dj_dw[j] += reg_cost

    return dj_dw, dj_db

def compute_new_params(weights, bias, dj_dw, dj_db, alpha):
    
    feature_number = len(weights)
    new_weights = []
    for j in range(feature_number):
        new_weight = weights[j] - alpha * dj_dw[j]
        new_weights.append(new_weight)
    new_bias = bias - alpha * dj_db
    return new_weights, new_bias

def compute_gradient(x, y, weights, bias, alpha, delta, iterations = 4000):

    cost_hist = []

    for i in range(iterations):

        cost = compute_cost(x, y, weights, bias, delta)
        cost_hist.append(cost)
        dj_dw, dj_db = compute_cost_derivative(x, y, weights, bias, delta)
        new_weights, new_bias = compute_new_params(weights, bias, dj_dw, dj_db, alpha)
        weights = new_weights
        bias = new_bias
    
    plt.plot(np.arange(len(cost_hist)), cost_hist)
    plt.xlabel("iterations")
    plt.ylabel("cost")
    plt.show()
    return weights, bias

def compute_vector_average(vector):
    m = len(vector)
    total = 0
    for i in range(m):
        total += vector[i]
    average = total / m
    return average

def compute_vector_std(vector):
    m = len(vector)
    average = compute_vector_average(vector)
    total = 0
    for i in range(m):
        total += (vector[i] - average) ** 2
    deviation = (total / m) ** (1/2)
    return deviation, average

def compute_vector_zscore_normalization(vector):

    m = len(vector)
    deviation, average = compute_vector_std(vector)
    vector_normalized = []
    for i in range(m):
        if deviation == 0:
            rescaled = 0
        else:
            rescaled = (vector[i] - average) / deviation
        vector_normalized.append(rescaled)
    return vector_normalized

def compute_matrix_average(matrix):

    m = len(matrix)
    feature_number = len(matrix[0])
    averages = []
    for j in range(feature_number):
        total = 0
        for i in range(m):
            total += matrix[i][j]
        average = total / m
        averages.append(average)
    return averages

def compute_matrix_std(matrix):

    m = len(matrix)
    feature_number = len(matrix[0])
    averages = compute_matrix_average(matrix)
    deviations = []
    for j in range(feature_number):
        total = 0
        for i in range(m):
            total += (matrix[i][j] - averages[j]) ** 2
        deviation = (total / m) ** (1/2)
        deviations.append(deviation)
    return deviations, averages

def compute_matrix_zscore_normalization(matrix):

    m = len(matrix)
    feature_number = len(matrix[0])
    deviations, averages = compute_matrix_std(matrix)
    matrix_normalized = []
    for i in range(m):
        cols = []
        for j in range(feature_number):
            if deviations[j] == 0:
                rescaled = 0
            else:
                rescaled = (matrix[i][j] - averages[j]) / deviations[j]
            cols.append(rescaled)
        matrix_normalized.append(cols)
    return matrix_normalized

def predict(value):
    result = value + 0.5
    return int(result)

feature_number = len(train_x[0])
weights = [0] * feature_number
bias = 0
alpha = 0.01
delta = 0.7

deviation, average = compute_vector_std(train_y)
x = compute_matrix_zscore_normalization(train_x)

weights, bias = compute_gradient(x, train_y, weights, bias, alpha, delta)

index = 0
target = x[index]
probability = hypothesis(target, weights, bias)
prediction = predict(probability)

print(f"expected is {train_y[index]}, probability is {probability} and prediction is {prediction}")