import numpy as np  # linear algebra
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')

x = data['OverallQual']
y = data['SalePrice']

x = (x - x.mean()) / x.std()
x = np.c_[np.ones(x.shape[0]), x]

# GRADIENT DESCENT

lr_rate = 0.1  # Step size
iterations = 100
# No. of iterations
m = y.size  # No. of data points
np.random.seed(123)  # Set the seed
weight = np.random.rand(2)  # Some random values to start with
bias = 1  # np.random.rand(1) (Bias as 1)


# GRADIENT DESCENT
def gradient_descent(x, y, weight, iterations, lr_rate):
    past_costs = []
    past_weights = [weight]
    for i in range(iterations):
        prediction = np.dot(x, weight)
        error = prediction - y
        cost = 1 / (2 * m) * np.dot(error.T, error)
        past_costs.append(cost)
        weight = weight - (lr_rate * (1 / m) * np.dot(x.T, error) * bias)
        past_weights.append(weight)
        print("Gradient Descent: {:.2f}, {:.2f}".format(weight[0], weight[1]))

    return past_weights, past_costs


# Pass the relevant variables to the function and get the new values back...
past_weights, past_costs = gradient_descent(x, y, weight, iterations, lr_rate)
weight = past_weights[-1]


# Plot the cost function...
plt.title('Cost Function ')
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.plot(past_costs)
plt.show()
