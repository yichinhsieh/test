import numpy as np

# activation function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

# derivative of sigmoid
def sigmoid_prime(x):
    return x * (1 - x)

if __name__ == '__main__':
    # Number of iterations
    epochs = 60000
    inputLayerSize = 2
    hiddenLayerSize = 3
    outputLayerSize = 1
    # learning rate
    a = 0.1

    # data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # weights on layer inputs
    w_hidden = np.random.uniform(size=(inputLayerSize, hiddenLayerSize))
    w_output = np.random.uniform(size=(hiddenLayerSize, outputLayerSize))

    for epoch in range(epochs):

        # Forward
        act_hidden = sigmoid(np.dot(X, w_hidden))
        output = np.dot(act_hidden, w_output)

        # Calculate error
        error = y - output

        if epoch % 5000 == 0:
            print(f'error sum {sum(error)}')

        # Backward
        dZ = error * a
        w_output += act_hidden.T.dot(dZ)
        dH = dZ.dot(w_output.T) * sigmoid_prime(act_hidden)
        w_hidden += X.T.dot(dH)

    X_test = X[1]  # [0, 1]

    act_hidden = sigmoid(np.dot(X_test, w_hidden))
    result = np.round(np.dot(act_hidden, w_output))
    print(result)