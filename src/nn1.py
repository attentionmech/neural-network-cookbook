# Perceptron for AND Gate Problem

# Activation function (step function)
def step_function(x):
    if x >= 0:
        return 1
    else:
        return 0

# Perceptron algorithm
def perceptron(X, y, learning_rate=0.1, epochs=100):
    # Initialize weights and bias to 0
    weights = [0, 0]  # Two inputs
    bias = 0
    # Training the perceptron
    for epoch in range(epochs):
        for i in range(len(X)):
            # Calculate the output
            output = step_function(weights[0] * X[i][0] + weights[1] * X[i][1] + bias)
            # Calculate the error
            error = y[i] - output
            # Update weights and bias
            weights[0] += learning_rate * error * X[i][0]
            weights[1] += learning_rate * error * X[i][1]
            bias += learning_rate * error
    return weights, bias

# AND Gate Input and Output
X = [[0, 0], [0, 1], [1, 0], [1, 1]]  # Input data
y = [0, 0, 0, 1]  # Expected output for AND gate

# Train the perceptron
weights, bias = perceptron(X, y)

# Print the trained weights and bias
print(f"Trained Weights: {weights}")
print(f"Trained Bias: {bias}")

# Test the perceptron on all inputs
for i in X:
    result = step_function(weights[0] * i[0] + weights[1] * i[1] + bias)
    print(f"Input: {i}, Output: {result}")
