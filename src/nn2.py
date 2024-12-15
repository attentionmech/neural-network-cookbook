# Step 1: Define the step function (activation function)
def step_function(x):
    return 1 if x >= 0 else 0

# Step 2: Perceptron training function
def perceptron_train(X, y, learning_rate=0.1, epochs=10):
    # Initialize weights and bias
    weights = [0, 0]  # for two inputs
    bias = 0
    
    # Training loop
    for epoch in range(epochs):
        for i in range(len(X)):
            # Calculate the weighted sum of inputs + bias
            linear_output = sum(x * w for x, w in zip(X[i], weights)) + bias
            # Apply the activation function
            prediction = step_function(linear_output)
            
            # Update weights and bias if there is an error
            error = y[i] - prediction
            weights = [w + learning_rate * error * x for w, x in zip(weights, X[i])]
            bias += learning_rate * error
    
    return weights, bias

# Step 3: Perceptron predict function
def perceptron_predict(X, weights, bias):
    predictions = []
    for i in range(len(X)):
        linear_output = sum(x * w for x, w in zip(X[i], weights)) + bias
        predictions.append(step_function(linear_output))
    return predictions

# Step 4: Define input-output pairs for the OR gate
X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]

y = [0, 1, 1, 1]  # Output for OR gate

# Step 5: Train the perceptron
weights, bias = perceptron_train(X, y, learning_rate=0.1, epochs=10)

# Step 6: Test the perceptron
print("Trained weights:", weights)
print("Trained bias:", bias)

# Make predictions
predictions = perceptron_predict(X, weights, bias)
print("Predictions on the training set:", predictions)
