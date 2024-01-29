import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score



# Define features and classes
features = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "roundnes"]
classes = ["BOMBAY", "CALI", "SIRA"]

# User Input
df = pd.read_excel("D:\Downloads/Dry_Bean_Dataset.xlsx")

# Preprocess the data
lab = LabelEncoder()
min_max_scaler = MinMaxScaler()

# Encode class labels
df['Class'] = lab.fit_transform(df['Class'])

# Fill missing values in MinorAxisLength with the mean
mean_value = df["MinorAxisLength"].mean()
df["MinorAxisLength"].fillna(mean_value, inplace=True)

# Scale features to [0, 1]
df[features] = min_max_scaler.fit_transform(df[features])

# Split data into classes
Drybean_BOMBAY = df[df['Class'] == 0].iloc[:50]
Drybean_CALI = df[df['Class'] == 1].iloc[:50]
Drybean_SIRA = df[df['Class'] == 2].iloc[:50]

# Combine data for training
training_data = pd.concat([Drybean_BOMBAY.iloc[:30], Drybean_CALI.iloc[:30], Drybean_SIRA.iloc[:30]])

# Test data
test_data = pd.concat([Drybean_BOMBAY.iloc[30:], Drybean_CALI.iloc[30:], Drybean_SIRA.iloc[30:]])

# Extract features and labels
X_train = training_data[features].values
y_train = training_data['Class'].values

X_test = test_data[features].values
y_test = test_data['Class'].values

num_hidden_layers = int(input("Enter the number of hidden layers: "))
hidden_layer_neurons = int(input("Enter the number of neurons in each hidden layer: "))
learning_rate = float(input("Enter the learning rate (eta): "))
epochs = int(input("Enter the number of epochs (m): "))
use_bias = input("Add bias? (yes/no): ").lower() == "yes"
activation_function = input("Choose activation function (sigmoid/tanh): ").lower()

# Initialization
num_features = len(features)
num_classes = len(classes)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def get_activation_function(activation_function):
    activation_function = activation_function.lower()  # Convert to lowercase for case-insensitivity

    if activation_function == "sigmoid":
        return sigmoid, sigmoid_derivative
    elif activation_function == "hyperbolic tangent" or activation_function == "tanh":
        return tanh, tanh_derivative
    else:
        raise ValueError(f"Unsupported activation function: {activation_function}")

import numpy as np
w=[]
b=[]
def train(hidden_layers, neurals_in_hidden_layer, activation_function, learning_rate, epochs, use_bias, train_samples, train_labels):
    activation_function, activation_function_derivative = get_activation_function(activation_function)

    # Initializing the weights and biases
    weightss, biasess = init(hidden_layers, neurals_in_hidden_layer, train_samples)
    w=weightss
    b=biasess
    if not use_bias:
        biasess = [np.zeros(bias.shape) for bias in biasess]

    for epoch in range(epochs):
        correct_predictions = 0

        for sample, label in zip(train_samples, train_labels):
            layer_output = forward_propagation(weightss, biasess, sample, activation_function, hidden_layers)

            errors = back_propagation(weightss, label, activation_function_derivative, hidden_layers, layer_output)

            weightss, biasess = update_weights(weightss, biasess, learning_rate, errors, layer_output, hidden_layers, use_bias)

            predicted_label = np.argmax(layer_output[-1])
            if label == predicted_label:
                correct_predictions += 1

        accuracy = correct_predictions / len(train_labels)
        # print(f"Epoch {epoch + 1}/{epochs}, Train Accuracy: {accuracy * 100}%")

    overall_accuracy = correct_predictions / len(train_labels)
    print("Train Accuracy:", overall_accuracy * 100, '%')
    print("-------------------------------------------------------------")

    return weightss, biasess
def init(hiddenLayers, neuralsInHiddenLayer, trainSamples):
    weights = []
    biases = []
    for i in range(hiddenLayers):
        if i == 0:
            weights.append(np.random.uniform(0, 1, (neuralsInHiddenLayer[i], len(trainSamples[0]))))
            biases.append(np.random.uniform(0, 1, (neuralsInHiddenLayer[i], 1)))
        else:
            weights.append(np.random.uniform(0, 1, (neuralsInHiddenLayer[i], neuralsInHiddenLayer[i - 1])))
            biases.append(np.random.uniform(0, 1, (neuralsInHiddenLayer[i], 1)))
    # initializing the weights and biases for the output layer
    if hiddenLayers == 0:
        weights.append(np.random.uniform(0, 1, (3, 5)))
        biases.append(np.random.uniform(0, 1, (3, 1)))
    else:
        weights.append(np.random.uniform(0, 1, (3, neuralsInHiddenLayer[-1])))
        biases.append(np.random.uniform(0, 1, (3, 1)))

    return weights, biases

def forward_propagation(weights, biases, trainSample, activationFunction, hiddenLayers):
    layerOutput = []
    layerOutput.append(trainSample.reshape(len(trainSample), 1))
    # layerOutput = [5*1]
    for k in range(hiddenLayers + 1):
        if k == 0:
            layerOutput.append(activationFunction(
                np.dot(weights[k], trainSample.reshape(len(trainSample), 1)) + biases[k]))
        else:
            layerOutput.append(activationFunction(np.dot(weights[k], layerOutput[k]) + biases[k]))
    return layerOutput
def back_propagation(weights, trainLabel, activationFunctionDerivative, hiddenLayers, layerOutput):
    errors = []
    expectedOutput = np.zeros((3, 1))
    for k in range(3):
        if k == trainLabel:
            expectedOutput[k] = 1
        else:
            expectedOutput[k] = 0
    errors.append((expectedOutput - layerOutput[-1]) * activationFunctionDerivative(layerOutput[-1]))
    # calculating the error for the hidden layers using the formula: error = (weights of the next layer * error of the next layer) * derivative of activation function
    for k in range(hiddenLayers):
        errors.append(np.dot(weights[-k - 1].T, errors[k]) * activationFunctionDerivative(layerOutput[-k - 2]))
    return errors


def update_weights(weights, biases, learningRate, errors, layerOutput, hiddenLayers, useBias):
    for k in range(hiddenLayers + 1):
        weights[k] += learningRate * np.dot(errors[hiddenLayers - k], layerOutput[k].T)  # 2 - 0 - 2
        if useBias:
            biases[k] += learningRate * errors[hiddenLayers - k]
    return weights, biases




def test(hidden_layers, neurals_in_hidden_layer, activation_function, weights, biases, test_samples, test_labels):
    activation_function, _ = get_activation_function(activation_function)

    confusion_matrix = np.zeros((len(np.unique(test_labels)), len(np.unique(test_labels))))
    accuracy = 0.0

    for sample, label in zip(test_samples, test_labels):
        layer_output = forward_propagation(weights, biases, sample, activation_function, hidden_layers)

        # Assuming labels are one-hot encoded
        predicted_label = np.argmax(layer_output[-1])

        confusion_matrix[label, predicted_label] += 1

        if label == predicted_label:
            accuracy += 1

    accuracy /= len(test_labels)

    print("Confusion Matrix:")
    print(confusion_matrix)
    print("-------------------------------------------------------------")
    print("\nTest Accuracy:", accuracy *100,'%')

    return confusion_matrix, accuracy

train(num_hidden_layers,hidden_layer_neurons,activation_function,learning_rate,epochs,use_bias,X_train,y_train)
test(num_hidden_layers,hidden_layer_neurons,activation_function,w,b,X_test,y_test)