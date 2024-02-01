import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Define features and classes
features = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "roundnes"]
classes = ["BOMBAY", "CALI", "SIRA"]

# Initialization
num_features = len(features)
num_classes = len(classes)


def initialize_weights_and_biases(hidden_layer_neurons, num_hidden_layers ):
    weights = []
    biases = []


    for i in range(num_hidden_layers):
        if i == 0:
            weights.append(np.random.uniform(0, 1, (hidden_layer_neurons[i], len(features))))
            biases.append(np.random.uniform(0, 1, (hidden_layer_neurons[i], 1)))
        else:
            weights.append(np.random.uniform(0, 1, (hidden_layer_neurons[i], hidden_layer_neurons[i - 1])))
            biases.append(np.random.uniform(0, 1, (hidden_layer_neurons[i], 1)))


    if num_hidden_layers == 0:

            weights.append(np.random.uniform(0, 1, (3, 5)))
            biases.append(np.random.uniform(0, 1, (3, 1)))
    else:
      weights.append(np.random.uniform(0, 1, (3, hidden_layer_neurons[-1])))
      biases.append(np.random.uniform(0, 1, (3, 1)))

    return weights, biases


def sigmoid_activation_fun(x):
    return 1 / (1 + np.exp(-x))


def tanh_activation_fun(x):
    return np.tanh(x)
def sigmoid_derivative(x):
    return sigmoid_activation_fun(x) * (1 - sigmoid_activation_fun(x))



def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def function_type(activation_function):
    activation_function = activation_function.lower()

    if activation_function == "sigmoid":
        return sigmoid_activation_fun, sigmoid_derivative
    elif activation_function == "hyperbolic tangent":
        return tanh_activation_fun, tanh_derivative

def activation_derivative(x, activation_func):
    if activation_func == 'sigmoid':
        return x * (1 - x)
    elif activation_func == 'tanh':
        return 1 - x ** 2


def one_hot_encode(y):
    encoded = np.zeros((len(y), num_classes))
    encoded[np.arange(len(y)), y] = 1
    return encoded


def feedforward(train_X, weights, biases, activationFunction, hiddenLayers):
    last_layer_out = []
    last_layer_out.append(train_X.reshape(len(train_X), 1))

    for h in range(hiddenLayers + 1):
        if h == 0:
            last_layer_out.append(activationFunction(
                np.dot(weights[h], train_X.reshape(len(train_X), 1)) + biases[h]))
        else:
            last_layer_out.append(activationFunction(np.dot(weights[h], last_layer_out[h]) + biases[h]))
    return last_layer_out


def backward_step(weights, Y_Train, activationFunctionDerivative, hiddenLayers, layerOutput):
    errors = []
    expectedOutput = np.zeros((3, 1))
    for k in range(3):
        if k == Y_Train:
            expectedOutput[k] = 1
        else:
            expectedOutput[k] = 0
    errors.append((expectedOutput - layerOutput[-1]) * activationFunctionDerivative(layerOutput[-1]))

    for k in range(hiddenLayers):
        errors.append(np.dot(weights[-k - 1].T, errors[k]) * activationFunctionDerivative(layerOutput[-k - 2]))
    return errors

def update_weights(weights, biases, learning_rate, errors, layer_output, hidden_layers, use_bias):
    for k in range(hidden_layers + 1):
        weights[k] += learning_rate * np.dot(errors[hidden_layers - k], layer_output[k].T)
        if use_bias:
            biases[k] += learning_rate * errors[hidden_layers - k]
    return weights, biases



def run_neural_network(epochs, num_hidden_layers, hidden_layer_neurons, activation_function, learning_rate, use_bias ):
    # Load the dataset
    activation_function, activation_function_derivative = function_type(activation_function)
    df = pd.read_excel("Dry_Bean_Dataset.xlsx")
    features = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "roundnes"]
    classes = ["BOMBAY", "CALI", "SIRA"]


    lab = LabelEncoder()
    min_max_scaler = MinMaxScaler()
    #
    #
    Dry_Bean_All = pd.DataFrame(df)
    # lables = (Dry_Bean_All['Class'])
    #www=(Dry_Bean_All['Class']).astype(str)
    Dry_Bean_All['Class'] = lab.fit_transform(Dry_Bean_All['Class'])
    mean_value = Dry_Bean_All["MinorAxisLength"].mean()
    Dry_Bean_All["MinorAxisLength"].fillna(mean_value, inplace=True)
    Dry_Bean_All[
        ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength"]] = min_max_scaler.fit_transform(
        Dry_Bean_All[["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength"]])
    """ 
    for i in range(len(lables)):
        if lables[i]=="BOMBAY":
            lables[i]=0
        elif lables[i]=="CALI":
            lables[i]=1
        elif lables[i]=="SIRA":
            lables[i]=2
    #Dry_Bean_All['Class']=lables"""
    Drybean_BOMBAY = Dry_Bean_All.iloc[:50]
    Drybean_CALI = Dry_Bean_All.iloc[50:100]
    Drybean_SIRA = Dry_Bean_All.iloc[100:150]

    # Split data into classes
    # Combine data for training
    training_data = pd.concat([Drybean_BOMBAY.iloc[:30], Drybean_CALI.iloc[:30], Drybean_SIRA.iloc[:30]])
    # Test data
    test_data = pd.concat([Drybean_BOMBAY.iloc[30:], Drybean_CALI.iloc[30:], Drybean_SIRA.iloc[30:]])
    #shaffel
    training_data = training_data.sample(frac=1)
    test_data = test_data.sample(frac=1)
    # Extract features and labels
    X_train = training_data[features].values
    y_train = training_data['Class'].values

    X_test = test_data[features] .values
    y_test = test_data['Class']

    # Initialize weights and biases
    weights, biases = initialize_weights_and_biases(hidden_layer_neurons, num_hidden_layers)
    if not use_bias:
        biases = [np.zeros(bias.shape) for bias in biases]

    for epoch in range(epochs):
        correct_predictions = 0
        for sample, targets in zip(X_train, y_train):
            layer_outputs = feedforward(sample, weights, biases, activation_function, num_hidden_layers)

            errors = backward_step(weights, targets, activation_function_derivative, num_hidden_layers,
                                   layer_outputs)

            # Update weights and biases
            weights, biases = update_weights(weights, biases, learning_rate, errors, layer_outputs,
                                              num_hidden_layers, use_bias)
            predicted_label = np.argmax(layer_outputs[-1])
            if targets == predicted_label:
                correct_predictions += 1
        overall_accuracy = correct_predictions / len(y_train)




        #y_pred = np.argmax([feedforward(x, weights, biases, activation_function, num_hidden_layers)[-1] for x in X_test], axis=1)


        #conf_matrix = confusion_matrix(y_test, y_pred)
        #accuracy = accuracy_score(y_test, y_pred)
    confusion_matrix = np.zeros((len(classes), len(classes)))
    accuracy = 0.0
    cc=0

    for sample, label in zip(X_test, y_test):
        layer_output = feedforward(sample, weights, biases, activation_function, num_hidden_layers)


        predicted_label = np.argmax(layer_output[-1] )
        if label == predicted_label:
             accuracy += 1
        confusion_matrix[label, predicted_label] += 1







    accuracy /= len(y_test)
    # Print confusion matrix and overall accuracy
    print("Train Accuracy:", overall_accuracy * 100, '%')
    print("Confusion Matrix:")
    print(confusion_matrix )
    print("Overall Accuracy (Testing ):", accuracy * 100, '%')
    return confusion_matrix, accuracy

