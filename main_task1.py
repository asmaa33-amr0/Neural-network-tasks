import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
import tkinter
from tkinter import ttk
from tkinter import *
from tkinter import messagebox
from sklearn.model_selection import train_test_split

features = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "roundnes"]
classes = ["BOMBAY", "CALI", "SIRA"]

df = pd.read_excel("Dry_Bean_Dataset.xlsx")
lab = LabelEncoder()
min_max_scaler = MinMaxScaler()
#
#
Dry_Bean_All = pd.DataFrame(df)
Dry_Bean_All['Class'] = lab.fit_transform(Dry_Bean_All['Class'])
#
mean_value=Dry_Bean_All["MinorAxisLength"].mean()
Dry_Bean_All["MinorAxisLength"].fillna(mean_value, inplace=True)
Dry_Bean_All[
     ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength"]] = min_max_scaler.fit_transform(
     Dry_Bean_All[["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength"]])
#
Drybean_BOMBAY = Dry_Bean_All.iloc[:50]
Drybean_CALI = Dry_Bean_All.iloc[50:100]
Drybean_SIRA = Dry_Bean_All.iloc[100:150]
#
#
# #
#
# #  classification algorithm
#
def testing(x_test, y_test, weights):
    true = 0
    false = 0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    size = len(x_test)
    for index, row in x_test.iterrows():
        row = row.to_numpy()
        row = row.reshape(1, 3)
        yi = np.dot(row, weights.transpose())
        if yi > 0:
            yi = 1
        elif yi < 0:
            yi = -1
        else:
            yi = 0

        if yi == y_test[index]:
            if (yi == 1):
                true_pos += 1
            else:
                true_neg += 1
        else:
            if (yi == 1):
                false_pos += 1
            else:
                false_neg += 1
    accuracy = ((true_pos + true_neg) / size) * 100

    conf_matrix = np.array([[true_pos, false_pos],[false_neg, true_neg]])
    # conf_matrix=conf_matrix.to_frame()
    return accuracy, conf_matrix


def adaline(class1, class2, f1, f2, bias, lr, epochs,threshold):
    if bias == False:
        bias = 0
    else:
        bias = 1

    if class1 == "BOMBAY":
        class1 = Drybean_BOMBAY
    elif class1 == "CALI":
        class1 = Drybean_CALI
    elif class1 == "SIRA":
        class1 = Drybean_SIRA

    if class2 == "BOMBAY":
        class2 = Drybean_BOMBAY
    elif class2 == "CALI":
        class2 = Drybean_CALI
    elif class2 == "SIRA":
        class2 = Drybean_SIRA

    class1["Class"] = -1
    class2["Class"] = 1
    x = pd.concat([class1[:30], class2[:30]])
    y = pd.concat([class1[30:], class2[30:]])
    x["bias"] = bias
    y["bias"] = bias
    x = x.sample(frac=1).reset_index(drop=True)
    y = y.sample(frac=1).reset_index(drop=True)
    x_train = x[[f1, f2, "bias"]]
    y_train = x["Class"]
#
    # bias???

    x_test = y[[f1, f2, "bias"]]
    y_test = y["Class"]
    iterations = len(x_train)

    # Algorithm----

    weights = pd.DataFrame(np.random.randn(1, 3))

    for i in range(epochs):
        loss_sum=0
        for index, row in x_train.iterrows():
            row = row.to_numpy()
            row = row.reshape(1, 3)
            yi = np.dot(row, weights.transpose())
            if yi != y_train[index]:
                loss = y_train[index] - yi
                loss_sum += loss**2
                weights = weights + (lr * loss * row)
        mse = loss_sum/(2*60)
        # the two lines below is the threshold but its  declared in gui
        # buthave some errors
        # if mse <threshold:
        #    break
    plott(class1, class2, f1, f2, weights, bias)
    return x_test, y_test, weights
#############################################################################################
def preceptron(class1, class2, f1, f2, bias, lr, epochs):
    if bias == False:
        bias = 0
    else:
        bias = 1

    if class1 == "BOMBAY":
        class1 = Drybean_BOMBAY
    elif class1 == "CALI":
        class1 = Drybean_CALI
    elif class1 == "SIRA":
        class1 = Drybean_SIRA

    if class2 == "BOMBAY":
        class2 = Drybean_BOMBAY
    elif class2 == "CALI":
        class2 = Drybean_CALI
    elif class2 == "SIRA":
        class2 = Drybean_SIRA

    class1["Class"] = -1
    class2["Class"] = 1
    x = pd.concat([class1[:30], class2[:30]])
    y = pd.concat([class1[30:], class2[30:]])
    x["bias"] = bias
    y["bias"] = bias
    x = x.sample(frac=1).reset_index(drop=True)
    y = y.sample(frac=1).reset_index(drop=True)
    x_train = x[[f1, f2, "bias"]]
    y_train = x["Class"]
#
    # bias???

    x_test = y[[f1, f2, "bias"]]
    y_test = y["Class"]
    iterations = len(x_train)

    # Algorithm----

    weights = pd.DataFrame(np.random.randn(1, 3))

    for i in range(epochs):
        loss_sum=0
        for index, row in x_train.iterrows():
            row = row.to_numpy()
            row = row.reshape(1, 3)
            yi = np.dot(row, weights.transpose())
            if yi>0:
                yi=1
            elif yi<0:
                yi=-1
            else:yi=0
            if yi != y_train[index]:
                loss = y_train[index] - yi
                loss_sum += loss**2
                weights = weights + (lr * loss * row)
    plott(class1, class2, f1, f2, weights, bias)
    return x_test, y_test, weights
##################################################################################################
def visualization(f1, f2):
    plt.scatter(x=f1[:50], y=f2[:50])
    plt.scatter(x=f1[50:100], y=f2[50:100])
    plt.scatter(x=f1[100:], y=f2[100:])

    plt.xlabel(f1.name)
    plt.ylabel(f2.name)

    plt.show()



def plott(class1, class2, f1, f2, weights, bias2):
   # %matplotlib inline
    xx = (-weights[2]) / weights[0]
    yy = (-weights[2]) / weights[1]
    p1 = [xx, 0]
    p2 = [0, yy]
    x, y = [p1[0], p2[0]], [p1[1], p2[1]]
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    plt.scatter(x=class1[f1], y=class1[f2])
    plt.scatter(x=class2[f1], y=class2[f2])
    plt.axline(x, y)
    plt.show()
#visualization
feature_size = len(features)
for i in range(feature_size):
    for j in range(i + 1, feature_size):
        visualization(Dry_Bean_All[features[i]], Dry_Bean_All[features[j]])
#
# scalling
Dry_Bean_All[
    ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength"]] = min_max_scaler.fit_transform(
    Dry_Bean_All[["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength"]])



Drybean_BOMBAY = Dry_Bean_All.iloc[:50]
Drybean_CALI = Dry_Bean_All.iloc[50:100]
Drybean_SIRA = Dry_Bean_All.iloc[100:150]



# print(Dry_Bean_All)

