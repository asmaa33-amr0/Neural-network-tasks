import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import messagebox
from mlxtend.plotting import plot_confusion_matrix
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy
import main_task1


master = Tk()
master.title('Task 1')
master.geometry("600x500")
master.resizable(False, False)

features =main_task1.features
classes = main_task1.classes
methods =["Adaline", "Preceptron"]

#main title
mainlabel = tk.Label(master, text = "DryBeans", font = ("Times New Roman", 18 , 'bold' , 'underline') , background = 'yellow', borderwidth=2, relief="solid")
mainlabel.place(x=200 , y=10)

#selecting features

combo_label1 = ttk.Label(master, text = "Select the First Features :", font = ("Times New Roman", 13))
combo_label1.place(x=10 , y=120)
combo_feature_1 = ttk.Combobox(master , value = features , width=40 )
combo_feature_1.place(x=200 , y=120)


combo_label2 = ttk.Label(master, text = "Select the Second Features :", font = ("Times New Roman", 13))
combo_label2.place(x=10 , y=150)
combo_feature_2 = ttk.Combobox(master , value = features, width=35 )
combo_feature_2.place(x=230 , y=150)


#selecting classes

combo_label3 = ttk.Label(master, text = "Select the First Class :", font = ("Times New Roman", 13))
combo_label3.place(x=10 , y=180)
combo_class_1 = ttk.Combobox(master , value = classes , width=40 )
combo_class_1.place(x=200 , y=180)


combo_label4 = ttk.Label(master, text = "Select the Second Class :", font = ("Times New Roman", 13) , )
combo_label4.place(x=10 , y=210)
combo_class_2 = ttk.Combobox(master , value = classes, width=40 )
combo_class_2.place(x=200 , y=210)



#selecting method
selected_option = tk.StringVar(value="Adaline")
radio_button1 = tk.Radiobutton(master, text="Adaline", variable=selected_option, value="Adaline")
radio_button2 = tk.Radiobutton(master, text="Preceptron", variable=selected_option, value="Preceptron")

radio_button1.pack()
radio_button2.pack()
radio_button1.place(x=10, y=240)
radio_button2.place(x=100, y=240)

# combo_label5 = ttk.Label(master, text="Select Method to use :", font=("Times New Roman", 13) , )
# combo_label5.place(x=10, y=240)
# combo_methods = ttk.Combobox(master, value=methods, width=40)
# combo_methods.place(x=200, y=240)

#selecting learning rate

elearn_label = ttk.Label(master, text = "Write the learning rate :", font = ("Times New Roman", 13))
elearn_label.place(x=10 , y=270)
elearn =Entry(master , width=43 , borderwidth=1, relief="solid")
elearn.place(x=200 , y=270)
elearn.focus_set()
elearn.insert(END,'0')

#selecting epochs

epochs_label = ttk.Label(master, text = "Write the Number of ephocs :", font = ("Times New Roman", 13))
epochs_label.place(x=10 , y=300)
epochs = Entry(master , width=40 , borderwidth=1, relief="solid")
epochs.place(x=220 , y=300)
epochs.focus_set()
epochs.insert(END,'0')

#select bias or not
vario = BooleanVar()
bias_label = ttk.Label(master, text = "Bias ", font = ("Times New Roman", 13))
bias_label.place(x=190 , y=350)
bias = Checkbutton(master, text="Add bias",height = 0, width = 10 , borderwidth=1, relief="solid" , variable = vario)
bias.place(x=230 , y=350)
############
threshold_label = ttk.Label(master, text="Threshold:", font=("Times New Roman", 13))
threshold_label.place(x=10, y=330)
threshold_entry = Entry(master, width=40, borderwidth=1, relief="solid")
threshold_entry.place(x=220, y=330)
threshold_entry.focus_set()
threshold_entry.insert(END, '0')

def submition():
    feature1 = str(combo_feature_1.get())
    feature2 = str(combo_feature_2.get())

    class1 = str(combo_class_1.get())
    class2 = str(combo_class_2.get())

    learning_rate = float(elearn.get())
    epochss = int(epochs.get())
    threshold= float
    biasss = vario.get()
    selected_value = selected_option.get()
    print(selected_value)
    wady = True
    if feature1 == feature2 == '':
        messagebox.showerror(title='ERROR', message='Cannot leave the Features Empty')
        wady = False

    elif feature1 == feature2 :
        messagebox.showwarning(title='Warning', message='Cannot Select the Same Feature twice, Please select two different Features')
        wady = False


    if class1 == class2 == '':
        messagebox.showerror(title='ERROR', message='Cannot leave the Classes Empty')
        wady = False

    elif class1 == class2:
        messagebox.showwarning(title='Warning', message='Cannot Select the Same CLASS twice, Please select two different CLASSES')
        wady = False


    if learning_rate == 0:
        messagebox.showerror(title='ERROR', message='Cannot leave the Learning Rate Empty')
        wady = False

        # elearn.set()

    if epochss == 0:
        messagebox.showerror(title='ERROR', message='Cannot leave the epochs Empty')
        wady = False

        # Retrieve the threshold value
        threshold_str =float(threshold_entry.get())
        try:
            threshold = float(threshold_str)
        except ValueError:
            messagebox.showerror(title='Invalid Threshold', message='Please enter a valid numerical threshold value.')
            return  # Exit the function if the threshold is not a valid float
    if wady == True:
        if selected_value == "Adaline":
            x_test, y_test, weights = main_task1.adaline(class1 , class2 , feature1 , feature2 , biasss , learning_rate , epochss,threshold)
            acc , conf_matrix = main_task1.testing(x_test, y_test, weights)
        elif selected_value == "Preceptron":
            x_test, y_test, weights = main_task1.preceptron(class1, class2, feature1, feature2, biasss, learning_rate, epochss)
            acc, conf_matrix = main_task1.testing(x_test, y_test, weights)
        #confusion matrix
        cm_display = plot_confusion_matrix(conf_mat = conf_matrix , cmap=plt.cm.Greens , colorbar=True)
        plt.show()

        messagebox.showinfo(title='Accuracy', message='Accuracy = ' + str(acc))


    #print(feature1 + " " + feature2 + " " + class1 + " " + class2 + " " +str(learning_rate)+ " " + str(epochss) + " " + str(biasss))


#start botton

B = Button(master , text = "START", font = ("Times New Roman", 18 , 'bold' , 'underline') , width=30 , command=submition , borderwidth=2, relief="solid")
B.place(x=40 , y=400)

#start
master.mainloop()