import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from sklearn.metrics import confusion_matrix, accuracy_score

import main


# Function to get user input and run the neural network
def get_user_input_and_run():
    try:
        # Get user input from the entry widgets
        num_hidden_layers = int(hidden_layers_entry.get())
        hidden_layer_neurons = int(neurons_entry.get())
        learning_rate = float(learning_rate_entry.get())
        epochs = int(epochs_entry.get())
        use_bias = bias_var.get() == 1
        activation_function = activation_var.get().lower()

        # Call your existing code to run the neural network
        conf_matrix,accuracy=main.run_neural_network(epochs,num_hidden_layers,hidden_layer_neurons,activation_function,learning_rate)
        # Show results in a messagebox
        result_message = f"Confusion Matrix:\n{conf_matrix}\nOverall Accuracy: {accuracy}"
        messagebox.showinfo("Results", result_message)

    except ValueError as ve:
        messagebox.showerror("Error", f"Invalid input: {ve}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Tkinter GUI setup
root = tk.Tk()
root.title("Neural Network GUI")

# Add Tkinter widgets for user input (Entry, Label, Button, etc.)
hidden_layers_label = ttk.Label(root, text="Number of Hidden Layers:")
hidden_layers_label.grid(row=0, column=0, padx=10, pady=5)
hidden_layers_entry = ttk.Entry(root)
hidden_layers_entry.grid(row=0, column=1, padx=10, pady=5)

neurons_label = ttk.Label(root, text="Number of Neurons in Each Hidden Layer:")
neurons_label.grid(row=1, column=0, padx=10, pady=5)
neurons_entry = ttk.Entry(root)
neurons_entry.grid(row=1, column=1, padx=10, pady=5)

learning_rate_label = ttk.Label(root, text="Learning Rate (eta):")
learning_rate_label.grid(row=2, column=0, padx=10, pady=5)
learning_rate_entry = ttk.Entry(root)
learning_rate_entry.grid(row=2, column=1, padx=10, pady=5)

epochs_label = ttk.Label(root, text="Number of Epochs (m):")
epochs_label.grid(row=3, column=0, padx=10, pady=5)
epochs_entry = ttk.Entry(root)
epochs_entry.grid(row=3, column=1, padx=10, pady=5)

bias_var = tk.IntVar()
bias_checkbox = ttk.Checkbutton(root, text="Add Bias", variable=bias_var)
bias_checkbox.grid(row=4, column=0, columnspan=2, padx=10, pady=5)

activation_var = tk.StringVar()
activation_var.set("Sigmoid")  # Default activation function
activation_label = ttk.Label(root, text="Choose Activation Function:")
activation_label.grid(row=5, column=0, padx=10, pady=5)
activation_combobox = ttk.Combobox(root, textvariable=activation_var, values=["Sigmoid", "Tanh"])
activation_combobox.grid(row=5, column=1, padx=10, pady=5)

run_button = ttk.Button(root, text="Run Neural Network", command=get_user_input_and_run)
run_button.grid(row=6, column=0, columnspan=2, pady=10)

# Run the Tkinter event loop
root.mainloop()
