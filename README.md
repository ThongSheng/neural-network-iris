# Neural Network Projects with PyTorch

This repository contains Python projects demonstrating the implementation of neural networks using PyTorch for different classification tasks.

---

## Project 1: Iris Flower Classification using a Feedforward Neural Network

This project demonstrates a simple feedforward neural network built with PyTorch to classify the species of Iris flowers based on their sepal and petal measurements. The popular Iris dataset is used for training and evaluation.

### Project Overview

The Jupyter Notebook (`nn_iris.ipynb`) covers the following key steps:
1.  **Data Loading and Preprocessing:** Loads the Iris dataset from a URL, maps categorical species names to numerical labels, and splits the data into training and testing sets.
2.  **Model Definition:** Defines a feedforward neural network with 4 input features (sepal length, sepal width, petal length, petal width), two hidden layers (with 8 and 9 neurons respectively), and an output layer predicting 3 classes of Iris flowers (Setosa, Versicolor, Virginica). ReLU activation is used in the hidden layers.
3.  **Model Training:**
    * Uses Cross-Entropy Loss as the criterion and the Adam optimizer.
    * Trains the model for 200 epochs.
    * Visualizes the training loss over epochs.
4.  **Model Evaluation:**
    * Evaluates the trained model on the test dataset.
    * Calculates and prints the test loss.
    * Displays detailed predictions against true labels for the test set and calculates the overall accuracy.
5.  **Prediction on New Data:** Shows an example of how to use the trained model to predict the species of a new Iris flower.
6.  **Saving and Loading Model:**
    * Saves the trained model's parameters (`state_dict`) to a file (`iris_nn_model.pt`).
    * Demonstrates how to load the saved model and use it for predictions, ensuring the model is set to evaluation mode (`model.eval()`).

### Key Libraries Used (Iris Project)

* **PyTorch:** For building and training the neural network.
* **Pandas:** For data manipulation and loading the dataset.
* **Scikit-learn:** For splitting the dataset into training and testing sets.
* **Matplotlib:** For plotting the training loss.

### Expected Output (Iris Project)

* The notebook will print the training loss at intervals.
* A plot visualizing the training loss over epochs will be displayed.
* The final test loss and accuracy will be printed.
* Predictions for individual test samples and a new data sample will be shown.
* The trained model parameters will be saved to `iris_nn_model.pt`.

---

## Project 2: MNIST Handwritten Digit Classification with a Convolutional Neural Network (CNN)

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset.

### Project Overview

The Python script (`convolutional_neural_network_on_mnist.py`) performs the following steps:

1.  **Data Loading and Preparation:**
    * Loads the MNIST dataset for training and testing using `torchvision.datasets.MNIST`.
    * Transforms images into PyTorch tensors.
    * Creates `DataLoader` instances for batch processing of training and testing data.
2.  **CNN Model Definition (Conceptual):**
    * A "Sample Model" section demonstrates the tensor transformations (shape changes) through a couple of convolutional and max-pooling layers to help understand the mechanics. This section is for illustrative purposes.
3.  **CNN Model Implementation (`ConvolutionalNetwork` class):**
    * Defines a `ConvolutionalNetwork` class that inherits from `nn.Module`.
    * The architecture consists of:
        * Two convolutional layers (`nn.Conv2d`) with ReLU activation.
        * Max-pooling layers (`F.max_pool2d`) after each convolutional layer.
        * Three fully connected layers (`nn.Linear`) with ReLU activation for the first two, leading to a 10-class output (for digits 0-9).
        * The forward pass uses `F.log_softmax` on the final output.
4.  **Model Training:**
    * Initializes the `ConvolutionalNetwork` model.
    * Sets the random seed for reproducibility using `torch.manual_seed(123)`.
    * Uses `nn.CrossEntropyLoss` as the loss function and `torch.optim.Adam` as the optimizer.
    * Trains the model for a specified number of epochs (10 in the script).
    * Tracks and prints training loss at intervals.
    * Calculates training accuracy.
5.  **Model Validation:**
    * After each training epoch, the model is evaluated on the test data.
    * Calculates validation loss and accuracy.
    * Records training and validation losses and accuracies for plotting.
    * Measures and prints the total training time.
6.  **Results Visualization:**
    * Plots the training and validation loss curves over epochs.
    * Plots the training and validation accuracy curves over epochs.
7.  **Final Evaluation:**
    * Evaluates the model's accuracy on the entire test dataset.
8.  **Prediction on a New Image:**
    * Selects and visualizes a sample image from the test dataset.
    * Sets the model to evaluation mode (`model.eval()`) and predicts its class.

### Key Libraries Used (MNIST Project)

* **PyTorch (`torch`, `torch.nn`, `torch.nn.functional`):** For building and training the CNN.
* **`torch.utils.data.DataLoader`:** For creating data loaders.
* **`torchvision.datasets`, `torchvision.transforms`:** For loading the MNIST dataset and applying transformations.
* **Matplotlib:** For plotting graphs and displaying images.

### Expected Output (MNIST Project)

* The script will download the MNIST dataset if not already present (default to `./cnn_data`).
* During training, it will print epoch number, batch number, and loss periodically.
* After training, it will print the total training time.
* Two plots will be displayed:
    * Training Loss vs. Validation Loss per epoch.
    * Training Accuracy vs. Validation Accuracy per epoch.
* The final accuracy on the test set will be printed.
* An example image from the test set will be displayed, followed by the model's prediction for that image.

---

These projects serve as basic examples of implementing different types of neural networks for classification tasks using PyTorch.
