# Iris Flower Classification using a Neural Network with PyTorch

This project demonstrates a simple neural network built with PyTorch to classify the species of Iris flowers based on their sepal and petal measurements. The popular Iris dataset is used for training and evaluation.

## Project Overview

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

## Key Libraries Used

* **PyTorch:** For building and training the neural network.
* **Pandas:** For data manipulation and loading the dataset.
* **Scikit-learn:** For splitting the dataset into training and testing sets.
* **Matplotlib:** For plotting the training loss.

## Files

* `nn_iris.ipynb`: The Jupyter Notebook containing all the Python code for the project.
* `iris_nn_model.pt`: The saved state dictionary of the trained neural network model (this file will be generated after running the notebook).

## How to Use

1.  **Prerequisites:** Ensure you have Python installed along with the necessary libraries. You can install them using pip:
    ```bash
    pip install torch pandas scikit-learn matplotlib jupyter
    ```
2.  **Run the Notebook:**
    * Open the `nn_iris.ipynb` file in a Jupyter Notebook environment (e.g., Jupyter Lab, Google Colab).
    * Run the cells sequentially from top to bottom.

## Expected Output

* The notebook will print the training loss at intervals.
* A plot visualizing the training loss over epochs will be displayed.
* The final test loss and accuracy will be printed.
* Predictions for individual test samples and a new data sample will be shown.
* The trained model parameters will be saved to `iris_nn_model.pt`.

This project serves as a basic example of implementing a neural network for a classification task using PyTorch.
