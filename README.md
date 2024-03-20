# Neural Network on Iris Dataset

## Overview
In this project, I've implemented a neural network with 2 hidden layers to perform classification on the Iris dataset. The Iris dataset is a popular dataset in machine learning and contains information about various iris flowers. The goal is to classify the iris flowers into one of three species based on features such as sepal length, sepal width, petal length, and petal width.

## Implementation Details
- **Neural Network Architecture**: 
    - Input Layer: Number of features in the Iris dataset (4 features)
    - Hidden Layers: 2 hidden layers
    - Output Layer: Number of classes in the Iris dataset (3 classes)
- **Activation Function**: ReLU (Rectified Linear Unit) activation function is used for the hidden layers.
- **Optimizer**: Adam optimizer is used to optimize the neural network's weights during training.
- **Training Parameters**:
    - Number of Epochs: 100
    - Learning Rate: 0.01

## How to Run
1. Ensure you have the necessary dependencies installed. You can install them via pip:
    ```bash
    pip install numpy pandas scikit-learn torch
    ```
2. Clone this repository:
    ```bash
    git clone https://github.com/anjaneyak10/IrisDataNeuralNetwork.git
    ```
3. Navigate to the project directory:
4. Run the Python script to train and evaluate the neural network:
    ```bash
    python main.py
    ```

