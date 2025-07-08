# MNIST Digit Classification with a Custom Deep Neural Network in C

This project implements a simple Deep Neural Network (DNN) from scratch in C to classify handwritten digits from the MNIST dataset.

---

## Overview

- **Language:** C (no external ML libraries)  
- **Dataset:** MNIST (28x28 grayscale images, 10 classes: digits 0-9)  
- **Model:** Fully connected neural network with:
  - 2 hidden layers
  - 16 neurons per hidden layer  
- **Training:**
  - 50 epochs
  - Learning rate: 0.000085  
- **Activation function:** ReLU for hidden layers, softmax for output

---

## Files

- `mnist_train.csv` — training data (CSV with labels and pixel values)  
- `mnist_test.csv` — test data  
- `deeplearning.c` — source code implementing the network, training, and testing

---

## How It Works

1. **Data Loading:** Reads MNIST data from CSV files.  
2. **Network Architecture:**
   - Input layer: 784 nodes (flattened 28x28 image)  
   - Hidden layers: 2 layers with 16 neurons each  
   - Output layer: 10 neurons (digit classes)  
3. **Forward Pass:** Matrix multiplications + ReLU activations for hidden layers, softmax for output.  
4. **Backpropagation:** Updates weights and biases using gradient descent.  
5. **Training:** Runs multiple epochs over training data, printing loss per epoch.  
6. **Testing:** Evaluates accuracy on test dataset.

---

## Usage

1. Compile the program:

```bash
gcc -o mnist_dnn deeplearning.c -lm
