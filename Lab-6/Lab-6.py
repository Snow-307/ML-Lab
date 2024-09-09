# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 23:36:06 2024

@author: sneha_xqbh6g1
"""

import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation function for MLP
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function for backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)

# Perceptron summation unit
def summation_unit(inputs, weights):
    return np.dot(inputs, weights)

# Activation functions for perceptron
def step_activation(x):
    return 1 if x >= 0 else 0

def bipolar_step_activation(x):
    return 1 if x >= 0 else -1

def relu_activation(x):
    return max(0, x)

def find_error(target, output):
    return target - output

# Perceptron Algorithm
def perceptron(inputs, targets, weights, activation_func, learning_rate, epochs=1000, convergence_threshold=0.002):
    errors = []
    bias = weights[0]
    for epoch in range(epochs):
        total_error = 0
        for i, input_vec in enumerate(inputs):
            summation = summation_unit(input_vec, weights[1:]) + bias
            output = activation_func(summation)
            error = find_error(targets[i], output)
            weights[1:] += learning_rate * error * input_vec
            bias += learning_rate * error
            total_error += error**2
        errors.append(total_error)
        if total_error <= convergence_threshold:
            break
    weights[0] = bias
    return weights, errors, epoch

# Multi-Layer Perceptron (MLP) Training for XOR Gate
def mlp_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # XOR inputs
    y = np.array([[0], [1], [1], [0]])  # XOR outputs

    # Initialize weights and biases for MLP
    np.random.seed(1)
    weights_input_hidden = np.random.uniform(size=(2, 2))
    weights_hidden_output = np.random.uniform(size=(2, 1))
    bias_hidden = np.random.uniform(size=(1, 2))
    bias_output = np.random.uniform(size=(1, 1))

    learning_rate = 0.1
    epochs = 10000

    for epoch in range(epochs):
        # Forward Propagation
        hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
        hidden_output = sigmoid(hidden_input)
        output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
        predicted_output = sigmoid(output_input)

        # Calculate error
        error = y - predicted_output

        # Backpropagation
        d_predicted_output = error * sigmoid_derivative(predicted_output)
        error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
        d_hidden_output = error_hidden_layer * sigmoid_derivative(hidden_output)

        # Update weights and biases
        weights_hidden_output += hidden_output.T.dot(d_predicted_output) * learning_rate
        weights_input_hidden += X.T.dot(d_hidden_output) * learning_rate
        bias_output += np.sum(d_predicted_output, axis=0) * learning_rate
        bias_hidden += np.sum(d_hidden_output, axis=0) * learning_rate

    print("Final output from MLP for XOR gate:")
    print(predicted_output)

# Plotting Function
def plot_graph(errors, title):
    plt.plot(range(len(errors)), errors)
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title(title)
    plt.show()

# AND Gate Logic for Perceptron
def and_gate_experiment(activation_func, learning_rate):
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([0, 0, 0, 1])
    weights = np.array([10, 0.2, -0.75])
    
    final_weights, errors, epoch_count = perceptron(inputs, targets, weights, activation_func, learning_rate)
    
    return final_weights, errors, epoch_count

# XOR Gate Logic for Perceptron
def xor_gate_experiment(activation_func, learning_rate):
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([0, 1, 1, 0])
    weights = np.array([10, 0.2, -0.75])
    
    final_weights, errors, epoch_count = perceptron(inputs, targets, weights, activation_func, learning_rate)
    
    return final_weights, errors, epoch_count

# Learning Rate Trials for Perceptron
def learning_rate_trials(inputs, targets, activation_func, learning_rates):
    convergence_epochs = []
    weights = np.array([10, 0.2, -0.75])
    
    for lr in learning_rates:
        _, _, epoch_count = perceptron(inputs, targets, weights, activation_func, lr)
        convergence_epochs.append(epoch_count)

    plt.plot(learning_rates, convergence_epochs)
    plt.xlabel('Learning Rate')
    plt.ylabel('Epochs to Converge')
    plt.title('Learning Rate vs Epochs to Converge')
    plt.show()

# Customer Data Classification using Perceptron
def customer_data_classification(activation_func, learning_rate):
    inputs = np.array([
        [20, 6, 2],
        [16, 3, 6],
        [27, 6, 2],
        [19, 1, 2],
        [24, 4, 2],
        [22, 1, 5],
        [15, 4, 2],
        [18, 4, 2],
        [21, 1, 4],
        [16, 2, 4]
    ])
    targets = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])

    initial_weights = np.random.randn(inputs.shape[1] + 1)
    final_weights, errors, epoch_count = perceptron(inputs, targets, initial_weights, activation_func, learning_rate)
    
    return final_weights, errors, epoch_count

# Main Function
def main():
    # AND Gate Perceptron Experiment
    perceptron_AND_weights, and_errors, and_epoch_count = and_gate_experiment(step_activation, learning_rate=0.05)
    plot_graph(and_errors, f'Epochs vs Error for AND Gate Step Activation')
    print(f"Observation for AND Gate (Step Activation)\nEpochs = {and_epoch_count}.")
    
    perceptron_bipolar_weights, bipolar_errors, bipolar_epoch_count = and_gate_experiment(bipolar_step_activation, learning_rate=0.05)
    plot_graph(bipolar_errors, f'Epochs vs Error for AND Gate Bipolar Step Activation')
    print(f"Bipolar Step Activation\nEpochs = {bipolar_epoch_count}.")
    
    perceptron_sigmoid_weights, sigmoid_errors, sigmoid_epoch_count = and_gate_experiment(sigmoid, learning_rate=0.05)
    plot_graph(sigmoid_errors, f'Epochs vs Error for AND Gate Sigmoid Activation')
    print(f"Sigmoid Activation\nEpochs = {sigmoid_epoch_count}.")
    
    perceptron_relu_weights, relu_errors, relu_epoch_count = and_gate_experiment(relu_activation, learning_rate=0.05)
    plot_graph(relu_errors, f'Epochs vs Error for AND Gate ReLU Activation')
    print(f"ReLU Activation\nEpochs = {relu_epoch_count}.")
    
    # Learning Rates Trials for AND Gate
    learning_rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([0, 0, 0, 1])
    learning_rate_trials(inputs, targets, step_activation, learning_rates)
    
    # XOR Gate Experiment
    perceptron_XOR_weights, xor_errors, xor_epoch_count = xor_gate_experiment(sigmoid, learning_rate=0.05)
    plot_graph(xor_errors, f'Epochs vs Error for XOR Gate (Activation: sigmoid_activation)')
    print(f"\nObservation for XOR Gate (Sigmoid Activation)\nEpochs = {xor_epoch_count}.")
    
    # Customer Data Classification
    customer_weights, customer_errors, customer_epoch_count = customer_data_classification(sigmoid, learning_rate=0.1)
    plot_graph(customer_errors, f'Epochs vs Error for Customer Data (Activation: sigmoid_activation)')
    print(f"Customer Data Classification - Weights converged in {customer_epoch_count} epochs using sigmoid_activation activation.")

    # XOR Gate Perceptron Experiment
    perceptron_XOR_weights, xor_errors, xor_epoch_count = xor_gate_experiment(sigmoid, learning_rate=0.05)
    plot_graph(xor_errors, f'Epochs vs Error for XOR Gate (Activation: Sigmoid)')
    print(f"\nObservation for XOR Gate (Sigmoid Activation)\nEpochs = {xor_epoch_count}.")

    # Learning Rates Trials for AND Gate
    learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([0, 0, 0, 1])
    learning_rate_trials(inputs, targets, step_activation, learning_rates)
    
    # Multi-Layer Perceptron XOR Experiment
    mlp_xor()
    
    # Customer Data Classification using Perceptron
    customer_weights, customer_errors, customer_epoch_count = customer_data_classification(sigmoid, learning_rate=0.1)
    plot_graph(customer_errors, f'Epochs vs Error for Customer Data (Activation: Sigmoid)')
    print(f"Customer Data Classification - Weights converged in {customer_epoch_count} epochs using sigmoid activation.")

if __name__ == '__main__':
    main()
