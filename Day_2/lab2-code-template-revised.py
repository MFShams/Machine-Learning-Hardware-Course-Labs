# Lab 2: Fully Connected Neural Networks (FCNN)
# Machine Learning Hardware Course
# Student Name: [YOUR NAME HERE]

# =========================================================
# PART 1: ENVIRONMENT SETUP
# =========================================================

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import psutil
import os

# Check TensorFlow version and GPU availability
print("TensorFlow version:", tf.__version__)
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# If psutil is not installed, uncomment this line:
# !pip install psutil

# Mount Google Drive (uncomment when running in Colab)
# from google.colab import drive
# drive.mount('/content/drive')
# !mkdir -p "/content/drive/My Drive/ML_Hardware_Course/Lab2"

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# =========================================================
# PART 2: DATASET PREPARATION
# =========================================================

def load_and_prepare_mnist():
    """
    Load and prepare the MNIST dataset for training FCNNs.
    
    Returns:
        tuple: Training, validation, and test data along with test labels
    """
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Print original data shapes
    print("Original MNIST shapes:")
    print(f"  X_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test: {x_test.shape}, y_test: {y_test.shape}")
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape data (flatten images)
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    
    # One-hot encode labels
    y_train_encoded = to_categorical(y_train, 10)
    y_test_encoded = to_categorical(y_test, 10)
    
    # Create validation set (10% of training data)
    val_size = 6000
    x_val = x_train_flat[-val_size:]
    y_val = y_train_encoded[-val_size:]
    x_train_final = x_train_flat[:-val_size]
    y_train_final = y_train_encoded[:-val_size]
    
    print("\nProcessed MNIST dataset:")
    print(f"  Training set: {x_train_final.shape}")
    print(f"  Validation set: {x_val.shape}")
    print(f"  Test set: {x_test_flat.shape}")
    
    return (x_train_final, y_train_final), (x_val, y_val), (x_test_flat, y_test_encoded), y_test

def load_and_prepare_fashion_mnist():
    """
    Load and prepare the Fashion MNIST dataset for training FCNNs.
    
    Returns:
        tuple: Training, validation, and test data along with test labels
    """
    # Load Fashion MNIST dataset
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # Print original data shapes
    print("\nOriginal Fashion MNIST shapes:")
    print(f"  X_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test: {x_test.shape}, y_test: {y_test.shape}")
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape data (flatten images)
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    
    # One-hot encode labels
    y_train_encoded = to_categorical(y_train, 10)
    y_test_encoded = to_categorical(y_test, 10)
    
    # Create validation set (10% of training data)
    val_size = 6000
    x_val = x_train_flat[-val_size:]
    y_val = y_train_encoded[-val_size:]
    x_train_final = x_train_flat[:-val_size]
    y_train_final = y_train_encoded[:-val_size]
    
    print("\nProcessed Fashion MNIST dataset:")
    print(f"  Training set: {x_train_final.shape}")
    print(f"  Validation set: {x_val.shape}")
    print(f"  Test set: {x_test_flat.shape}")
    
    return (x_train_final, y_train_final), (x_val, y_val), (x_test_flat, y_test_encoded), y_test

# Define class names for Fashion MNIST
fashion_class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Define class names for MNIST
mnist_class_names = [str(i) for i in range(10)]

# Load both datasets
print("Loading MNIST dataset...")
mnist_train, mnist_val, mnist_test, mnist_y_test = load_and_prepare_mnist()

print("\nLoading Fashion MNIST dataset...")
fashion_train, fashion_val, fashion_test, fashion_y_test = load_and_prepare_fashion_mnist()

# Display sample images from both datasets
def display_samples(dataset_name, x_data, y_data, class_names, num_samples=5):
    """
    Display sample images from a dataset.
    
    Args:
        dataset_name: Name of the dataset
        x_data: Image data (flattened)
        y_data: One-hot encoded labels
        class_names: List of class names
        num_samples: Number of samples to display
    """
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        idx = np.random.randint(0, len(x_data))
        plt.subplot(1, num_samples, i+1)
        plt.imshow(x_data[idx].reshape(28, 28), cmap='gray')
        class_idx = np.argmax(y_data[idx])
        plt.title(f"{class_names[class_idx]}")
        plt.axis('off')
    plt.suptitle(f"{dataset_name} Samples")
    plt.tight_layout()
    plt.show()

# Display sample images
print("\nDisplaying MNIST samples:")
display_samples("MNIST", mnist_train[0], mnist_train[1], mnist_class_names)

print("\nDisplaying Fashion MNIST samples:")
display_samples("Fashion MNIST", fashion_train[0], fashion_train[1], fashion_class_names)

# =========================================================
# PART 3: FULLY CONNECTED NEURAL NETWORK IMPLEMENTATION
# =========================================================

def create_fcnn(input_dim, hidden_layers, hidden_units, activation='relu', dropout_rate=0.0):
    """
    Create a Fully Connected Neural Network with specified architecture.
    
    Args:
        input_dim: Input dimension (e.g., 784 for MNIST)
        hidden_layers: Number of hidden layers
        hidden_units: List or int specifying neurons in each hidden layer
        activation: Activation function to use
        dropout_rate: Dropout rate (0 = no dropout)
    
    Returns:
        model: Compiled Keras model
    """
    model = Sequential(name=f"FCNN_L{hidden_layers}_U{hidden_units if isinstance(hidden_units, int) else '-'.join(map(str, hidden_units))}")
    
    # Convert hidden_units to list if it's an integer
    if isinstance(hidden_units, int):
        hidden_units = [hidden_units] * hidden_layers
    
    # Ensure we have enough hidden_units specified
    if len(hidden_units) < hidden_layers:
        hidden_units = hidden_units + [hidden_units[-1]] * (hidden_layers - len(hidden_units))
    
    # Add input layer
    model.add(Dense(hidden_units[0], activation=activation, input_shape=(input_dim,),
                   name=f'dense_1_{hidden_units[0]}'))
    
    # Add dropout if specified
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate, name=f'dropout_1_{dropout_rate}'))
    
    # Add additional hidden layers
    for i in range(1, hidden_layers):
        model.add(Dense(hidden_units[i], activation=activation, 
                       name=f'dense_{i+1}_{hidden_units[i]}'))
        
        # Add dropout if specified
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate, name=f'dropout_{i+1}_{dropout_rate}'))
    
    # Add output layer
    model.add(Dense(10, activation='softmax', name='output'))
    
    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_and_evaluate_model(model, train_data, val_data, test_data, model_name, 
                            batch_size=128, epochs=20, patience=3, verbose=1):
    """
    Train and evaluate a model, and calculate performance metrics.
    
    Args:
        model: Compiled Keras model
        train_data: Tuple of (x_train, y_train)
        val_data: Tuple of (x_val, y_val)
        test_data: Tuple of (x_test, y_test)
        model_name: Name for logging
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        patience: Early stopping patience
        verbose: Verbosity level for training
    
    Returns:
        results: Dictionary of results and metrics
    """
    x_train, y_train = train_data
    x_val, y_val = val_data
    x_test, y_test = test_data
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=patience,
        restore_best_weights=True,
        verbose=(1 if verbose > 0 else 0)
    )
    
    print(f"\nTraining {model_name}...")
    
    # Record start time
    start_time = time.time()
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping],
        verbose=verbose
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    # Measure inference time (average over 1000 samples)
    inference_samples = min(1000, len(x_test))
    start_time = time.time()
    _ = model.predict(x_test[:inference_samples], verbose=0)
    inference_time = (time.time() - start_time) / inference_samples  # per sample
    
    # Count parameters
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    # Calculate train-validation gap (for overfitting analysis)
    train_acc = max(history.history['accuracy'])
    val_acc = max(history.history['val_accuracy'])
    train_val_gap = train_acc - val_acc
    
    # Calculate efficiency metrics
    params_per_second = total_params / training_time
    accuracy_per_million_params = test_accuracy * 100 / (total_params / 1e6)
    
    # Store results
    results = {
        'model_name': model_name,
        'history': history,
        'training_time': training_time,
        'test_accuracy': test_accuracy * 100,  # convert to percentage
        'test_loss': test_loss,
        'inference_time': inference_time * 1000,  # convert to milliseconds
        'total_params': total_params,
        'trainable_params': trainable_params,
        'params_per_second': params_per_second,
        'accuracy_per_million_params': accuracy_per_million_params,
        'epochs_trained': len(history.history['accuracy']),
        'train_val_gap': train_val_gap,
        'batch_size': batch_size
    }
    
    print(f"\n--- {model_name} Results ---")
    print(f"Test Accuracy: {results['test_accuracy']:.2f}%")
    print(f"Training Time: {results['training_time']:.2f} seconds")
    print(f"Inference Time: {results['inference_time']:.4f} ms")
    print(f"Total Parameters: {results['total_params']:,}")
    print(f"Epochs Trained: {results['epochs_trained']}")
    print(f"Train-Val Gap: {results['train_val_gap']:.4f}")
    
    return results

# =========================================================
# PART 4: VISUALIZATION AND ANALYSIS FUNCTIONS
# =========================================================

def plot_training_history(histories, labels=None):
    """
    Plot training history for multiple models.
    
    Args:
        histories: List of history objects or results dictionaries
        labels: List of model names
    """
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    for i, hist in enumerate(histories):
        history = hist['history'] if isinstance(hist, dict) else hist
        label = labels[i] if labels else f"Model {i+1}"
        plt.plot(history.history['accuracy'], label=f"{label} - Train")
        plt.plot(history.history['val_accuracy'], label=f"{label} - Val")
    
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    for i, hist in enumerate(histories):
        history = hist['history'] if isinstance(hist, dict) else hist
        label = labels[i] if labels else f"Model {i+1}"
        plt.plot(history.history['loss'], label=f"{label} - Train")
        plt.plot(history.history['val_loss'], label=f"{label} - Val")
    
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, x_test, y_test_true, class_names, title):
    """
    Plot confusion matrix for model predictions.
    
    Args:
        model: Trained Keras model
        x_test: Test inputs
        y_test_true: True labels (not one-hot encoded)
        class_names: List of class names
        title: Plot title
    
    Returns:
        tuple: Confusion matrix and most confused pair
    """
    # Generate predictions
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test_true, y_pred_classes)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # Find most confused pairs
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)  # Ignore correct classifications
    max_confusion = np.unravel_index(np.argmax(cm_copy), cm_copy.shape)
    print(f"Most confused pair: {class_names[max_confusion[0]]} mistaken for {class_names[max_confusion[1]]} ({cm_copy[max_confusion]} times)")
    
    return cm, max_confusion

def profile_memory_usage(model, x_input, batch_size=32):
    """
    Profile memory usage during model inference.
    
    Args:
        model: Keras model
        x_input: Input data
        batch_size: Batch size for inference
    
    Returns:
        dict: Memory usage metrics
    """
    # Record baseline memory usage
    baseline_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
    
    # Warm-up run
    _ = model.predict(x_input[:batch_size], verbose=0)
    
    # Record memory usage during inference
    peak_memory = baseline_memory
    
    for i in range(0, min(1000, len(x_input)), batch_size):
        batch = x_input[i:i+batch_size]
        _ = model.predict(batch, verbose=0)
        current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        peak_memory = max(peak_memory, current_memory)
    
    memory_results = {
        'baseline_memory_mb': baseline_memory,
        'peak_memory_mb': peak_memory,
        'memory_increase_mb': peak_memory - baseline_memory
    }
    
    print(f"Memory profiling results:")
    print(f"  Baseline Memory: {memory_results['baseline_memory_mb']:.2f} MB")
    print(f"  Peak Memory: {memory_results['peak_memory_mb']:.2f} MB")
    print(f"  Memory Increase: {memory_results['memory_increase_mb']:.2f} MB")
    
    return memory_results

def is_pareto_efficient(costs):
    """
    Find the Pareto-efficient points.
    
    Args:
        costs: An (n_points, n_costs) array
    
    Returns:
        np.array: A boolean array of Pareto-efficient points
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep any point with a lower cost in at least one dimension
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1) | np.all(costs[is_efficient] == c, axis=1)
    return is_efficient

def plot_metric_comparison(results_df, x_metric, y_metric, title, annotate=True):
    """
    Plot a comparison of two metrics across models.
    
    Args:
        results_df: DataFrame with results
        x_metric: Column name for x-axis
        y_metric: Column name for y-axis
        title: Plot title
        annotate: Whether to annotate points with model names
    """
    plt.figure(figsize=(12, 8))
    plt.scatter(results_df[x_metric], results_df[y_metric], s=100, alpha=0.7)
    
    if annotate:
        for i, row in results_df.iterrows():
            model_name = row['Model'].replace('FCNN_', '')
            plt.annotate(model_name, 
                        (row[x_metric], row[y_metric]),
                        xytext=(5, 5), textcoords='offset points')
    
    plt.title(title)
    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.grid(True)
    plt.show()

# =========================================================
# PART 5: EXPERIMENT - NETWORK DEPTH VARIATION
# =========================================================

print("\n" + "="*50)
print("EXPERIMENT 1: VARYING NETWORK DEPTH")
print("="*50)

# Experiment with different network depths
depth_results = []
depth_histories = []
depth_models = []
depth_names = []

# Create models with different depths (1, 2, 3, 4 hidden layers)
for num_layers in [1, 2, 3, 4]:
    model_name = f"FCNN_Depth_{num_layers}"
    
    model = create_fcnn(
        input_dim=784,
        hidden_layers=num_layers,
        hidden_units=128,
        activation='relu',
        dropout_rate=0.2
    )
    
    result = train_and_evaluate_model(
        model=model,
        train_data=mnist_train,
        val_data=mnist_val,
        test_data=mnist_test,
        model_name=model_name,
        verbose=1
    )
    
    depth_results.append(result)
    depth_histories.append(result['history'])
    depth_models.append(model)
    depth_names.append(model_name)

# Plot training history for different depths
print("\nTraining history comparison for different network depths:")
plot_training_history(depth_histories, depth_names)

# Create results table
depth_df = pd.DataFrame([
    {
        'Model': result['model_name'],
        'Depth': i+1,
        'Accuracy (%)': result['test_accuracy'],
        'Training Time (s)': result['training_time'],
        'Inference Time (ms)': result['inference_time'],
        'Parameters': result['total_params'],
        'Params/Second': result['params_per_second'],
        'Accuracy/Million Params': result['accuracy_per_million_params'],
        'Train-Val Gap': result['train_val_gap']
    }
    for i, result in enumerate(depth_results)
])

print("\nDepth Experiment Results:")
print(depth_df.to_string(index=False))

# =========================================================
# PART 6: EXPERIMENT - NETWORK WIDTH VARIATION
# =========================================================

print("\n" + "="*50)
print("EXPERIMENT 2: VARYING NETWORK WIDTH")
print("="*50)

# Experiment with different network widths
width_results = []
width_histories = []
width_models = []
width_names = []

# Create models with different widths (64, 128, 256, 512 neurons)
for width in [64, 128, 256, 512]:
    model_name = f"FCNN_Width_{width}"
    
    model = create_fcnn(
        input_dim=784,
        hidden_layers=2,
        hidden_units=width,
        activation='relu',
        dropout_rate=0.2
    )
    
    result = train_and_evaluate_model(
        model=model,
        train_data=mnist_train,
        val_data=mnist_val,
        test_data=mnist_test,
        model_name=model_name,
        verbose=1
    )
    
    width_results.append(result)
    width_histories.append(result['history'])
    width_models.append(model)
    width_names.append(model_name)

# Plot training history for different widths
print("\nTraining history comparison for different network widths:")
plot_training_history(width_histories, width_names)

# Create results table
width_df = pd.DataFrame([
    {
        'Model': result['model_name'],
        'Width': [64, 128, 256, 512][i],
        'Accuracy (%)': result['test_accuracy'],
        'Training Time (s)': result['training_time'],
        'Inference Time (ms)': result['inference_time'],
        'Parameters': result['total_params'],
        'Params/Second': result['params_per_second'],
        'Accuracy/Million Params': result['accuracy_per_million_params'],
        'Train-Val Gap': result['train_val_gap']
    }
    for i, result in enumerate(width_results)
])

print("\nWidth Experiment Results:")
print(width_df.to_string(index=False))

# =========================================================
# PART 7: EXPERIMENT - ACTIVATION FUNCTIONS
# =========================================================

print("\n" + "="*50)
print("EXPERIMENT 3: VARYING ACTIVATION FUNCTIONS")
print("="*50)

# Experiment with different activation functions
activation_results = []
activation_histories = []
activation_models = []
activation_names = []

# Create models with different activation functions
for activation in ['relu', 'sigmoid', 'tanh', 'elu']:
    model_name = f"FCNN_Activation_{activation}"
    
    model = create_fcnn(
        input_dim=784,
        hidden_layers=2,
        hidden_units=128,
        activation=activation,
        dropout_rate=0.2
    )
    
    result = train_and_evaluate_model(
        model=model,
        train_data=mnist_train,
        val_data=mnist_val,
        test_data=mnist_test,
        model_name=model_name,
        verbose=1
    )
    
    activation_results.append(result)
    activation_histories.append(result['history'])
    activation_models.append(model)
    activation_names.append(model_name)

# Plot training history for different activation functions
print("\nTraining history comparison for different activation functions:")
plot_training_history(activation_histories, activation_names)

# Create results table
activation_df = pd.DataFrame([
    {
        'Model': result['model_name'],
        'Activation': ['relu', 'sigmoid', 'tanh', 'elu'][i],
        'Accuracy (%)': result['test_accuracy'],
        'Training Time (s)': result['training_time'],
        'Epochs': result['epochs_trained'],
        'Inference Time (ms)': result['inference_time'],
        'Parameters': result['total_params'],
        'Params/Second': result['params_per_second'],
        'Train-Val Gap': result['train_val_gap']
    }
    for i, result in enumerate(activation_results)
])

print("\nActivation Function Experiment Results:")
print(activation_df.to_string(index=False))

# =========================================================
# PART 8: EXPERIMENT - DROPOUT REGULARIZATION
# =========================================================

print("\n" + "="*50)
print("EXPERIMENT 4: VARYING DROPOUT REGULARIZATION")
print("="*50)

# Experiment with different dropout rates
dropout_results = []
dropout_histories = []
dropout_models = []
dropout_names = []

# Create models with different dropout rates
for dropout_rate in [0.0, 0.2, 0.4, 0.6]:
    model_name = f"FCNN_Dropout_{dropout_rate}"
    
    model = create_fcnn(
        input_dim=784,
        hidden_layers=2,
        hidden_units=128,
        activation='relu',
        dropout_rate=dropout_rate
    )
    
    result = train_and_evaluate_model(
        model=model,
        train_data=mnist_train,
        val_data=mnist_val,
        test_data=mnist_test,
        model_name=model_name,
        verbose=1
    )
    
    dropout_results.append(result)
    dropout_histories.append(result['history'])
    dropout_models.append(model)
    dropout_names.append(model_name)

# Plot training history for different dropout rates
print("\nTraining history comparison for different dropout rates:")
plot_training_history(dropout_histories, dropout_names)

# Create results table
dropout_df = pd.DataFrame([
    {
        'Model': result['model_name'],
        'Dropout Rate': [0.0, 0.2, 0.4, 0.6][i],
        'Accuracy (%)': result['test_accuracy'],
        'Training Time (s)': result['training_time'],
        'Epochs': result['epochs_trained'],
        'Train-Val Gap': result['train_val_gap'],
        'Parameters': result['total_params']
    }
    for i, result in enumerate(dropout_results)
])

print("\nDropout Experiment Results:")
print(dropout_df.to_string(index=False))


# =========================================================
# PART 9: MEMORY PROFILING
# =========================================================

print("\n" + "="*50)
print("EXPERIMENT 5: MEMORY PROFILING")
print("="*50)

# Profile memory usage for models with different sizes
memory_results = []

# Select representative models from width experiment
for i, width in enumerate([64, 128, 256, 512]):
    model = width_models[i]
    model_name = width_names[i]
    
    print(f"\nProfiling memory usage for {model_name}...")
    memory_profile = profile_memory_usage(model, mnist_test[0][:1000])
    
    memory_results.append({
        'Model': model_name,
        'Width': width,
        'Parameters': width_results[i]['total_params'],
        'Baseline Memory (MB)': memory_profile['baseline_memory_mb'],
        'Peak Memory (MB)': memory_profile['peak_memory_mb'],
        'Memory Increase (MB)': memory_profile['memory_increase_mb']
    })

# Create memory results table
memory_df = pd.DataFrame(memory_results)
print("\nMemory Usage Results:")
print(memory_df.to_string(index=False))

# Plot memory vs. model size
plt.figure(figsize=(10, 6))
plt.scatter(memory_df['Parameters'], memory_df['Memory Increase (MB)'], s=100)
for i, row in memory_df.iterrows():
    plt.annotate(f"Width={row['Width']}", 
                 (row['Parameters'], row['Memory Increase (MB)']),
                 xytext=(10, 5), textcoords='offset points')

plt.title('Memory Usage vs. Model Size')
plt.xlabel('Number of Parameters')
plt.ylabel('Memory Increase (MB)')
plt.grid(True)
plt.show()

# =========================================================
# PART 10: DATASET COMPARISON
# =========================================================

print("\n" + "="*50)
print("EXPERIMENT 6: DATASET COMPARISON")
print("="*50)

# Compare performance on different datasets
dataset_results = []
dataset_histories = []
dataset_names = []

# Create a standard model architecture
model_architecture = {
    'input_dim': 784,
    'hidden_layers': 2,
    'hidden_units': 128,
    'activation': 'relu',
    'dropout_rate': 0.2
}

# Train on MNIST
print("\nTraining on MNIST dataset...")
mnist_model = create_fcnn(**model_architecture)
mnist_result = train_and_evaluate_model(
    model=mnist_model,
    train_data=mnist_train,
    val_data=mnist_val,
    test_data=mnist_test,
    model_name="FCNN_MNIST",
    verbose=1
)
dataset_results.append(mnist_result)
dataset_histories.append(mnist_result['history'])
dataset_names.append("MNIST")

# Train on Fashion MNIST
print("\nTraining on Fashion MNIST dataset...")
fashion_model = create_fcnn(**model_architecture)
fashion_result = train_and_evaluate_model(
    model=fashion_model,
    train_data=fashion_train,
    val_data=fashion_val,
    test_data=fashion_test,
    model_name="FCNN_Fashion",
    verbose=1
)
dataset_results.append(fashion_result)
dataset_histories.append(fashion_result['history'])
dataset_names.append("Fashion MNIST")

# Plot training history comparison
print("\nTraining history comparison between datasets:")
plot_training_history(dataset_histories, dataset_names)

# Create results table
dataset_df = pd.DataFrame([
    {
        'Dataset': name,
        'Accuracy (%)': result['test_accuracy'],
        'Training Time (s)': result['training_time'],
        'Epochs': result['epochs_trained'],
        'Inference Time (ms)': result['inference_time'],
        'Train-Val Gap': result['train_val_gap']
    }
    for name, result in zip(dataset_names, dataset_results)
])

print("\nDataset Comparison Results:")
print(dataset_df.to_string(index=False))

# Plot confusion matrices
print("\nMNIST Confusion Matrix:")
mnist_cm, mnist_confused = plot_confusion_matrix(
    mnist_model, mnist_test[0], mnist_y_test, 
    mnist_class_names, "MNIST Confusion Matrix"
)

print("\nFashion MNIST Confusion Matrix:")
fashion_cm, fashion_confused = plot_confusion_matrix(
    fashion_model, fashion_test[0], fashion_y_test, 
    fashion_class_names, "Fashion MNIST Confusion Matrix"
)

# =========================================================
# PART 11: COMPREHENSIVE ANALYSIS
# =========================================================

print("\n" + "="*50)
print("COMPREHENSIVE ANALYSIS OF ALL EXPERIMENTS")
print("="*50)

# Consolidate results from all experiments
all_results = depth_results + width_results + activation_results + dropout_results

# Create comprehensive results table
all_df = pd.DataFrame([
    {
        'Model': result['model_name'],
        'Accuracy (%)': result['test_accuracy'],
        'Training Time (s)': result['training_time'],
        'Inference Time (ms)': result['inference_time'],
        'Parameters': result['total_params'],
        'Params/Second': result['params_per_second'],
        'Accuracy/Million Params': result['accuracy_per_million_params'],
        'Train-Val Gap': result['train_val_gap'],
        'Epochs': result['epochs_trained']
    }
    for result in all_results
])

print("\nAll Experiment Results:")
print(all_df.to_string(index=False))

# Plot accuracy vs. parameters
print("\nModel Accuracy vs. Parameter Count:")
plot_metric_comparison(all_df, 'Parameters', 'Accuracy (%)', 
                      'Model Accuracy vs. Parameter Count')

# Plot training time vs. parameter count
print("\nTraining Time vs. Parameter Count:")
plot_metric_comparison(all_df, 'Parameters', 'Training Time (s)', 
                      'Training Time vs. Parameter Count')

# Plot inference time vs. parameter count
print("\nInference Time vs. Parameter Count:")
plot_metric_comparison(all_df, 'Parameters', 'Inference Time (ms)', 
                      'Inference Time vs. Parameter Count')

# Plot efficiency metrics
print("\nEfficiency Metrics Comparison:")
plt.figure(figsize=(15, 10))

# Accuracy per million parameters
plt.subplot(2, 2, 1)
plt.bar(all_df['Model'], all_df['Accuracy/Million Params'])
plt.title('Accuracy per Million Parameters')
plt.xticks(rotation=90)
plt.grid(axis='y')

# Parameters trained per second
plt.subplot(2, 2, 2)
plt.bar(all_df['Model'], all_df['Params/Second'])
plt.title('Parameters Trained per Second')
plt.xticks(rotation=90)
plt.grid(axis='y')

# Accuracy vs. Training Time
plt.subplot(2, 2, 3)
plt.scatter(all_df['Training Time (s)'], all_df['Accuracy (%)'], s=100, alpha=0.7)
for i, row in all_df.iterrows():
    model_name = row['Model'].replace('FCNN_', '')
    plt.annotate(model_name, 
                 (row['Training Time (s)'], row['Accuracy (%)']),
                 xytext=(5, 0), textcoords='offset points')
plt.title('Accuracy vs. Training Time')
plt.xlabel('Training Time (seconds)')
plt.ylabel('Test Accuracy (%)')
plt.grid(True)

# Accuracy vs. Inference Time
plt.subplot(2, 2, 4)
plt.scatter(all_df['Inference Time (ms)'], all_df['Accuracy (%)'], s=100, alpha=0.7)
for i, row in all_df.iterrows():
    model_name = row['Model'].replace('FCNN_', '')
    plt.annotate(model_name, 
                 (row['Inference Time (ms)'], row['Accuracy (%)']),
                 xytext=(5, 0), textcoords='offset points')
plt.title('Accuracy vs. Inference Time')
plt.xlabel('Inference Time (ms)')
plt.ylabel('Test Accuracy (%)')
plt.grid(True)

plt.tight_layout()
plt.show()

# =========================================================
# PART 12: IDENTIFY OPTIMAL ARCHITECTURES
# =========================================================

print("\n" + "="*50)
print("IDENTIFYING OPTIMAL ARCHITECTURES")
print("="*50)

# Find models with best metrics
best_accuracy_model = all_df.loc[all_df['Accuracy (%)'].idxmax()]
print("\nModel with Best Accuracy:")
print(best_accuracy_model.to_string())

# Find model with best accuracy per parameter
best_efficiency_model = all_df.loc[all_df['Accuracy/Million Params'].idxmax()]
print("\nModel with Best Accuracy/Parameter Ratio:")
print(best_efficiency_model.to_string())

# Find model with fastest inference
fastest_inference_model = all_df.loc[all_df['Inference Time (ms)'].idxmin()]
print("\nModel with Fastest Inference:")
print(fastest_inference_model.to_string())

# Find model with fastest training
fastest_training_model = all_df.loc[all_df['Training Time (s)'].idxmin()]
print("\nModel with Fastest Training:")
print(fastest_training_model.to_string())

# Calculate Pareto frontier (non-dominated models in terms of accuracy and inference time)
print("\nCalculating Pareto Frontier for Accuracy vs. Inference Time...")

# Convert to efficiency problem (higher accuracy is better, lower inference time is better)
costs = np.column_stack([
    -all_df['Accuracy (%)'].values,  # Negative because we want to maximize accuracy
    all_df['Inference Time (ms)'].values  # We want to minimize inference time
])

# Find Pareto-efficient models
pareto_efficient = is_pareto_efficient(costs)
pareto_df = all_df[pareto_efficient].copy()

print("\nPareto-Efficient Models (Accuracy vs. Inference Time):")
print(pareto_df.to_string(index=False))

# Plot Pareto frontier
plt.figure(figsize=(12, 8))
plt.scatter(all_df['Inference Time (ms)'], all_df['Accuracy (%)'], s=100, alpha=0.5, label='All Models')
plt.scatter(pareto_df['Inference Time (ms)'], pareto_df['Accuracy (%)'], s=150, color='red', label='Pareto Frontier')

# Annotate Pareto frontier points
for i, row in pareto_df.iterrows():
    model_name = row['Model'].replace('FCNN_', '')
    plt.annotate(model_name, 
                 (row['Inference Time (ms)'], row['Accuracy (%)']),
                 xytext=(10, 5), textcoords='offset points',
                 color='red', fontweight='bold')

plt.title('Pareto Frontier: Accuracy vs. Inference Time')
plt.xlabel('Inference Time (ms)')
plt.ylabel('Test Accuracy (%)')
plt.grid(True)
plt.legend()
plt.show()

# =========================================================
# PART 13: RESULTS SUMMARY FOR WORKSHEET
# =========================================================

print("\n" + "="*50)
print("RESULTS SUMMARY FOR WORKSHEET")
print("="*50)

print("\nPart 1: Network Depth Experiment")
for i, result in enumerate(depth_results):
    depth = i + 1
    print(f"\nDepth {depth}:")
    print(f"  Test Accuracy: {result['test_accuracy']:.2f}%")
    print(f"  Training Time: {result['training_time']:.2f} seconds")
    print(f"  Inference Time: {result['inference_time']:.4f} ms")
    print(f"  Total Parameters: {result['total_params']:,}")

print("\nPart 2: Network Width Experiment")
for i, width in enumerate([64, 128, 256, 512]):
    result = width_results[i]
    print(f"\nWidth {width}:")
    print(f"  Test Accuracy: {result['test_accuracy']:.2f}%")
    print(f"  Training Time: {result['training_time']:.2f} seconds")
    print(f"  Inference Time: {result['inference_time']:.4f} ms")
    print(f"  Total Parameters: {result['total_params']:,}")

print("\nPart 3: Activation Functions")
for i, activation in enumerate(['relu', 'sigmoid', 'tanh', 'elu']):
    result = activation_results[i]
    print(f"\n{activation.upper()}:")
    print(f"  Test Accuracy: {result['test_accuracy']:.2f}%")
    print(f"  Training Time: {result['training_time']:.2f} seconds")
    print(f"  Epochs to Converge: {result['epochs_trained']}")
    print(f"  Inference Time: {result['inference_time']:.4f} ms")

print("\nPart 4: Dropout Regularization")
for i, dropout_rate in enumerate([0.0, 0.2, 0.4, 0.6]):
    result = dropout_results[i]
    print(f"\nDropout Rate {dropout_rate}:")
    print(f"  Test Accuracy: {result['test_accuracy']:.2f}%")
    print(f"  Training Time: {result['training_time']:.2f} seconds")
    print(f"  Epochs to Converge: {result['epochs_trained']}")
    print(f"  Train-Val Gap: {result['train_val_gap']:.4f}")

print("\nPart 5: Memory Profiling")
for _, row in memory_df.iterrows():
    print(f"\n{row['Model']}:")
    print(f"  Parameters: {row['Parameters']:,}")
    print(f"  Baseline Memory: {row['Baseline Memory (MB)']:.2f} MB")
    print(f"  Peak Memory: {row['Peak Memory (MB)']:.2f} MB")
    print(f"  Memory Increase: {row['Memory Increase (MB)']:.2f} MB")

print("\nPart 6: Dataset Comparison")
for i, dataset in enumerate(['MNIST', 'Fashion MNIST']):
    result = dataset_results[i]
    print(f"\n{dataset}:")
    print(f"  Test Accuracy: {result['test_accuracy']:.2f}%")
    print(f"  Training Time: {result['training_time']:.2f} seconds")
    print(f"  Epochs to Converge: {result['epochs_trained']}")

print("\nPart 7: Efficiency Metrics")
print(f"\nBest Accuracy Model ({best_accuracy_model['Model']}):")
print(f"  Accuracy/Million Params: {best_accuracy_model['Accuracy/Million Params']:.2f}")
print(f"  Params/Second: {best_accuracy_model['Params/Second']:.2f}")

print(f"\nFastest Training Model ({fastest_training_model['Model']}):")
print(f"  Accuracy/Million Params: {fastest_training_model['Accuracy/Million Params']:.2f}")
print(f"  Params/Second: {fastest_training_model['Params/Second']:.2f}")

print(f"\nFastest Inference Model ({fastest_inference_model['Model']}):")
print(f"  Accuracy/Million Params: {fastest_inference_model['Accuracy/Million Params']:.2f}")
print(f"  Params/Second: {fastest_inference_model['Params/Second']:.2f}")

print(f"\nMost Parameter-Efficient Model ({best_efficiency_model['Model']}):")
print(f"  Accuracy/Million Params: {best_efficiency_model['Accuracy/Million Params']:.2f}")
print(f"  Params/Second: {best_efficiency_model['Params/Second']:.2f}")

# =========================================================
# PART 14: SAVE RESULTS (UNCOMMENT TO USE)
# =========================================================

# Save results to Google Drive
# results_path = "/content/drive/My Drive/ML_Hardware_Course/Lab2/fcnn_results.csv"
# all_df.to_csv(results_path, index=False)
# print(f"Results saved to {results_path}")

# Save the best model
# best_model_path = "/content/drive/My Drive/ML_Hardware_Course/Lab2/best_fcnn_model.h5"
# model_idx = all_df[all_df['Model'] == best_accuracy_model['Model']].index[0]
# model_group = model_idx // len(depth_results)
# model_within_group = model_idx % len(depth_results)

# if model_group == 0:
#     depth_models[model_within_group].save(best_model_path)
# elif model_group == 1:
#     width_models[model_within_group].save(best_model_path)
# elif model_group == 2:
#     activation_models[model_within_group].save(best_model_path)
# else:
#     dropout_models[model_within_group].save(best_model_path)

# print(f"Best model saved to {best_model_path}")

print("\nLab 2 completed successfully!")

# =========================================================
# END OF LAB 2
# =========================================================
