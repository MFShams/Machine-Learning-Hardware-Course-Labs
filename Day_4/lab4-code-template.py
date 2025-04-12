# Lab 4: Machine Learning Hardware Optimization
# Machine Learning Hardware Course
# Student Name: [YOUR NAME HERE]

# =========================================================
# PART 1: ENVIRONMENT SETUP
# =========================================================

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
import pandas as pd
import seaborn as sns
from pathlib import Path

# Check TensorFlow version and available hardware
print("TensorFlow version:", tf.__version__)
print("Eager execution enabled:", tf.executing_eagerly())
print("GPU available:", bool(tf.config.list_physical_devices('GPU')))
#print("TPU available:", bool(tf.distribute.cluster_resolver.TPUClusterResolver()))

# Try to detect and configure TPU if available
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
except:
    # No TPU detected, use default strategy
    strategy = tf.distribute.get_strategy()

print("Distribution strategy:", strategy.__class__.__name__)
print("Number of replicas:", strategy.num_replicas_in_sync)

# Install TensorFlow Model Optimization Toolkit if not already installed
try:
    import tensorflow_model_optimization as tfmot
    print("TensorFlow Model Optimization Toolkit version:", tfmot.__version__)
except ImportError:
    !pip install -q tensorflow-model-optimization
    import tensorflow_model_optimization as tfmot
    print("Installed TensorFlow Model Optimization Toolkit version:", tfmot.__version__)

# Mount Google Drive (optional, for saving results)
# from google.colab import drive
# drive.mount('/content/drive')
# !mkdir -p "/content/drive/My Drive/ML_Hardware_Course/Lab4"

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# =========================================================
# PART 2: MODEL AND DATASET PREPARATION
# =========================================================

def create_cnn_model(input_shape, num_classes):
    """Create a simple CNN model for classification."""
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_fcnn_model(input_shape, num_classes):
    """Create a simple fully connected neural network model for classification."""
    input_dim = np.prod(input_shape)  # Flatten the input shape
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_and_prepare_mnist():
    """Load and prepare MNIST dataset."""
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape for CNN (add channel dimension)
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # One-hot encode labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

# Load and prepare dataset
print("\nLoading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = load_and_prepare_mnist()

# Create a validation set
val_size = 5000
x_val, y_val = x_test[:val_size], y_test[:val_size]
x_test, y_test = x_test[val_size:], y_test[val_size:]
mnist_data = ((x_train, y_train), (x_val, y_val))

# Define input shape and number of classes
input_shape = (28, 28, 1)
num_classes = 10

# Create models
print("\nCreating models...")
cnn_model = create_cnn_model(input_shape, num_classes)
fcnn_model = create_fcnn_model(input_shape, num_classes)

# Print model summaries
print("\nCNN Model Summary:")
cnn_model.summary()

print("\nFCNN Model Summary:")
fcnn_model.summary()

# =========================================================
# PART 3: HARDWARE PERFORMANCE BENCHMARKING
# =========================================================

def benchmark_training(model, train_data, val_data, device, epochs=3, batch_size=128):
    """Benchmark model training on specified device."""
    x_train, y_train = train_data
    x_val, y_val = val_data

    # Use a small subset for benchmarking to save time
    x_train_subset = x_train[:10000]
    y_train_subset = y_train[:10000]

    with tf.device(device):
        # Warm-up
        model.fit(
            x_train_subset[:100], y_train_subset[:100],
            batch_size=batch_size,
            epochs=1,
            verbose=0
        )

        # Actual timing
        start_time = time.time()
        history = model.fit(
            x_train_subset, y_train_subset,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            verbose=1
        )
        training_time = time.time() - start_time

    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)

    return {
        'device': device,
        'training_time': training_time,
        'epochs': epochs,
        'samples': len(x_train_subset),
        'batch_size': batch_size,
        'samples_per_second': (len(x_train_subset) * epochs) / training_time,
        'val_accuracy': val_accuracy * 100,  # convert to percentage
        'val_loss': val_loss
    }

def benchmark_inference(model, data, device, num_runs=1000, batch_size=1):
    """Benchmark model inference on specified device."""
    x_data = data[0]

    # Use a subset for benchmarking to save time
    if batch_size == 1:
        x_subset = x_data[:num_runs]
    else:
        # Ensure we have enough samples for the requested batch size and runs
        required_samples = batch_size * num_runs
        if required_samples <= len(x_data):
            x_subset = x_data[:required_samples]
        else:
            # Repeat the data if necessary
            repeats = (required_samples // len(x_data)) + 1
            x_subset = np.tile(x_data, (repeats, 1, 1, 1))[:required_samples]

    with tf.device(device):
        # Warm-up
        if batch_size == 1:
            for _ in range(10):
                _ = model.predict(x_subset[0:1], verbose=0)
        else:
            for _ in range(3):
                _ = model.predict(x_subset[:batch_size], verbose=0)

        # Actual timing for single sample inference
        if batch_size == 1:
            start_time = time.time()
            for i in range(num_runs):
                _ = model.predict(x_subset[i:i+1], verbose=0)
            inference_time = (time.time() - start_time) / num_runs

        # Actual timing for batch inference
        else:
            start_time = time.time()
            for i in range(0, num_runs * batch_size, batch_size):
                end_idx = min(i + batch_size, len(x_subset))
                _ = model.predict(x_subset[i:end_idx], verbose=0)
            inference_time = (time.time() - start_time) / num_runs

    return {
        'device': device,
        'batch_size': batch_size,
        'inference_time_ms': inference_time * 1000,  # convert to ms
        'samples_per_second': batch_size / inference_time
    }

# Define device strings
cpu_device = "/device:CPU:0"
gpu_device = "/device:GPU:0" if tf.config.list_physical_devices('GPU') else None

# Benchmark training on different devices
training_results = []

# CPU benchmarking
print("\nBenchmarking CNN training on CPU...")
cnn_cpu_training = benchmark_training(cnn_model, mnist_data[0], mnist_data[1], cpu_device)
training_results.append({**cnn_cpu_training, 'model': 'CNN'})

print("\nBenchmarking FCNN training on CPU...")
fcnn_cpu_training = benchmark_training(fcnn_model, mnist_data[0], mnist_data[1], cpu_device)
training_results.append({**fcnn_cpu_training, 'model': 'FCNN'})

# GPU benchmarking (if available)
if gpu_device:
    print("\nBenchmarking CNN training on GPU...")
    cnn_gpu_training = benchmark_training(cnn_model, mnist_data[0], mnist_data[1], gpu_device)
    training_results.append({**cnn_gpu_training, 'model': 'CNN'})

    print("\nBenchmarking FCNN training on GPU...")
    fcnn_gpu_training = benchmark_training(fcnn_model, mnist_data[0], mnist_data[1], gpu_device)
    training_results.append({**fcnn_gpu_training, 'model': 'FCNN'})

# Create and display training results table
training_df = pd.DataFrame(training_results)
print("\nTraining Benchmark Results:")
print(training_df)

# Plot training benchmarks
plt.figure(figsize=(12, 5))

# Plot training time comparison
plt.subplot(1, 2, 1)
sns.barplot(data=training_df, x='model', y='training_time', hue='device')
plt.title('Training Time Comparison')
plt.ylabel('Training Time (seconds)')
plt.xlabel('Model')

# Plot samples per second comparison
plt.subplot(1, 2, 2)
sns.barplot(data=training_df, x='model', y='samples_per_second', hue='device')
plt.title('Training Throughput Comparison')
plt.ylabel('Samples per Second')
plt.xlabel('Model')

plt.tight_layout()
plt.show()

# Benchmark inference on different devices
inference_results = []

# Single sample inference (batch_size=1)
print("\nBenchmarking single sample inference...")
for device in [cpu_device] + ([gpu_device] if gpu_device else []):
    print(f"Benchmarking CNN inference on {device} (batch_size=1)...")
    cnn_inference = benchmark_inference(cnn_model, (x_test, None), device, num_runs=100, batch_size=1)
    inference_results.append({**cnn_inference, 'model': 'CNN'})

    print(f"Benchmarking FCNN inference on {device} (batch_size=1)...")
    fcnn_inference = benchmark_inference(fcnn_model, (x_test, None), device, num_runs=100, batch_size=1)
    inference_results.append({**fcnn_inference, 'model': 'FCNN'})

# Batch inference (batch_size=32)
print("\nBenchmarking batch inference...")
for device in [cpu_device] + ([gpu_device] if gpu_device else []):
    print(f"Benchmarking CNN inference on {device} (batch_size=32)...")
    cnn_batch_inference = benchmark_inference(cnn_model, (x_test, None), device, num_runs=10, batch_size=32)
    inference_results.append({**cnn_batch_inference, 'model': 'CNN'})

    print(f"Benchmarking FCNN inference on {device} (batch_size=32)...")
    fcnn_batch_inference = benchmark_inference(fcnn_model, (x_test, None), device, num_runs=10, batch_size=32)
    inference_results.append({**fcnn_batch_inference, 'model': 'FCNN'})

# Create and display inference results table
inference_df = pd.DataFrame(inference_results)
print("\nInference Benchmark Results:")
print(inference_df)

# Plot inference benchmarks
plt.figure(figsize=(12, 5))

# Plot inference time comparison (log scale)
plt.subplot(1, 2, 1)
sns.barplot(data=inference_df, x='model', y='inference_time_ms', hue='device')
plt.title('Inference Time Comparison')
plt.ylabel('Inference Time (ms, log scale)')
plt.yscale('log')
plt.xlabel('Model')

# Plot samples per second comparison (log scale)
plt.subplot(1, 2, 2)
sns.barplot(data=inference_df, x='model', y='samples_per_second', hue='device')
plt.title('Inference Throughput Comparison')
plt.ylabel('Samples per Second (log scale)')
plt.yscale('log')
plt.xlabel('Model')

plt.tight_layout()
plt.show()

# Compare batch size impact on inference
plt.figure(figsize=(12, 5))
batch_comparison = inference_df.copy()
batch_comparison['device_batch'] = batch_comparison['device'] + ' (batch=' + batch_comparison['batch_size'].astype(str) + ')'

# Plot inference time comparison by batch size
plt.subplot(1, 2, 1)
sns.barplot(data=batch_comparison, x='model', y='inference_time_ms', hue='device_batch')
plt.title('Inference Time by Batch Size')
plt.ylabel('Inference Time (ms, log scale)')
plt.yscale('log')
plt.xlabel('Model')
plt.legend(title='Device (Batch Size)')

# Plot samples per second comparison by batch size
plt.subplot(1, 2, 2)
sns.barplot(data=batch_comparison, x='model', y='samples_per_second', hue='device_batch')
plt.title('Throughput by Batch Size')
plt.ylabel('Samples per Second (log scale)')
plt.yscale('log')
plt.xlabel('Model')
plt.legend(title='Device (Batch Size)')

plt.tight_layout()
plt.show()

# Save the trained CNN model for later use
cnn_model.save('original_cnn_model.h5')
print("\nOriginal CNN model saved as 'original_cnn_model.h5'")

# =========================================================
# PART 4: MODEL QUANTIZATION
# =========================================================

# Load the saved CNN model
original_model = load_model('original_cnn_model.h5')

# Function to evaluate model performance
def evaluate_model(model, x_test, y_test, model_name="Model"):
    """Evaluate model accuracy and size."""
    # Measure model size
    model_size = 0
    if hasattr(model, 'weights'):
        for weight in model.weights:
            model_size += weight.numpy().nbytes
    else:
        model.save('temp_model.h5')
        model_size = os.path.getsize('temp_model.h5')
        os.remove('temp_model.h5')

    # Evaluate model accuracy
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f"{model_name} Evaluation:")
    print(f"  Accuracy: {accuracy * 100:.2f}%")
    print(f"  Model Size: {model_size / 1024 / 1024:.2f} MB")

    return {
        'model_name': model_name,
        'accuracy': accuracy * 100,
        'model_size_mb': model_size / 1024 / 1024
    }

# Evaluate original model
original_results = evaluate_model(original_model, x_test, y_test, "Original Model")

# Apply post-training quantization with TensorFlow Lite
# First, create a representative dataset for quantization
def representative_dataset():
    for i in range(100):
        yield [x_train[i:i+1]]

# Convert to TFLite model
converter = tf.lite.TFLiteConverter.from_keras_model(original_model)
tflite_model = converter.convert()

# Save the TFLite model
with open('original_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Check the size of the TFLite model
tflite_model_size = os.path.getsize('original_model.tflite')
print(f"TFLite model size: {tflite_model_size / 1024 / 1024:.2f} MB")

# Quantize the model to float16
converter = tf.lite.TFLiteConverter.from_keras_model(original_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16_model = converter.convert()

# Save the quantized model
with open('quantized_fp16_model.tflite', 'wb') as f:
    f.write(tflite_fp16_model)

# Check the size of the quantized TFLite model
tflite_fp16_model_size = os.path.getsize('quantized_fp16_model.tflite')
print(f"Float16 quantized TFLite model size: {tflite_fp16_model_size / 1024 / 1024:.2f} MB")
print(f"Size reduction: {(1 - tflite_fp16_model_size / tflite_model_size) * 100:.2f}%")

# Quantize the model to int8 (full integer quantization)
converter = tf.lite.TFLiteConverter.from_keras_model(original_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_int8_model = converter.convert()

# Save the int8 quantized model
with open('quantized_int8_model.tflite', 'wb') as f:
    f.write(tflite_int8_model)

# Check the size of the int8 quantized TFLite model
tflite_int8_model_size = os.path.getsize('quantized_int8_model.tflite')
print(f"Int8 quantized TFLite model size: {tflite_int8_model_size / 1024 / 1024:.2f} MB")
print(f"Size reduction: {(1 - tflite_int8_model_size / tflite_model_size) * 100:.2f}%")

# Function to evaluate TFLite model accuracy
def evaluate_tflite_model(tflite_model_path, x_test, y_test):
    """Evaluate a TFLite model on the test dataset."""
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check if the model is quantized (int8)
    input_scale, input_zero_point = input_details[0].get('quantization', (0, 0))
    is_quantized = input_scale != 0

    # Test model on random input data
    correct_predictions = 0
    num_samples = min(1000, len(x_test))  # Limit to 1000 samples for speed

    for i in range(num_samples):
        # Get test sample
        test_image = x_test[i:i+1]
        true_label = np.argmax(y_test[i])

        # Quantize input if necessary
        if is_quantized:
            test_image = test_image / input_scale + input_zero_point
            test_image = test_image.astype(np.int8)

        # Resize tensor shape
        input_shape = input_details[0]['shape']
        test_image = np.reshape(test_image, input_shape)

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], test_image)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        output = interpreter.get_tensor(output_details[0]['index'])

        # Get predicted label
        if output.shape[-1] == 10:  # If output is class probabilities
            predicted_label = np.argmax(output)
        else:  # If output is already the class index
            predicted_label = output[0]

        if predicted_label == true_label:
            correct_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / num_samples

    # Get model size
    model_size = os.path.getsize(tflite_model_path)

    print(f"TFLite Model: {tflite_model_path}")
    print(f"  Accuracy: {accuracy * 100:.2f}%")
    print(f"  Model Size: {model_size / 1024 / 1024:.2f} MB")

    return {
        'model_name': os.path.basename(tflite_model_path),
        'accuracy': accuracy * 100,
        'model_size_mb': model_size / 1024 / 1024
    }

# Evaluate TFLite models
print("\nEvaluating TFLite models...")
tflite_original_results = evaluate_tflite_model('original_model.tflite', x_test, y_test)
tflite_fp16_results = evaluate_tflite_model('quantized_fp16_model.tflite', x_test, y_test)
tflite_int8_results = evaluate_tflite_model('quantized_int8_model.tflite', x_test, y_test)

# Collect all quantization results
quantization_results = [
    original_results,
    tflite_original_results,
    tflite_fp16_results,
    tflite_int8_results
]

# Create a dataframe and plot results
quantization_df = pd.DataFrame(quantization_results)
print("\nQuantization Results:")
print(quantization_df)

# Plot quantization results
plt.figure(figsize=(12, 5))

# Plot accuracy comparison
plt.subplot(1, 2, 1)
sns.barplot(data=quantization_df, x='model_name', y='accuracy')
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.xlabel('Model')
plt.xticks(rotation=45)

# Plot model size comparison
plt.subplot(1, 2, 2)
sns.barplot(data=quantization_df, x='model_name', y='model_size_mb')
plt.title('Model Size Comparison')
plt.ylabel('Model Size (MB)')
plt.xlabel('Model')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Plot trade-off between accuracy and model size
plt.figure(figsize=(10, 6))
sns.scatterplot(data=quantization_df, x='model_size_mb', y='accuracy', s=100)

# Annotate points
for i, row in quantization_df.iterrows():
    plt.annotate(row['model_name'],
                 (row['model_size_mb'], row['accuracy']),
                 xytext=(5, 5), textcoords='offset points')

plt.title('Accuracy vs. Model Size Trade-off')
plt.xlabel('Model Size (MB)')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.show()

# =========================================================
# PART 5: MODEL PRUNING
# =========================================================

# Import tensorflow_model_optimization again to ensure it's available
import tensorflow_model_optimization as tfmot

# Define a function for manual pruning to replace the problematic API
def create_prunable_cnn_model(input_shape, num_classes):
    """Create a functional CNN model for pruning compatibility."""
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Define a function to manually prune weights based on magnitude
def prune_weights(model, sparsity=0.5):
    """
    Prune model weights based on magnitude.
    
    Args:
        model: Keras model to prune
        sparsity: Target sparsity level (fraction of weights to be pruned)
    
    Returns:
        Pruned model
    """
    print(f"\nApplying magnitude-based pruning with {sparsity:.0%} sparsity...")
    
    pruned_model = tf.keras.models.clone_model(model)
    pruned_model.set_weights(model.get_weights())
    
    # Prune eligible layers (Conv and Dense)
    for i, layer in enumerate(pruned_model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            weights = layer.get_weights()
            
            # Only prune the weight matrix, not the bias
            weight_matrix = weights[0]
            
            # Flatten the weight matrix to identify the threshold
            flat_weights = weight_matrix.flatten()
            abs_weights = np.abs(flat_weights)
            
            # Calculate the threshold value based on the sparsity level
            k = int(flat_weights.size * sparsity)
            if k > 0:
                threshold = np.partition(abs_weights, k)[k]
                
                # Create a mask for weights with magnitude below threshold
                mask = np.abs(weight_matrix) > threshold
                
                # Apply the mask to the weights
                pruned_weights = weight_matrix * mask
                
                # Update the weights
                weights[0] = pruned_weights
                layer.set_weights(weights)
                
                non_zero = np.count_nonzero(pruned_weights)
                total = pruned_weights.size
                print(f"  Layer {i} ({layer.name}): {non_zero}/{total} weights retained ({non_zero/total:.2%})")
    
    # Recompile the model with the same settings
    pruned_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return pruned_model

# Create a new model for pruning
cnn_model_for_pruning = create_prunable_cnn_model(input_shape, num_classes)

# First, train the model before pruning
print("\nTraining model before pruning...")
cnn_model_for_pruning.fit(
    x_train[:10000], y_train[:10000],  # Using a subset for speed
    batch_size=128,
    epochs=1,
    validation_data=(x_val, y_val),
    verbose=1
)

# Apply manual pruning
pruned_model = prune_weights(cnn_model_for_pruning, sparsity=0.5)

# Fine-tune the pruned model
print("\nFine-tuning the pruned model...")
pruned_model.fit(
    x_train[:10000], y_train[:10000],  # Using a subset for speed
    batch_size=128,
    epochs=2,
    validation_data=(x_val, y_val),
    verbose=1
)

# Evaluate the pruned model
pruned_model_results = evaluate_model(pruned_model, x_test, y_test, "Pruned Model")

# Save the pruned model
pruned_model.save('pruned_model.keras')
stripped_model = pruned_model  # Since we're using manual pruning, no stripping is needed
stripped_model_results = evaluate_model(stripped_model, x_test, y_test, "Stripped Pruned Model")

# Convert the stripped model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(stripped_model)
tflite_stripped_model = converter.convert()

# Save the TFLite model
with open('stripped_pruned_model.tflite', 'wb') as f:
    f.write(tflite_stripped_model)

# Evaluate the TFLite stripped model
tflite_stripped_results = evaluate_tflite_model('stripped_pruned_model.tflite', x_test, y_test)

# Apply both pruning and quantization

# First, convert to TFLite with float16 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(stripped_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_pruned_fp16_model = converter.convert()

# Save the quantized pruned model
with open('pruned_quantized_fp16_model.tflite', 'wb') as f:
    f.write(tflite_pruned_fp16_model)

# Then convert to TFLite with int8 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(stripped_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_pruned_int8_model = converter.convert()

# Save the int8 quantized pruned model
with open('pruned_quantized_int8_model.tflite', 'wb') as f:
    f.write(tflite_pruned_int8_model)

# Evaluate the quantized pruned models
tflite_pruned_fp16_results = evaluate_tflite_model('pruned_quantized_fp16_model.tflite', x_test, y_test)
tflite_pruned_int8_results = evaluate_tflite_model('pruned_quantized_int8_model.tflite', x_test, y_test)

# Collect all pruning results
pruning_results = [
    original_results,
    pruned_model_results,
    stripped_model_results,
    tflite_stripped_results,
    tflite_pruned_fp16_results,
    tflite_pruned_int8_results
]

# Create a dataframe and plot results
pruning_df = pd.DataFrame(pruning_results)
print("\nPruning and Quantization Results:")
print(pruning_df)

# Plot pruning results
plt.figure(figsize=(12, 5))

# Plot accuracy comparison
plt.subplot(1, 2, 1)
sns.barplot(data=pruning_df, x='model_name', y='accuracy')
plt.title('Accuracy Comparison')
plt.xticks(rotation=45)
plt.ylabel('Accuracy (%)')

# Plot model size comparison
plt.subplot(1, 2, 2)
sns.barplot(data=pruning_df, x='model_name', y='model_size_mb')
plt.title('Model Size Comparison')
plt.xticks(rotation=45)
plt.ylabel('Model Size (MB)')

plt.tight_layout()
plt.show()

# =========================================================
# PART 6: DEPLOYMENT FORMAT CONVERSION (FIXED)
# =========================================================

# For TensorFlow 2.18.0, we need to use a different approach for ONNX conversion
print("\n== Model Format Conversion ==")

# Try converting to ONNX format or skip if not compatible
try:
    import tf2onnx
    
    # For TensorFlow 2.18.0, we need to convert the model to a saved model first
    # and then convert the saved model to ONNX
    print("Converting original model to SavedModel format first...")
    original_saved_model_dir = "original_saved_model"
    tf.saved_model.save(original_model, original_saved_model_dir)
    
    print("Converting SavedModel to ONNX...")
    # Use the command-line converter approach
    import subprocess
    
    # Convert the original model
    result = subprocess.run(
        ["python", "-m", "tf2onnx.convert", 
         "--saved-model", original_saved_model_dir, 
         "--output", "original_model.onnx"],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        print("ONNX conversion successful")
        onnx_model_exists = True
    else:
        print("ONNX conversion failed with error:")
        print(result.stderr)
        onnx_model_exists = False
    
    # If the first conversion was successful, try the pruned model
    if onnx_model_exists:
        # Convert the pruned model
        pruned_saved_model_dir = "pruned_saved_model"
        tf.saved_model.save(stripped_model, pruned_saved_model_dir)
        
        result = subprocess.run(
            ["python", "-m", "tf2onnx.convert", 
             "--saved-model", pruned_saved_model_dir, 
             "--output", "pruned_model.onnx"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            print("Pruned model ONNX conversion successful")
            pruned_onnx_exists = True
        else:
            print("Pruned model ONNX conversion failed with error:")
            print(result.stderr)
            pruned_onnx_exists = False
    else:
        pruned_onnx_exists = False
        
    # Calculate ONNX model sizes if conversion was successful
    if onnx_model_exists and os.path.exists('original_model.onnx'):
        onnx_model_size = os.path.getsize('original_model.onnx')
        print(f"ONNX model size: {onnx_model_size / 1024 / 1024:.2f} MB")
    else:
        onnx_model_size = 0
        
    if pruned_onnx_exists and os.path.exists('pruned_model.onnx'):
        onnx_pruned_model_size = os.path.getsize('pruned_model.onnx')
        print(f"Pruned ONNX model size: {onnx_pruned_model_size / 1024 / 1024:.2f} MB")
        if onnx_model_size > 0:
            print(f"Size reduction: {(1 - onnx_pruned_model_size / onnx_model_size) * 100:.2f}%")
    else:
        onnx_pruned_model_size = 0
        
except (ImportError, Exception) as e:
    print(f"ONNX conversion skipped due to error: {str(e)}")
    onnx_model_size = 0
    onnx_pruned_model_size = 0
    
# Convert to TensorFlow SavedModel format (optimized for TensorFlow Serving)
print("\nConverting to SavedModel format...")
tf.saved_model.save(original_model, "saved_model")
saved_model_size = sum(f.stat().st_size for f in Path("saved_model").glob('**/*') if f.is_file())
print(f"SavedModel size: {saved_model_size / 1024 / 1024:.2f} MB")

# Try converting to TensorFlow.js format
try:
    print("\nConverting to TensorFlow.js format...")
    # Use subprocess to call the converter to avoid dependency issues
    result = subprocess.run(
        ["tensorflowjs_converter", "--input_format=keras", 
         "original_cnn_model.h5", "tfjs_model"],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        print("TensorFlow.js conversion successful")
        tfjs_model_exists = True
        # Calculate the size of the TensorFlow.js model
        tfjs_model_size = sum(f.stat().st_size for f in Path("tfjs_model").glob('**/*') if f.is_file())
        print(f"TensorFlow.js model size: {tfjs_model_size / 1024 / 1024:.2f} MB")
    else:
        print("TensorFlow.js conversion failed with error:")
        print(result.stderr)
        tfjs_model_exists = False
        tfjs_model_size = 0
except Exception as e:
    print(f"TensorFlow.js conversion skipped due to error: {str(e)}")
    tfjs_model_exists = False
    tfjs_model_size = 0

# Summarize all model formats
format_results = [
    {
        'model_name': 'Original Model (H5)',
        'format': 'Keras H5',
        'model_size_mb': os.path.getsize('original_cnn_model.h5') / 1024 / 1024
    },
    {
        'model_name': 'Original Model (TFLite)',
        'format': 'TFLite',
        'model_size_mb': tflite_model_size / 1024 / 1024
    },
    {
        'model_name': 'Quantized Model (TFLite FP16)',
        'format': 'TFLite FP16',
        'model_size_mb': tflite_fp16_model_size / 1024 / 1024
    },
    {
        'model_name': 'Quantized Model (TFLite INT8)',
        'format': 'TFLite INT8',
        'model_size_mb': tflite_int8_model_size / 1024 / 1024
    },
    {
        'model_name': 'Pruned & Quantized Model (TFLite INT8)',
        'format': 'TFLite INT8',
        'model_size_mb': os.path.getsize('pruned_quantized_int8_model.tflite') / 1024 / 1024
    }
]

# Add ONNX models if they were converted successfully
if onnx_model_size > 0:
    format_results.append({
        'model_name': 'Original Model (ONNX)',
        'format': 'ONNX',
        'model_size_mb': onnx_model_size / 1024 / 1024
    })

if onnx_pruned_model_size > 0:
    format_results.append({
        'model_name': 'Pruned Model (ONNX)',
        'format': 'ONNX',
        'model_size_mb': onnx_pruned_model_size / 1024 / 1024
    })

# Add SavedModel
format_results.append({
    'model_name': 'Original Model (SavedModel)',
    'format': 'SavedModel',
    'model_size_mb': saved_model_size / 1024 / 1024
})

# Add TensorFlow.js model if it was converted successfully
if tfjs_model_exists:
    format_results.append({
        'model_name': 'Original Model (TensorFlow.js)',
        'format': 'TensorFlow.js',
        'model_size_mb': tfjs_model_size / 1024 / 1024
    })

# Create a dataframe and plot results
format_df = pd.DataFrame(format_results)
print("\nModel Format Comparison:")
print(format_df)

# Plot model format comparison
plt.figure(figsize=(12, 6))
sns.barplot(data=format_df, x='model_name', y='model_size_mb', hue='format')
plt.title('Model Size by Format')
plt.ylabel('Model Size (MB)')
plt.xlabel('Model')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Format')
plt.tight_layout()
plt.show()

# Group by format and calculate average size
format_summary = format_df.groupby('format')['model_size_mb'].mean().reset_index()
format_summary = format_summary.rename(columns={'model_size_mb': 'average_size_mb'})

plt.figure(figsize=(10, 6))
sns.barplot(data=format_summary, x='format', y='average_size_mb')
plt.title('Average Model Size by Format')
plt.ylabel('Average Size (MB)')
plt.xlabel('Format')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# =========================================================
# PART 7: COMPREHENSIVE ANALYSIS
# =========================================================

# Combine all optimization results
all_models = []

# Add original models
all_models.append({
    'model_name': 'Original (Keras)',
    'optimization': 'None',
    'accuracy': original_results['accuracy'],
    'model_size_mb': original_results['model_size_mb'],
    'relative_size': 1.0  # baseline
})

# Add quantized models
all_models.append({
    'model_name': 'TFLite (Float32)',
    'optimization': 'Format Conversion',
    'accuracy': tflite_original_results['accuracy'],
    'model_size_mb': tflite_original_results['model_size_mb'],
    'relative_size': tflite_original_results['model_size_mb'] / original_results['model_size_mb']
})

all_models.append({
    'model_name': 'TFLite (Float16)',
    'optimization': 'Quantization',
    'accuracy': tflite_fp16_results['accuracy'],
    'model_size_mb': tflite_fp16_results['model_size_mb'],
    'relative_size': tflite_fp16_results['model_size_mb'] / original_results['model_size_mb']
})

all_models.append({
    'model_name': 'TFLite (Int8)',
    'optimization': 'Quantization',
    'accuracy': tflite_int8_results['accuracy'],
    'model_size_mb': tflite_int8_results['model_size_mb'],
    'relative_size': tflite_int8_results['model_size_mb'] / original_results['model_size_mb']
})

# Add pruned models
all_models.append({
    'model_name': 'Pruned (Keras)',
    'optimization': 'Pruning',
    'accuracy': pruned_model_results['accuracy'],
    'model_size_mb': pruned_model_results['model_size_mb'],
    'relative_size': pruned_model_results['model_size_mb'] / original_results['model_size_mb']
})

all_models.append({
    'model_name': 'Pruned Stripped (Keras)',
    'optimization': 'Pruning',
    'accuracy': stripped_model_results['accuracy'],
    'model_size_mb': stripped_model_results['model_size_mb'],
    'relative_size': stripped_model_results['model_size_mb'] / original_results['model_size_mb']
})

# Add pruned and quantized models
all_models.append({
    'model_name': 'Pruned (TFLite Float16)',
    'optimization': 'Pruning + Quantization',
    'accuracy': tflite_pruned_fp16_results['accuracy'],
    'model_size_mb': tflite_pruned_fp16_results['model_size_mb'],
    'relative_size': tflite_pruned_fp16_results['model_size_mb'] / original_results['model_size_mb']
})

all_models.append({
    'model_name': 'Pruned (TFLite Int8)',
    'optimization': 'Pruning + Quantization',
    'accuracy': tflite_pruned_int8_results['accuracy'],
    'model_size_mb': tflite_pruned_int8_results['model_size_mb'],
    'relative_size': tflite_pruned_int8_results['model_size_mb'] / original_results['model_size_mb']
})

# Create a comprehensive dataframe
all_models_df = pd.DataFrame(all_models)
print("\nComprehensive Model Optimization Results:")
print(all_models_df)

# Create a scatter plot of accuracy vs. model size
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=all_models_df,
    x='model_size_mb',
    y='accuracy',
    hue='optimization',
    size='model_size_mb',
    sizes=(50, 400),
    alpha=0.7
)

# Annotate points
for i, row in all_models_df.iterrows():
    plt.annotate(
        row['model_name'],
        (row['model_size_mb'], row['accuracy']),
        xytext=(5, 5),
        textcoords='offset points'
    )

plt.title('Model Optimization: Accuracy vs. Size')
plt.xlabel('Model Size (MB)')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend(title='Optimization Technique')
plt.tight_layout()
plt.show()

# Plot relative size reduction vs. accuracy loss
all_models_df['accuracy_loss'] = original_results['accuracy'] - all_models_df['accuracy']
all_models_df['size_reduction'] = (1 - all_models_df['relative_size']) * 100

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=all_models_df[all_models_df['model_name'] != 'Original (Keras)'],  # Exclude original
    x='size_reduction',
    y='accuracy_loss',
    hue='optimization',
    size='model_size_mb',
    sizes=(50, 400),
    alpha=0.7
)

# Annotate points
for i, row in all_models_df[all_models_df['model_name'] != 'Original (Keras)'].iterrows():
    plt.annotate(
        row['model_name'],
        (row['size_reduction'], row['accuracy_loss']),
        xytext=(5, 5),
        textcoords='offset points'
    )

plt.title('Optimization Trade-off: Size Reduction vs. Accuracy Loss')
plt.xlabel('Size Reduction (%)')
plt.ylabel('Accuracy Loss (%)')
plt.grid(True)
plt.legend(title='Optimization Technique')
plt.tight_layout()
plt.show()

# Calculate the efficiency metric: accuracy per MB
all_models_df['accuracy_per_mb'] = all_models_df['accuracy'] / all_models_df['model_size_mb']

plt.figure(figsize=(10, 6))
sns.barplot(data=all_models_df, x='model_name', y='accuracy_per_mb', hue='optimization')
plt.title('Model Efficiency: Accuracy per MB')
plt.xlabel('Model')
plt.ylabel('Accuracy (%) per MB')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Optimization Technique')
plt.tight_layout()
plt.show()

# Summarize by optimization technique
optimization_summary = all_models_df.groupby('optimization').agg({
    'accuracy': 'mean',
    'model_size_mb': 'mean',
    'size_reduction': 'mean',
    'accuracy_loss': 'mean',
    'accuracy_per_mb': 'mean'
}).reset_index()

print("\nOptimization Technique Summary:")
print(optimization_summary)

plt.figure(figsize=(12, 6))
sns.barplot(data=optimization_summary, x='optimization', y='accuracy_per_mb')
plt.title('Average Efficiency by Optimization Technique')
plt.xlabel('Optimization Technique')
plt.ylabel('Average Accuracy per MB')
plt.tight_layout()
plt.show()

# Find the optimal model for different scenarios
best_accuracy = all_models_df.loc[all_models_df['accuracy'].idxmax()]
print(f"\nBest Model for Accuracy:")
print(f"  Model: {best_accuracy['model_name']}")
print(f"  Accuracy: {best_accuracy['accuracy']:.2f}%")
print(f"  Size: {best_accuracy['model_size_mb']:.2f} MB")

smallest_size = all_models_df.loc[all_models_df['model_size_mb'].idxmin()]
print(f"\nSmallest Model:")
print(f"  Model: {smallest_size['model_name']}")
print(f"  Accuracy: {smallest_size['accuracy']:.2f}%")
print(f"  Size: {smallest_size['model_size_mb']:.2f} MB")

best_efficiency = all_models_df.loc[all_models_df['accuracy_per_mb'].idxmax()]
print(f"\nMost Efficient Model (Accuracy/MB):")
print(f"  Model: {best_efficiency['model_name']}")
print(f"  Accuracy: {best_efficiency['accuracy']:.2f}%")
print(f"  Size: {best_efficiency['model_size_mb']:.2f} MB")
print(f"  Efficiency: {best_efficiency['accuracy_per_mb']:.2f} %/MB")

# For mobile deployment (balancing size and accuracy)
# Create a custom score: (accuracy - 95) / size
# This prioritizes models with >95% accuracy but weights heavily toward small size
all_models_df['mobile_score'] = (all_models_df['accuracy'] - 95) / all_models_df['model_size_mb']
all_models_df.loc[all_models_df['accuracy'] < 95, 'mobile_score'] = 0  # Penalize models with <95% accuracy

best_mobile = all_models_df.loc[all_models_df['mobile_score'].idxmax()]
print(f"\nBest Model for Mobile Deployment:")
print(f"  Model: {best_mobile['model_name']}")
print(f"  Accuracy: {best_mobile['accuracy']:.2f}%")
print(f"  Size: {best_mobile['model_size_mb']:.2f} MB")
print(f"  Size Reduction: {best_mobile['size_reduction']:.2f}%")
print(f"  Accuracy Loss: {best_mobile['accuracy_loss']:.2f}%")

# Show the final comparison of the "winners" in each category
winners = pd.DataFrame([best_accuracy, smallest_size, best_efficiency, best_mobile])
print("\nComparison of Optimal Models:")
print(winners[['model_name', 'optimization', 'accuracy', 'model_size_mb', 'accuracy_per_mb']])

# =========================================================
# PART 8: SAVE RESULTS
# =========================================================

# Save results to CSV (uncomment if you want to save)
# all_models_df.to_csv('model_optimization_results.csv', index=False)
# format_df.to_csv('model_format_comparison.csv', index=False)
# training_df.to_csv('hardware_training_benchmark.csv', index=False)
# inference_df.to_csv('hardware_inference_benchmark.csv', index=False)

print("\nLab 4 completed successfully!")

# =========================================================
# END OF LAB 4
# =========================================================