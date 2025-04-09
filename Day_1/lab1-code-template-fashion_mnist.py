# Lab 1: Working with Pre-trained Models for MNIST Classification
# Student Name: [YOUR NAME HERE]

# =========================================================
# PART 1: ENVIRONMENT SETUP
# =========================================================

# Import basic libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D  # Added for model modification
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

fashion_class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Mount Google Drive (uncomment when running in Colab)
# from google.colab import drive
# drive.mount('/content/drive')
# !mkdir -p "/content/drive/My Drive/ML_Hardware_Course/Lab1"

# Check TensorFlow version and GPU availability
print("TensorFlow version:", tf.__version__)
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
# If running in Colab, check GPU details
try:
    !nvidia-smi
except:
    print("nvidia-smi command not available (likely not running on GPU)")

# =========================================================
# PART 2: MNIST DATASET PREPARATION
# =========================================================

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Print dataset shapes
print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Test data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)

# Create a function to display multiple images
# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Print dataset shapes
print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Test data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)

# Create a function to display multiple images
def display_sample_images(X, y, num_samples=10):
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(X[i], cmap='gray')
        plt.title(f"{fashion_class_names[y[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Display 10 sample images
display_sample_images(X_train, y_train)

# Simple preprocessing function for MNIST
def preprocess_mnist_simple(X_train, X_test):
    """
    Simple preprocessing for MNIST, normalizing and reshaping to add channel dimension
    """
    # Normalize to [0,1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Add channel dimension (28x28 -> 28x28x1)
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    
    return X_train, X_test

# Prepare the labels
y_train_encoded = to_categorical(y_train, 10)
y_test_encoded = to_categorical(y_test, 10)

# Create a validation set (20% of training data)
val_size = 12000  # 20% of 60,000
X_val = X_train[-val_size:]
y_val = y_train_encoded[-val_size:]
X_train_final = X_train[:-val_size]
y_train_final = y_train_encoded[:-val_size]

print("Training set size:", X_train_final.shape[0])
print("Validation set size:", X_val.shape[0])
print("Test set size:", X_test.shape[0])

# =========================================================
# PART 3: MOBILENETV2 MODEL PREPARATION (MODIFIED)
# =========================================================

def create_mobilenet_model():
    """
    Create MobileNetV2 model with properly padded input for MNIST
    """
    # Preprocess data
    X_train_mobilenet, X_test_mobilenet = preprocess_mnist_simple(X_train_final, X_test)
    X_val_mobilenet, _ = preprocess_mnist_simple(X_val, np.zeros((1, 28, 28)))
    
    # Create model architecture
    inputs = Input(shape=(28, 28, 1))
    
    # Pad the input from 28x28 to 32x32 using zero padding
    x = tf.keras.layers.ZeroPadding2D(padding=2)(inputs)  # Add 2 pixels on each side: 28x28 -> 32x32
    
    # Convert single-channel grayscale to 3-channel RGB format required by MobileNetV2
    x = Conv2D(16, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(3, kernel_size=1, padding='same', activation='relu')(x)  # Output 3 channels
    
    # Create sub-model with MobileNetV2
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(32, 32, 3),
        pooling='avg'
    )
    
    # Freeze the base model layers
    base_model.trainable = True
    
    # Continue with the model architecture
    x = base_model(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(10, activation='softmax')(x)
    
    mobilenet_model = Model(inputs, outputs)
    
    # Compile the model
    mobilenet_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model summary
    mobilenet_model.summary()
    
    return mobilenet_model, X_train_mobilenet, X_val_mobilenet, X_test_mobilenet

# Create MobileNetV2 model
mobilenet_model, X_train_mobilenet, X_val_mobilenet, X_test_mobilenet = create_mobilenet_model()

# =========================================================
# PART 4: RESNET50 MODEL PREPARATION (MODIFIED)
# =========================================================

def create_resnet_model():
    """
    Create ResNet50 model with properly padded input for MNIST
    """
    # Preprocess data
    X_train_resnet, X_test_resnet = preprocess_mnist_simple(X_train_final, X_test)
    X_val_resnet, _ = preprocess_mnist_simple(X_val, np.zeros((1, 28, 28)))
    
    # Create model architecture
    inputs = Input(shape=(28, 28, 1))
    
    # Pad the input from 28x28 to 32x32
    x = tf.keras.layers.ZeroPadding2D(padding=2)(inputs)  # Add 2 pixels on each side
    
    # Convert single-channel to 3-channel input
    x = Conv2D(16, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(3, kernel_size=1, padding='same', activation='relu')(x)  # Output 3 channels
    
    # Load ResNet50 with proper input shape
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(32, 32, 3),
        pooling='avg'
    )
    
    # Freeze the base model layers
    base_model.trainable = True
    
    # Continue with model architecture
    x = base_model(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(10, activation='softmax')(x)
    
    resnet_model = Model(inputs, outputs)
    
    # Compile the model
    resnet_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model summary
    resnet_model.summary()
    
    return resnet_model, X_train_resnet, X_val_resnet, X_test_resnet
    
    # Create ResNet50 model
resnet_model, X_train_resnet, X_val_resnet, X_test_resnet = create_resnet_model()

# =========================================================
# PART 5: VGG16 MODEL PREPARATION (MODIFIED)
# =========================================================

def create_vgg_model():
    """
    Create VGG16 model with properly padded input for MNIST
    """
    # Preprocess data
    X_train_vgg, X_test_vgg = preprocess_mnist_simple(X_train_final, X_test)
    X_val_vgg, _ = preprocess_mnist_simple(X_val, np.zeros((1, 28, 28)))
    
    # Create model architecture
    inputs = Input(shape=(28, 28, 1))
    
    # Pad the input from 28x28 to 32x32
    x = tf.keras.layers.ZeroPadding2D(padding=2)(inputs)  # Add 2 pixels on each side
    
    # Convert single-channel to 3-channel
    x = Conv2D(16, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(3, kernel_size=1, padding='same', activation='relu')(x)  # Output 3 channels
    
    # Load VGG16 with proper input shape
    base_model = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(32, 32, 3),
        pooling='avg'
    )
    
    # Freeze the base model layers
    base_model.trainable = True
    
    # Continue with model architecture
    x = base_model(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(10, activation='softmax')(x)
    
    vgg_model = Model(inputs, outputs)
    
    # Compile the model
    vgg_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model summary
    vgg_model.summary()
    
    return vgg_model, X_train_vgg, X_val_vgg, X_test_vgg


# Create VGG16 model
vgg_model, X_train_vgg, X_val_vgg, X_test_vgg = create_vgg_model()

# =========================================================
# PART 6: MODEL TRAINING
# =========================================================

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

# Train MobileNetV2 model
print("\n--- Training MobileNetV2 Model ---")
start_time = time.time()
mobilenet_history = mobilenet_model.fit(
    X_train_mobilenet,
    y_train_final,
    epochs=10,
    batch_size=64,
    validation_data=(X_val_mobilenet, y_val),
    callbacks=[early_stopping],
    verbose=1
)
mobilenet_training_time = time.time() - start_time
print(f"MobileNetV2 - Training completed in {mobilenet_training_time:.2f} seconds")

# Train ResNet50 model
print("\n--- Training ResNet50 Model ---")
start_time = time.time()
resnet_history = resnet_model.fit(
    X_train_resnet,
    y_train_final,
    epochs=10,
    batch_size=32,  # Smaller batch size due to larger model
    validation_data=(X_val_resnet, y_val),
    callbacks=[early_stopping],
    verbose=1
)
resnet_training_time = time.time() - start_time
print(f"ResNet50 - Training completed in {resnet_training_time:.2f} seconds")

# Train VGG16 model
print("\n--- Training VGG16 Model ---")
start_time = time.time()
vgg_history = vgg_model.fit(
    X_train_vgg,
    y_train_final,
    epochs=10,
    batch_size=64,
    validation_data=(X_val_vgg, y_val),
    callbacks=[early_stopping],
    verbose=1
)
vgg_training_time = time.time() - start_time
print(f"VGG16 - Training completed in {vgg_training_time:.2f} seconds")

# =========================================================
# PART 7: MODEL EVALUATION
# =========================================================

# Evaluate models on test set
print("\n--- Model Evaluation on Test Set ---")
mobilenet_loss, mobilenet_accuracy = mobilenet_model.evaluate(X_test_mobilenet, y_test_encoded)
print(f"MobileNetV2 - Test accuracy: {mobilenet_accuracy*100:.2f}%")

resnet_loss, resnet_accuracy = resnet_model.evaluate(X_test_resnet, y_test_encoded)
print(f"ResNet50 - Test accuracy: {resnet_accuracy*100:.2f}%")

vgg_loss, vgg_accuracy = vgg_model.evaluate(X_test_vgg, y_test_encoded)
print(f"VGG16 - Test accuracy: {vgg_accuracy*100:.2f}%")

# Plot training history
def plot_training_history(histories, titles):
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    for history, title in zip(histories, titles):
        plt.plot(history.history['accuracy'], label=f'{title} - Training')
        plt.plot(history.history['val_accuracy'], label=f'{title} - Validation')
    
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    for history, title in zip(histories, titles):
        plt.plot(history.history['loss'], label=f'{title} - Training')
        plt.plot(history.history['val_loss'], label=f'{title} - Validation')
    
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Plot training history for all models
plot_training_history(
    [mobilenet_history, resnet_history, vgg_history],
    ['MobileNetV2', 'ResNet50', 'VGG16']
)

# =========================================================
# PART 8: CONFUSION MATRICES AND CLASSIFICATION REPORTS
# =========================================================

# Function to generate predictions and confusion matrix
def analyze_model_performance(model, X_test, y_test, model_name):
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Find the most confused pairs
    cm_normalized = cm.copy()
    np.fill_diagonal(cm_normalized, 0)  # Ignore correct predictions
    max_confusion = np.unravel_index(np.argmax(cm_normalized), cm_normalized.shape)
    print(f"Most confused pair: True digit {max_confusion[0]} predicted as {max_confusion[1]} ({cm_normalized[max_confusion]} times)")
    
    # Generate classification report
    report = classification_report(y_true_classes, y_pred_classes, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print(f"{model_name} Classification Report:")
    print(report_df.round(3))
    
    return y_pred_classes, report, max_confusion

# Analyze each model's performance
print("\n--- MobileNetV2 Performance Analysis ---")
mobilenet_pred, mobilenet_report, mobilenet_confused_pair = analyze_model_performance(
    mobilenet_model, X_test_mobilenet, y_test_encoded, 'MobileNetV2'
)

print("\n--- ResNet50 Performance Analysis ---")
resnet_pred, resnet_report, resnet_confused_pair = analyze_model_performance(
    resnet_model, X_test_resnet, y_test_encoded, 'ResNet50'
)

print("\n--- VGG16 Performance Analysis ---")
vgg_pred, vgg_report, vgg_confused_pair = analyze_model_performance(
    vgg_model, X_test_vgg, y_test_encoded, 'VGG16'
)

# =========================================================
# PART 9: MODEL METRICS COMPARISON
# =========================================================

def count_model_parameters(model):
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    return trainable_params, non_trainable_params, total_params

# Get parameter counts for each model
mobilenet_trainable, mobilenet_non_trainable, mobilenet_total = count_model_parameters(mobilenet_model)
resnet_trainable, resnet_non_trainable, resnet_total = count_model_parameters(resnet_model)
vgg_trainable, vgg_non_trainable, vgg_total = count_model_parameters(vgg_model)

# Function to measure inference time
def measure_inference_time(model, X_test, batch_size=1, num_runs=50):
    # Warm-up
    for _ in range(10):
        _ = model.predict(X_test[:batch_size])
    
    # Measure time for inference
    start_time = time.time()
    for _ in range(num_runs):
        _ = model.predict(X_test[:batch_size])
    total_time = time.time() - start_time
    
    # Calculate average inference time per batch
    avg_time = total_time / num_runs
    return avg_time * 1000  # Convert to milliseconds

# Measure inference time for each model (single image)
mobilenet_inference_time = measure_inference_time(mobilenet_model, X_test_mobilenet)
resnet_inference_time = measure_inference_time(resnet_model, X_test_resnet)
vgg_inference_time = measure_inference_time(vgg_model, X_test_vgg)

print("\n--- Single Image Inference Time ---")
print(f"MobileNetV2 - Inference time (1 image): {mobilenet_inference_time:.2f} ms")
print(f"ResNet50 - Inference time (1 image): {resnet_inference_time:.2f} ms")
print(f"VGG16 - Inference time (1 image): {vgg_inference_time:.2f} ms")

# Measure inference time for batch of 32 images
mobilenet_batch_time = measure_inference_time(mobilenet_model, X_test_mobilenet, batch_size=32, num_runs=20)
resnet_batch_time = measure_inference_time(resnet_model, X_test_resnet, batch_size=32, num_runs=20)
vgg_batch_time = measure_inference_time(vgg_model, X_test_vgg, batch_size=32, num_runs=20)

print("\n--- Batch Inference Time (32 images) ---")
print(f"MobileNetV2 - Inference time (32 images): {mobilenet_batch_time:.2f} ms")
print(f"ResNet50 - Inference time (32 images): {resnet_batch_time:.2f} ms")
print(f"VGG16 - Inference time (32 images): {vgg_batch_time:.2f} ms")

# Create comparison table
model_metrics = {
    'Model': ['MobileNetV2', 'ResNet50', 'VGG16'],
    'Test Accuracy (%)': [
        mobilenet_accuracy * 100,
        resnet_accuracy * 100,
        vgg_accuracy * 100
    ],
    'Trainable Parameters': [
        mobilenet_trainable,
        resnet_trainable,
        vgg_trainable
    ],
    'Total Parameters': [
        mobilenet_total,
        resnet_total,
        vgg_total
    ],
    'Training Time (s)': [
        mobilenet_training_time,
        resnet_training_time,
        vgg_training_time
    ],
    'Inference Time (ms)': [
        mobilenet_inference_time,
        resnet_inference_time,
        vgg_inference_time
    ],
    'Batch Inference Time (ms)': [
        mobilenet_batch_time,
        resnet_batch_time,
        vgg_batch_time
    ],
    'Parameters/Second': [
        mobilenet_total / mobilenet_training_time,
        resnet_total / resnet_training_time,
        vgg_total / vgg_training_time
    ],
    'Accuracy/Million Params': [
        (mobilenet_accuracy * 100) / (mobilenet_total / 1e6),
        (resnet_accuracy * 100) / (resnet_total / 1e6),
        (vgg_accuracy * 100) / (vgg_total / 1e6)
    ],
    'Most Confused Pair': [
        f"{mobilenet_confused_pair[0]}-{mobilenet_confused_pair[1]}",
        f"{resnet_confused_pair[0]}-{resnet_confused_pair[1]}",
        f"{vgg_confused_pair[0]}-{vgg_confused_pair[1]}"
    ]
}

# Create and display DataFrame
metrics_df = pd.DataFrame(model_metrics).set_index('Model')
print("\n--- Model Comparison Metrics ---")
pd.set_option('display.float_format', '{:.2f}'.format)
print(metrics_df)

# =========================================================
# PART 10: VISUALIZATION OF MODEL COMPARISONS
# =========================================================

# Visualize the metrics
plt.figure(figsize=(15, 10))

# Accuracy comparison
plt.subplot(2, 2, 1)
plt.bar(model_metrics['Model'], model_metrics['Test Accuracy (%)'])
plt.title('Test Accuracy (%)')
plt.ylim(90, 100)  # Adjust as needed
plt.grid(axis='y')

# Training time comparison
plt.subplot(2, 2, 2)
plt.bar(model_metrics['Model'], model_metrics['Training Time (s)'])
plt.title('Training Time (seconds)')
plt.grid(axis='y')

# Parameter count comparison (log scale)
plt.subplot(2, 2, 3)
plt.bar(model_metrics['Model'], [np.log10(p) for p in model_metrics['Total Parameters']])
plt.title('Log10(Total Parameters)')
plt.grid(axis='y')

# Efficiency comparison
plt.subplot(2, 2, 4)
plt.bar(model_metrics['Model'], model_metrics['Accuracy/Million Params'])
plt.title('Accuracy/Million Parameters')
plt.grid(axis='y')

plt.tight_layout()
plt.show()

# Inference time comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(model_metrics['Model'], model_metrics['Inference Time (ms)'])
plt.title('Single Image Inference Time (ms)')
plt.grid(axis='y')

plt.subplot(1, 2, 2)
plt.bar(model_metrics['Model'], model_metrics['Batch Inference Time (ms)'])
plt.title('Batch Inference Time (32 images, ms)')
plt.grid(axis='y')

plt.tight_layout()
plt.show()

# =========================================================
# PART 11: WORKSHEET VALUES SUMMARY
# =========================================================

print("\n===== WORKSHEET VALUES =====")

print("\n1.1 Basic Performance Metrics:")
for model_name, trainable, total, accuracy, train_time, infer_time in zip(
    model_metrics['Model'],
    model_metrics['Trainable Parameters'],
    model_metrics['Total Parameters'],
    model_metrics['Test Accuracy (%)'],
    model_metrics['Training Time (s)'],
    model_metrics['Inference Time (ms)']
):
    print(f"\n{model_name}:")
    print(f"  Trainable Parameters: {trainable}")
    print(f"  Total Parameters: {total}")
    print(f"  Test Accuracy: {accuracy:.2f}%")
    print(f"  Training Time: {train_time:.2f} seconds")
    print(f"  Inference Time: {infer_time:.2f} ms")

print("\n1.2 Efficiency Metrics:")
for model_name, params_per_sec, acc_per_mil, batch_time in zip(
    model_metrics['Model'],
    model_metrics['Parameters/Second'],
    model_metrics['Accuracy/Million Params'],
    model_metrics['Batch Inference Time (ms)']
):
    print(f"\n{model_name}:")
    print(f"  Parameters/Second: {params_per_sec:.2f}")
    print(f"  Accuracy/Million Params: {acc_per_mil:.2f}")
    print(f"  Batch Inference Time: {batch_time:.2f} ms")

print("\n2.1 Most Confused Digit Pairs:")
for model_name, confused_pair in zip(
    model_metrics['Model'],
    model_metrics['Most Confused Pair']
):
    print(f"  {model_name}: {confused_pair}")

# Get the best performing model
best_model_idx = np.argmax(model_metrics['Test Accuracy (%)'])
best_model_name = model_metrics['Model'][best_model_idx]
best_model_report = [mobilenet_report, resnet_report, vgg_report][best_model_idx]

print(f"\n2.2 Per-Class Precision for Best Model ({best_model_name}):")
for digit in range(10):
    print(f"  Digit {digit}: {best_model_report[str(digit)]['precision']:.4f}")

# =========================================================
# PART 12: SAVE MODELS AND NOTEBOOK
# =========================================================

# Uncomment the following to save in Google Colab
# Save models
# mobilenet_model.save("/content/drive/My Drive/ML_Hardware_Course/Lab1/mobilenet_mnist.h5")
# resnet_model.save("/content/drive/My Drive/ML_Hardware_Course/Lab1/resnet_mnist.h5")
# vgg_model.save("/content/drive/My Drive/ML_Hardware_Course/Lab1/vgg_mnist.h5")
# print("Models saved successfully!")

# =========================================================
# PART 13: SAVE RESULTS FOR WORKSHEET
# =========================================================

# Create a results summary file
results_summary = {
    "basic_metrics": {
        "mobilenet": {
            "trainable_params": mobilenet_trainable,
            "total_params": mobilenet_total,
            "test_accuracy": mobilenet_accuracy * 100,
            "training_time": mobilenet_training_time,
            "inference_time": mobilenet_inference_time,
        },
        "resnet": {
            "trainable_params": resnet_trainable,
            "total_params": resnet_total,
            "test_accuracy": resnet_accuracy * 100,
            "training_time": resnet_training_time,
            "inference_time": resnet_inference_time,
        },
        "vgg": {
            "trainable_params": vgg_trainable,
            "total_params": vgg_total,
            "test_accuracy": vgg_accuracy * 100,
            "training_time": vgg_training_time,
            "inference_time": vgg_inference_time,
        }
    },
    "efficiency_metrics": {
        "mobilenet": {
            "params_per_second": mobilenet_total / mobilenet_training_time,
            "accuracy_per_million_params": (mobilenet_accuracy * 100) / (mobilenet_total / 1e6),
            "batch_inference_time": mobilenet_batch_time,
        },
        "resnet": {
            "params_per_second": resnet_total / resnet_training_time,
            "accuracy_per_million_params": (resnet_accuracy * 100) / (resnet_total / 1e6),
            "batch_inference_time": resnet_batch_time,
        },
        "vgg": {
            "params_per_second": vgg_total / vgg_training_time,
            "accuracy_per_million_params": (vgg_accuracy * 100) / (vgg_total / 1e6),
            "batch_inference_time": vgg_batch_time,
        }
    },
    "confusion_pairs": {
        "mobilenet": {
            "pair": f"{mobilenet_confused_pair[0]}-{mobilenet_confused_pair[1]}",
            "count": int(confusion_matrix(np.argmax(y_test_encoded, axis=1), mobilenet_pred)[mobilenet_confused_pair])
        },
        "resnet": {
            "pair": f"{resnet_confused_pair[0]}-{resnet_confused_pair[1]}",
            "count": int(confusion_matrix(np.argmax(y_test_encoded, axis=1), resnet_pred)[resnet_confused_pair])
        },
        "vgg": {
            "pair": f"{vgg_confused_pair[0]}-{vgg_confused_pair[1]}",
            "count": int(confusion_matrix(np.argmax(y_test_encoded, axis=1), vgg_pred)[vgg_confused_pair])
        }
    },
    "best_model": {
        "name": best_model_name,
        "precision_by_digit": {str(digit): best_model_report[str(digit)]['precision'] for digit in range(10)}
    }
}

# Convert to DataFrame for easier viewing
results_df = pd.DataFrame({
    "Model": model_metrics['Model'],
    "Test Accuracy (%)": model_metrics['Test Accuracy (%)'],
    "Total Parameters": model_metrics['Total Parameters'],
    "Training Time (s)": model_metrics['Training Time (s)'],
    "Inference Time (ms)": model_metrics['Inference Time (ms)'],
    "Accuracy/Million Params": model_metrics['Accuracy/Million Params']
})

print("\n--- Results Summary for Worksheet ---")
print(results_df)