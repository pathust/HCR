import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report

np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Dataset information:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Visualize some samples
data_dir = 'data'
fig, axes = plt.subplots(3, 10, figsize=(15, 5))
fig.suptitle('30 samples from MNIST', fontsize=14, y=1.02)
for i in range(30):
    ax = axes[i//10, i%10]
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(f'Label: {y_train[i]}', fontsize=10)
    ax.axis('off')
plt.tight_layout()
plt.savefig(data_dir + '/mnist_samples.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Đã lưu: {data_dir}/mnist_samples.png")

# Distribution of labels
plt.figure(figsize=(10, 5))
uniques, counts = np.unique(y_train, return_counts=True)
plt.bar(uniques, counts, color='steelblue', alpha=0.8)
plt.xlabel('Label', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Distribution of Labels in Training Set', fontsize=14)
plt.grid(axis='y', alpha=0.3)
for i, (x, y) in enumerate(zip(uniques, counts)):
    plt.text(x, y + 50, str(y), ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(data_dir + '/label_distribution.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Đã lưu: {data_dir}/label_distribution.png")

# Normalize the data
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
print(f"✓ Normalization: pixel range [{X_train.min():.1f}, {X_train.max():.1f}]")

# One-hot encoding cho labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)
print(f"✓ One-hot encoding: {y_train_cat.shape}")
print(f"  Ví dụ: label {y_train[0]} → {y_train_cat[0]}")

def create_cnn_model():
    """
    Kiến trúc CNN chuẩn cho MNIST:
    - 2 khối Conv-Pool để trích xuất đặc trưng
    - Fully connected layers để phân loại
    - Dropout để tránh overfitting
    """
    model = models.Sequential([
        # Block 1: Conv-Pool
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv1'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Block 2: Conv-Pool
        layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
        layers.MaxPooling2D((2, 2), name='pool2'),

        # Block 3: Convolution
        layers.Conv2D(64, (3, 3), activation='relu', name='conv3'),
        
        # Fully connected layers
        layers.Flatten(name='flatten'),
        layers.Dense(128, activation='relu', name='dense1'),
        layers.Dropout(0.5, name='dropout'),
        layers.Dense(10, activation='softmax', name='output')
    ])
    
    return model

model = create_cnn_model()

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Visualize the model
keras.utils.plot_model(model, to_file=data_dir + '/model.png', show_shapes=True, show_layer_names=True)
print(f"\n✅ Đã lưu: {data_dir}/model.png")

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_mnist_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train the model
EPOCHS = 20
BATCH_SIZE = 128

history = model.fit(
    X_train, y_train_cat,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Training completed")

# Load the best model
model.load_weights('best_mnist_model.h5')

# Evaluate the model on test set
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print("Result on test set:")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Prediction on test set
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test_cat, axis=1)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix:")
print(conf_matrix)

# Classification report
class_report = classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)])
print("\nClassification report:")
print(class_report)

# Visualize the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion matrix')
plt.savefig(data_dir + '/conf_matrix.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Đã lưu: {data_dir}/conf_matrix.png")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.legend()
plt.title('Training and validation accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.title('Training and validation loss')
plt.savefig(data_dir + '/training_history.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Đã lưu: {data_dir}/training_history.png")

# Visualize predictions
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(X_test[0], cmap='gray')
plt.title(f'Predicted: {y_pred[0]}, True: {y_test[0]}')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.bar(range(10), y_pred[0], color='steelblue', alpha=0.8)
plt.xlabel('Label', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title('Prediction probabilities', fontsize=14)
plt.savefig(data_dir + '/predictions.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Đã lưu: {data_dir}/predictions.png")