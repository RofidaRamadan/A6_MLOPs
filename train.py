import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1 (speeds up training)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a simple, lightweight neural network
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten 2D image to 1D array
  tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons
  tf.keras.layers.Dropout(0.2),                   # Prevents overfitting
  tf.keras.layers.Dense(10, activation='softmax') # Output layer (10 digits: 0-9)
])

# 4. Compile the model with an optimizer and loss function
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train the model for exactly 5 epochs
print("--- Starting Training ---")
model.fit(x_train, y_train, epochs=5)

# 6. Test the model's accuracy on unseen data
print("\n--- Evaluating Model ---")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.4f}")