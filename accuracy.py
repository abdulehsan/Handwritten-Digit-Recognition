import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load the trained model
model = tf.keras.models.load_model('digit_recognition_model.h5')

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the test data
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0  # Normalize
y_test = tf.keras.utils.to_categorical(y_test, 10)  # One-hot encoding

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

# Display the accuracy
print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")
