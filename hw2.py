import tensorflow as tensorflow
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

x_val = x_train[-12000:]
y_val = y_train[-12000:]
x_train = x_train[-12000:]
y_train = y_train[-12000:]

model = tf.keras.Sequential([
	tf.keras.layers.Conv2D(28, (3,3), activation = 'relu', input_shape = (28,28,1), padding = 'same'),
	tf.keras.layers.MaxPooling2D((2,2)),
	tf.keras.layers.Conv2D(56, (3,3), activation = 'relu', input_shape=(28,28,1), padding = 'same'),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(56, activation = 'relu'),
	tf.keras.layers.Dense(10, activation = 'softmax')
	])


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

print("Number of trainable parameters:", model.count_params())


history = model.fit(x_train.reshape(-1,28,28,1), y_train, batch_size = 32, epochs = 10, validation_data = (x_val.reshape(-1,28,28,1), y_val))


history.history

loss = history.history['loss']
accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

predictions = np.argmax(model.predict(x_test.reshape(-1,28,28,1)), axis = 1)
misclassified_indices = []

for i in range(10):
	for j in range(predictions.shape[0]):
		if((y_test[j] == i) and (predictions[j] != i)):
			misclassified_indices.append(j)
			break

print("Misclassified indices:", misclassified_indices)

plt.plot(range(1,11), accuracy, label = 'Training Accuracy')
plt.plot(range(1,11), val_accuracy, label = 'Validation Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


for i in range(10):
	plt.figure(figsize=[3,3])
	plt.imshow(x_test[misclassified_indices[i]], cmap = 'gray')
	plt.title(f'Predicted: {predictions[misclassified_indices[i]]}, Actual: {y_test[misclassified_indices[i]]}')
	plt.show