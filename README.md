# Handwritten Digit Recognition using CNN

## Project Overview

This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits (0–9) using the MNIST dataset.
The model is trained to classify grayscale images of size 28×28 pixels and can also predict custom handwritten digit images.

---

## Features

* Load and preprocess MNIST dataset
* Build CNN model using TensorFlow and Keras
* Train and evaluate the model
* Achieve approximately 99% accuracy
* Predict individual test images
* Generate confusion matrix
* Test custom handwritten digit images

---

## Dataset

The project uses the MNIST dataset, which contains:

* 60,000 training images
* 10,000 testing images
* 28×28 grayscale images
* 10 classes (digits 0–9)

---

## Model Architecture

The CNN model consists of:

* Convolutional Layer (32 filters, 5×5 kernel, ReLU activation)
* MaxPooling Layer
* Dropout Layer (0.2)
* Flatten Layer
* Dense Layer (128 neurons, ReLU activation)
* Output Layer (Softmax activation)

---

## Technologies Used

* Python 3.10
* TensorFlow / Keras
* NumPy
* Matplotlib
* Scikit-learn
* OpenCV (for custom image testing)

---

## Data Preprocessing

* Reshape images to (28, 28, 1)
* Normalize pixel values from 0–255 to 0–1
* One-hot encode labels
* Resize and center custom images before prediction

---

## Model Training

```python
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=10,
          batch_size=200)
```

---

## Model Evaluation

```python
scores = model.evaluate(X_test, y_test)
print("Test Accuracy:", scores[1] * 100)
```

The model achieves approximately 99% accuracy on test data.

---

## Testing Custom Handwritten Image

Steps:

1. Write a digit on white paper
2. Capture and crop the image
3. Convert to grayscale
4. Resize to 28×28
5. Normalize and reshape
6. Predict using:

```python
prediction = model.predict(image)
print("Predicted Digit:", np.argmax(prediction))
```

---

## Confusion Matrix

Used to analyze classification performance:

```python
from sklearn.metrics import confusion_matrix
```

This helps visualize model accuracy for each digit class.

---

## Results

* High accuracy (approximately 99%)
* Good generalization on test data
* Works with properly preprocessed custom images



If you want, I can also create a shorter GitHub version or a resume-ready project description.
