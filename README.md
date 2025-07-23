# CNN-Based COVID-19 Chest X-ray Classification

This project is a deep learning application using Convolutional Neural Networks (CNNs) to classify chest X-ray images into categories such as COVID-19, Pneumonia, and Normal. The implementation covers the entire pipeline from data loading, preprocessing, CNN model design, training, evaluation, and deployment using FastAPI and Streamlit.

---

## 🧠 Theory

- **ReLU Activation Function:** ReLU introduces non-linearity into the CNN model by replacing negative pixel values with zero, allowing the network to learn complex patterns in the data.
- **Data Augmentation:** Data augmentation increases the diversity of the training data by applying random transformations (like rotation, zoom, flips), which helps prevent overfitting and improves generalization—especially important in medical imaging tasks with limited data.

---

## 📥 Dataset Loading & Splitting

The COVID-19 Radiography Dataset is loaded using TensorFlow’s `image_dataset_from_directory`. It is split into training, validation, and test sets while maintaining class balance and applying standard preprocessing such as resizing and normalization.

---

## 🧾 Class Information

The number of classes and their labels are printed after loading the dataset to confirm the target categories (e.g., COVID-19, Pneumonia, Normal).

---

## 🖼️ Data Visualization

A visualization function is implemented to display five randomly selected images from any given class. Each image is displayed using Matplotlib with its corresponding filename as the title.

---

## 🧩 Image Grid Preview

One batch from the training dataset is visualized in a 3x3 grid layout. This gives a quick overview of the images and their diversity within the dataset.

---

## 🧱 CNN Model Architecture

A custom CNN is designed with:
- At least three convolutional layers followed by ReLU activation and MaxPooling layers.
- A flattening layer.
- Dense layers for classification.
- A final softmax activation for multi-class output.

---

## ⚙️ Model Compilation & Summary

The CNN model is compiled using the Adam optimizer, categorical cross-entropy loss, and accuracy as the evaluation metric. The model architecture and parameters are displayed using `model.summary()`.

---

## 🏋️ Model Training

The model is trained for at least five epochs while logging the training and validation accuracy and loss. This provides insights into the learning behavior of the model over time.

---

## 📈 Training & Validation Curves

Training and validation performance is visualized using accuracy and loss graphs, helping to diagnose overfitting or underfitting during model training.

---

## ✅ Model Evaluation

The trained CNN is evaluated on the test dataset to measure its generalization ability. Test accuracy is reported to summarize performance.

---

## 🔢 Confusion Matrix

A confusion matrix is generated using scikit-learn to visualize the performance of the model across all classes. It shows the true vs. predicted classifications.

---

## 🔮 Prediction Function

A function `predict(model, image)` is created to:
- Preprocess a single input image (resize, normalize).
- Predict the class using the trained model.
- Return the predicted label and confidence score.

---

## 💾 Model Saving

The trained model is saved to disk in HDF5 format using `model.save('covid_classifier.h5')`. This saved model can be reused during deployment.

---

## 🖥️ Deployment with FastAPI and Streamlit

### 🔧 FastAPI Backend

A FastAPI app (`main.py`) is developed that:
- Accepts a chest X-ray image via a POST request (`/predict`).
- Resizes and preprocesses the image to match the model's input shape.
- Loads the saved CNN model.
- Returns the predicted class and confidence score in JSON format.

### 🌐 Streamlit Frontend

A user-friendly interface (`app.py`) built using Streamlit allows:
- Uploading chest X-ray images.
- Sending images to the FastAPI backend for prediction.
- Displaying the predicted class and confidence.
- Showing a loading spinner or progress bar during prediction.

---

## 🚀 Technologies Used

- TensorFlow & Keras for model building
- scikit-learn for metrics and evaluation
- FastAPI for backend API
- Streamlit for interactive frontend
- Matplotlib & Seaborn for visualization
