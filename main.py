# Import Library
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Muat dataset iris dari file iris.data
dataset = pd.read_csv('iris.data', header=None, sep=',')

# Menyusun data X (fitur) dan y (label)
X = dataset.iloc[:, :-1].values   # 4 kolom pertama
y = dataset.iloc[:, -1].values    # kolom terakhir (string)

# Mengonversi label dari string menjadi numerik
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Iris-setosa=0, versicolor=1, virginica=2

# Memisahkan dataset menjadi data latih dan data validasi dengan rasio 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat Model Neural Network
model = Sequential([
    Input(shape=(X_train.shape[1],)),  # input layer (4 fitur)
    Dense(1000, activation='relu'),
    Dense(500, activation='relu'),
    Dense(300, activation='relu'),
    Dense(3, activation='softmax')     # output 3 kelas
])

# Tampilkan arsitektur model
model.summary()

# Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # karena y integer (0,1,2)
    metrics=['accuracy']
)

# Training Model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Evaluasi Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nLoss pada data test : {loss:.4f}")
print(f"Accuracy pada data test : {accuracy:.4f}")

# Visualisasi History Training (Loss & Accuracy)
pd.DataFrame(history.history).plot(figsize=(10, 6))
plt.title("Training & Validation Metrics")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.grid(True)
plt.show()   # GAMBAR 1

# Prediksi pada data test & Confusion Matrix
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()   # GAMBAR 2

# Fungsi Prediksi Data Baru Interaktif
# Fungsi untuk memprediksi data input baru
def predict_new_data():
    sepal_length = float(input("Masukkan sepal length: "))
    sepal_width = float(input("Masukkan sepal width: "))
    petal_length = float(input("Masukkan petal length: "))
    petal_width = float(input("Masukkan petal width: "))
    # Membuat data array baru
    new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Melakukan prediksi
    prediction = model.predict(new_data)
    predicted_class = prediction.argmax(axis=1)
    
    # Mengonversi hasil prediksi numerik menjadi label asli
    predicted_label = label_encoder.inverse_transform(predicted_class)
    print(f"Prediksi kelas: {predicted_label[0]}")

predict_new_data()