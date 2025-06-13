import os
import cv2
import sys
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Définir le chemin du dossier de données
zip_path = "amhcd-data-64.zip"
extract_path = "amhcd-data-64"

# Extraire le dataset si nécessaire
if not os.path.exists(extract_path):
    if os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(extract_path))
                print(f"Extracted dataset to {extract_path}")
        except zipfile.BadZipFile:
            print(f"Error: Corrupted ZIP file at {zip_path}")
    else:
        print(f"ZIP file not found at {zip_path}")
else:
    print(f"Dataset already exists at {extract_path}")

# Path to use later
data_dir = extract_path

# Fonction de prétraitement des images
def load_and_preprocess_image(image_path, data_dir, target_size=(32, 32)):
    """Load and preprocess image: grayscale, resize, normalize"""
    full_path = os.path.join(data_dir, image_path)
    assert os.path.exists(full_path), f"Image not found: {full_path}"
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Failed to load image: {full_path}"
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return img.flatten()

# Charger les étiquettes depuis labels-map.csv
labels_csv_path = os.path.join(data_dir, "labels-map.csv")
if not os.path.exists(labels_csv_path):
    print(f"labels-map.csv not found. Building DataFrame from directory.")
    image_paths, labels = [], []
    tifinagh_dir = os.path.join(data_dir, "tifinagh-images")
    if not os.path.exists(tifinagh_dir):
        raise FileNotFoundError(f"Tifinagh images directory not found: {tifinagh_dir}")
    for label_dir in os.listdir(tifinagh_dir):
        label_path = os.path.join(tifinagh_dir, label_dir)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                image_paths.append(os.path.join("tifinagh-images", label_dir, img_name))
                labels.append(label_dir)
    labels_df = pd.DataFrame({'image_path': image_paths, 'label': labels})
else:
    print(f"Loading labels from {labels_csv_path}")
    labels_df = pd.read_csv(labels_csv_path, names=['image_path', 'label'])

assert not labels_df.empty, "No data loaded. Check dataset files."
labels_df['image_path'] = labels_df['image_path'].apply(lambda x: x.replace('./images-data-64/', '') if isinstance(x, str) else x)

# Debugging: Afficher des exemples de chemins d'images
print("Sample image paths:")
for path in labels_df['image_path'].head(5):
    full_path = os.path.join(data_dir, path)
    print(f"{full_path} -> Exists: {os.path.exists(full_path)}")

print(f"\nLoaded {len(labels_df)} samples with {labels_df['label'].nunique()} unique classes.")

# Analyse de la distribution des classes
class_counts = labels_df['label'].value_counts()
print("Class distribution summary:")
print(f"Min: {class_counts.min()}, Max: {class_counts.max()}, Mean: {class_counts.mean():.2f}")

# Encodage des étiquettes
print("Label encoding ...")
label_encoder = LabelEncoder()
labels_df['label_encoded'] = label_encoder.fit_transform(labels_df['label'])
num_classes = len(label_encoder.classes_)

print("==================== Data with label encoded: ======================")
print(labels_df)

# Charger toutes les images
X = np.array([load_and_preprocess_image(path, data_dir) for path in labels_df['image_path']])
y = labels_df['label_encoded'].values

# Vérifier les dimensions
assert X.shape[0] == y.shape[0], "Mismatch between number of images and labels"
assert X.shape[1] == 32 * 32, f"Expected flattened image size of {32 * 32}, got {X.shape[1]}"

# Diviser en ensembles d'entraînement, validation et test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

one_hot_encoder = OneHotEncoder(sparse_output=False)
y_train_one_hot = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
y_val_one_hot = one_hot_encoder.transform(y_val.reshape(-1, 1))
y_test_one_hot = one_hot_encoder.transform(y_test.reshape(-1, 1))
print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# Définir le modèle de réseau de neurones multiclasses
class MulticlassNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01 
                       for i in range(len(layer_sizes) - 1)]
        self.biases = [np.zeros((1, layer_sizes[i+1])) 
                      for i in range(len(layer_sizes) - 1)]
        self.z_values = []
        self.activations = []

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stabilité numérique
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.z_values = []
        self.activations = [X]
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            self.activations.append(self.relu(z))
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        output = self.softmax(z)
        self.activations.append(output)
        return output

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        epsilon = 1e-15  # Éviter log(0)
        return -np.sum(y_true * np.log(y_pred + epsilon)) / m

    def compute_accuracy(self, y_true, y_pred):
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        return np.mean(predictions == true_labels)

    def backward(self, X, y, outputs):
        m = X.shape[0]
        self.d_weights = [np.zeros_like(w) for w in self.weights]
        self.d_biases = [np.zeros_like(b) for b in self.biases]
        dZ = outputs - y
        self.d_weights[-1] = np.dot(self.activations[-2].T, dZ) / m
        self.d_biases[-1] = np.sum(dZ, axis=0, keepdims=True) / m
        for i in range(len(self.weights) - 2, -1, -1):
            dZ = np.dot(dZ, self.weights[i+1].T) * self.relu_derivative(self.z_values[i])
            self.d_weights[i] = np.dot(self.activations[i].T, dZ) / m
            self.d_biases[i] = np.sum(dZ, axis=0, keepdims=True) / m
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.d_weights[i]
            self.biases[i] -= self.learning_rate * self.d_biases[i]

    def train(self, X, y, X_val, y_val, epochs, batch_size):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            epoch_loss = 0
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                outputs = self.forward(X_batch)
                epoch_loss += self.compute_loss(y_batch, outputs)
                self.backward(X_batch, y_batch, outputs)
            train_loss = epoch_loss / (X.shape[0] // batch_size)
            train_pred = self.forward(X)
            train_accuracy = self.compute_accuracy(y, train_pred)
            val_pred = self.forward(X_val)
            val_loss = self.compute_loss(y_val, val_pred)
            val_accuracy = self.compute_accuracy(y_val, val_pred)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        return train_losses, val_losses, train_accuracies, val_accuracies

    def predict(self, X):
        outputs = self.forward(X)
        return np.argmax(outputs, axis=1)

# Créer et entraîner le modèle
layer_sizes = [X_train.shape[1], 64, 32, num_classes]  # 1024 entrées, 64 et 32 neurones cachés, 33 classes
nn = MulticlassNeuralNetwork(layer_sizes, learning_rate=0.01)
train_losses, val_losses, train_accuracies, val_accuracies = nn.train(
    X_train, y_train_one_hot, X_val, y_val_one_hot, epochs=100, batch_size=32
)

# Évaluation
y_pred = nn.predict(X_test)
print("\nRapport de classification (Test set) :")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de confusion (Test set)')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.savefig('confusion_matrix.png')
plt.close()

# Courbes de perte et précision
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses, label='Validation Loss')
ax1.set_title('Courbe de perte')
ax1.set_xlabel('Époque')
ax1.set_ylabel('Perte')
ax1.legend()
ax2.plot(train_accuracies, label='Train Accuracy')
ax2.plot(val_accuracies, label='Validation Accuracy')
ax2.set_title('Courbe de précision')
ax2.set_xlabel('Époque')
ax2.set_ylabel('Précision')
ax2.legend()
plt.tight_layout()
plt.savefig('loss_accuracy_plot.png')
plt.close()