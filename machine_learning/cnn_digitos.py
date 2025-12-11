'''
CNN para Reconhecimento de Dígitos Escritos à Mão (MNIST)
Dataset: 60.000 treino + 10.000 teste
Classes: 0-9 (10 dígitos)
Imagens: 28×28 pixels (grayscale)
'''

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

(X_train,y_train), (X_test, y_test) = mnist.load_data()

# Reshape: adiciona canal (28,28) → (28,28,1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Normalização: [0-255] → [0-1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encoding: 5 → [0,0,0,0,0,1,0,0,0,0]
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Arquitetura da cnn
model = Sequential()

# Bloco Convolucional 1
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Bloco Convolucional 2
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Camadas Densas
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))  # 10 classes (0-9)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

callbacks = [early_stop, reduce_lr]

# Treinamento
epochs = 20
batch_size = 128

history = model.fit(
    X_train, y_train_cat,
    validation_split=0.1,  # 10% dos dados de treino para validação
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks,
    verbose=1
)

model.save('mnist_cnn_model.h5')

# Avaliação no conjunto de teste e predições
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)


# Estatisticas
erros = np.sum(y_pred_classes != y_test)
acertos = len(y_test) - erros

print(f"\nTotal de testes: {len(y_test)}")
print(f"Acertos: {acertos} ({acertos/len(y_test)*100:.2f}%)")
print(f"Erros: {erros} ({erros/len(y_test)*100:.2f}%)")

# Acurácia por dígito
print("\nAcurácia por dígito:")
for digit in range(10):
    mask = y_test == digit
    digit_accuracy = np.sum((y_pred_classes == y_test) & mask) / np.sum(mask)
    print(f"  Dígito {digit}: {digit_accuracy:.4f} ({digit_accuracy*100:.2f}%)")
    
# Matriz de confusão
cm = confusion_matrix(y_test, y_pred_classes)

# Plot da Matriz de Confusão
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predito', fontsize=14)
plt.ylabel('Real', fontsize=14)
plt.title('Matriz de Confusão - MNIST CNN', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()