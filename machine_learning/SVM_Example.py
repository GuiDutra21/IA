import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Carrega dataset
dados = pd.read_csv('cars_dataset.csv')

# Codifica todas as variáveis em numeros, pois o SVM so trabalha com valores numericos
label_encoders = {}
for coluna in dados.columns:
    le = LabelEncoder()
    dados[coluna] = le.fit_transform(dados[coluna])
    label_encoders[coluna] = le
    print(f"\n{coluna}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
# Separa features (X) e classes (y)
X = dados.drop('car', axis=1).values
y = dados['car'].values

# Dvidir os dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Diferentes kernels para comparar
kernels = ['linear', 'rbf', 'poly']
modelos = {}
acuracias = {}

# Treina o modelo para cada kernel
for kernel in kernels:
    print(f"\nTreinando SVM com kernel {kernel}...")

    if kernel == 'poly':
        clf = svm.SVC(kernel=kernel, degree=3, random_state=42)
    else:
        clf = svm.SVC(kernel=kernel, random_state=42)

    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)

    acuracia = accuracy_score(y_test, y_pred)
    modelos[kernel] = clf
    acuracias[kernel] = acuracia
    
    print(f"  Acurácia: {acuracia:.4f} ({acuracia*100:.2f}%)")

# Escolhe o melhor modelo
melhor_kernel = max(acuracias,key=acuracias.get)
melhor_modelo = modelos[melhor_kernel]

print(f"\nMelhor modelo: {melhor_kernel} (Acurácia: {acuracias[melhor_kernel]:.4f})")

# Matriz de confusão
matriz = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusão:")
print(matriz)