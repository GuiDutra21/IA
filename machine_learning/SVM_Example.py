import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
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

# Dvidir os dados de treino e teste, 20% para teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Diferentes kernels para comparar
kernels = ['linear', 'rbf', 'poly']
modelos_svm = {}
acuracias_svm = {}

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
    modelos_svm[kernel] = clf
    acuracias_svm[kernel] = acuracia
    
    print(f"  Acurácia: {acuracia:.4f} ({acuracia*100:.2f}%)")

# Escolhe o melhor modelo
melhor_kernel_svm = max(acuracias_svm,key=acuracias_svm.get)
melhor_modelo_svm = modelos_svm[melhor_kernel_svm]

print(f"\nMelhor modelo: {melhor_kernel_svm} (Acurácia: {acuracias_svm[melhor_kernel_svm]:.4f})")

y_pred = melhor_modelo_svm.predict(X_test)

# Matriz de confusão
matriz = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusão:")
print(matriz)

# Relatório de classificação
print("\nRelatório de Classificação:")
target_names = label_encoders['car'].classes_
print(classification_report(y_test, y_pred, target_names=target_names))

fig = plt.figure(figsize=(12, 6))

# Subplot 4: Importância das Features (aproximada via permutação)
ax4 = plt.subplot(1, 2, 1)
perm_importance = permutation_importance(melhor_modelo_svm, X_test, y_test, n_repeats=10, random_state=42)
feature_names = ['Buying', 'Maint', 'Doors', 'Persons', 'Lug_boot', 'Safety']
indices = np.argsort(perm_importance.importances_mean)[::-1]
ax4.barh(range(len(indices)), perm_importance.importances_mean[indices], color='blue', edgecolor='black')
ax4.set_yticks(range(len(indices)))
ax4.set_yticklabels([feature_names[i] for i in indices])
ax4.set_xlabel('Importância', fontsize=12)
ax4.set_title('Importância das Features', fontsize=14, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

# Subplot 5: Predições Corretas vs Incorretas
ax5 = plt.subplot(1, 2, 2)
corretas = np.sum(y_pred == y_test)
incorretas = len(y_test) - corretas
ax5.bar(['Corretas', 'Incorretas'], [corretas, incorretas], 
        color=['green', 'red'], edgecolor='black', linewidth=2, alpha=0.7)
ax5.set_ylabel('Quantidade', fontsize=12)
ax5.set_title('Predições do Modelo', fontsize=14, fontweight='bold')

for i, v in enumerate([corretas, incorretas]):
    ax5.text(i, v + 5, str(v), ha='center', fontsize=12, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

plt.show()