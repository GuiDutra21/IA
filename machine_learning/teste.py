import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

# Dividir os dados de treino e teste, 20% para teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "="*70)
print("TREINAMENTO DOS MODELOS SVM")
print("="*70)

# Diferentes kernels para comparar (SVM)
kernels = ['linear', 'rbf', 'poly']
modelos_svm = {}
acuracias_svm = {}

# Treina o modelo SVM para cada kernel
for kernel in kernels:
    print(f"\nTreinando SVM com kernel {kernel}...")

    if kernel == 'poly':
        clf = svm.SVC(kernel=kernel, degree=3, random_state=42)
    else:
        clf = svm.SVC(kernel=kernel, random_state=42)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acuracia = accuracy_score(y_test, y_pred)
    modelos_svm[kernel] = clf
    acuracias_svm[kernel] = acuracia
    
    print(f"  Acurácia: {acuracia:.4f} ({acuracia*100:.2f}%)")

# Escolhe o melhor modelo SVM
melhor_kernel_svm = max(acuracias_svm, key=acuracias_svm.get)
melhor_modelo_svm = modelos_svm[melhor_kernel_svm]

print(f"\nMelhor SVM: {melhor_kernel_svm} (Acurácia: {acuracias_svm[melhor_kernel_svm]:.4f})")

print("\n" + "="*70)
print("TREINAMENTO DO RANDOM FOREST")
print("="*70)

# Treina Random Forest com diferentes configurações
rf_configs = [
    {'n_estimators': 50, 'max_depth': 5},
    {'n_estimators': 100, 'max_depth': 10},
    {'n_estimators': 200, 'max_depth': None}
]

modelos_rf = {}
acuracias_rf = {}

for i, config in enumerate(rf_configs):
    print(f"\nTreinando Random Forest (n_estimators={config['n_estimators']}, max_depth={config['max_depth']})...")
    
    clf_rf = RandomForestClassifier(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        random_state=42
    )
    
    clf_rf.fit(X_train, y_train)
    y_pred_rf = clf_rf.predict(X_test)
    
    acuracia_rf = accuracy_score(y_test, y_pred_rf)
    nome_config = f"RF_{config['n_estimators']}_{config['max_depth']}"
    modelos_rf[nome_config] = clf_rf
    acuracias_rf[nome_config] = acuracia_rf
    
    print(f"  Acurácia: {acuracia_rf:.4f} ({acuracia_rf*100:.2f}%)")

# Escolhe o melhor modelo Random Forest
melhor_config_rf = max(acuracias_rf, key=acuracias_rf.get)
melhor_modelo_rf = modelos_rf[melhor_config_rf]

print(f"\nMelhor Random Forest: {melhor_config_rf} (Acurácia: {acuracias_rf[melhor_config_rf]:.4f})")

print("\n" + "="*70)
print("COMPARAÇÃO FINAL")
print("="*70)
print(f"SVM ({melhor_kernel_svm}): {acuracias_svm[melhor_kernel_svm]:.4f}")
print(f"Random Forest ({melhor_config_rf}): {acuracias_rf[melhor_config_rf]:.4f}")

# Predições dos melhores modelos
y_pred_svm = melhor_modelo_svm.predict(X_test)
y_pred_rf = melhor_modelo_rf.predict(X_test)

# Matrizes de confusão
matriz_svm = confusion_matrix(y_test, y_pred_svm)
matriz_rf = confusion_matrix(y_test, y_pred_rf)

print("\n--- Relatório SVM ---")
target_names = label_encoders['car'].classes_
print(classification_report(y_test, y_pred_svm, target_names=target_names))

print("\n--- Relatório Random Forest ---")
print(classification_report(y_test, y_pred_rf, target_names=target_names))

# VISUALIZAÇÕES COMPARATIVAS
fig = plt.figure(figsize=(16, 10))

# 1. Comparação de Acurácia entre todos os modelos
ax1 = plt.subplot(2, 3, 1)
todos_modelos = list(acuracias_svm.keys()) + list(acuracias_rf.keys())
todas_acuracias = list(acuracias_svm.values()) + list(acuracias_rf.values())
cores = ['blue']*len(acuracias_svm) + ['green']*len(acuracias_rf)

bars = ax1.bar(range(len(todos_modelos)), todas_acuracias, color=cores, edgecolor='black', alpha=0.7)
ax1.set_xticks(range(len(todos_modelos)))
ax1.set_xticklabels(todos_modelos, rotation=45, ha='right')
ax1.set_ylabel('Acurácia', fontsize=12)
ax1.set_title('Comparação de Acurácia: SVM vs Random Forest', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=max(todas_acuracias), color='red', linestyle='--', alpha=0.5)

# Legenda
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='blue', alpha=0.7, label='SVM'),
                   Patch(facecolor='green', alpha=0.7, label='Random Forest')]
ax1.legend(handles=legend_elements, loc='lower right')

# 2. Importância das Features - SVM
ax2 = plt.subplot(2, 3, 2)
perm_importance_svm = permutation_importance(melhor_modelo_svm, X_test, y_test, n_repeats=10, random_state=42)
feature_names = ['Buying', 'Maint', 'Doors', 'Persons', 'Lug_boot', 'Safety']
indices_svm = np.argsort(perm_importance_svm.importances_mean)[::-1]
ax2.barh(range(len(indices_svm)), perm_importance_svm.importances_mean[indices_svm], color='blue', edgecolor='black')
ax2.set_yticks(range(len(indices_svm)))
ax2.set_yticklabels([feature_names[i] for i in indices_svm])
ax2.set_xlabel('Importância', fontsize=12)
ax2.set_title(f'Importância das Features - SVM ({melhor_kernel_svm})', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# 3. Importância das Features - Random Forest
ax3 = plt.subplot(2, 3, 3)
# Random Forest tem importância nativa
importancias_rf = melhor_modelo_rf.feature_importances_
indices_rf = np.argsort(importancias_rf)[::-1]
ax3.barh(range(len(indices_rf)), importancias_rf[indices_rf], color='green', edgecolor='black')
ax3.set_yticks(range(len(indices_rf)))
ax3.set_yticklabels([feature_names[i] for i in indices_rf])
ax3.set_xlabel('Importância', fontsize=12)
ax3.set_title(f'Importância das Features - Random Forest', fontsize=14, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# 4. Matriz de Confusão - SVM
ax4 = plt.subplot(2, 3, 4)
sns.heatmap(matriz_svm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names, ax=ax4, cbar=False)
ax4.set_xlabel('Predito', fontsize=12)
ax4.set_ylabel('Real', fontsize=12)
ax4.set_title(f'Matriz de Confusão - SVM\nAcurácia: {acuracias_svm[melhor_kernel_svm]:.4f}', 
              fontsize=14, fontweight='bold')

# 5. Matriz de Confusão - Random Forest
ax5 = plt.subplot(2, 3, 5)
sns.heatmap(matriz_rf, annot=True, fmt='d', cmap='Greens', 
            xticklabels=target_names, yticklabels=target_names, ax=ax5, cbar=False)
ax5.set_xlabel('Predito', fontsize=12)
ax5.set_ylabel('Real', fontsize=12)
ax5.set_title(f'Matriz de Confusão - Random Forest\nAcurácia: {acuracias_rf[melhor_config_rf]:.4f}', 
              fontsize=14, fontweight='bold')

# 6. Predições Corretas vs Incorretas - Comparação
ax6 = plt.subplot(2, 3, 6)
corretas_svm = np.sum(y_pred_svm == y_test)
incorretas_svm = len(y_test) - corretas_svm
corretas_rf = np.sum(y_pred_rf == y_test)
incorretas_rf = len(y_test) - corretas_rf

x = np.arange(2)
width = 0.35

bars1 = ax6.bar(x - width/2, [corretas_svm, incorretas_svm], width, 
                label='SVM', color='blue', edgecolor='black', alpha=0.7)
bars2 = ax6.bar(x + width/2, [corretas_rf, incorretas_rf], width,
                label='Random Forest', color='green', edgecolor='black', alpha=0.7)

ax6.set_ylabel('Quantidade', fontsize=12)
ax6.set_title('Predições: SVM vs Random Forest', fontsize=14, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(['Corretas', 'Incorretas'])
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

# Adicionar valores nas barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("ANÁLISE COMPARATIVA")
print("="*70)

if acuracias_svm[melhor_kernel_svm] > acuracias_rf[melhor_config_rf]:
    diff = acuracias_svm[melhor_kernel_svm] - acuracias_rf[melhor_config_rf]
    print(f"✓ SVM ({melhor_kernel_svm}) é MELHOR por {diff:.4f} ({diff*100:.2f}%)")
elif acuracias_rf[melhor_config_rf] > acuracias_svm[melhor_kernel_svm]:
    diff = acuracias_rf[melhor_config_rf] - acuracias_svm[melhor_kernel_svm]
    print(f"✓ Random Forest ({melhor_config_rf}) é MELHOR por {diff:.4f} ({diff*100:.2f}%)")
else:
    print("✓ Empate técnico entre os modelos")

print(f"\nPredições corretas:")
print(f"  SVM: {corretas_svm}/{len(y_test)}")
print(f"  Random Forest: {corretas_rf}/{len(y_test)}")