import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# ============================================
# 1. CARREGAR E EXPLORAR OS DADOS
# ============================================

# Carregar dataset
df = pd.read_csv('  cars_dataset.csv')

print("="*60)
print("ANÁLISE EXPLORATÓRIA DO DATASET DE CARROS")
print("="*60)
print(f"\nShape: {df.shape}")
print(f"\nPrimeiras linhas:")
print(df.head(10))

print(f"\nInformações das colunas:")
print(df.info())

print(f"\nDistribuição das classes:")
print(df['car'].value_counts())
print(f"\nPercentual:")
print(df['car'].value_counts(normalize=True) * 100)

print(f"\nValores únicos por coluna:")
for col in df.columns:
    print(f"  {col}: {df[col].unique()}")

# ============================================
# 2. PRÉ-PROCESSAMENTO
# ============================================

print("\n" + "="*60)
print("PRÉ-PROCESSAMENTO DOS DADOS")
print("="*60)

# Criar cópia para não modificar original
df_processed = df.copy()

# Codificar todas as variáveis categóricas
label_encoders = {}
for column in df_processed.columns:
    le = LabelEncoder()
    df_processed[column] = le.fit_transform(df_processed[column])
    label_encoders[column] = le
    print(f"\n{column}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Separar features (X) e target (y)
X = df_processed.drop('car', axis=1).values
y = df_processed['car'].values

print(f"\nShape de X: {X.shape}")
print(f"Shape de y: {y.shape}")
print(f"Classes: {np.unique(y)}")

# ============================================
# 3. DIVIDIR DADOS TREINO/TESTE
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDados de treino: {X_train.shape[0]} amostras")
print(f"Dados de teste: {X_test.shape[0]} amostras")

# ============================================
# 4. TREINAR MODELOS SVM
# ============================================

print("\n" + "="*60)
print("TREINAMENTO DE MODELOS SVM")
print("="*60)

# Diferentes kernels para comparar
kernels = ['linear', 'rbf', 'poly']
models = {}
accuracies = {}

for kernel in kernels:
    print(f"\nTreinando SVM com kernel {kernel}...")
    
    if kernel == 'poly':
        clf = svm.SVC(kernel=kernel, degree=3, random_state=42)
    else:
        clf = svm.SVC(kernel=kernel, random_state=42)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    models[kernel] = clf
    accuracies[kernel] = accuracy
    
    print(f"  Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Vetores de suporte: {len(clf.support_vectors_)}")

# Escolher melhor modelo
best_kernel = max(accuracies, key=accuracies.get)
best_model = models[best_kernel]

print(f"\n🏆 Melhor modelo: {best_kernel} (Acurácia: {accuracies[best_kernel]:.4f})")

# ============================================
# 5. AVALIAÇÃO DETALHADA DO MELHOR MODELO
# ============================================

print("\n" + "="*60)
print(f"AVALIAÇÃO DETALHADA - KERNEL {best_kernel.upper()}")
print("="*60)

y_pred = best_model.predict(X_test)

# Relatório de classificação
print("\nRelatório de Classificação:")
target_names = label_encoders['car'].classes_
print(classification_report(y_test, y_pred, target_names=target_names))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusão:")
print(cm)

# ============================================
# 6. VISUALIZAÇÕES
# ============================================

fig = plt.figure(figsize=(18, 12))

# Subplot 1: Comparação de Acurácias
ax1 = plt.subplot(2, 3, 1)
kernels_list = list(accuracies.keys())
accs = list(accuracies.values())
colors = ['green' if k == best_kernel else 'steelblue' for k in kernels_list]
bars = ax1.bar(kernels_list, accs, color=colors, edgecolor='black', linewidth=2)
ax1.set_ylabel('Acurácia', fontsize=12)
ax1.set_title('Comparação de Kernels SVM', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 1)
for bar, acc in zip(bars, accs):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Subplot 2: Distribuição das Classes
ax2 = plt.subplot(2, 3, 2)
class_counts = df['car'].value_counts()
colors_pie = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
ax2.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
        startangle=90, colors=colors_pie, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax2.set_title('Distribuição das Classes no Dataset', fontsize=14, fontweight='bold')

# Subplot 3: Matriz de Confusão (Heatmap)
ax3 = plt.subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names,
            cbar_kws={'label': 'Contagem'}, ax=ax3)
ax3.set_xlabel('Predito', fontsize=12)
ax3.set_ylabel('Real', fontsize=12)
ax3.set_title(f'Matriz de Confusão - {best_kernel}', fontsize=14, fontweight='bold')

# Subplot 4: Importância das Features (aproximada via permutação)
ax4 = plt.subplot(2, 3, 4)
from sklearn.inspection import permutation_importance
perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
feature_names = ['Buying', 'Maint', 'Doors', 'Persons', 'Lug_boot', 'Safety']
indices = np.argsort(perm_importance.importances_mean)[::-1]
ax4.barh(range(len(indices)), perm_importance.importances_mean[indices], color='coral', edgecolor='black')
ax4.set_yticks(range(len(indices)))
ax4.set_yticklabels([feature_names[i] for i in indices])
ax4.set_xlabel('Importância', fontsize=12)
ax4.set_title('Importância das Features', fontsize=14, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

# Subplot 5: Predições Corretas vs Incorretas
ax5 = plt.subplot(2, 3, 5)
correct = np.sum(y_pred == y_test)
incorrect = len(y_test) - correct
ax5.bar(['Corretas', 'Incorretas'], [correct, incorrect], 
        color=['green', 'red'], edgecolor='black', linewidth=2, alpha=0.7)
ax5.set_ylabel('Quantidade', fontsize=12)
ax5.set_title('Predições do Modelo', fontsize=14, fontweight='bold')
for i, v in enumerate([correct, incorrect]):
    ax5.text(i, v + 5, str(v), ha='center', fontsize=12, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

# Subplot 6: Número de Vetores de Suporte por Kernel
ax6 = plt.subplot(2, 3, 6)
n_support = [len(models[k].support_vectors_) for k in kernels_list]
ax6.bar(kernels_list, n_support, color='purple', edgecolor='black', linewidth=2, alpha=0.6)
ax6.set_ylabel('Nº Vetores de Suporte', fontsize=12)
ax6.set_title('Complexidade dos Modelos', fontsize=14, fontweight='bold')
for i, (k, v) in enumerate(zip(kernels_list, n_support)):
    ax6.text(i, v + 10, str(v), ha='center', fontsize=11, fontweight='bold')
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('svm_car_evaluation_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# 7. TESTAR NOVOS CARROS
# ============================================

print("\n" + "="*60)
print("TESTANDO NOVOS CARROS")
print("="*60)

# Criar exemplos de carros para testar
test_cars = [
    {'buying': 'low', 'maint': 'low', 'doors': 'four', 'persons': 'more', 'lug_boot': 'big', 'safety': 'high'},
    {'buying': 'vhigh', 'maint': 'vhigh', 'doors': 'two', 'persons': 'two', 'lug_boot': 'small', 'safety': 'low'},
    {'buying': 'med', 'maint': 'med', 'doors': 'four', 'persons': 'four', 'lug_boot': 'med', 'safety': 'med'},
    {'buying': 'high', 'maint': 'low', 'doors': 'three', 'persons': 'four', 'lug_boot': 'big', 'safety': 'high'},
]

for i, car in enumerate(test_cars, 1):
    # Codificar
    car_encoded = []
    for col in ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']:
        car_encoded.append(label_encoders[col].transform([car[col]])[0])
    
    # Predizer
    prediction = best_model.predict([car_encoded])[0]
    prediction_label = label_encoders['car'].inverse_transform([prediction])[0]
    
    # Mostrar resultado
    print(f"\n🚗 Carro {i}:")
    print(f"  Características: {car}")
    print(f"  → Avaliação prevista: {prediction_label.upper()}")
    
    # Interpretação
    interpretations = {
        'unacc': '❌ Inaceitável',
        'acc': '✅ Aceitável',
        'good': '⭐ Bom',
        'vgood': '🌟 Muito Bom'
    }
    print(f"  → {interpretations.get(prediction_label, 'Desconhecido')}")

print("\n" + "="*60)
print("ANÁLISE CONCLUÍDA!")
print("="*60)