import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# Codificar a coluna Gender (sexo) para números
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['Gender'] = le.fit_transform(dataset['Gender'])  # Male=1, Female=0 (ou vice-versa)

# Agora incluir Gender (coluna 1), Age (coluna 2) e EstimatedSalary (coluna 3)
X = dataset.iloc[:, [1, 2, 3]].values  # ← mudou de [2,3] para [1,2,3]
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 153)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) # aprende a escala do treino e transforma
X_test = sc.transform(X_test) #  aplica a mesma escala no teste (não aprende de novo!)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
ac = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Acurácia: {ac:.2f}")
print(f"Matriz de Confusão:\n{cm}")