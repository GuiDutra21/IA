from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target
gnb = GaussianNB()
gnb.fit(X, y)
predicted = gnb.predict(X)
print(accuracy_score(y, predicted))