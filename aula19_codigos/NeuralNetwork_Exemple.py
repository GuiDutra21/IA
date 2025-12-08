# Libraries used
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Creating random samples with 4 labels
n_samples = 200
blob_centers = ([1, 1], [3, 4], [1, 3.3], [3.5, 1.8])
data, labels = make_blobs(n_samples=n_samples, 
                          centers=blob_centers, 
                          cluster_std=0.5,
                          random_state=0)

# Print samples
colours = ('green', 'orange', "blue", "magenta")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for n_class in range(len(blob_centers)):
    ax1.scatter(data[labels==n_class][:, 0], 
               data[labels==n_class][:, 1], 
               c=colours[n_class], 
               s=30, 
               label=str(n_class))

ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_title('Dados Originais (4 Clusters)')
ax1.legend()

#Creates training set and testing set
from sklearn.model_selection import train_test_split
datasets = train_test_split(data, 
                            labels,
                            test_size=0.2,
                            random_state=42)

train_data, test_data, train_labels, test_labels = datasets

#Creates the Feedforward Neural Network.
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(activation='relu', solver='lbfgs', 
                    alpha=1e-5,
                    hidden_layer_sizes=(6,), 
                    random_state=1)

clf.fit(train_data, train_labels) 
'''

activation: {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’
Activation function for the hidden layer.
‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
‘relu’, the rectified linear unit function, returns f(x) = max(0, x)

solver: The weight optimization can be influenced with the solver parameter. 
Three solver modes are available:
'lbfgs' is an optimizer in the family of quasi-Newton methods.
'sgd' refers to stochastic gradient descent.
'adam' refers to a stochastic gradient-based optimizer proposed by Kingma, 
       Diederik, and Jimmy Ba

hidden_layer_sizes: tuple, length = n_layers - 2, default=(100,)
The ith element represents the number of neurons in the ith hidden layer.
(6,) means one hidden layer with 6 neurons

alpha: This parameter can be used to control possible 'overfitting' and 
'underfitting'. Increasing alpha may fix high variance (a sign of overfitting) 
by encouraging smaller weights, resulting in a decision boundary plot that 
appears with lesser curvatures. Similarly, decreasing alpha may fix high bias 
(a sign of underfitting) by encouraging larger weights, potentially resulting 
in a more complicated decision boundary.
'''

from sklearn.metrics import accuracy_score

predictions_train = clf.predict(train_data)
predictions_test = clf.predict(test_data)

train_score = accuracy_score(predictions_train, train_labels)
print("score on train data: ", train_score)

test_score = accuracy_score(predictions_test, test_labels)
print("score on test data: ", test_score)

import numpy as np
# Cria grid de pontos
x_min, x_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
y_min, y_max = data[:, 1].min() - 0.5, data[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Prediz classe para cada ponto do grid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plota
ax2.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
ax2.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', 
            edgecolors='k', s=50)
ax2.set_title('Fronteiras de Decisão da Rede Neural')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
plt.show()