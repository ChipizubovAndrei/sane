import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import normalize
from sklearn.metrics import log_loss, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


from sane import Sane
from model import Model

"""
Датасет и подготовка данных
"""
data = load_wine()
X = data["data"]
Y = data["target"]

# Нормализация входных векторов
X = normalize(X, norm='max')
X, Y = shuffle(X, Y, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                            test_size=0.2, 
                                            random_state=0)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                            test_size=0.4, 
                                            random_state=0)

"""
Общие настройки
"""
# Число эпох
n_epoches = 10000
# Общее количество нейронов в популяции
total_number_neuron = 2000
# Общее количество комбинаций нейронов
total_number_blueprints = 500
# Количество скрытых нейронов в нейронной сети
number_hidden_neurons = 10
# Количество связей у нейрона
number_neuron_connection = 8

patience = 30

# Количество нейронов входного слоя
num_input_neurons = X.shape[1]
# Количество нейронов выходного слоя
num_output_neurons = np.unique(Y).shape[0]

model_id = 1

ga_instance = Sane(
    n_epoches, log_loss, 
    total_number_neuron, 
    num_input_neurons,
    num_output_neurons,
    number_hidden_neurons,
    number_neuron_connection, 
    total_number_blueprints,
    patience=patience, act_func='relu'
)

_, loss_arr = ga_instance.run(X_train, y_train, X_val, y_val, model_id)

best_model = np.load(f"./outputs/models/model_{model_id}.npy")
model = Model(
            best_model, 
            num_input_neurons, 
            num_output_neurons, 
            number_hidden_neurons
        )
preds = model.forward(X_train)

print(f"Loss train = {log_loss(y_train, preds)}")
print(f"F1-measure train = {f1_score(y_train, np.argmax(preds, axis=1), average='micro')}")

preds = model.forward(X_test)
print(f"Loss test = {log_loss(y_test, preds)}")
print(f"F1-measure test = {f1_score(y_test, np.argmax(preds, axis=1), average='micro')}")

preds = model.forward(X_val)
print(f"Loss val = {log_loss(y_val, preds)}")
print(f"F1-measure val = {f1_score(y_val, np.argmax(preds, axis=1), average='micro')}")