import numpy as np
import pygad
from sklearn.datasets import load_wine
from sklearn.preprocessing import normalize
from sklearn.metrics import log_loss

"""
Датасет и погдотовка данных
"""
data = load_wine()
X = data["data"]
Y = data["target"]


# Количество входных и выходных нейронов
number_input_neurons = X.shape[1]
number_output_neurons = Y.shape[0]

# Нормализация входных векторов
X = normalize(X, norm='max')

"""
Общие настройки
"""
# Общее количество нейронов в популяции
total_number_neuron = 100
# Общее количество комбинаций нейронов
total_number_blueprints = 20
# Количество скрытых нейронов в нейронной сети
number_hidden_neurons = 10
# Количество связей у нейрона
number_neuron_connection = 20

"""
Настройки генетического алгоритма для популяции нейронов
"""
# Список возможных значений индексов нейронов
neuron_index_range = [_ for _ in range(0, number_input_neurons + number_output_neurons)]
# Ограничение на возможные значения весов связей
min_max_weigths_values = {"low": -100, "high": 100}
# Список с ограничениями на типы значений в генах
neuron_gene_type = [None for _ in range(number_neuron_connection*2)]
for i in range(number_hidden_neurons):
    if (i % 2 == 0):
        neuron_gene_type[i] = int
    else:
        neuron_gene_type[i] = float

"""
Настройки генетического алгоритма для популяции комбинаций нейронов (ИНС)
"""
# Список возможных значений индексов нейронов
neuron_index_range = [_ for _ in range(0, number_hidden_neurons)]
# Список с ограничениями на типы значений в генах
neuron_gene_type = [int for _ in range(number_hidden_neurons)]
