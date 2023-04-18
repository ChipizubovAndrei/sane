import numpy as np


from utils import ReLU, SoftMax

"""
Класс модели однослойной ИНС
Аргументы:
    - neuron_list - указатели на ID нейронов, входящих в ИНС
    - neuron_population - Вся популяция нейронов
    - n_input_neurons - Количество выходных нейронов
    - n_output_neurons - Количество выходных нейронов
"""
class Model:

    def __init__(self, neuron_list, neuron_population,
                n_input_neurons, n_output_neurons):

        # Количество нейронов входного слоя
        self.n_input_neurons = n_input_neurons
        # Количество нейронов выходного слоя
        self.n_output_neurons = n_output_neurons
        # Количество нейронов скрытого слоя
        self.n_hidden_neurons = len(neuron_list)

        # Список нейронов входящих в нейронную сеть
        self.neural_network_neurons = neuron_population[neuron_list]     

    """
    Функция для вычисления прямого 
    распространения сигнала по сети
    """
    def forward(self, x):
        W1, W2 = self.create_weight_matrix()
        # return np.argmax(ReLU(np.matmul(ReLU(np.matmul(x, self.W1), self.W2))), dim=1)
        return SoftMax(ReLU(np.matmul(ReLU(np.matmul(x, self.W1), self.W2))))

    """
    Функция для формирования матриц весов
    скрытого и выходного слоя
    """
    def create_weight_matrix(self):

        W1 = np.zeros((self.n_input_neurons, self.n_hidden_neurons))
        W2 = np.zeros((self.n_hidden_neurons, self.n_output_neurons))

        # Заполнение матрицы весов
        for i in range(0, self.n_hidden_neurons):
            hn_connections = self.neural_network_neurons[i]
            for j in range(len(hn_connections) / 2):
                """
                Если ID нейрона относится к входному слою 
                (меньше количества нейронов во входном слое),
                то заполняется матрицы весов скрытого слоя
                """
                if (hn_connections[2*j] < self.n_input_neurons):
                    in_n_id = hn_connections[2*j]
                    W1[in_n_id, i] = hn_connections[2*j+1]
                else:
                    out_n_id = hn_connections[2*j] - self.n_input_neurons
                    W2[i, out_n_id] = hn_connections[2*j+1]
        return W1, W2
