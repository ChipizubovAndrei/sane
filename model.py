import numpy as np


class Model:
    """
    Класс модели однослойной ИНС
    Аргументы:
        - neural_network_neurons - параметры нейронной сети
        - n_input_neurons - Количество входных нейронов
        - n_output_neurons - Количество выходных нейронов
        - n_hidden_neurons - количество нейронов скрытого слоя
    """
    def __init__(
        self, neural_network_neurons,
        n_input_neurons, n_output_neurons, 
        n_hidden_neurons
    ):

        # Количество нейронов входного слоя
        self.n_input_neurons = n_input_neurons
        # Количество нейронов выходного слоя
        self.n_output_neurons = n_output_neurons
        # Количество нейронов скрытого слоя
        self.n_hidden_neurons = n_hidden_neurons

        # Список нейронов входящих в нейронную сеть
        self.neural_network_neurons = neural_network_neurons # neuron_population[neuron_list]     


    def forward(self, x, f='relu'):
        """
        Функция для вычисления прямого 
        распространения сигнала по сети
        """
        if f == 'sigmoid':
            act = self.Sigmoid
        else:
            act = self.ReLU
        
        W1, W2 = self.create_weight_matrix()
        return self.SoftMax(np.matmul(act(np.matmul(x, W1)), W2))

    def ReLU(self, x):
        """
        Функция активации ReLU
        """
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i, j] < 0:
                    x[i, j] = 0
        return x

    def Sigmoid(self, x):
        """
        Функция активации ReLU
        """
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i, j] =  1 / (1 + np.exp(-x[i, j]))
        return x

    def SoftMax(self, x):
        """
        Функция активации SoftMax
        """
        out = np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                out[i, j] = np.exp(x[i, j]) / np.sum(np.exp(x[i]))
        return out
    
    def create_weight_matrix(self):
        """
        Функция для формирования матриц весов
        скрытого и выходного слоя
        """
        W1 = np.zeros((self.n_input_neurons, self.n_hidden_neurons))
        W2 = np.zeros((self.n_hidden_neurons, self.n_output_neurons))

        # Заполнение матрицы весов
        for i in range(0, self.n_hidden_neurons):
            hn_connections = self.neural_network_neurons[i]
            for j in range(0, len(hn_connections), 2):
                """
                Если ID нейрона относится к входному слою 
                (меньше количества нейронов во входном слое),
                то заполняется матрицы весов скрытого слоя
                """
                if (hn_connections[j] < self.n_input_neurons):
                    in_n_id = int(hn_connections[j])
                    W1[in_n_id, i] = hn_connections[j+1]
                else:
                    out_n_id = int(hn_connections[j] - self.n_input_neurons)
                    W2[i, out_n_id] = hn_connections[j+1]
        return W1, W2
