import numpy as np


from model import Model

"""
Класс нейроэволюционного алгоритма Sane
Аргументы:
    - n_epoches - количество эволюционных эпох
    - fitness_func - Оптимизуруемая функция
    - num_neuron_pop - Количество нейронов в популяции
    - num_hidden_neurons - Количество скрытых нейронов в сети
    - num_neuron_connect - Количество связей нейрона (= кол-во генов*2 )
    - num_blueprints - Количетство сохраняемых лучших ИНС
    - X - набор входных данных (narray)
    - Y - набор целевых значений (narray)
"""
class Sane:
    def __init__(self, n_epoches, fitness_func, num_neuron_pop, 
                num_hidden_neurons, num_neuron_connect,
                num_blueprints, X, Y):
        try:
            self.n_epoches = n_epoches
            self.fitness_func = fitness_func
            self.num_neuron_pop = num_neuron_pop
            self.num_hidden_neurons = num_hidden_neurons
            self.num_neuron_connect = num_neuron_connect
            self.num_blueprints = num_blueprints
            self.X = X
            self.Y = Y
            # Количество нейронов входного слоя
            self.num_input_neurons = X.shape[1]
            # Количество нейронов выходного слоя
            self.num_output_neurons = np.unique(Y).shape[0]

            if (num_neuron_connect > (self.num_input_neurons + 
                                        self.num_output_neurons)):
                raise Exception("Number The number of neuron connections is greater than the number of input and output neurons")
        except Exception as e:
            print(str(e))

    """Функция оптимизации весов"""
    def run(self):

        count = 0
        # Генерация популяции нейронов
        population = self.generate_population()
        
        # Приспособленности ИНС
        blueprints = np.zeros((self.num_blueprints, 
                                        self.num_hidden_neurons))
        while (count < self.n_epoches):
            count += 1
            """
            1. Сбрасываем приспособленности нейронов.
            """
            # Приспособленности ИНС
            blueprint_fitness = np.zeros(self.num_blueprints) + 1000
            # Приспособленности нейронов
            neuron_fitness = np.zeros(population.shape[0])
            # Количество вхождений нейронов в ИНС
            num_neuron_include = np.zeros(population.shape[0])
            """
            1.1 Пересчитать приспособленности комбинаций 
            """
            for bp_id in range(blueprints.shape[0]):
                model = Model(blueprints[bp_id], population, 
                            self.num_input_neurons, self.num_output_neurons)
                preds = model.forward(self.X)
                loss = self.fitness_func(self.Y, preds)
                blueprint_fitness[bp_id] = loss
            """
            2. Найти среднее значение приспособленности нейронов
            """
            for _ in range(self.num_blueprints):
                """
                2.1. Случайным образом выберается ~ нейронов из популяции.
                """
                random_NN = np.random.randint(0, self.num_hidden_neurons)
                # Увеличить число вхождений для каждого участвующего нейрона
                for n_id in np.unique(random_NN):
                    num_neuron_include[n_id] += 1
                """
                2.2. Создается нейронная сеть из выбранных нейронов.
                """
                model = Model(random_NN, population, 
                                self.num_input_neurons, self.num_output_neurons)
                """
                2.3. Оценка сети на задаче
                """
                preds = model.forward(self.X)
                loss = self.fitness_func(self.Y, preds)
                blueprints, blueprint_fitness = self.update_blueprints(blueprints, 
                                                                        blueprint_fitness, 
                                                                        random_NN, loss)
                """
                2.4. Добавить оценку сети к приспособленности каждого нейрона, 
                входящего в сеть.
                """
                neuron_fitness = self.update_neuron_fitness(neuron_fitness, 
                                                            random_NN, loss)
            """
            3. Усреднить полученные значения приспособленности нейронов
            """
            neuron_fitness /= num_neuron_include
            """
            4. Сортировка популяции нейронов по приспособленности
            """
            sort_id = np.argsort(neuron_fitness)
            population = population[sort_id]
            neuron_fitness = neuron_fitness[sort_id]

    """
    Функция обновления комбинаций нейронов (ИНС)
    Аргументы:
        - blueprints - Массив комбинаций нейронов (narray)
        - b_fit - Массив приспособленностей комбинаций (narray)
        - n_ids_in - Массив индексов нейронов вошедших в нейронную сеть (narray)
        - loss - Значение приспособленности нейронной сети
    """
    def update_blueprints(self, blueprints, b_fit, n_ids_in,  loss):
        for i in range(b_fit.shape[0]):
            if (b_fit[i] > loss):
                blueprints[i] = n_ids_in
                b_fit[i] = loss
        return blueprints, b_fit

    """
    Функция обновления значений приспособленности нейронов
    Аргументы:
        - n_fit - Массив приспособленностей нейронов (narray)
        - n_ids_in - Массив индексов нейронов вошедших в нейронную сеть (narray)
        - loss - Значение приспособленности нейронной сети
    """
    def update_neuron_fitness(self, n_fit, n_ids_in, loss):
        for i in np.unique(n_ids_in):
            n_fit[i] += loss
        return n_fit

    """
    Функция мутации нейрона
    """
    def neuron_mutation(self):
        None

    """
    Функция скрещивания нейронов
    """
    def neuron_crossover(self):
        None

    """
    Функция мутации ИНС
    """
    def blueprint_mutation(self):
        None

    """
    Функция скрещивания ИНС
    """
    def blueprint_crossover(self):
        None

    """
    Функция инициализации популяции нейронов
    """
    def generate_population(self):

        # Добавочный коэффициент, чтобы при заполнениями индексами не было повторяющихся
        io_neuron = self.num_input_neurons + self.num_output_neurons
        population = np.zeros((self.num_neuron_pop, 
                                self.num_neuron_connect*2)) + io_neuron
        for i in range(0, self.num_neuron_pop):
            for j in range(0, self.num_neuron_connect):
                neuron_id = np.random.randint(0, self.num_input_neurons + 
                                                        self.num_output_neurons)
                if (neuron_id in population[i]):
                    while (neuron_id in population[i]):
                        neuron_id = np.random.randint(0, self.num_input_neurons + 
                                                        self.num_output_neurons)
                population[i][j] = neuron_id
                population[i][j+1] = np.random.random()*100
        return population