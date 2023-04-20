import sys


import numpy as np


from model import Model
from utils import LoggerCSV, EarlyStopping


class Sane:
    """
    Класс нейроэволюционного алгоритма Sane
    Аргументы:
        - n_epoches - количество эволюционных эпох
        - fitness_func - Оптимизуруемая функция
        - num_neuron_pop - Количество нейронов в популяции
        - num_hidden_neurons - Количество скрытых нейронов в сети
        - num_neuron_connect - Количество связей нейрона (= кол-во генов*2 )
        - num_blueprints - Количество сохраняемых лучших ИНС
    """
    def __init__(
        self, n_epoches, fitness_func, 
        num_neuron_pop, num_input_neurons, 
        num_output_neurons, num_hidden_neurons, 
        num_neuron_connect, num_blueprints
    ):
        try:
            self.n_epoches = n_epoches
            self.fitness_func = fitness_func
            self.num_neuron_pop = num_neuron_pop
            self.num_hidden_neurons = num_hidden_neurons
            self.num_neuron_connect = num_neuron_connect
            self.num_blueprints = num_blueprints
            # Количество нейронов входного слоя
            self.num_input_neurons = num_input_neurons
            # Количество нейронов выходного слоя
            self.num_output_neurons = num_output_neurons

            self.logger = LoggerCSV("model")
            self.early_stopping = EarlyStopping("model", 30)

            if (num_neuron_connect >= (self.num_input_neurons + 
                                        self.num_output_neurons)):
                raise Exception("""Number The number of neuron connections is greater than the number of input and output neurons""")
        except Exception as e:
            print(str(e))
            sys.exit()

    def run(
        self, X_train, 
        y_train, X_val, 
        y_val, model_id
    ):
        """Функция оптимизации весов"""
        epoch = 0
        # Генерация популяции нейронов
        print("Генерация популяции")
        population, blueprints = self.generate_population()
        while (epoch < self.n_epoches and not self.early_stopping.early_stop):
            epoch += 1
            """
            1. Сбрасываем приспособленности нейронов.
            """
            # Приспособленности ИНС
            blueprint_fitness = np.zeros(self.num_blueprints)
            # Приспособленности нейронов
            neuron_fitness = np.zeros(self.num_neuron_pop)
            # Количество вхождений нейронов в ИНС
            num_neuron_include = np.ones(self.num_neuron_pop)
            """
            1.1 Пересчитать приспособленности комбинаций 
            """
            for bp_id in range(blueprints.shape[0]):
                net = population[blueprints[bp_id]]
                model = Model(net, self.num_input_neurons, 
                                self.num_output_neurons,
                                self.num_hidden_neurons)
                preds = model.forward(X_train)
                loss = self.fitness_func(y_train, preds)
                blueprint_fitness[bp_id] = loss
            
            """
            1.2 Сортировка моделей от лучшей к худшей
            """
            sort_id = np.argsort(blueprint_fitness)
            blueprints = blueprints[sort_id]
            blueprint_fitness = blueprint_fitness[sort_id]
            """
            1.3 Сохранение лучшей модели
            """
            best_loss_train = blueprint_fitness = blueprint_fitness[0]
            net = population[blueprints[0]]
            model = Model(
                    net, self.num_input_neurons, 
                    self.num_output_neurons,
                    self.num_hidden_neurons
                )
            preds = model.forward(X_val)
            best_loss_val = self.fitness_func(y_val, preds)
            self.logger(epoch, model_id, [best_loss_train, best_loss_val])
            self.early_stopping(best_loss_val, net, epoch, model_id)
            """
            2. Найти среднее значение приспособленности нейронов
            """
            for _ in range(1000):
                """
                2.1. Случайным образом выберается ~ нейронов из популяции.
                """
                random_NN = np.random.randint(0, self.num_neuron_pop, 
                                                size=(self.num_hidden_neurons))
                # Увеличить число вхождений для каждого участвующего нейрона
                for n_id in np.unique(random_NN):
                    num_neuron_include[n_id] += 1
                """
                2.2. Создается нейронная сеть из выбранных нейронов.
                """
                net = population[random_NN]
                model = Model(
                        net, self.num_input_neurons, 
                        self.num_output_neurons, 
                        self.num_hidden_neurons
                    )
                """
                2.3. Оценка сети на задаче
                """
                preds = model.forward(X_train)
                loss = self.fitness_func(y_train, preds)
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

            # Обновление указателей на изменившиеся индексы нейронов
            for i in range(self.num_blueprints):
                for j in range(self.num_hidden_neurons):
                    blueprints[i, j] = np.where(sort_id == blueprints[i, j])[0][0]
            """
            5. Скрещивание нейронов
            """
            population = self.neuron_crossover(population)
            """
            6. Мутация нейронов
            """
            population = self.neuron_mutation(population, 0.001)
            """
            7. Скрещивание комбинаций
            """
            blueprints = self.blueprint_crossover(blueprints)
            """
            8. Мутация комбинаций
            """
            blueprints = self.blueprint_mutation(blueprints, 0.01, 0.5)
        return population[blueprints[0]]

    def update_neuron_fitness(self, n_fit, n_ids_in, loss):
        """
        Функция обновления значений приспособленности нейронов
        Аргументы:
            - n_fit - Массив приспособленностей нейронов (ndarray)
            - n_ids_in - Массив индексов нейронов вошедших в нейронную сеть (ndarray)
            - loss - Значение приспособленности нейронной сети
        """
        for i in np.unique(n_ids_in):
            n_fit[i] += loss
        return n_fit

    def neuron_mutation(self, populat, p):
        """
        Функция мутации нейрона
        Аргументы:
            - populat - Популяция нейронов (ndarray)
            - p - Вероятность мутации 1 бита гена
        """
        # Вероятность мутации индекса нейрона
        pid = 1 - (1 - p)**8
        # Вероятность мутации связи
        pw = 1 - (1 - p)**16
        for i in range(self.num_neuron_pop):
            for j in range(populat[i].shape[0]):
                if (j % 2 == 0):
                    if (np.random.random() < pid):
                        n_id = np.random.randint(0, self.num_input_neurons + 
                                                    self.num_output_neurons)
                        # while (n_id in populat[i]):
                        #     n_id = np.random.randint(0, self.num_input_neurons + 
                        #                                 self.num_output_neurons)
                        populat[i, j] = n_id
                else:
                    if (np.random.random() < pw):
                        populat[i, j] = np.random.random() - 0.5
        return populat
        
    def neuron_crossover(self, populat):
        """
        Функция скрещивания нейронов
        Аргументы:
            - populat - Популяция нейронов (ndarray)
        """
        # Количество нейронов, которые учавcтсвуют в скрещивании
        n_cross_neurons = int(0.25*populat.shape[0])
        for i in range(1, n_cross_neurons, 2):
            # Индексы скрещиваемых нейронов
            n1_id = np.random.randint(0, n_cross_neurons)
            n2_id = np.random.randint(0, n_cross_neurons)
            # Точка разрыва
            p = np.random.randint(1, self.num_neuron_connect*2 - 1)
            if (n1_id != n2_id):
                populat[-i] = np.concatenate((populat[n1_id, 0:p],
                                                populat[n2_id, p:self.num_neuron_connect*2]), 
                                                axis=0)
                # populat[-(i+1)] = np.concatenate((populat[n2_id, 0:p],
                #                                     populat[n1_id, p:self.num_neuron_connect*2]), 
                #                                     axis=0)
                populat[-(i+1)] = populat[n1_id]
        return populat

    def blueprint_mutation(self, bp, p1, p2):
        """
        Функция мутации ИНС
        Аргументы:
            - bp - Массив комбинаций нейронов (ИНС) (ndarray)
            - p1 - вероятность перенаправить указатель на
            другой нейрон
            - p2 - вероятность перенаправить указатель на нейрон потомок
        """
        n_mut_bp = int(0.75*self.num_blueprints)
        n_mut_neurons = int(0.25*self.num_neuron_pop)
        # Вероятность мутации индекса нейрона
        p2 = p2 + p1
        for i in range(n_mut_bp, self.num_blueprints):
            for j in range(self.num_hidden_neurons):
                pm = np.random.random()
                if (pm < p1):
                    n_id = np.random.randint(0, self.num_neuron_pop - n_mut_neurons)
                    bp[i, j] = n_id
                if (pm > p1 and pm < p2):
                    n_id = np.random.randint(self.num_neuron_pop - n_mut_neurons, self.num_neuron_pop)
                    bp[i, j] = n_id
        return bp

    def blueprint_crossover(self, bp):
        """
        Функция скрещивания ИНС
        Аргументы:
            - bp - Массив комбинаций нейронов (ИНС) (ndarray) 
        """
        n_cross_bp = int(0.25*self.num_blueprints)
        for i in range(1, n_cross_bp, 2):
            # Индексы скрещиваемых нейронов
            n1_id = np.random.randint(0, n_cross_bp)
            n2_id = np.random.randint(0, n_cross_bp)
            # Точка разрыва
            p = np.random.randint(1, self.num_hidden_neurons)
            if (n1_id != n2_id):
                bp[-i] = np.concatenate((bp[n1_id, 0:p], 
                                        bp[n2_id, p:self.num_hidden_neurons]), 
                                        axis=0)
                # bp[-(i+1)] = np.concatenate((bp[n2_id, 0:p],
                #                             bp[n1_id, p:self.num_hidden_neurons]), 
                #                             axis=0)
                bp[-(i+1)] = bp[n1_id]
        return bp

    def generate_population(self):
        """
        Функция инициализации популяции нейронов
        """
        # Добавочный коэффициент, чтобы при заполнениями индексами не было повторяющихся
        io_neuron = self.num_input_neurons + self.num_output_neurons
        population = np.zeros((self.num_neuron_pop, 
                                self.num_neuron_connect*2))
        for i in range(0, self.num_neuron_pop):
            for j in range(0, self.num_neuron_connect*2 - 1, 2):
                neuron_id = np.random.randint(0, self.num_input_neurons + 
                                                self.num_output_neurons)
                # while (neuron_id in population[i]):
                #     neuron_id = np.random.randint(0, self.num_input_neurons + 
                #                                     self.num_output_neurons)
                population[i][j] = neuron_id
                population[i][j+1] = np.random.random() - 0.5
        blueprints = np.zeros((self.num_blueprints, 
                                self.num_hidden_neurons), dtype=int)
        for i in range(0, self.num_blueprints):
            blueprints[i] = np.random.randint(0, self.num_neuron_pop, 
                                              size=(self.num_hidden_neurons))

        return population, blueprints