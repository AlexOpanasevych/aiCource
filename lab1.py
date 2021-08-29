import numpy as np


class neural_ware:
    def __init__(self, arr, row, w_h, w_o):  # Ініціалізація значень
        # Виділяємо 13 и 14 элементы
        self.test_13_14 = np.array([[1.34, 5.8, 1.61],
                                    [5.8, 1.61, 5.97]]) / 10
        # Описуємо цільові(необхідні значення)
        self.aim_15 = np.array([[5.97, 1.95]]).reshape(2, 1) / 10

        # Вхід нейрона
        self.input_arr = arr

        # Ряд всіх значень
        self.row = row

        # Приховані ваги
        self.w_hidden = w_h

        # Вихідні ваги
        self.w_output = w_o
        self.n = 0.005  # швидкість навчання
        self.delta = 1000000  # помилка
        self.w_hidden_current = self.w_hidden  # Присвоюємо поточні приховані ваги
        self.w_output_current = self.w_output  # Присвоюємо поточні вихідні ваги
        self.epoch_current = -1  # Початкова епоха
        self.delta_current = 11311311  # поточна помилка

    @staticmethod
    def sigmoid_func(x):  # сігмоїд
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative_func(x):  # Похідна сігмоїда
        return np.exp(-x) / ((1 + np.exp(-x)) ** 2)

    def func(self, s):  # Функція навчання
        for epoch in range(420000):
            hidden_input_layer = np.dot(self.input_arr, self.w_hidden)  # Перемножуємо вхід нейронів на приховані ваги

            hidden_output_layer = self.sigmoid_func(hidden_input_layer)  # Рахуємо сігмоїд прихованого шару

            input_layer = np.dot(hidden_output_layer, self.w_output)  # Перемножуємо скрытый слой на выходящие веса

            output_layer = self.sigmoid_func(input_layer)  # Рахуємо сігмоїд вихідного шару

            delta_current = np.sum(np.power((output_layer - self.row), 2))  # Рахуємо помилку

            # Backpropagation
            bp_delta_output = output_layer - self.row  # Рахуємо вихідну помилку
            bp_sigmoid_der_input = self.sigmoid_derivative_func(input_layer)  # Рахуємо похідну вхідного шару
            bp_hidden_output_layer = hidden_output_layer  # Присвоюємо прихований шар

            # Перемножуємо прихований вихідний шар на вхідну похідну на помилку вихідного шару
            bp_delta_w_output = np.dot(bp_hidden_output_layer.T, bp_delta_output * bp_sigmoid_der_input)

            bp_delta_input = bp_delta_output * bp_sigmoid_der_input  # Обчислюємо вхідну помилку
            bp_w_output = self.w_output  # Присвоюємо вихідні ваги
            bp_delta_out_hidden = np.dot(bp_delta_input, bp_w_output.T)  # Перемножуємо помилку на вихідні ваги

            # Обчислюємо похідну прихованого шару
            bp_sigmoid_der_in_hidden = self.sigmoid_derivative_func(hidden_input_layer)
            bp_input = self.input_arr  # Присвоюємо вхідні дані

            # Перемножуємо вхідні дані на приховану похідну на помилку прихованого шару
            bp_delta_w_hidden = np.dot(bp_input.T, bp_sigmoid_der_in_hidden * bp_delta_out_hidden)
            # Backpropagation

            self.w_hidden -= self.n * bp_delta_w_hidden  # Перемножуємо помилку прихованих вагів на швидкість навчання
            self.w_output -= self.n * bp_delta_w_output  # Перемножуємо помилку вихідних вагів на швидкість навчання
            if s == "neuro":
                if np.sum(delta_current) < self.delta:  # Якщо сума поточних помилок менше заданої
                    # Переписуємо поточні дані для нового обчислення
                    self.w_hidden_current = self.w_hidden
                    self.w_output_current = self.w_output
                    self.epoch_current = epoch
                    self.delta_current = np.sum(delta_current)
            self.delta = np.sum(delta_current)
            if epoch % 10000 == 0:
                print("Епоха: ", epoch, " | Помилка: ", np.sum(delta_current))

    def print_epoch_error(self):  # Друк кращої епохи и вагів
        print("Краща епоха: ", self.epoch_current)
        print("Минімальна помилка: ", self.delta_current)
        print("Ваги: ", self.w_output)
        print("Приховані ваги: ", self.w_hidden)
        self.w_hidden = self.w_hidden_current
        self.w_output = self.w_output_current

    def print_series(self, s):  # Друк результатів: для числового ряда - x4 и x13, для логічних функцій - відповіді
        print("\n\nВідповіді:")
        for i in range(len(self.input_arr)):
            res1 = np.dot(self.input_arr[i], self.w_hidden)
            res2 = self.sigmoid_func(res1)
            res3 = np.dot(res2, self.w_output)
            res4 = self.sigmoid_func(res3)
            if s == "neuro":
                print("Припущення: ", (res4 * 10), " | Потрібний результат: ", (self.row[i] * 10))
            else:
                print("Припущення: ", res4, " | Потрібний результат: ", self.row[i])

    def print_result(self):  # Друк останніх двох тестованих значень x14 и x15
        print("\n\nx14 и x15:")
        for i in range(len(self.test_13_14)):
            res1 = np.dot(self.test_13_14[i], self.w_hidden)
            res2 = self.sigmoid_func(res1)
            res3 = np.dot(res2, self.w_output)
            res4 = self.sigmoid_func(res3)
            print("Припущення: ", (res4 * 10), " Потрібний результат: ", (self.aim_15[i] * 10))

    def processing(self, s):  # Запуск процесу для числового ряда
        self.func(s)
        self.print_epoch_error()
        self.print_series(s)
        self.print_result()

    def processing_logic(self, s):  # Запуск процесу для логічних функцій
        self.func(s)
        self.print_series(s)


if __name__ == '__main__':
    #  Числовий ряд

    initial = (2.37, 4.85, 1.97, 4.17, 1.39, 4.66, 1.26, 4.40,
               0.46, 5.54, 1.34, 5.80, 1.61, 5.97, 1.95)
    input_arr1 = (np.array([[initial[j + i] for i in range(3)] for j in range(len(initial) - 5)])) / 10

    row_4_10 = (np.array([[4.17, 1.39, 4.66, 1.26, 4.4, 0.46, 5.54, 1.34, 5.8, 1.36]])
                .reshape(10, 1)) / 10

    w_hidden1 = np.array([[2, 3, 1],
                          [0.5, 0.3, 0.4],
                          [0.7, 1, 0.6]]).T

    w_output1 = np.array([[0.4], [0.5], [0.3]])

    s1 = "neuro"
    neuro = neural_ware(input_arr1, row_4_10, w_hidden1, w_output1)
    neuro.processing(s1)

    # Логічні функції AND, OR, XOR, NOT
    input_arr2 = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]])
    w_hidden2 = np.array([[0.3, 0.2],
                          [0.5, 0.4],
                          [0.7, 0.6]]).T
    w_output2 = np.array([[0.8], [0.4], [0.6]])

    row_AND = np.array([[0, 0, 0, 1]]).reshape(4, 1)
    row_OR = np.array([[0, 1, 1, 1]]).reshape(4, 1)
    row_XOR = np.array([[0, 1, 1, 0]]).reshape(4, 1)

    s2 = "logic"
    logic_AND = neural_ware(input_arr2, row_AND, w_hidden2, w_output2)
    # logic_AND.processing_logic(s2)

    logic_OR = neural_ware(input_arr2, row_OR, w_hidden2, w_output2)
    # logic_OR.processing_logic(s2)

    logic_XOR = neural_ware(input_arr2, row_XOR, w_hidden2, w_output2)
    # logic_XOR.processing_logic(s2)

    input_arr2 = np.array([[0],
                           [1]])
    w_hidden2 = np.array([[0.5],
                          [0.3],
                          [0.1]]).T
    w_output2 = np.array([[0.6], [1], [0.9]])

    row_NOT = np.array([[0, 1]]).reshape(2, 1)

    logic_NOT = neural_ware(input_arr2, row_NOT, w_hidden2, w_output2)
# logic_NOT.processing_logic(s2)
