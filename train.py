import os
from skopt import forest_minimize
from tqdm import tqdm
from sudoku_breaker import SudokuBreaker
import mlflow
from helper import load_and_process, save_params, load_params, plot, np, load_hardcore
import random
from colored import fg
from settings import layer_combination
import datetime
import tensorflow as tf

tf.random.set_seed(5)


class Train:
    def __init__(self, path, seed=5, custom=True, validation_portion=0.15, test_portion=0.1, exp_name='tmp',
                 minimize_scaler=0.7, hardcore_path=None):
        self.minimize_scaler = minimize_scaler
        if hardcore_path:
            self.test_x, self.test_y = load_hardcore()
        else:
            self.train_x, self.train_y, self.val_x, self.val_y, \
            self.test_x, self.test_y = load_and_process(path, seed, test_portion=test_portion,
                                                        validation_portion=validation_portion)
        self.exp_name = exp_name
        self.custom = custom
        self.sudoku_model = None

    def objective(self, space):
        layers, average_pooling, batch, epochs, learning_rate = space
        layers = layer_combination[layers]
        print(fg('green') + 'Parameters : layers : {0},  average_pooling : {1}, batch : {2}, '
                            'epochs : {3}, '
                            'learning rate : {4}'.format(layers, average_pooling, batch, epochs,
                                                         learning_rate))
        sudoku_model = SudokuBreaker(layers=layers, average_pooling=average_pooling)
        id_mlflow = random.randint(1, 2542314)
        if self.custom:
            sudoku_model.fit_custom(self.train_x, self.train_y, self.val_x, self.val_y,
                                    batch=batch, epochs=epochs, learning_rate=learning_rate, id_mlflow=id_mlflow)
        else:
            sudoku_model.fit_inbuilt(self.train_x, self.train_y, self.val_x, self.val_y,
                                     batch=batch, epochs=epochs, learning_rate=learning_rate)
        if self.custom:
            return self.minimize_loss(sudoku_model.get_val_masked_acc(), sudoku_model.get_val_acc())
        else:
            return self.minimize_loss(sudoku_model.get_val_acc(), sudoku_model.get_train_acc())

    def minimize_loss(self, a, b):
        return -(a + b * self.minimize_scaler) / 2

    def minimize(self, space, ncalls, minimize_seed, path_params='best_params.json'):
        exp_name = self.exp_name + '_{}'.format(datetime.datetime.now())
        mlflow.create_experiment(exp_name)
        mlflow.set_experiment(exp_name)
        best_params = forest_minimize(self.objective, space, n_calls=ncalls, random_state=minimize_seed)['x']
        save_params(best_params, path_params=path_params)

    def train(self, path_params='best_params.json', path_model='model.h5',
              plot_chart=False, handmade_params=None):
        if os.path.exists(path_params) and handmade_params:
            layers, average_pooling, batch, epochs, learning_rate = load_params(path_params)
        else:
            layers, average_pooling, batch, epochs, learning_rate = handmade_params
        layers = layer_combination[layers]
        print(fg(
            'green') + 'Parameters : layers : {0}, average_pooling : {1}, batch : {2}, epochs : '
                       '{3}, '
                       'learning rate : {4}'.format(layers, average_pooling, batch, epochs,
                                                    learning_rate))
        sudoku_model = SudokuBreaker(layers=layers, average_pooling=average_pooling)
        id_mlflow = random.randint(1, 2542314)
        if self.custom:
            sudoku_model.fit_custom(self.train_x, self.train_y, self.val_x, self.val_y,
                                    batch=batch, epochs=epochs, learning_rate=learning_rate, id_mlflow=id_mlflow)
        else:
            sudoku_model.fit_inbuilt(self.train_x, self.train_y, self.val_x, self.val_y,
                                     batch=batch, epochs=epochs, learning_rate=learning_rate)
        sudoku_model.save(path_model)
        if plot_chart:
            plot(sudoku_model.hist)

    def evaluate(self, path_model, batch_size=None):
        if batch_size:
            number_parts = int(np.ceil(self.test_x.shape[0] / batch_size))
            print('Parts of data to be processed : {}'.format(number_parts))
            self.test_x, self.test_y = np.array_split(self.test_x, number_parts), \
                                       np.array_split(self.test_y, number_parts)
        self.sudoku_model = SudokuBreaker(path=path_model)
        mean_acc_per_cell = []
        mean_acc = []
        for x, y in tqdm(zip(self.test_x, self.test_y)):
            if batch_size:
                predicted = self.sudoku_model.predict_on_batch(x)
                y = y.reshape((predicted.shape[0], 9, 9)) + 1
            else:
                predicted = self.sudoku_model.predict(x)
                y = y.reshape((9, 9)) + 1
            mean_acc_per_cell.append(np.equal(predicted, y).astype(int).mean())
            if (predicted - y).sum() == 0:
                mean_acc.append(1)
            else:
                mean_acc.append(0)
        mean_acc = np.mean(mean_acc)
        mean_acc_per_cell = np.mean(mean_acc_per_cell)
        print('Mean accuracy per cell on test data : {}'.format(mean_acc_per_cell))
        print('Mean accuracy on test data : {}'.format(mean_acc))
