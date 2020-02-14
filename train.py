from skopt import forest_minimize
from sudoku_breaker import SudokuBreaker
import mlflow
from helper import load_and_process, save_params, load_params, plot, np
import random
from colored import fg
from settings import layer_combination
import datetime


class Train:
    def __init__(self, path, seed=5, custom=True, validation_portion=0.15, test_portion=0.1, exp_name='tmp'):
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
        exp_name = self.exp_name + '_{}'.format(datetime.datetime.now())
        mlflow.create_experiment(exp_name)
        mlflow.set_experiment(exp_name)
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
        return -(a + b * 0.7) / 2

    def minimize(self, space, ncalls, minimize_seed, path_params='best_params.json'):
        best_params = forest_minimize(self.objective, space, n_calls=ncalls, random_state=minimize_seed)['x']
        save_params(best_params, path_params=path_params)

    def train(self, path_params='best_params.json', path_model='model.h5',
              plot_chart=False):
        layers, average_pooling, batch, epochs, learning_rate = load_params(path_params)
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

    def sample_generator(self, X, y, batch_size=64):
        X, y = np.array_split(X, int(np.ceil(len(X) / batch_size))), \
               np.array_split(y, int(np.ceil(len(y) / batch_size)))
        for sample_x, sample_y in (X, y):
            yield sample_x, sample_y

    def evaluate(self, path_model, batch_size=64):
        self.sudoku_model = SudokuBreaker(path=path_model)
        mean_acc = []
        for x, y in self.sample_generator(self.test_x, self.test_y, batch_size=batch_size):
            mean_acc.append(np.equal(self.sudoku_model.predict(x), y).astype(int).mean())
        mean_acc = np.mean(mean_acc)
        print('Mean accuracy on test data : {}'.format(mean_acc))
