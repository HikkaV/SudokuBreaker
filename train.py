from skopt import forest_minimize
from sudoku_breaker import SudokuBreaker
from helper import load_and_process, save_params, load_params, plot
import random
from colored import fg


class Train:
    def __init__(self, path, seed=5, custom=True, validation_portion=0.15, test_portion=0.1):
        self.train_x, self.train_y, self.val_x, self.val_y, \
        self.test_x, self.test_y = load_and_process(path, seed, test_portion=test_portion,
                                                    validation_portion=validation_portion)
        self.custom = custom

    def objective(self, space):
        layers, average_pooling, batch, epochs, learning_rate = space
        print(fg('green') + 'Parameters : layers : {0}, average_pooling : {1}, batch : {2}, epochs : {3},'
                            'learning rate : {4}'.format(layers, average_pooling, batch, epochs, learning_rate))
        sudoku_model = SudokuBreaker(layers=layers, average_pooling=average_pooling)
        id_mlflow = random.randint(1, 2542314)
        if self.custom:
            sudoku_model.fit_custom(self.train_x, self.train_y, self.val_x, self.val_y,
                                    batch=batch, epochs=epochs, learning_rate=learning_rate, id_mlflow=id_mlflow)
        else:
            sudoku_model.fit_inbuilt(self.train_x, self.train_y, self.val_x, self.val_y,
                                     batch=batch, epochs=epochs, learning_rate=learning_rate)
        return self.minimize_loss(sudoku_model.get_val_masked_acc(), sudoku_model.get_val_acc())

    def minimize_loss(self, masked_acc, acc):
        return -(masked_acc + acc * 0.7) / 2

    def minimize(self, space, ncalls, minimize_seed, path='best_params.json'):
        best_params = forest_minimize(self.objective, space, n_calls=ncalls, random_state=minimize_seed)['x']
        save_params(best_params, path_params=path)

    def train(self, path='best_params.json', path_model='model.h5',
              plot_chart=False):
        layers, average_pooling, batch, epochs, learning_rate = load_params(path)
        print(fg('green') + 'Parameters : layers : {0}, average_pooling : {1}, batch : {2}, epochs : {3},'
                            'learning rate : {4}'.format(layers, average_pooling, batch, epochs, learning_rate))
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
