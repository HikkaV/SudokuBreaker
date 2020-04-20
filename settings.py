layer_combination = [[64, 64, 64],
                     [64, 64, 64, 32, 16], [64, 64, 64, 64, 64],
                     [64, 64, 64, 64], [64, 64, 32, 16]]
conf = {
    "path_data": 'sudoku.csv',
    "path_model": '/home/volodymyr/Downloads/sudoku.model',
    "path_params": 'best_params.json',
    "val_portion": 0.01,
    "test_portion": 0.01,
    "seed": 10,
    "random_state": 4,
    "ncalls": 10,
    "exp_name": "final_exp",
    "custom": True,
    "hardcore_path": "data/hardcore_sudoku.csv",
    "space": [
        list(range(len(layer_combination))),  # layer type
        [False],  # average pooling
        list(range(32, 88, 8)),  # batch
        (5, 40),  # epochs
        (0.0001, 0.001)  # eta
    ],
    "eval_batch_size": 64,
    "plot_chart": False,
    # layers, average_pooling, batch, epochs, learning_rate
    "handmade_params": [1, False, 64, 19, 0.000755]
}
