layer_combination = [[64, 32, 64], [64, 64, 64], [256, 64, 32], [32, 32, 32], [64, 128, 256],
                          [64, 64, 64, 32, 16], [64, 64, 32, 16]]
conf = {
    "path_data": '../sudoku.csv',
    "path_model": 'sudoku_breaker.h5',
    "path_params": 'best_params.json',
    "val_portion": 0.15,
    "test_portion": 0.1,
    "seed": 5,
    "random_state": 10,
    "ncalls": 10,
    "custom": True,
    "space": [
        list(range(len(layer_combination))),  # layer type
        [True, False],  # average pooling
        list(range(8, 88, 8)),  # batch
        (5, 40),  # epochs
        (0.0001, 0.001)  # eta
    ],
    "eval_batch_size": 64,
    "plot_chart": False
}
