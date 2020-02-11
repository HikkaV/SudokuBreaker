path_data = '../sudoku.csv'
val_partion = 0.15
test_portion = 0.1
seed = 5
random_state = 10
ncalls = 10
custom = True
space = [
    [[64, 32, 64], [64, 64, 64], [256, 64, 32], [32, 32, 32], [64, 128, 256],
     [64, 64, 64, 32, 16], [64, 64, 32, 16]],  # layers
    [True, False],  # average pooling
    list(range(8, 88, 8)),  # batch
    (5, 40),  # epochs
    (0.0001, 0.001)  # eta
]
