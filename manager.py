import argparse
from train import Train
from settings import conf

ap = argparse.ArgumentParser()


def train_wrapper(args):
    train_cls = Train(path=conf["path_data"], seed=conf["seed"], custom=conf["custom"],
                      validation_portion=conf["val_partion"], test_portion=conf["test_portion"])
    train_cls.train(path_params=conf["path_params"], path_model=conf["path_model"],
                    plot_chart=conf["plot_chart"])


def define_params_wrapper(args):
    train_cls = Train(path=conf["path_data"], seed=conf["seed"], custom=conf["custom"],
                      validation_portion=conf["val_portion"], test_portion=conf["test_portion"],
                      exp_name=conf['exp_name'])
    train_cls.minimize(space=conf["space"], ncalls=conf["ncalls"], minimize_seed=conf["random_state"],
                       path_params=conf["path_params"])


def evaluate_wrapper(args):
    train_cls = Train(path=conf["path_data"], seed=conf["seed"], custom=conf["custom"],
                      validation_portion=conf["val_partion"], test_portion=conf["test_portion"]
                      )
    train_cls.evaluate(path_model=conf['path_model'], batch_size=conf['eval_batch_size'])


def parse_args():
    subparsers = ap.add_subparsers()
    define_parser = subparsers.add_parser('define_params', help='Define best hyper parameters for a model')
    define_parser.set_defaults(func=define_params_wrapper)
    train_parser = subparsers.add_parser('train', help='Train a model with best hyper parameters')
    train_parser.set_defaults(func=train_wrapper)
    evaluate_parser = subparsers.add_parser('evaluate',help='Evaluate model on test data')
    evaluate_parser.set_defaults(func=evaluate_wrapper)
    return ap.parse_args()
