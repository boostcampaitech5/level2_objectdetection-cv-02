import torch.optim as optim


def get_sgd(params, lr, momentum, weight_decay):
    sgd = optim.SGD(params, lr, momentum, weight_decay)

    return sgd


def get_adam():
    pass