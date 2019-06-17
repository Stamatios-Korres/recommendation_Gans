import torch.optim as optim


def sgd_optimizer(model_params, lr=1e-2, weight_decay=1e-6):
    
    return optim.SGD(model_params,
                                lr=lr,
                                weight_decay=weight_decay)

def adam_optimizer(model_params, lr=1e-2, betas = (0.9,0.999),weight_decay=1e-6):
    
    return optim.Adam(model_params,
                            lr=lr,
                            betas = betas,
                            weight_decay=weight_decay)

