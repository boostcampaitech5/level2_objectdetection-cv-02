import torch.optim as optim


class MyOptimizer:
    def __init__(self, params, opt_cfg):
        self.params = params
        self.opt_cfg = opt_cfg

        self.opt_name = self.opt_cfg['name']
        self.opt_lr = self.opt_cfg['lr']
    
    def __call__(self):
        if self.opt_name == "adam":
            return self.load_adam(self.params, self.opt_lr, self.opt_cfg['weight_decay'])
        
        if self.opt_name == 'sgd':
            return self.load_sgd(self.params, self.opt_lr,
                                 self.opt_cfg['momentum'], self.opt_cfg['weight_decay'])
            

    def load_adam(self, params, lr=0.001, weight_decay=0, betas=(0.9, 0.999), eps=1e-08):
        adam = optim.Adam(params, lr, betas, eps, weight_decay)

        return adam

    def load_sgd(self, params, lr=0.001, momentum=0, weight_decay=0):
        sgd = optim.SGD(params, lr, momentum, weight_decay)

        return sgd
