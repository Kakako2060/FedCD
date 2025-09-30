from torch.optim import Optimizer

class pFedIBOptimizer(Optimizer):
    def __init__(self, params, lr=0.01):
        # self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults=dict(lr=lr)
        super(pFedIBOptimizer, self).__init__(params, defaults)

    def step(self, apply=True, lr=None, allow_unused=False):
        grads = []
        # apply gradient to model.parameters, and return the gradients
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None :
                    continue
                grads.append(p.grad.data)       #没有处理 p.grad is None 且 allow_unused 为 False 的情况，这可能会导致抛出异常或未定义的行为。
                if apply:
                    if lr == None:
                        p.data= p.data - group['lr'] * p.grad.data
                    else:
                        p.data=p.data - lr * p.grad.data
        return grads


    def apply_grads(self, grads, beta=None, allow_unused=False):    #允许独立地应用梯度列表来更新模型参数，这在联邦学习中可能是必要的，因为客户端可能需要在不同的时间点应用梯度。
        #apply gradient to model.parameters
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None and allow_unused:
                    continue
                p.data= p.data - group['lr'] * grads[i] if beta == None else p.data - beta * grads[i]
                i += 1
        return
class ScaffoldOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(ScaffoldOptimizer, self).__init__(params, defaults)

    def step(self, server_controls, client_controls, closure=None):

        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            for p, c, ci in zip(group['params'], server_controls.values(), client_controls.values()):
                if p.grad is None:
                    continue
                dp = p.grad.data + c.data - ci.data
                p.data = p.data - dp.data * group['lr']

        return loss
class FedAPENOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super(FedAPENOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        # Implement the step function logic according to FedAPEN requirements
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)  # 简单的SGD更新
class FedProxOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.01, mu=0.001):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults=dict(lr=lr, lamda=lamda, mu=mu)
        super(FedProxOptimizer, self).__init__(params, defaults)

    def step(self, vstar, closure=None):
        loss=None
        if closure is not None:
            loss=closure
        for group in self.param_groups:
            for p, pstar in zip(group['params'], vstar):
                # w <=== w - lr * ( w'  + lambda * (w - w* ) + mu * w )
                if p.grad is None:
                    continue
                p.data=p.data - group['lr'] * (
                            p.grad.data + group['lamda'] * (p.data - pstar.data.clone()) + group['mu'] * p.data)
        return group['params'], loss
