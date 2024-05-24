import math
import torch
from torch.optim.optimizer import Optimizer

class SSDH(Optimizer):
    """
    Implementation of SSDH: Second-Order Step Size Methods Using the Smoothed Squared Diagonal Hessian Estimates
    """
    def __init__(self, params, lr:float=6e-2, betas:tuple=(0.96, 0.99),rho:float=1e-2, weight_decay:float=0.0, eps:float=1e-12, update_period:int=10, diag_hessian_distribution:str='rademacher'):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= rho:
            raise ValueError('Invalid rho: {}'.format(rho))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(
                    betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(
                    betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError('Invalid weight decay: {}'.format(weight_decay))
        if not 0 < update_period:
            raise ValueError('Invalid update period: {}'.format(update_period))
        if not (diag_hessian_distribution == 'gaussian' or diag_hessian_distribution == 'rademacher'):
            raise NotImplementedError('Only Support Gaussian or Rademacher Hessian distribution: {}'.format(diag_hessian_distribution))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            rho=rho,

        )
        self.update_period = update_period
        self.hessian_distribution = diag_hessian_distribution

        super(SSDH, self).__init__(params, defaults)

    def __str__(self) -> str:
        return 'ssdh'
    
    @torch.no_grad()
    def diagonal_hessian_estimate(self):
        
        params = [p for group in self.param_groups for p in group['params'] if p.requires_grad]
        if len(params) == 0:
            return
        
        gradients = [p.grad for p in params]

        if self.hessian_distribution == 'rademacher':
            v = [torch.randint_like(p, 0, 1)*2.0-1.0 for p in params]
        else:
            v = [torch.randn_like(p) for p in params]

        with torch.enable_grad():
            hvp = torch.autograd.grad(gradients, params, grad_outputs=v, retain_graph=True)
        
        for h, vec, p in zip(hvp, v, params):
            self.state[p]['hessian'] = h*vec

    @torch.no_grad()
    def set_hessian(self, hessian):
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.size() != hessian[i].size():
                    raise ValueError('The shape of parameters and hessian does not match: {} vs {}'.format(p.size(), hessian[i].size()))
                self.state[p]['hessian'] = hessian[i]
                i += 1
    

    @torch.no_grad()
    def step(self, closure=None, hessian=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss=closure()
        step = self.param_groups[0].get('step', 0)

        if hessian is not None:
            self.set_hessian(hessian)
        elif step % self.update_period == 0:
            self.diagonal_hessian_estimate()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError('No Sparse Gradient Error')
                
                state = self.state[p]
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p)
                    state['hessian_term'] = torch.zeros_like(p)
                    
                momentum, hessian_term = state['momentum'], state['hessian_term']
                momentum.mul_(beta1).add_(grad, alpha=1.0-beta1)

       
                if 'hessian' in state and (step % self.update_period == 0 or hessian is not None):
                    hessian_term.mul_(beta2).addcmul_(state['hessian'], state['hessian'], value=1.0-beta2)
                denom = torch.sqrt(hessian_term) + group['eps']
                
                without_clip = momentum /denom
                update = (momentum/denom).clamp_(min=-group['rho'], max=group['rho'])
                state['clips']=(without_clip != update).sum().item()

                p.mul_(1.0-group['lr']*group['weight_decay'])
                p.add_(update, alpha=-group['lr'])


        return loss

    