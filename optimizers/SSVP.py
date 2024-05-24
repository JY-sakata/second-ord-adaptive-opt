import math
import torch
from torch.optim.optimizer import Optimizer
import torch.utils

class SSVP(Optimizer):
    """
    Implementation of SSVP: Second-order adaptive step size method using the Hessian vector product
    """
    def __init__(self, params, lr:float=6e-2, betas:tuple=(0.96, 0.99),rho:float=1e-2, weight_decay:float=0.0, eps:float=1e-12, update_period:int=10,sign = True):
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
 
        self.sign = sign
        super(SSVP, self).__init__(params, defaults)

    def __str__(self) -> str:
        return 'ssvp'
    
    @torch.no_grad()
    def hvp_estimate(self, v,):
        
        params = [p for group in self.param_groups for p in group['params'] if p.requires_grad]
        if len(params) == 0:
            return
        
        gradients = [p.grad for p in params]


        with torch.enable_grad():
            hvp = torch.autograd.grad(gradients, params, grad_outputs=v, retain_graph=True)
        
        for h, p in zip(hvp, params):
            self.state[p]['hvp'] = h

    @torch.no_grad()
    def set_hvp(self, hvp):
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.size() != hvp[i].size():
                    raise ValueError('The shape of parameters and hvp does not match: {} vs {}'.format(p.size(), hvp[i].size()))
                self.state[p]['hvp'] = hvp[i]
                i += 1
    

    @torch.no_grad()
    def step(self, closure=None, hvp=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss=closure()
        step = self.param_groups[0].get('step', 0)

      
        v = []
    
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
                    state['momentum_vec'] = torch.sign(-1*grad.detach())
                

                momentum= state['momentum']
                
                momentum.mul_(beta1).add_(grad, alpha=1.0-beta1)
                if self.sign:
                    if step % self.update_period == 0:
                        v.append(torch.sign(momentum)*(-1*group['rho']))
                else:
                    momentum_vec = state['momentum_vec']
                    momentum_vec.add_(torch.sign(momentum))
                    if step % self.update_period == 0:
                        state['momentum_vec'] = torch.sign(momentum)
                        v.append(-1*(group['rho']/self.update_period)*momentum_vec.detach())




        if hvp is not None:
            self.set_hvp(hvp)
        elif step % self.update_period == 0:
            # if not self.sign:
            #     norm = torch.linalg.norm(torch.cat([vec.view(-1) for vec in v])).item()
            #     for vec in v:
            #         vec = (1/norm)*vec
            self.hvp_estimate(v)

        for group in self.param_groups:

            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError('No Sparse Gradient Error')
                
                state = self.state[p]

                momentum, hessian_term = state['momentum'], state['hessian_term']
                # momentum /= 1-beta1**group['step']

                if 'hvp' in state and (step % self.update_period == 0 or hvp is not None):
                    hessian_term.mul_(beta2).addcmul_(state['hvp'], state['hvp'], value=1.0-beta2)
                denom = torch.sqrt(hessian_term) + group['eps']
                
                without_clip = momentum/denom
                update = (momentum/denom).clamp_(min=-group['rho'], max=group['rho'])
                state['clips']=(without_clip != update).sum().item()

                p.mul_(1.0-group['lr']*group['weight_decay'])
                p.add_(update, alpha=-group['lr'])
        return loss
