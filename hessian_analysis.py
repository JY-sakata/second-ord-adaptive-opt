import torch



class Hessian_Analysis:
    """
    Compute the extact Hessian matrix and record statistics for the second-order characteristics of function
    
    This class is primarily used with the Rahimi-Recht function
    """

    def __init__(self, parameters,  closure) -> None:
      
        self.parameters_list = [p for p in parameters if p.requires_grad]

        with torch.enable_grad():
            loss = closure()
            self.loss = loss
            self.gradient_list = torch.autograd.grad(loss, self.parameters_list, create_graph=True, retain_graph=True)
        self.n_parameters = sum(p.numel() for p in self.parameters_list if p.requires_grad)
        self.compute_hessian_squared_matrix()


    def compute_hessian_list(self):
        """
        Compute the hessian matrix as a list of blocks of matrix. 
        """
        hessian_list = []
        for gradient in self.gradient_list:
            grad = gradient.view(-1)
            for i in range(len(grad)):
                grad2 = torch.autograd.grad(grad[i], self.parameters_list, retain_graph=True)
                hessian_list.extend([g.view(-1).clone() for g in grad2])
        return hessian_list
    

    def compute_hessian_squared_matrix(self):
        """
        Convert the list of the blocks of Hessian to the squared Hessian matrix for eigendecomposition.
        """
        hessian_list = self.compute_hessian_list()
        flatten_hessian_list = torch.cat(hessian_list)
        n = self.n_parameters
        square_hessian = torch.empty((n,n))
        idx = 0
        for i in range(n):
            for j in range(n):
                square_hessian[i, j] = flatten_hessian_list[idx]
                idx+=1
        self.hessian_squared_matrix = square_hessian
        # all_gradients = torch.cat([g.contiguous().view(-1) for g in self.gradient_list])
        # hessian = torch.autograd.grad(all_gradients, self.parameters_list, create_graph=True, retain_graph=True)



    @torch.no_grad()
    def get_diagonal_hessian(self):
        diag_hessian = torch.diag(self.hessian_squared_matrix).view(-1)
        return self.recover_hessian_block(diag_hessian)
    
    @torch.no_grad()
    def hessian_inverse_abs_eigenvals(self): # |H|^-1 saddle-free newton step size
        # square abosulute hessian
        e, V = torch.linalg.eigh(self.hessian_squared_matrix)
        self.eigenvalues = torch.diag(e)
        return -V @ (torch.diag(1/e.abs()) @V.T)
    
    @torch.no_grad()
    def get_saddle_free_newton_step_size(self):
        e, V = torch.linalg.eigh(self.hessian_squared_matrix)
        self.eigenvalues = torch.diag(e)
        flatten_grad = [g.view(-1) for g in self.gradient_list]
        flatten_grad = torch.cat(flatten_grad)
        step_size =  -V @ (torch.diag(1/e.abs()) @ (V.t() @flatten_grad))
        return self.recover_hessian_block(step_size)
    

    @torch.no_grad()
    def recover_hessian_block(self, flatten_element):
        # flatten_element should be flatten
        # recover flatten tensor (diagonal hessian/diagonal hessian estimate) to tensor with shape the same as parameter tensor shape
        reshaped_tensor = []
        idx = 0
        for parameter in self.parameters_list:
            n_elements = len(parameter.view(-1))
            tensor = flatten_element[idx: idx+n_elements].reshape(parameter.size())
            reshaped_tensor.append(tensor)
            idx+=n_elements
        return reshaped_tensor
    

    def get_hutchinson_hessian_diag_estimator(self, n_samples=1, alpha=1.0):
        hutchison_hessian_list = [torch.zeros_like(p) for p in self.parameters_list]
        gradient_clone = [g.clone() for g in self.gradient_list]
        for i in range(n_samples):
            zs = [torch.randint_like(p, 0, 1) * 2.0 - 1.0 for p in self.parameters_list]
            h_zs = torch.autograd.grad(gradient_clone, self.parameters_list, grad_outputs=zs, retain_graph=i < n_samples - 1)
            with torch.no_grad():
                for j in range(len(hutchison_hessian_list)):
                    hutchison_hessian_list[j].add_(h_zs[j] * zs[j], alpha=alpha/n_samples)
        return hutchison_hessian_list