import time
import os
import shutil

import torch
import torch.nn as nn

from pathlib import Path
from models import *
from optimizers import *
import time
from optimiser_configuration import *
import ray
from ray import train, tune, air
from torch.optim.lr_scheduler import CosineAnnealingLR
from ray.tune.search import ConcurrencyLimiter

from ray.tune.search.bayesopt import BayesOptSearch

from datasets.RR_data import get_RR_dataloader
from models import *
from optimizers import *

from hessian_analysis import Hessian_Analysis
from torch.linalg import norm
import copy
import time

from optimiser_configuration import *



def tune_func_RR(config):
    # config = config_dict['best_config']
    lr = config['lr']
    net_path = config['net_path']
    net = copy.deepcopy(torch.load(net_path)).to(device)
    train_loader, test_loader, full_batch_loader = get_RR_dataloader(net, batch_size)
    is_second_order_opt = (config['optimizer_name'] in ['sophia', 'ssdh', 'sadh', 'ssvp', 'ssvp_cum_sign'])
    n_epochs = config['n_epochs']

    optimiser = get_optimizer(config, net)
    lr_scheduler = CosineAnnealingLR(optimiser, T_max=n_epochs+1, eta_min=0.1*lr)


    for i in range(n_epochs):
        if not is_second_order_opt:
            training_loss = train_first_ord_opt(net, optimiser, train_loader, loss_func)
            clips = 0
        else:
            training_loss, clips = train_second_ord_opt(net, optimiser, train_loader, loss_func,)
        test_loss = evaluate(net, test_loader, loss_func)

        metrics = {
            'training_loss': training_loss,
            'clips': clips,
            'test_loss': test_loss,
            'last_lr': lr_scheduler.get_last_lr()[0],

        }
        lr_scheduler.step()
    
        train.report(metrics)
    


def train_RR_with_hessian_analysis(config_dict):
    # config = config_dict['best_config']
    config=config_dict
    lr = config['lr']
    # net_path = config['net_path']
    net = copy.deepcopy(torch.load(net_path)).to(device)
    train_loader, test_loader, full_batch_loader = get_RR_dataloader(net, batch_size)
    opt_name = config['optimizer_name']
    is_second_order_opt = (config['optimizer_name'] in ['sophia', 'ssdh', 'sadh', 'ssvp', 'ssvp_cum_sign'])
    rho = config.get('rho', None)
    n_epochs = config['n_epochs']

    optimiser = get_optimizer(config, net)
    lr_scheduler = CosineAnnealingLR(optimiser, T_max=n_epochs+1, eta_min=0.1*lr)
    print('Training RR with xdim{}, wdim{}, ydim{}, n_layers{}, cond_num{}, optimiser_{}, lr{}, rho{}'.format(net.xdim, net.wdim, net.ydim, net.num_layers, torch.linalg.cond(net.Atrue).item(), opt_name, lr, rho))
    iteration = 0
    n_params = sum([p.numel() for p in net.parameters() if p.requires_grad])
    for i in range(1, n_epochs+1):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            net.train()
            optimiser.zero_grad(set_to_none=True)
            out = net(x)
            loss = loss_func(out, y)    
            if is_second_order_opt:
                loss.backward(create_graph=True)
            else:
                loss.backward()

            optimiser.step()  
            metrics = {
                'training_loss': loss.item(),
            }                   


            if iteration % 100 == 0:
                # net.eval()
                optimiser.zero_grad(set_to_none=True)
                def closure():
                    out = net(x)
                    loss = loss_func(out, y)
                    return loss
                
                hess = Hessian_Analysis(net.parameters(), closure)
                exact_hessian_diag = torch.diag(hess.hessian_squared_matrix)
                e, V = torch.linalg.eigh(hess.hessian_squared_matrix)
                gradient = torch.cat([g.detach().view(-1) for g in hess.gradient_list])  
   
                metrics['hessian_spectral_norm'] = max(abs(e)).item()
                metrics['diag_hessian_norm'] = norm(exact_hessian_diag.view(-1)).item()
                metrics['gradient_norm'] =  norm(gradient).item()
                metrics['minimum_abs_eigenval']= min(abs(e)).item()
                metrics['eignvals_msqrt']= norm(e).item()
                metrics['positive_eigenval_pct'] = (e > 0).sum().item()/n_params
                metrics['positive_exact_hess_diag_pct'] = (exact_hessian_diag.view(-1) > 0).sum().item()/n_params
                if is_second_order_opt:
                    if opt_name == 'ssvp':
                        hvp_estimate = []
                    else:
                        hutchinson_diag_hess_estimate = []
                    hessian_term = []
                    clip_list = []

                    for group in optimiser.param_groups:
                        for p in group['params']:
                            if not p.requires_grad:
                                continue
                            state = optimiser.state[p]
                            if opt_name == 'ssvp':
                                hvp_estimate.extend([state['hvp'].view(-1)])
                            else:
                                hutchinson_diag_hess_estimate.extend([state['hessian'].view(-1)])
                            hessian_term.extend([state['hessian_term'].view(-1)])
                            clip_list.append(state['clips'])
            
                    if opt_name == 'ssvp':
                        hvp_estimate = torch.cat(hvp_estimate)
                    else:
                        hutchinson_diag_hess_estimate = torch.cat(hutchinson_diag_hess_estimate)
                    hessian_term = torch.cat(hessian_term)
                    clips_pct = sum(clip_list)/n_params
                    metrics['clips_pct'] = clips_pct
                    if opt_name == 'ssvp':
                        metrics['hvp_estimate_norm'] = norm(hvp_estimate).item()
                    else:
                        metrics['hutchinson_diag_hessian_norm'] = norm(hutchinson_diag_hess_estimate).item()
                        metrics['positive_hutchinson_diag_hess_pct'] = (hutchinson_diag_hess_estimate > 0).sum().item()/n_params
                    metrics['hessian_term_norm'] = norm(hessian_term).item()
                elif opt_name == 'adam':
                    denominator = torch.cat([optimiser.state[p]['exp_avg_sq'].view(-1) for group in optimiser.param_groups for p in group['params'] if p.requires_grad])
                    metrics['exp_avg_sq_norm'] = norm(denominator).item()
   

            train.report(metrics)
            iteration += 1

        lr_scheduler.step()


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss




def evaluate(model, dataloader, loss_func):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_func(output, y)
            test_loss += loss.item()

    return test_loss/len(dataloader)





def train_second_ord_opt(model, optimizer, dataloader, loss_func, train_with_exact_diag_hess=False):
    model.train()
    training_loss = 0
    clip_list = []
    n_param = sum([p.numel() for p in model.parameters() if p.requires_grad])
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        if train_with_exact_diag_hess:

            optimizer.zero_grad()
            def closure():
                out = model(x)
                loss = loss_func(out, y)
                return loss
            
            hess = Hessian_Analysis(model.parameters(), closure)
            exact_hessian_diag = torch.diag(hess.hessian_squared_matrix)

        optimizer.zero_grad(set_to_none=True)
        
        out = model(x)
        loss = loss_func(out, y)

        # input = torch.randn_like(x)
        # target = model.compute_ytrue(input)
        # out = model(input)
        # loss = loss_func(out, target)

        loss.backward(create_graph = True)
        if train_with_exact_diag_hess:

            hessian_block = hess.recover_hessian_block(exact_hessian_diag)
            optimizer.step(hessian = hessian_block)
        else:
            optimizer.step()
        
        clips = [optimizer.state[p]['clips'] for group in optimizer.param_groups for p in group['params'] if p.requires_grad]
        clip_list.append(sum(clips)/n_param)
        
  
        training_loss += loss.item()

    training_loss/= len(dataloader)
    clipping = sum(clip_list)/len(clip_list)

    return training_loss, clipping



def train_first_ord_opt(model, optimizer, dataloader, loss_func):
    model.train()
    training_loss = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)


        out = model(x)
        loss = loss_func(out, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        training_loss += loss.item()

    training_loss/= len(dataloader)

    return training_loss



def train_RR_with_best_config(optimiser_name, n_samples, large_cond=True):

    torch.manual_seed(10100202)
    # torch.cuda.manual_seed(10100202)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    abs_path = os.path.abspath('./results')
    configs = {
       

        "n_epochs": 3000,
        "net_path": "/home/remote/u7529050/saddle-free-opt/net_RR_cond29_xdim12_wdim13_ydim14_nlayers2.pth",
        "optimizer_name": optimiser_name,
    
        "update_period": 10,
        "weight_decay": 0.0
    }
    if large_cond:
        configs=get_best_optimiser_config_RR_cond15099(configs)
        configs['net_path'] = 'net_RR_cond15099_xdim12_wdim13_ydim14_nlayers2.pth'
    else:
        configs=get_best_optimiser_config_RR_cond29(configs)
        configs['net_path'] = 'net_RR_cond29_xdim12_wdim13_ydim14_nlayers2.pth'


    model = torch.load(configs['net_path'])
    experiment_path_name = 'train_RR_cond{}_{}epochs'.format(int(torch.linalg.cond(model.Atrue).item()), configs['n_epochs'])
    
    resource_group = tune.PlacementGroupFactory([{'CPU': 1, }])

    # print('Best configuration. optimiser: {}, lr: {}, rho: {}, condition number of Atrue: {}'.format(best_config['lr'], best_config['rho'], torch.linalg.cond(model.Atrue).item()))
    tuner = tune.Tuner(
        tune.with_resources(train_RR_with_hessian_analysis, resource_group),
        run_config=air.RunConfig(storage_path=abs_path, name=experiment_path_name),
        tune_config=tune.TuneConfig(num_samples=n_samples),
        param_space=configs
    )
    train_with_hessian_analysis_result = tuner.fit()
    df = train_with_hessian_analysis_result.get_dataframe()

    path_name = os.path.join(abs_path, experiment_path_name)
    csv_path_name = '{}.csv'.format(path_name)
    file_exists = os.path.isfile(csv_path_name)


    with open(csv_path_name, mode='a', newline='') as file:
        df.to_csv(file, header=not file_exists, index=False)
            


      


def tune_with_bo(optimiser_name, net_path, n_samples, update_period=10, n_epochs=150, lr=None, rho=None, beta2=None):
    param_space = {
        'optimizer_name': optimiser_name,
        'net_path': net_path,
        'lr': tune.uniform(5e-4, 8e-1) if lr is None else lr, # 0.2 - 0.0005
        'rho': tune.uniform(1e-4, 5e-1) if rho is None and optimiser_name in ['sophia', 'ssdh', 'sadh', 'ssvp', 'ssvp_cum_sign'] else rho, # 0.05 - 0.0001
        'update_period': update_period,
        'beta2': tune.uniform(0.985, 0.9999) if beta2 is None and optimiser_name in ['sophia', 'ssdh', 'sadh', 'ssvp', 'ssvp_cum_sign', 'adam'] else beta2,
        'n_epochs': n_epochs,
        'net_path': net_path,
        'weight_decay': 0.0,
    }
    torch.manual_seed(10100202) 
    torch.cuda.manual_seed(10100202)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = torch.load(net_path)
    print('Hyperparameter Tuning. Optimiser: {}; Net: xdim-{}, wdim-{}, ydim-{}, cond_number-{}, batch_size-{}'.format(optimiser_name, model.xdim, model.wdim, model.ydim,int(torch.linalg.cond(model.Atrue).item()), batch_size))

    algo = BayesOptSearch(metric='training_loss', mode='min',skip_duplicate= True)
    algo = ConcurrencyLimiter(algo, max_concurrent=4)
    ray.shutdown()
    ray.init(include_dashboard=False)
    abs_path = os.path.abspath('./results')

    resource_group = tune.PlacementGroupFactory([{'CPU': 1}])

    experiment_path_name = 'hp_tuning_RR_{}_update{}_{}epochs_cond_num{}_bs{}'.format(optimiser_name, update_period, n_epochs,int(torch.linalg.cond(model.Atrue).item()), batch_size)

    tuner = tune.Tuner(

        tune.with_resources(tune_func_RR, resource_group),
        run_config=train.RunConfig(
            storage_path=abs_path, 
            name=experiment_path_name,
            # stop=max_loss_stopper,

        ),
        tune_config=tune.TuneConfig(
            metric='training_loss',
            mode='min',
            search_alg=algo,
            num_samples=n_samples,
        ),
        param_space=param_space,
    )        


    result = tuner.fit()
    df = result.get_dataframe()
 
    path_name = os.path.join(abs_path, experiment_path_name)
    relative_path = os.path.join('./results', experiment_path_name)
    for root, dirs, files in os.walk(relative_path, topdown=False):
        for name in dirs:
            if name.startswith('checkpoint'):
                shutil.rmtree(os.path.join(root, name))
                print(f"Deleted {os.path.join(root, name)}")

    print(f"Loading results from {experiment_path_name}...")
    csv_path_name = '{}.csv'.format(path_name)
    file_exists = os.path.isfile(csv_path_name)

    with open(csv_path_name, mode='a', newline='') as file:
        df.to_csv(file, header=not file_exists, index=False)

    best_result_config = result.get_best_result(metric='training_loss', mode='min').config
    best_result_config_path = 'best_optimiser_config_optimiser_{}_x{}w{}y{}_{}layers_cond_num{}_bs{}.pth'.format(optimiser_name, model.xdim, model.wdim, model.ydim, model.num_layers,int(torch.linalg.cond(model.Atrue).item()), batch_size)
    p = os.path.join(abs_path, best_result_config_path)
    torch.save((best_result_config, model), p)

    return result





def create_net_with_large_Atrue(xdim=12, wdim=13, ydim=14, cond_num=1e+4, nsamples=1000, n_layers=2,):
    
    max_cond_num = -float('inf')


    while max_cond_num < cond_num:
        sample_net = RahimiRechtFuncModule(xdim=xdim, wdim=wdim, ydim=ydim, nsamples=nsamples, num_layers=n_layers, A_condition_number=cond_num)
        max_cond_num = torch.linalg.cond(sample_net.Atrue).item()

    net_path = 'net_RR_cond{}_xdim{}_wdim{}_ydim{}_nlayers{}.pth'.format(int(max_cond_num), xdim, wdim, ydim, n_layers)
    torch.save(sample_net, net_path)
    print('created net with Atrue condition number: {}'.format(max_cond_num))
    
    return net_path

def create_net_with_small_Atrue(xdim=12, wdim=13, ydim=14, cond_num=1e+4, nsamples=1000, n_layers=2,):
    min_cond_num = float('inf')
    while min_cond_num > cond_num:
        sample_net = RahimiRechtFuncModule(xdim=xdim, wdim=wdim, ydim=ydim, nsamples=nsamples, num_layers=n_layers, A_condition_number=cond_num)
        min_cond_num = torch.linalg.cond(sample_net.Atrue).item()

    net_path = 'net_RR_cond{}_xdim{}_wdim{}_ydim{}_nlayers{}.pth'.format(int(min_cond_num), xdim, wdim, ydim, n_layers)
    torch.save(sample_net, net_path)
    print('created net with Atrue condition number: {}'.format(min_cond_num))
    
    return net_path
         




if __name__ == '__main__':
    print("Is CUDA or ROCm available? ", torch.cuda.is_available())
    if torch.cuda.is_available():
        
        print(torch.cuda.device_count(), " GPUs are available")
        print("Using GPU ", torch.cuda.current_device(), " ", torch.cuda.get_device_name(0))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    xdim = 12
    wdim = 13
    ydim=14
    nsamples=1000
    batch_size = 128
    n_layers = 2
    loss_func = RMSELoss()
    abs_path = os.path.abspath('./')


    ########### train Rahimi-Recht function with a condition number of 15099
    ###### supported optimisers: adam, sophia, sgdm, ssdh, sadh, ssvp

    net_path = os.path.join(abs_path, 'net_RR_cond15099_xdim12_wdim13_ydim14_nlayers2.pth')

    train_RR_with_best_config('ssdh', n_samples=2)




    ########## train Rahimi-Recht function with a condition number of 29
    # net_path = os.path.join(abs_path, 'net_RR_cond29_xdim12_wdim13_ydim14_nlayers2.pth')

    # train_RR_with_best_config('ssdh', n_samples=2)

