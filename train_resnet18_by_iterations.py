import time
import os
from ray import tune, air, train
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import StepLR
from datasets.cifar10_data import get_cifar10_dataloader, get_cifar10_dataset
from models import *
from optimizers import *
from ray.tune.stopper import Stopper

import time
from optimiser_configuration import *
import ray
from ray import train, tune
from ray.tune import Trainable
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from torch.optim.lr_scheduler import CosineAnnealingLR
from ray.tune.search import ConcurrencyLimiter
from ray.tune.stopper import Stopper, TrialPlateauStopper, CombinedStopper
from torch.utils.data import DataLoader, Subset
from torch.linalg import norm
import tempfile
# from ray.train import Checkpoint



def train_resnet18(config, datasets=None):
    lr = config['lr']
    opt_name = config['optimizer_name']
    is_second_order_opt = (opt_name in ['sophia', 'ssdh', 'sadh', 'ssvp', 'ssvp_sign', 'ssvp_cum_sign'])
    rho = config['rho'] if is_second_order_opt else None
    trainset = datasets['trainset']
    testset = datasets['testset']

    trainloader = DataLoader(trainset,batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    n_epochs = config['n_epochs']

    loss_func = nn.CrossEntropyLoss()
    net = ResNet18().to(device)
    optimiser = get_optimizer(config, net)
    lr_scheduler = CosineAnnealingLR(optimiser, T_max=n_epochs+1,eta_min = 0.1*lr)

    itr = 0
    n_param = sum([p.numel() for p in net.parameters() if p.requires_grad])


    for i in range(n_epochs):


        for inputs, targets in trainloader:
            net.train()
            itr += 1
            optimiser.zero_grad(set_to_none=True)
            inputs, targets = inputs.to(device), targets.to(device)
            
            start_time = time.time()
            output = net(inputs)
            loss = loss_func(output, targets)
            if not is_second_order_opt:
                loss.backward()
            else:
                loss.backward(create_graph=True)
            optimiser.step()

            runtime_per_iteration = time.time() - start_time
            _, predicted = output.max(1)   
            correct =  predicted.eq(targets).sum().item()/targets.size()[0]

            
            metrics = {
                'training_loss': loss.item(),
                'training_acc': correct,   
                'last_lr': lr_scheduler.get_last_lr()[0],
                'runtime_per_iteration': runtime_per_iteration, 
                'test_loss': None,
                'test_acc': None,
            }    


            if opt_name in ['sophia', 'ssvp', 'sadh', 'ssdh']:
                clips_list = []
                denominator_list = []
                gradient_list = []
                for group in optimiser.param_groups:
                    for p in group['params']:
                        if not p.requires_grad:
                            continue
                        state = optimiser.state[p]    
                        clips_list.append(state['clips'])
                        denominator_list.extend([state['hessian_term'].view(-1)])
                        gradient_list.extend([p.grad.view(-1)])
                    

                gradients = torch.cat(gradient_list)
                hessian_term = torch.cat(denominator_list)
                clip_pct = sum(clips_list)/n_param
                if opt_name in ['ssdh', 'ssvp']:
                    metrics['hessian_term_norm'] = norm(hessian_term.sqrt_()).item()
                else:
                    metrics['hessian_term_norm'] = norm(hessian_term).item()
                metrics['gradient_norm'] = norm(gradients).item()
                metrics['clip_pct'] = clip_pct


            else:
                if opt_name == 'adam':
                    denominator = torch.cat([optimiser.state[p]['exp_avg_sq'].view(-1) for group in optimiser.param_groups for p in group['params'] if p.requires_grad])
                    metrics['exp_avg_sq_norm'] = norm(denominator.sqrt_()).item()
                gradient = torch.cat([p.grad.view(-1) for p in net.parameters() if p.requires_grad])
                metrics['gradient_norm'] = norm(gradient).item()

            if itr % 196 == 0:
                test_loss, test_acc = test(net, testloader, loss_func)
                metrics['test_loss'] = test_loss
                metrics['test_acc'] = test_acc

            # if itr == 25088:
            #     with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            #         path = os.path.join(temp_checkpoint_dir, 'end_ckpt.pth')
            #         torch.save(
            #            net.state_dict(), path
            #         )
            #         train.report(metrics = metrics, checkpoint=Checkpoint.from_directory(temp_checkpoint_dir))  
            # else:              
            train.report(metrics)
        
        lr_scheduler.step()



def test(model, test_loader, loss_func):

    model.eval()
    correct = 0
    total_num = 0
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_func(output, target)
            test_loss += loss.item()*target.size()[0]
            # get the index of the max log-probability
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()
            total_num += target.size()[0]
    test_loss/= total_num
    correct/=total_num

    # print('testing_correct: ', correct / total_num, '\n')
    return test_loss, correct

def train_resnet18_cifar10(optimiser_name, update_period=10, n_epochs=128, weight_decay=0.0, lr=None, rho=None, random_seed = 10100202):
    torch.cuda.manual_seed(random_seed) # second version
    # torch.cuda.manual_seed(10100202) # first version
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    param_space = {
        'optimizer_name': optimiser_name,
        'update_period': update_period,
        'n_epochs': n_epochs,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'random_seed': random_seed,
        
    }

    if lr is not None and rho is not None:
        param_space['lr'] = lr
        param_space['rho'] = rho
    else:
        get_best_optimiser_config_resnet18_cifar10(param_space)

    trainset, testset = get_cifar10_dataset()

    datasets = {
        'trainset': trainset,
        'testset': testset,
    }
    ray.shutdown()
    ray.init(include_dashboard=False)
    abs_path = os.path.abspath('./results')
    experiment_path_name = 'train_resnet18_{}_{}epochs_rs_{}_update{}'.format(optimiser_name, n_epochs,random_seed, update_period)
    train_with_parameters = tune.with_parameters(train_resnet18, datasets=datasets)

    ############ uncomment the following block of code when trainining on gpu

    # resource_group = tune.PlacementGroupFactory([{'GPU': 1, }])
    # train_with_resource = tune.with_resources(train_with_parameters, resource_group)
    ##########################

    tuner = tune.Tuner(
        # train_with_resource, # uncomment out this when training on cpu
        train_with_parameters, # comment this when training on gpu
        run_config=air.RunConfig(storage_path=abs_path, name=experiment_path_name,),
        tune_config=tune.TuneConfig(num_samples=1),
        param_space=param_space,
    )
    result = tuner.fit()
    df = result.get_dataframe()
    path_name = os.path.join(abs_path, experiment_path_name)
    print(f"Loading results from {experiment_path_name}...")
    csv_path_name = 'train_resnet18_cifar10_results.csv'
    file_exists = os.path.isfile(csv_path_name)

    with open(csv_path_name, mode='a', newline='') as file:
        df.to_csv(file, header=not file_exists, index=False)

    os.system('rm -rf /tmp/tmp*') 

    return result





if __name__ == '__main__':
    batch_size = 256
    weight_decay=0.0


    print("Is CUDA or ROCm available? ", torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        
        print(torch.cuda.device_count(), " GPUs are available")
        print("Using GPU ", torch.cuda.current_device(), " ", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(device)
        print(torch.version.cuda)
    


#################################
###### Run this to train the ResNet18 on the cifar10 dataset
###### Supported optimisers: adam, sgdm, sophia, ssdh, sadh, ssvp
###### random seeds have been used: 10100202, 111011, 110111, 1010



    result = train_resnet18_cifar10(optimiser_name='ssvp', n_epochs=128, update_period=10)
    

