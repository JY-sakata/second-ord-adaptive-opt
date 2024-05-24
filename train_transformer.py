import time
import os

import numpy as np
import shutil
import ray
from ray import train, tune, air
from ray.tune import Trainable
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from models import *
from optimizers import *
from datasets import *

import math
import tempfile
from ray.train import Checkpoint
from optimiser_configuration import *
from torch.linalg import norm
from ray.air.integrations.wandb import WandbLoggerCallback


def train_func(config, data=None):

    lr = config['lr']
    opt_name =config['optimizer_name']
    is_second_order_opt = (opt_name in ['sophia', 'ssdh', 'sadh', 'ssvp', 'ssvp_sign', 'ssvp_cum_sign'])
    rho = config['rho'] if is_second_order_opt else None

    train_data = data['train_data']
    valid_data = data['valid_data']
    test_data = data['test_data']
    log_interval = 100
    n_tokens = data['n_tokens']
    max_batch_idx = (train_data.size(0)-1)//bptt
    net = TransformerModel(n_tokens, d_model, n_head, d_hid,n_layers, dropout).to(device)
    loss_func = nn.CrossEntropyLoss()
    optimiser = get_optimizer(config, net)
    n_epochs = config['n_epochs']
    lr_scheduler = CosineAnnealingLR(optimiser, T_max=n_epochs+1, eta_min=0.1*lr)

    n_param = sum([p.numel() for p in net.parameters() if p.requires_grad])
    for j in range(1, n_epochs+1):


        sum_seq_length = 0
        total_sec = 0
        unclipped_gradient_norm_list = []
        training_loss = 0
        denominator_norm_list = []
        clip_list = []

        for batch, i in enumerate(range(0, train_data.size(0)-1, bptt)):
            net.train()
            data, target = get_batch(train_data, i, bptt)
            data, target = data.to(device), target.to(device)
            seq_len = data.size(0)
            sum_seq_length += seq_len    
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            start_time = time.time()    
            out = net(data)
            output = out.view(-1, n_tokens)
            loss = loss_func(output, target)
            if not is_second_order_opt:
                loss.backward()
            else:
                loss.backward(create_graph=True)
            unclipped_gradient_norm = torch.nn.utils.clip_grad_norm_(net.parameters(),max_norm=1)
            optimiser.step()
                    
            total_sec += time.time() - start_time
            training_loss += seq_len*loss.item()
            unclipped_gradient_norm_list.append(unclipped_gradient_norm.item())
            optimiser.zero_grad(set_to_none=True)
            if opt_name == 'adam':
                denominator = torch.cat([optimiser.state[p]['exp_avg_sq'].view(-1) for group in optimiser.param_groups for p in group['params'] if p.requires_grad])
                denominator_norm_list.append(norm(denominator.sqrt_()).item())
            elif is_second_order_opt:
                clips_list = []
                denominator_list = []

                for group in optimiser.param_groups:
                    for p in group['params']:
                        if not p.requires_grad:
                            continue
                        state = optimiser.state[p]    
                        clips_list.append(state['clips'])
                        denominator_list.extend([state['hessian_term'].view(-1)])
           
                hessian_term = torch.cat(denominator_list)
                clip_pct = sum(clips_list)/n_param 
                clip_list.append(clip_pct)
                if opt_name in ['ssvp', 'ssdh']:
                    denominator_norm_list.append(norm(hessian_term.sqrt_()).item())
                else:
                    denominator_norm_list.append(norm(hessian_term).item())
            
            if (batch % log_interval == 0 or batch == max_batch_idx) and batch > 0:
                training_loss /= sum_seq_length
                ppl = math.exp(training_loss)
                clip_pct = sum(clip_list)/len(clip_list) if len(clip_list) > 0 else 0

                grad_norm = sum(unclipped_gradient_norm_list)/len(unclipped_gradient_norm_list)
                denom = sum(denominator_norm_list)/len(denominator_norm_list) if len(denominator_norm_list) >0 else 0
                valid_loss, valid_ppl = evaluate(net, valid_data, loss_func, n_tokens)
                test_loss, test_ppl = evaluate(net, test_data, loss_func, n_tokens)


                metrics = {
                    'training_loss': training_loss,
                    'training_ppl': ppl,
                    'gradient_norm': grad_norm,
                    'denominator_norm': denom,
                    'clip_pct': clip_pct,
                    'runtime_per_step': total_sec,
                    'last_lr': lr_scheduler.get_last_lr()[0], 
                    'valid_loss': valid_loss,
                    'valid_ppl': valid_ppl,
                    'test_loss': test_loss,
                    'test_ppl': test_ppl,
                }
   
                sum_seq_length = 0
                total_sec = 0
                unclipped_gradient_norm_list.clear()
                training_loss = 0
                denominator_norm_list.clear()
                clip_list.clear()
    
                train.report(metrics)

        lr_scheduler.step()





def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target



def evaluate(net, eval_data, loss_func, n_token):
    net.eval()
    eval_loss = 0

    with torch.no_grad():
        for i in range(0, eval_data.size(0)-1, bptt):
            data, target = get_batch(eval_data, i, bptt)
            data, target = data.to(device), target.to(device)
            seq_len = data.size(0)
            out = net(data)
            output = out.view(-1, n_token)
            loss = loss_func(output, target)
            eval_loss += seq_len*loss.item()
    
    eval_loss /= len(eval_data)-1
    ppl = math.exp(eval_loss)
    return eval_loss, ppl


def train_transformer_ptb(optimiser_name, n_epochs, batch_size, update_period=10, lr=None, rho=None, weight_decay=0.0, random_seed = 10101010):
    param_space = {
        'optimizer_name': optimiser_name,
        'update_period': update_period if optimiser_name in ['sophia', 'ssdh', 'sadh', 'ssvp', ] else None,
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'random_seed': random_seed
    }

    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


   
    if batch_size == 128:
        get_best_optimiser_config_transformer_ptb_bs128(param_space)
  
    

    train_data, valid_data, test_data, n_tokens = get_ptb_dataset(batch_size, batch_size)
    data = {
        'train_data': train_data,
        'valid_data': valid_data,
        'test_data': test_data,
        'n_tokens': n_tokens,
    }


    # ray.shutdown()
    # ray.init(include_dashboard=False)
    abs_path = os.path.abspath('./results')
    train_with_parameters = tune.with_parameters(train_func, data=data)


    ############ comment out the following block of code when trainining on cpu
    resource_group = tune.PlacementGroupFactory([{'GPU': 1, }])
    train_with_resource = tune.with_resources(train_with_parameters, resource_group)
    ###########################
    
    experiment_path_name = 'train_transformer_{}_bs{}_rs{}'.format(optimiser_name, batch_size, random_seed)

    tuner = tune.Tuner(
        train_with_resource, # comment out this when training on cpu
        # train_with_parameters,  # uncomment this when training on cpu
        run_config=air.RunConfig(storage_path=abs_path, name=experiment_path_name, 
    #      callbacks=[WandbLoggerCallback(
    #     project='train_transformer',
    #     api_key="adbf7946ea0814850ef843b51beb20153b3c55dc",
    #     log_config=True
    # )]
    ),
        tune_config=tune.TuneConfig(num_samples=1),
        param_space=param_space,
    )
    result = tuner.fit()
    df = result.get_dataframe()
    path_name = os.path.join(abs_path, experiment_path_name)
    print(f"Loading results from {experiment_path_name}...")
    csv_path_name = '{}.csv'.format(path_name)
    file_exists = os.path.isfile(csv_path_name)

    with open(csv_path_name, mode='a', newline='') as file:
        df.to_csv(file, header=not file_exists, index=False)

    # os.system('rm -rf /tmp/tmp*') 

    return result










if __name__ == '__main__':

    weight_decay=0.0
    bptt=35
    d_model=256
    n_head=4
    d_hid=256
    n_layers=4
    dropout=0.2



    print("Is CUDA or ROCm available? ", torch.cuda.is_available())
    if torch.cuda.is_available():
        
        print(torch.cuda.device_count(), " GPUs are available")
        print("Using GPU ", torch.cuda.current_device(), " ", torch.cuda.get_device_name(0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)



#################################
###### Run this to train the transformer on the PTB dataset
###### Supported optimisers: adam, sgdm, sophia, ssdh, sadh, ssvp
###### random seeds have been used: 10100202, 111011, 110111, 1010

    result = train_transformer_ptb('ssvp', n_epochs=64,batch_size=128,)
