import time
import os

import numpy as np
import shutil
import ray
from ray import train, tune
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
from ray.tune.stopper import Stopper, TrialPlateauStopper, CombinedStopper
import math

from optimiser_configuration import *

from ray.tune.search import ConcurrencyLimiter
from ray.tune.search import ConcurrencyLimiter
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ray.tune.logger import DEFAULT_LOGGERS
from ray.air.integrations.wandb import WandbLoggerCallback




class Trainable_transformer(Trainable):
    """
    class for tuning hyperparameters for optimisers training on the transformer
    """


    def setup(self, config, data=None):

        self.lr = config['lr']
        self.is_second_order_opt = (config['optimizer_name'] in ['sophia', 'ssdh', 'sadh', 'ssvp', 'asvp', 'ssvp_cum_sign'])
        self.rho = config['rho'] if self.is_second_order_opt else None

        self.T_max = config['T_max']
        self.train_data = data['train_data']
        self.valid_data = data['valid_data']
   
        self.n_tokens = data['n_tokens']
        self.max_batch_idx = len(self.train_data)-1
        self.net = TransformerModel(self.n_tokens, d_model, n_head, d_hid,n_layers, dropout).to(device)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimiser = get_optimizer(config, self.net)
        self.lr_scheduler = CosineAnnealingLR(self.optimiser, T_max=self.T_max+1, eta_min=0.1*self.lr)

    
    def step(self):

        if not self.is_second_order_opt:
            training_loss, training_ppl= train_first_order_opt_transformer(self.net, self.train_data, self.loss_func, self.optimiser, self.n_tokens)
            clips = 0
        else:
            training_loss, training_ppl, clips = train_second_order_opt_transformer(self.net, self.train_data, self.loss_func, self.optimiser, self.n_tokens)

        valid_loss, valid_ppl = evaluate(self.net, self.valid_data,self.loss_func, self.n_tokens)

        metrics = {
            'training_loss': training_loss,
            'valid_loss': valid_loss,
            'training_ppl': training_ppl,
            'valid_ppl': valid_ppl,
            'last_lr': self.lr_scheduler.get_last_lr()[0],
            'clips': clips,
            'T_max': self.T_max,
 
        }  
        self.lr_scheduler.step()

        return metrics
    
    def save_checkpoint(self, tmp_checkpoint_dir):
        path = os.path.join(tmp_checkpoint_dir, 'checkpoint.pth')
        torch.save((self.net.state_dict(), self.optimiser.state_dict(), self.lr_scheduler.state_dict()), path)
        return tmp_checkpoint_dir
    
    def load_checkpoint(self, tmp_checkpoint_dir):
        path = os.path.join(tmp_checkpoint_dir, 'checkpoint.pth')
        net_state_dict, opt_state_dict, lr_scheduler_state_dict,= torch.load(path)
        self.net.load_state_dict(net_state_dict)
        self.optimiser.load_state_dict(opt_state_dict)
        self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
        del net_state_dict
        del opt_state_dict
        del lr_scheduler_state_dict
    
    @classmethod
    def default_resource_request(cls, config):
        return tune.PlacementGroupFactory([{"GPU": 1}])




def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target



def train_first_order_opt_transformer(net, train_data, loss_func, optimiser, n_token):
    net.train()
    training_loss = 0
    ppl_list = []
    total_loss = 0
    log_interval = 100

    for batch, i in enumerate(range(0, train_data.size(0)-1, bptt)):
        optimiser.zero_grad()
        data, target = get_batch(train_data, i, bptt)
        data, target = data.to(device), target.to(device)
        
        seq_len = data.size(0)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        out = net(data)
        output = out.view(-1, n_token)
        loss = loss_func(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
        optimiser.step()
        total_loss += loss.item()

        training_loss += seq_len*loss.item()

        if batch % log_interval == 0 and batch > 0:
            current_loss= total_loss / log_interval
            ppl = math.exp(current_loss)
            ppl_list.append(ppl)
            total_loss = 0
            # print('batch: {}, current loss: {}, ppl: {}'.format(i, current_loss, ppl))

    training_loss /= train_data.size(0)-1
    avg_ppl = sum(ppl_list)/len(ppl_list) if len(ppl_list) != 0 else math.exp(training_loss)


    return training_loss, avg_ppl


def train_second_order_opt_transformer(net, train_data, loss_func, optimiser, n_token):
    net.train()
    training_loss = 0
    ppl_list = []
    total_loss = 0
    log_interval = 100
    n_param = sum([p.numel() for p in net.parameters() if p.requires_grad])
    clip_list = []

    for batch, i in enumerate(range(0, train_data.size(0)-1, bptt)):
        optimiser.zero_grad(set_to_none=True)
        data, target = get_batch(train_data, i, bptt)
        data, target = data.to(device), target.to(device)
        seq_len = data.size(0)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        out = net(data)
        output = out.view(-1, n_token)
        loss = loss_func(output, target)

        loss.backward(create_graph = True)
        torch.nn.utils.clip_grad_norm_(net.parameters(),max_norm=1)
        optimiser.step()
        total_loss += loss.item()
        training_loss += seq_len*loss.item()
        clips = [optimiser.state[p]['clips'] for group in optimiser.param_groups for p in group['params'] if p.requires_grad]
        clip_list.append(sum(clips)/n_param)

        if batch % log_interval == 0 and batch > 0:
            current_loss = total_loss / log_interval
            ppl = math.exp(current_loss)
            ppl_list.append(ppl)
            total_loss = 0
            # print('batch: {}, current loss: {}, ppl: {}'.format(i, current_loss, ppl))

    
    training_loss /= train_data.size(0)-1
    avg_ppl = sum(ppl_list)/len(ppl_list) if len(ppl_list) != 0 else math.exp(training_loss)
    c = sum(clip_list) / len(clip_list)

    return training_loss, avg_ppl, c


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

def tune_with_bohb(optimiser_name, n_samples, n_epochs=32, beta2=None, update_period=10, lr=None, rho=None, weight_decay=0.0):
    param_space = {
        'optimizer_name': optimiser_name,
        'weight_decay': weight_decay,
        'T_max': n_epochs,
        'batch_size': batch_size,
        'beta2': tune.uniform(0.985, 0.9999) if beta2 is None and optimiser_name in ['sophia', 'ssdh', 'sadh', 'ssvp', 'asvp', 'ssvp_cum_sign', 'adam'] else beta2,
        'rho': tune.uniform(1e-4, 1e-1) if rho is None and optimiser_name in ['sophia', 'ssdh', 'sadh', 'ssvp', 'asvp', 'ssvp_cum_sign'] else rho, # 0.1 to 0.0001
        'lr': tune.uniform(5e-5, 5e-2) if lr is None else lr, # 0.01 to 0.00005
        'update_period': update_period,
    }

    bohb_hyperband = HyperBandForBOHB(
        time_attr='training_iteration',
        max_t=n_epochs,
        reduction_factor=2,
        
    )
    bohb_search = TuneBOHB()
    bohb_search = ConcurrencyLimiter(bohb_search, max_concurrent=1)

    train_data, valid_data, test_data, n_tokens = get_ptb_dataset(batch_size, batch_size)
    data = {
        'train_data': train_data,
        'valid_data': valid_data,
        'test_data': test_data,
        'n_tokens': n_tokens,


    }

    # torch.cuda.manual_seed(2333333)
    torch.cuda.manual_seed(10100202) # random seed for tuining adam and sophia, sadh
    #torch.cuda.manual_seed(111011) # random seed for tuning ssvp, ssdh, and sgdm. dont work for adam and sophia for unknown reason
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    ray.shutdown()
    ray.init(include_dashboard=False)
    # resource_group = tune.PlacementGroupFactory([{'GPU': 1, }])
    # train_with_resource = tune.with_resources(Trainable_transformer, resource_group)
    abs_path = os.path.abspath('./results')
    
    experiment_path_name = 'hp_tuning_transformer_new_{}_testbatchsize{}_add_beta2_update{}'.format(optimiser_name,batch_size, update_period)

    tuner = tune.Tuner(
        tune.with_parameters(Trainable_transformer, data=data),
        run_config=train.RunConfig(
            storage_path=abs_path, 
            name=experiment_path_name,
            checkpoint_config=train.CheckpointConfig(num_to_keep=1),
            # callbacks=[WandbLoggerCallback(
            #         project="tune_transformer",
            #         api_key="adbf7946ea0814850ef843b51beb20153b3c55dc",
            #         log_config=True
            #     )]
        ),
        tune_config=tune.TuneConfig(
            metric='training_loss',
            mode='min',
            scheduler=bohb_hyperband,
            search_alg=bohb_search,
            num_samples=n_samples
        ),
        param_space=param_space,
    )
    result = tuner.fit()

    df = result.get_dataframe()
 
    path_name = os.path.join(abs_path, experiment_path_name)
    print(f"Loading results from {experiment_path_name}...")
    csv_path_name = '{}.csv'.format(path_name)
    file_exists = os.path.isfile(csv_path_name)

    path_name = os.path.join(abs_path, experiment_path_name)
    relative_path = os.path.join('./results', experiment_path_name)
    for root, dirs, files in os.walk(relative_path, topdown=False):
        for name in dirs:
            if name.startswith('checkpoint'):
                shutil.rmtree(os.path.join(root, name))
                print(f"Deleted {os.path.join(root, name)}")

    os.system('rm -rf /tmp/tmp*') 

    with open(csv_path_name, mode='a', newline='') as file:
        df.to_csv(file, header=not file_exists, index=False)

    return result

if __name__ == '__main__':
    # batch_size = 20 # small batch size
    batch_size=128  # large batch size
    # batch_size = 256 # much large batch size
    # batch_size = 512 # much much large batch size
    # weight_decay=0.0
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


    results = tune_with_bohb('sophia', n_samples=32, n_epochs=64)
