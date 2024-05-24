import time
import os
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
from ray.train import Checkpoint
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



class Traianable_resnet18(Trainable):
    """
    Class for tuning hyperparameters for optimiser on training the ResNet18
    """



    def setup(self, config, data=None):
        self.lr = config['lr']
        self.is_second_order_opt = (config['optimizer_name'] in ['sophia', 'ssdh', 'sadh', 'ssvp', 'ssvp_sign', 'ssvp_cum_sign', 'asvp'])
        self.rho = config['rho'] if self.is_second_order_opt else None
        trainset = data['trainset']
        testset = data['testset']

        self.trainloader = DataLoader(trainset,batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
        # self.validloader = DataLoader(valid_subset,batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
        self.testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
        # self.n_epochs = config['n_epochs']
        self.T_max = config['T_max']
        self.loss_func = nn.CrossEntropyLoss()
        self.net = ResNet18().to(device)
        self.optimiser = get_optimizer(config, self.net)
        self.lr_scheduler = CosineAnnealingLR(self.optimiser, T_max=self.T_max+1,eta_min = 0.1*self.lr)

    def step(self):
        if not self.is_second_order_opt:
            training_loss, training_acc = train_first_ord_opt(self.net, self.optimiser, self.trainloader, self.loss_func)
            clips = 0
        else:
            training_loss, training_acc, clips = train_second_ord_opt(self.net, self.optimiser, self.trainloader, self.loss_func)
     
        test_loss, test_acc = test(self.net, self.testloader, self.loss_func)
        
        metrics = {
            'training_loss': training_loss,
            'clips': clips,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'training_acc': training_acc,
            'T_max': self.T_max}
        
        self.lr_scheduler.step()
        return metrics



    def save_checkpoint(self, tmp_checkpoint_dir):
        path = os.path.join(tmp_checkpoint_dir, 'checkpoint.pth')
        torch.save((self.net.state_dict(), self.optimiser.state_dict(), self.lr_scheduler.state_dict()), path)
        return tmp_checkpoint_dir
    
    def load_checkpoint(self, tmp_checkpoint_dir):
        path = os.path.join(tmp_checkpoint_dir, 'checkpoint.pth')
        net_state_dict, opt_state_dict, lr_scheduler_state_dict = torch.load(path)
        self.net.load_state_dict(net_state_dict)
        self.optimiser.load_state_dict(opt_state_dict)
        self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
        del net_state_dict
        del opt_state_dict
        del lr_scheduler_state_dict

    @classmethod
    def default_resource_request(cls, config):
        return tune.PlacementGroupFactory([{"GPU": 1}])



class Maximum_iteration_stopper(Stopper):
    def __init__(self, n_epochs):
        self._iter = 0
        self._n_epochs = n_epochs
    
    def __call__(self, trial_id, result):
        self._iter += 1
        if result['training_iteration'] >= result['T_max'] or result['training_iteration'] >= self._n_epochs:
            return True
        return False

    def stop_all(self) -> bool:
        return False
    



def test(model, test_loader, loss_func):
    # print('Testing')
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

def train_second_ord_opt(model, optimizer, train_loader, loss_func):
    training_loss = 0
    total_num  = 0
    correct = 0
    clip_list = []
    n_param = sum([p.numel() for p in model.parameters() if p.requires_grad])
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = loss_func(output, target)
        loss.backward(create_graph=True)

        training_loss += loss.item()*target.size()[0]
        total_num += target.size()[0]
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


        clips = [optimizer.state[p]['clips'] for group in optimizer.param_groups for p in group['params'] if p.requires_grad]
        clip_list.append(sum(clips)/n_param)

    training_loss /= total_num
    training_acc = correct/total_num
    c = sum(clip_list)/len(train_loader)

    return training_loss, training_acc, c

def train_first_ord_opt(model, optimizer, train_loader, loss_func):
    training_loss = 0
    total_num  = 0
    correct = 0

    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
  
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        training_loss += loss.item()*target.size()[0]
        total_num += target.size()[0]
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        optimizer.step()

        optimizer.zero_grad(set_to_none=True)
    training_loss /= total_num
    training_acc = correct/total_num
  

    return training_loss, training_acc, 


def tune_with_bohb(optimiser_name, n_samples,T_max, update_period=10, n_epochs=150,weight_decay=0.0, lr=None, rho=None, beta2=None):
    param_space = {
        'optimizer_name': optimiser_name,
        'lr': tune.uniform(5e-4, 2e-1) if lr is None else lr, # 0.2 - 0.0005
        'T_max': tune.choice([30, 60, 100, 180, 400, 1000, ]) if T_max is None else T_max,
        'rho': tune.uniform(1e-4, 1e-1) if rho is None and optimiser_name in ['sophia', 'ssdh', 'sadh', 'ssvp', 'ssvp_sign', 'ssvp_cum_sign', 'asvp'] else rho, # 0.02 - 0.0001
        'beta2': tune.uniform(0.985, 0.9999) if beta2 is None and optimiser_name in ['sophia', 'ssdh', 'sadh', 'ssvp', 'adam'] else beta2,
        'update_period': update_period,
        'weight_decay': weight_decay,
    }

    torch.cuda.manual_seed(111011)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    bohb_hyperband = HyperBandForBOHB(
        time_attr='training_iteration',
        max_t=min(T_max, n_epochs),
        reduction_factor=2,
    )
    bohb_search = TuneBOHB()
    bohb_search = ConcurrencyLimiter(bohb_search, max_concurrent=1)
    combined_stopper = CombinedStopper(
        Maximum_iteration_stopper(n_epochs),

    )

    trainset, testset = get_cifar10_dataset()


    datasets = {
        'trainset': trainset,
        'testset': testset,
    }

    ray.shutdown()
    ray.init(include_dashboard=False)
    # resource_group = tune.PlacementGroupFactory([{'GPU': 1, }])
    # train_with_resource = tune.with_resources(Traianable_resnet18, resource_group)

    abs_path = os.path.abspath('./results')

    experiment_path_name = 'hp_tuning_resnet18_{}_addbeta2_nepochs{}_update{}'.format(optimiser_name, update_period, T_max)

    tuner = tune.Tuner(
        tune.with_parameters(Traianable_resnet18, data=datasets),
        # Traianable_resnet18,
        run_config=train.RunConfig(
            storage_path=abs_path, 
            name=experiment_path_name,
            stop=combined_stopper,
            checkpoint_config=train.CheckpointConfig(num_to_keep=1),
        ),
        tune_config=tune.TuneConfig(
            metric='training_loss',
            mode='min',
            scheduler=bohb_hyperband,
            search_alg=bohb_search,
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

    os.system('rm -rf /tmp/tmp*') 

    print(f"Loading results from {experiment_path_name}...")
    csv_path_name = '{}.csv'.format(path_name)
    file_exists = os.path.isfile(csv_path_name)

    with open(csv_path_name, mode='a', newline='') as file:
        df.to_csv(file, header=not file_exists, index=False)
    
 

    return result


if __name__ == '__main__':
    batch_size = 256
    weight_decay=0.0


    print("Is CUDA or ROCm available? ", torch.cuda.is_available())
    if torch.cuda.is_available():
        
        print(torch.cuda.device_count(), " GPUs are available")
        print("Using GPU ", torch.cuda.current_device(), " ", torch.cuda.get_device_name(0))
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    props = torch.cuda.get_device_properties(device)
    print(torch.version.cuda)
    

    for opt_name in ['sadh', 'sophia', 'adam']:
        result = tune_with_bohb(opt_name, n_samples=32, T_max=128, n_epochs=128)

 