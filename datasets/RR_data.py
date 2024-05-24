from torch.utils.data.dataset import Dataset
import torch
from torch.utils.data.dataloader import DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_list, labels_list):
        self.data = data_list
        self.labels = labels_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = self.data[idx]
        sample_label = self.labels[idx]
        return sample_data, sample_label



def get_RR_dataloader(net, batch_size):
    trainset = CustomDataset(net.X, net.Ytrue)
    n_test_samples = net.nsamples // 4
    test_X = torch.randn(n_test_samples, net.xdim)
    test_Y = net.Atrue@test_X.T
    

    testset = CustomDataset(test_X, test_Y.T)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle= True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    full_batch_loader = DataLoader(trainset, batch_size=len(trainset),)
    return train_loader, test_loader, full_batch_loader