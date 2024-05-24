from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import dataset
from torch.utils.data.dataloader import DataLoader

def yield_tokens(data_iter, tokenizer):
    for text in data_iter:
        yield tokenizer(text)

def batchify(data, bsz, device):
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N]``
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)


def get_ptb_dataset(train_bs, eval_bs, device=None):
    train_iter = PennTreebank(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocabulary = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), specials=['<unk>'])
    vocabulary.set_default_index(vocabulary["<unk>"])

    def data_process(raw_text_iter: dataset.IterableDataset):
        """Converts raw text into a flat Tensor."""
        data = [torch.tensor(vocabulary(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    train_iter, valid_iter, test_iter = PennTreebank()
    train_data = data_process(train_iter)
    valid_data = data_process(valid_iter)
    test_data = data_process(test_iter)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = batchify(train_data, train_bs, device)
    valid_data = batchify(valid_data, eval_bs, device)
    test_data = batchify(test_data, eval_bs, device)
    return train_data, valid_data, test_data, len(vocabulary)

