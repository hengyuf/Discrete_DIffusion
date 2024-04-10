import torch
from torch.utils.data import DataLoader, ConcatDataset
from .dataset_text8 import Text8Dataset
from .dataset_enwik8 import EnWik8Dataset
from .dataset_bandit import BanditDataset

dataset_choices = {'text8_256', 'enwik8_blocksparse','bandit'}

def add_data_args(parser):

    # Data params
    parser.add_argument('--dataset', type=str, default='bandit', choices=dataset_choices)
    parser.add_argument('--validation', type=eval, default=True)

    # Train params
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=eval, default=False)


def get_data_id(args):
    return args.dataset


dataset_choices = {'text8_256', 'enwik8_blocksparse', 'bandit'}


def add_data_args(parser):

    # Data params
    parser.add_argument('--dataset', type=str,
                        default='bandit', choices=dataset_choices)
    parser.add_argument('--validation', type=eval, default=True)

    # Train params
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=eval, default=False)


def get_data_id(args):
    return args.dataset


def get_data(args):
    assert args.dataset in dataset_choices

    # Dataset
    if args.dataset == 'bandit':
        train = BanditDataset(seq_len=512, split='train')
        valid = BanditDataset(seq_len=512, split='valid')
        test = BanditDataset(seq_len=512, split='test')
        data_shape = (512,)  # T
        num_classes = 4


        train_loader = DataLoader(train, batch_size=args.batch_size,
                                  shuffle=True)
        eval_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)

    return train_loader, eval_loader, data_shape, num_classes

