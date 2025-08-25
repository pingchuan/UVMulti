import models
import torch
import os
#from .. import models
import numpy as np
from torch.utils.data.sampler import Sampler
import itertools
import random


def generate_model(option, ema=False):
    if option.model == 'PolypUU':
        model = getattr(models, option.model)(option.num_class, option.feat_level, option.dropout)
    else:
        model = getattr(models, option.model)(option.num_class)
    if option.use_gpu:
        model.cuda()

    if option.load_ckpt is not None:
        model_dict = model.state_dict()

        if option.select_checkpoint is not None:
            root_dir = option.select_checkpoint
        elif option.pretrain is not None:
            root_dir = option.pretrain
        else:
            root_dir = f"{option.dataset}_{option.suffix}"
        load_ckpt_path = ''
        #load_ckpt_path = os.path.join(
         #   option.checkpoints,
         #   root_dir,
          #  f'{option.method}_{option.model}',
           # f'exp{option.expID}',
            #f'{option.load_ckpt}.pth'
      #  )
        #load_ckpt_path = r'E:/python project/Icme_polyp/PolypMix-main/checkpoints/kvasir_SEG1_normal/Supervised_UNet/exp8888/checkpoint_best.pth'
        if os.path.isfile(load_ckpt_path):
            print(f'Loading {option.method} checkpoint: {load_ckpt_path}')
            checkpoint = torch.load(load_ckpt_path)
            new_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            model.load_state_dict(model_dict)
            print('Done')
        else:
            print('No checkpoint found.')

    if ema:
        for param in model.parameters():
            param.detach_()
    return model





class TwoStreamBatchSampler_sup(Sampler):
    """Iterate two sets of indices, allowing incomplete batches."""

    def __init__(self, total_count, primary_count, primary_batch_size,secondary_batch_size, shuffle=False):
        super().__init__(data_source=None)

        self.indices = list(range(total_count))
        if shuffle:
            random.shuffle(self.indices)

        self.primary_indices = self.indices[:primary_count]
        self.primary_batch_size = primary_batch_size
        self.secondary_indices = self.indices[primary_count:]
        assert len(self.primary_indices) >= self.primary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)

        # Custom grouper to handle the final incomplete batch
        return self.custom_grouper(primary_iter, self.primary_batch_size)

    def custom_grouper(self, iterable, n):
        """Custom grouper that ensures the last batch is complete with all remaining elements."""
        batch = []
        for item in iterable:
            batch.append(item)
            if len(batch) == n:
                yield batch
                batch = []

        # If there are remaining items in the last batch, return them all
        if batch:
            yield batch  # Use the entire remaining batch if not enough for a full one

    def __len__(self):
        return (len(self.primary_indices) + self.primary_batch_size - 1) // self.primary_batch_size  # ceil(len/n)


class TwoStreamBatchSampler_2(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, total_count, primary_count, primary_batch_size, secondary_batch_size, shuffle=False):
        super().__init__(data_source=None)

        # Split the data into two parts
        first_part_count = 550  # The first 550 belong to dataset 1
        second_part_count = total_count - first_part_count  # The rest belong to dataset 2

        # Apply separate random seed for each part and split accordingly
        self.first_part_indices = list(range(first_part_count))
        self.second_part_indices = list(range(first_part_count, total_count))

        if shuffle:
            # Shuffle the first part with seed 1888
            random.seed(3911)
            random.shuffle(self.first_part_indices)

            # Shuffle the second part with seed 8888
            random.seed(10911)
            random.shuffle(self.second_part_indices)

        # Split first part by 10% into primary and secondary sets
        first_primary_count = int(54)
        self.first_primary_indices = self.first_part_indices[:first_primary_count]
        self.first_secondary_indices = self.first_part_indices[first_primary_count:]

        # Split second part by 10% into primary and secondary sets
        second_primary_count = int(0.1 * second_part_count)
        self.second_primary_indices = self.second_part_indices[:second_primary_count]
        self.second_secondary_indices = self.second_part_indices[second_primary_count:]

        # Combine both parts
        self.primary_indices = self.first_primary_indices + self.second_primary_indices
        self.secondary_indices = self.first_secondary_indices + self.second_secondary_indices

        self.primary_batch_size = primary_batch_size
        self.secondary_batch_size = secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
            grouper(primary_iter, self.primary_batch_size),
            grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

class TwoStreamBatchSampler1(Sampler):
    """Sampler that iterates over two sets of indices, with one set looping endlessly and the other looping once per epoch."""

    def __init__(self, total_count, primary_count, primary_batch_size, secondary_batch_size, shuffle=False):
        super().__init__(data_source=None)

        self.indices = list(range(total_count))
        if shuffle:
            random.shuffle(self.indices)

        self.primary_indices = self.indices[:primary_count]
        self.secondary_indices = self.indices[primary_count:]

        self.primary_batch_size = primary_batch_size
        self.secondary_batch_size = secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
            grouper(primary_iter, self.primary_batch_size),
            grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


class TwoStreamBatchSampler_ben(Sampler):
    def __init__(self, dataset, batch_size):
        super().__init__(data_source=dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.labeled_batch_size = batch_size // 2
        self.unlabeled_batch_size = batch_size // 2

        self.indices = list(range((self.dataset.get_unlabeled_length()) + (self.dataset.get_labeled_length())))

        # Shuffle the indices
        self.labeled_indices = self.indices[:self.dataset.get_labeled_length()]
        self.unlabeled_indices = self.indices[self.dataset.get_labeled_length():]


    def __iter__(self):
        #primary_iter = iterate_once(self.labeled_indices)
        primary_iter = iterate_eternally(self.labeled_indices)#
        secondary_iter = iterate_eternally(self.unlabeled_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
            grouper(primary_iter, self.labeled_batch_size),
            grouper(secondary_iter, self.unlabeled_batch_size))
        )
    def __len__(self):
        return min(
            len(self.dataset.get_unlabeled_length()) // self.unlabeled_batch_size
        )

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, total_count, primary_count, primary_batch_size, secondary_batch_size, shuffle=False):
        super().__init__(data_source=None)

        self.indices = list(range(total_count))
        if shuffle:
            random.shuffle(self.indices)

        self.primary_indices = self.indices[:primary_count]
        self.secondary_indices = self.indices[primary_count:]
        #print(self.primary_indices)
        #print(self.secondary_indices)
        self.primary_batch_size = primary_batch_size
        self.secondary_batch_size = secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0


    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
            grouper(primary_iter, self.primary_batch_size),
            grouper(secondary_iter, self.secondary_batch_size))
        )


    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    """
    Collect data into fixed-length chunks or blocks
    """
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


# base_seed should be large enough to keep 0 and 1 bits balanced
def set_seed(inc, base_seed=2023):
    # cuDNN
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enable = False

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    seed = base_seed + inc
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    os.environ['PYTHONHASHSEED'] = str(seed + 4)

    # # cuDNN
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = True
    #
    # seed = base_seed + inc
    # random.seed(seed)
    # np.random.seed(seed + 1)
    # torch.manual_seed(seed + 2)
    # torch.cuda.manual_seed(seed + 3)
