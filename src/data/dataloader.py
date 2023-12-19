import torch.utils.data
import torchaudio
import os
from utils import *
import random
from natsort import natsorted

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from torch.utils.data.distributed import DistributedSampler


class DemandDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, cut_len=16000 * 2):
        self.cut_len = cut_len
        self.clean_dir = os.path.join(data_dir, "source1")
        self.clean_dir2 = os.path.join(data_dir, "source2")
        self.noisy_dir = os.path.join(data_dir, "mixture")
        self.clean_wav_name = os.listdir(self.clean_dir)
        self.clean_wav_name = natsorted(self.clean_wav_name)
        self.clean_wav_name2 = os.listdir(self.clean_dir2)
        self.clean_wav_name2 = natsorted(self.clean_wav_name2)
        self.noisy_wav_name = os.listdir(self.noisy_dir)
        self.noisy_wav_name = natsorted(self.noisy_wav_name)

    def __len__(self):
        return len(self.clean_wav_name)

    def __getitem__(self, idx):
        clean_file = os.path.join(self.clean_dir, self.clean_wav_name[idx])
        clean_file2 = os.path.join(self.clean_dir2, self.clean_wav_name2[idx])
        noisy_file = os.path.join(self.noisy_dir, self.noisy_wav_name[idx])

        clean_ds, _ = torchaudio.load(clean_file)
        clean_ds2, _ = torchaudio.load(clean_file2)
        noisy_ds, _ = torchaudio.load(noisy_file)
        clean_ds = clean_ds.squeeze()
        clean_ds2 = clean_ds2.squeeze()
        noisy_ds = noisy_ds.squeeze()
        length = len(clean_ds)
        assert length == len(noisy_ds)

        if length > self.cut_len:
            # randomly cut 2 seconds segment
            wav_start = random.randint(0, length - self.cut_len)
            noisy_ds = noisy_ds[wav_start : wav_start + self.cut_len]
            clean_ds = clean_ds[wav_start : wav_start + self.cut_len]
            clean_ds2 = clean_ds2[wav_start : wav_start + self.cut_len]

        return clean_ds, clean_ds2, noisy_ds, length


def load_data(ds_dir, batch_size, n_cpu, cut_len):
    torchaudio.set_audio_backend("sox_io")  # in linux
    train_dir = os.path.join(ds_dir, "train")
    test_dir = os.path.join(ds_dir, "test")

    train_ds = DemandDataset(train_dir, cut_len)
    test_ds = DemandDataset(test_dir, cut_len)

    train_dataset = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(train_ds),
        drop_last=True,
        num_workers=n_cpu,
    )
    test_dataset = torch.utils.data.DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(test_ds),
        drop_last=False,
        num_workers=n_cpu,
    )

    return train_dataset, test_dataset
