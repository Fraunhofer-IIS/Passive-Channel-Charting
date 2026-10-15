import h5py
from numpy import random
import torch
from torch.utils.data import Dataset
import numpy as np

class CCDataset(Dataset):

    def __init__(self, fp_data, cc_data, mask=None, epoch_size = 10000, limit = None):
        """Dataset for Channel Charting

        Args:
            fp_data (numpy.ndarray): Fingerprinting data (CIR, reference, timestamp)
            cc_data (tuple): Channel Charting Data (indices, distances, weights)
            epoch_size (int, optional): Set dataset size to epoch_size. Will sample random from the whole dataset. Defaults to 50000.
            limit (_type_, optional): Set dataset size to limit. Will sample at fixed indices distributed equally across the whole dataset. Will override epoch_size. Defaults to None.
        """
        self.limit = limit
        self.epoch_size = epoch_size
        self.fp_data = fp_data
        self.cc_data = cc_data

        self.mask = mask

        
    def __len__(self):
        if self.limit:
            return self.limit
        return self.epoch_size
    
    def __getitem__(self, index):
        if self.limit:
            step_size = int(len(self.cc_data) / self.limit)
            index = index * step_size
        else:
            index = random.randint(0, len(self.cc_data))

        i1, i2 = self.cc_data["indices"][index]

        mask = self.mask if self.mask is not None else slice(None)
        cir1 = self.fp_data["CIR"][i1][ :, mask, :]
        cir2 = self.fp_data["CIR"][i2][ :, mask, :]

        distance = self.cc_data["distance"][index]

        weight = self.cc_data["weight"][index]

        return cir1.astype(np.float32), cir2.astype(np.float32), distance.astype(np.float32), weight.astype(np.float32)
    

def parse_link_list(antenna_pairs):
    """
    Makes sure, that the links are in both directions
    """
    link_list = []
    for links in antenna_pairs:
        link_list.append(f"{links[0]}-{links[1]}")
        link_list.append(f"{links[1]}-{links[0]}")
    return link_list

class FPDataset(Dataset):

    def __init__(self, fp_data, mask=None, epoch_size = 10000):
        """Dataset for Channel Charting

        Args:
            data (tuple): Channel Charting Data
            epoch_size (int, optional): Set dataset size to epoch_size. Will sample random from the whole dataset. Defaults to 50000.
            limit (_type_, optional): Set dataset size to limit. Will sample at fixed indices distributed equally across the whole dataset. Will override epoch_size. Defaults to None.
        """
        self.epoch_size = epoch_size
        self.fp_data = fp_data

        self.cirs = torch.tensor(fp_data["CIR"], dtype=torch.float32)
        self.mask = mask
        self.labels = torch.tensor(fp_data["reference"], dtype=torch.float32)
        
    def __len__(self):
        return len(self.fp_data)
    
    def __getitem__(self, index):
        cir = self.fp_data["CIR"][index]
        if self.mask is not None:
            cir = cir[ :, self.mask, :]
        position = self.fp_data["reference"][index]

        return cir.astype(np.float32), position.astype(np.float32)

def load_dataset(path: str, cc_limit: int = None, cc_epoch: int = None, force_fp = False, filter_links : list = None) -> tuple[torch.utils.data.Dataset, bool, int]:

    with h5py.File(path, "r") as f:
        links = f["fp"].attrs["a_ids"]
        if filter_links is not None:
            mask = np.bitwise_or.reduce(links == np.array(parse_link_list(filter_links))[:, None])
        else:
            mask = None

        if not force_fp and "cc" in f.keys():
            cc = True
            dataset = CCDataset(f["fp"][:], f["cc"][:], mask=mask, limit=cc_limit, epoch_size=cc_epoch)
        else:
            cc = False
            dataset = FPDataset(f["fp"][:], mask=mask)
    return dataset, cc, dataset.fp_data["CIR"].shape[1]
