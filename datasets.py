# datasets.py
import torch
from torch.utils.data import Dataset
import h5py
import json
import os


def _pick_image_key(h5_file):
    """
    Try to find the dataset key holding image tensors.
    Heuristics: prefer names that look like images, else pick the only large 4D/3D dataset.
    """
    keys = list(h5_file.keys())
    # First pass: common names
    common = [
        "images",
        "imgs",
        "train_images",
        "val_images",
        "test_images",
        "encoded_images",
        "image_features",
    ]
    for k in common:
        if k in keys:
            return k

    # Second pass: look for datasets with plausible shapes
    candidates = []
    for k in keys:
        obj = h5_file[k]
        if isinstance(obj, h5py.Dataset):
            shape = obj.shape
            # Typical raw image shapes: (N, H, W, 3), (N, 3, H, W), or features (N, C, H, W)
            if len(shape) == 4 or len(shape) == 3:
                candidates.append((k, shape))
    if len(candidates) == 1:
        return candidates[0][0]

    return None


class CaptionDataset(Dataset):
    def __init__(
        self, data_folder, data_name, split, transform=None, hdf5_images_key=None
    ):
        """
        If hdf5_images_key is None, we will auto-detect the key.
        """
        self.split = split
        assert self.split in {"TRAIN", "VAL", "TEST"}

        self.data_folder = data_folder
        self.data_name = data_name
        self.hdf5_path = os.path.join(
            data_folder, f"{self.split}_IMAGES_{data_name}.hdf5"
        )
        self.captions_path = os.path.join(
            data_folder, f"{self.split}_CAPTIONS_{data_name}.json"
        )
        self.caplens_path = os.path.join(
            data_folder, f"{self.split}_CAPLENS_{data_name}.json"
        )

        if not os.path.isfile(self.hdf5_path):
            raise FileNotFoundError(f"HDF5 not found: {self.hdf5_path}")
        if not os.path.isfile(self.captions_path):
            raise FileNotFoundError(f"Captions JSON not found: {self.captions_path}")
        if not os.path.isfile(self.caplens_path):
            raise FileNotFoundError(f"Caplens JSON not found: {self.caplens_path}")

        # Load small JSONs
        with open(self.captions_path, "r") as j:
            self.captions = json.load(j)
        with open(self.caplens_path, "r") as j:
            self.caplens = json.load(j)

        self.transform = transform
        self.dataset_size = len(self.captions)

        # Resolve the image dataset key (auto-detect if not provided)
        self.images_key = hdf5_images_key
        with h5py.File(self.hdf5_path, "r") as h:
            if self.images_key is None:
                picked = _pick_image_key(h)
                if picked is None:
                    keys = list(h.keys())
                    raise KeyError(
                        "Could not locate an image dataset inside the HDF5 file.\n"
                        f"Available top-level keys: {keys}\n"
                        "If the image dataset is nested inside a group, rearrange your file or pass "
                        '`hdf5_images_key="<key>"` explicitly to CaptionDataset.'
                    )
                self.images_key = picked

            if self.images_key not in h:
                raise KeyError(
                    f"HDF5 key '{self.images_key}' not found in {self.hdf5_path}. "
                    f"Available keys: {list(h.keys())}"
                )

            # Optionally print structure once to help debugging
            # print("HDF5 keys:", list(h.keys()))
            # print("Using images key:", self.images_key, "with shape", h[self.images_key].shape)

    def __getitem__(self, i):
        # Open HDF5 lazily per item (safe for workers; no pickling of open handles)
        with h5py.File(self.hdf5_path, "r") as h:
            arr = h[self.images_key][i]  # numpy array
        img = torch.from_numpy(arr).float()
        # If raw images in 0..255, normalize to 0..1 for torchvision-like transforms
        if img.max() > 1.0:
            img = img / 255.0

        # Ensure channel-first (C,H,W)
        if img.ndim == 3 and img.shape[-1] in (1, 3):  # H,W,C -> C,H,W
            img = img.permute(2, 0, 1)

        if self.transform is not None:
            img = self.transform(img)

        cap = torch.tensor(self.captions[i], dtype=torch.long)
        caplen = torch.tensor([self.caplens[i]], dtype=torch.long)

        if self.split == "TRAIN":
            return img, cap, caplen
        else:
            # For VAL/TEST, return all captions for this image index if your JSON is structured that way.
            # If your VAL JSON stores multiple captions per index, this will work.
            # Otherwise, adjust to load the group of 5 captions matching the image.
            allcaps = torch.tensor(self.captions[i], dtype=torch.long)
            return img, cap, caplen, allcaps

    def __len__(self):
        return self.dataset_size
