# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from PIL import Image
import torch.utils.data as data
import sys
import numpy as np
import six
import torch.utils.data as data
import pickle

try:
    import pyarrow as pa
    import lmdb
except Exception as e:
    print(e)


def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist, root=None):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            if (root is not None):
                impath = os.path.join(root, impath)
            imlist.append((impath, int(imlabel)))

    return imlist


class ImageFromList(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.samples = flist_reader(flist, root=root)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target = self.samples[index]
        try:
            for i in range(5):
                try:
                    img = self.loader(impath)
                    break
                except:
                    continue
        except:
            print("corrupted image path: {}".format(impath))
            return None
            #sys.exit(1)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.samples)


class LMDBReader(object):
    def __init__(self, db_path):
        self.env = None
        self.txn = None
        self.db_path = db_path
        db_info = db_path.replace('.lmdb', '.info')
        self.length, self.keys = pickle.load(open(db_info, 'rb'))

    def _init(self):
        self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)

        self.txn = self.env.begin(write=False)

    def get_keys(self):
        return self.keys

    def __len__(self):
        return self.length

    def get(self, idx):
        if self.env is None:
            self._init()
        assert (idx >= 0 and idx < self.length)
        img, imlabel, vid = None, None, None

        byteflow = self.txn.get(self.keys[idx])
        unpacked = pa.deserialize(byteflow)
        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        imlabel = int(unpacked[1])

        return img, imlabel


class ImageFromLMDB(data.Dataset):
    def __init__(
            self,
            lmdb_file,
            flist,
            pair_size=0,
            transform=None,
            target_transform=None,
            flist_reader=default_flist_reader,
    ):
        self.db = LMDBReader(lmdb_file)
        self.samples = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        try:
            img, imlabel = self.db.get(index)
        except Exception as e:
            print(e)
            print('BAD DATA... ')
            sys.stdout.flush()
            return self.__getitem__(index - 1)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            imlabel = self.target_transform(imlabel)

        return img, imlabel

    def __len__(self):
        return len(self.db)
