from pathlib import Path
from itertools import product
from numpy import ceil as np_ceil
from numpy import stack as np_stack
from rasterio import open as rio_open
from numpy import moveaxis as np_moveaxis
from tensorflow.keras.utils import Sequence
from random import shuffle as random_shuffle
from window_generator import generate_windows
from tensorflow.keras.utils import to_categorical

class RasterDataGenerator(Sequence):
    def __init__(
        self,  
        map_dict,
        channels,
        img_height,
        img_width,
        win_height,
        win_width,
        min_hoverlap,
        min_woverlap,
        cls_count,
        # <augs>: list of tuples like (fn, flag), 
            # fn: aug function works on channel first image
            # flag: wheather fn should be applied on labels
        augs=None, 
        boundless=False,
        shuffle=True,
        batch_size=1,
    ):
        assert isinstance(
            map_dict, dict
        ), 'Invalid type for parameter <map_dict>, expected type `dict`!'
        
        assert all(
            [
                set(map_dict[k].keys()) == {'IMAGE', 'LABEL'} 
                for k in map_dict.keys()
            ]
        ), "Invalid map <dict_map>, Key Mismatch!"

        if augs is None:
            augs = (((lambda m : m), True),)
        else:
            assert isinstance(augs, (tuple, list)) and all(
                [
                    callable(fn) and isinstance(flag, bool) 
                    for (fn, flag) in augs
                ]
            )

        couples =  [
            (
                Path(couple['IMAGE']).as_posix(),
                Path(couple['LABEL']).as_posix()
            ) 
            for couple in map_dict.values()
        ]

        windows = list(
            generate_windows(
                img_height=img_height,
                img_width=img_width,
                win_height=win_height,
                win_width=win_width,
                min_hoverlap=min_hoverlap,
                min_woverlap=min_woverlap,
                boundless=boundless
            )
        )

        dat = list(product(couples, windows, augs))
        if shuffle:
            random_shuffle(dat)

        self.data = dat
        self.channels = channels
        self.class_count = cls_count
        self.batch_size = batch_size
        self.boundless_flag = boundless
    
    def __len__(self):
        return int(np_ceil(len(self.data) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        current_batch = self.data[
            idx * self.batch_size:(idx + 1) * self.batch_size
        ]
        islices = list()
        lslices = list()
        for (im, lb), (_, w), (aug, af) in current_batch:
            with rio_open(im, 'r') as isrc:
                islice = isrc.read(
                    indexes=self.channels, 
                    window=w, 
                    boundless=self.boundless_flag, 
                    masked=False
                )
                islice = aug(islice)
                islice = np_moveaxis(a=islice, source=0, destination=-1)
                islices.append(islice)
            with rio_open(lb, 'r') as lsrc:
                lslice = lsrc.read(window=w, boundless=self.boundless_flag, masked=False)
                if af is True:
                    lslice = aug(lslice)
                lslice = np_moveaxis(a=lslice, source=0, destination=-1)
                lslice =to_categorical(
                    y=(lslice-1), 
                    num_classes=self.class_count
                )
                lslices.append(lslice)
        ibatch = np_stack(islices, axis=0)
        lbatch = np_stack(lslices, axis=0)
        return ibatch, lbatch
