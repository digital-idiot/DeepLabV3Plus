import numpy as np
from tqdm import tqdm
import rasterio as rio
import tensorflow as tf
from pathlib import Path
from advanced_losses import DiceLossVariants
from window_generator import generate_windows
from tensorflow.keras.models import load_model
from tensorflow_addons.activations import mish

bands = (1, 2, 3, 4, 5)
win_height = 256
win_width = 256

color_map = {
    0: (0, 0, 0, 0),
    1: (255, 255, 255, 255),
    2: (0, 0, 255, 255),
    3: (0, 255, 255, 255),
    4: (0, 255, 0, 255),
    5: (255, 255, 0, 255),
    6: (255, 0, 0, 255)
}

data_dir = Path(r"Data/")
data_subdirs = (
    "Train_Data/",
    "Validation_Data/",
    "Test_Data/"
)

model_path = Path(r"Models_Rnd/Model_MaxAccuracy.h5")
pred_dir = Path(r"Prediction_Rnd/")

src_files = list()
dst_files = list()

for sd in data_subdirs:
    simgs = list((data_dir / sd / 'Images').glob('*.vrt'))
    dimgs = list()
    for sf in simgs:
        sname = sf.stem
        parts = sname.split('_', 1)
        parts[0] = 'PredictedLabel'
        parts[1] = parts[1] + '.tif'
        dname = '_'.join(parts)
        ddir = pred_dir / sd
        ddir.mkdir(parents=True, exist_ok=True)
        df = ddir / dname
        dimgs.append(df)
    src_files += simgs
    dst_files += dimgs

trained_model = load_model(
    str(model_path),
    custom_objects={
        'DiceLossVariants': DiceLossVariants(
            name='log-cosh',
            alpha=0.3
        ),
        'tf' : tf
    }
)

for src_img, dst_img in tqdm(list(zip(src_files, dst_files))):
    with rio.open(src_img, 'r') as src:
        wins = list(
            generate_windows(
                img_height=src.height,
                img_width=src.width,
                win_height=256,
                win_width=256,
                min_hoverlap=1,
                min_woverlap=1,
            )
        )

        meta = dict()
        meta['driver'] = 'GTiff'
        meta['BIGTIFF'] = False
        meta['nodata'] = 0
        meta['count'] = 1
        meta['dtype'] = np.uint8
        meta['width'] = src.meta['width']
        meta['height'] = src.meta['height']
        meta['crs'] = src.meta['crs']
        meta['transform'] = src.meta['transform']
        meta['tiled'] = True
        meta['BLOCKXSIZE'] = 256
        meta['BLOCKYSIZE'] = 256
        meta['compress'] = 'zstd'
        meta['ZSTD_LEVEL']= 10
        meta['MAX_Z_ERROR'] = 0
        meta['predictor'] = 2

        with rio.open(dst_img, 'w', **meta) as dst:
            for _, w in tqdm(wins):
                im_array = np.stack(
                    [
                        np.moveaxis(
                            src.read(window=w, indexes=bands), 0, -1
                        ),
                    ], axis=0
                )
                pred = trained_model.predict(
                    im_array
                )
                pred_array = np.argmax(pred[0], axis=-1).astype(np.uint8)
                pred_img = pred_array + 1
                dst.write(pred_img, indexes=1, window=w)

            dst.write_colormap(1, color_map)
