import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from random import choice
from deeplab import Deeplabv3
from datetime import datetime
from functools import partial
from string import ascii_uppercase, digits
from advanced_losses import DiceLossVariants
from tensorflow.keras.optimizers import Adam
from imgblock_generator import RasterDataGenerator
from tensorflow.python.keras import metrics as kmetrics
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '-n', '--name', 
    type=str,
    dest='model_name', 
    required=False,
    help='Specify an unique name for the model'
)
args = arg_parser.parse_args()
if args.model_name is None:
    model_id = ''.join(choice(ascii_uppercase + digits) for i in range(8))
else:
    model_id = args.model_name

# Specify Inputs
log_dir = Path('Logs')
model_dir = Path('Models')
config_dir = Path('Configs')
train_config = config_dir / 'Train_Map.json'
valid_config = config_dir / 'Validation_Map.json'
test_config = config_dir / 'Test_Map.json'
input_config = config_dir/ 'Input_Spec.json'

with open(train_config.as_posix(), 'r') as tm:
    train_map = json.load(tm)

with open(valid_config.as_posix(), 'r') as tm:
    valid_map = json.load(tm)

with open(input_config.as_posix(), 'r') as ic:
    input_spec = json.load(ic)

# Configs
image_height= input_spec['image_height']
image_width = input_spec['image_width']
window_height = input_spec['window_height']
window_width = input_spec['window_width']
bands = input_spec['bands']
min_height_overlap = input_spec['min_height_overlap']
min_width_overlap = input_spec['min_width_overlap']
boundless_flag = input_spec['boundless_flag']
class_count = input_spec['class_count']
data_shuffle = input_spec['data_shuffle']
batchsize = input_spec['batchsize']
    
# Define Augmentations
geo_augs = (
    (partial(np.rot90, k=1, axes=(1, 2)), True),
    (partial(np.rot90, k=2, axes=(1, 2)), True),
    (partial(np.rot90, k=3, axes=(1, 2)), True),
    (partial(np.flip, axis=1), True),
    (partial(np.flip, axis=2), True)
)

# Prepare Data Generators
train_generator = RasterDataGenerator( 
    map_dict=train_map,
    channels=bands,
    img_height=image_height,
    img_width=image_width,
    win_height=window_height,
    win_width=window_width,
    min_hoverlap=min_height_overlap,
    min_woverlap=min_width_overlap,
    cls_count=class_count,
    boundless=boundless_flag,
    augs=geo_augs,
    shuffle=data_shuffle,
    batch_size=batchsize
)
valid_generator = RasterDataGenerator(
    map_dict=valid_map,
    channels=bands,
    img_height=image_height,
    img_width=image_width,
    win_height=window_height,
    win_width=window_width,
    min_hoverlap=1,
    min_woverlap=1,
    cls_count=class_count,
    augs=None,
    boundless=boundless_flag,
    shuffle=data_shuffle,
    batch_size=batchsize
)

# Stop Traing if Loss is Unchanged
stopper_vloss = EarlyStopping(
    monitor='val_loss', 
    mode='min',
    min_delta=1e-4,
    patience=5, 
    restore_best_weights=True,
    verbose=1
)

tb_monitor = TensorBoard(
    log_dir=log_dir, 
    histogram_freq=1, 
    write_graph=False, 
    write_images=False,
    update_freq='batch',
    profile_batch=0,
    embeddings_freq=0,
    embeddings_metadata=None
)

# Create Model
dv3 = Deeplabv3(
    input_shape=(window_height, window_width, len(bands)), 
    classes=class_count, 
    backbone='xception', 
    OS=8, 
    activation='softmax'
)

# Compile Model
dv3.compile(
        optimizer=Adam(learning_rate=1e-4), 
        loss=DiceLossVariants(
            name='log-cosh',
            alpha=0.3
        ),
        metrics=[
            kmetrics.categorical_accuracy,
            kmetrics.TruePositives(),
            kmetrics.TrueNegatives(),
            kmetrics.FalsePositives(),
            kmetrics.FalseNegatives(),
            kmetrics.Precision(),
            kmetrics.Recall(),
            kmetrics.AUC(curve='ROC'),
            kmetrics.AUC(curve='PR'),
            kmetrics.MeanIoU(num_classes=class_count)
        ]
    )

# Save Model @ Max Validation Accuracy
t_start = datetime.now()
start_stamp = t_start.strftime("D%d%m%YT%H%M%S.%0.4f")

model_path = str(model_dir) + '/' + model_id +'_VA{val_categorical_accuracy:0.4f}_VL{val_loss:0.4f}_EP{epoch:03d}_' + start_stamp + '.h5' 

backup_maxaccuracy = ModelCheckpoint(
    filepath=model_path, 
    monitor='val_categorical_accuracy', 
    mode='max', 
    verbose=1, 
    save_best_only=True
)

# Train Model
hist = dv3.fit(
    x=train_generator, 
    epochs=100, 
    validation_data=valid_generator,
    use_multiprocessing=False,
    callbacks=[
        tb_monitor,
        backup_maxaccuracy,
        stopper_vloss
    ]
)

t_end = now()
end_stamp = t_end.strftime("D%d%m%YT%H%M%S.%f")

hist_df = pd.DataFrame(hist.history)
hist_path = model_dir / 'TrainingHistory_{}_{}.csv'.format(
    start_stamp, end_stamp
)
with open(hist_path, mode='w') as fp:
    hist_df.to_csv(fp, index=False)
