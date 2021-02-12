# DeepLabV3+ Model focusing on Earth Observation Imagery

* Example of config files are located at `Configs` directory.
* It is **recommended** to organize the data according to the provided directory structure
* Update the Data Configs accordingly
	+ `Configs/Train_Map.json`
	+ `Configs/Validation_Map.json`
	+ `Configs/Test_Map.json`
* For Input Specification See `Configs/Input_Spec.json` and modify as necessary
* `deeplab_train.py` contains an example script for training
	+ Model with highest validation accuracy will be saved in `Models` directory
	+ Training history will be saved in `csv` format along side the model
* Training Logs will be saved under `Logs` directory
	+ During Training, the process can be visualized by starting a `Tensorboard` server
* Prediction can be performed using the `predict_image.py` script
