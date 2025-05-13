# PACFNet-for-CVDs-detection
The code respository for PACFnet from PeerJ computer science paper < A progressive attention-based cross-modal fusion network for cardiovascular disease detection using synchronized ECG and PCG signals >

Dataset is Physionet2016 from https://physionet.org/content/challenge-2016/1.0.0/

## Dataset Preparation
Download training-a dataset from https://physionet.org/content/challenge-2016/1.0.0/

And change the path in the config.py

## Running the Code
### Environment & Libraries
The full libraries list is provided as a requirements.txt in this repo. Please create a virtual environment with conda or venv and run:
```
pip install -r requirements.txt
```
### Preprocessing the dataset
```
python load_data.py
```
### Training & Sampling
```
python training.py
```
