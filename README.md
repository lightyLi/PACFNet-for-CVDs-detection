# A progressive attention-based cross-modal fusion network for cardiovascular disease detection using synchronized ECG and PCG signals

>Abstract: Synchronized electrocardiogram (ECG) and phonocardiogram (PCG) signals offer complementary diagnostic insights crucial for enhancing cardiovascular disease (CVD) detection accuracy. However, existing deep learning methods often utilize single-modal data or employ simplistic early/late fusion strategies, which inadequately capture the complex, hierarchical interdependencies between these modalities, thereby limiting detection performance. This study introduces PACFNet, a novel Progressive Attention-based Cross-modal Feature Fusion Network, for end-to-end CVD detection. PACFNet features a three-branch architecture: two modality-specific encoders for ECG and PCG, and a central progressive selective attention-based cross-modal fusion encoder. A key innovation is its four-layer progressive fusion mechanism, which integrates multi-modal information from low-level morphological details to high-level semantic representations. This is achieved by Selective Attention Cross-Modal Fusion (SACMF) modules at each progressive level, employing cascaded spatial and channel attention to dynamically emphasize salient feature contributions across modalities, thus significantly enhancing feature learning. Signals are pre-processed using a beat-to-beat segmentation approach to analyze individual cardiac cycles. Experimental validation on the public PhysioNet 2016 dataset demonstrates PACFNet's state-of-the-art performance, achieving an accuracy of 97.7%, sensitivity of 98%, specificity of 97.3%, and an F1-score of 99.7%. Notably, PACFNet not only excels in multi-modal settings but also maintains robust diagnostic capabilities even with missing modalities, underscoring its practical effectiveness and reliability. The source code is publicly available on GitHub to ensure reproducibility and facilitate further research.



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
