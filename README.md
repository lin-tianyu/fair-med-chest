<p align="center">
  <img src="https://github.com/user-attachments/assets/4f31a7a5-2e2a-4a9d-9e4f-a0ccb933c222" width="150" />
</p>

<h1 align="center">Benchmarking demographic fairness for models in radiology</h1>
<h3 align="center">Yike Guo, Chengjie Lin, Tianyu Lin</h3>
<h4 align="center">Johns Hopkins University</h4>

> This is the course project repo of **EN.580.464.01 : Advanced Data Science for Biomedical Engineering**

## Selected dataset
- MIMIC-CXR
- CheXpert
- NIH-Lung
## Selected model
- CLIP
- BiomedCLIP
- MedCLIP
- PubMedCLIP
## Selected disease
- 7 common diseases: ['Edema', 'Atelectasis', 'Pneumothorax', 'Consolidation', 'No Finding', 'Cardiomegaly', 'Pneumonia']
- Steps
  - `preprocessing`: find corresponding `datasets.py` and change to the disease name (no finding -> disease name)
  - `config`: ['diseased','normal'] -> ['normal',disease name]
    - 'no finding' is 1
    - disease_name is 1
## How to evaluate model utility and fairness using demo webpage
1. Go to `metriceval` folder
2. Go to disease folder,e.g. `no_finding`
3. Organize your pkl files and metata files as
```
├── CXP
│   ├── CHEXPERT DEMO.xlsx
│   ├── predictions_BiomedCLIP.pkl
│   ├── predictions_CLIP.pkl
│   ├── predictions_MedCLIP.pkl
│   ├── predictions_PubMedCLIP.pkl
│   └── test_with_metadata.csv
├── MIMIC
│   ├── predictions_BiomedCLIP.pkl
│   ├── predictions_CLIP.pkl
│   ├── predictions_MedCLIP.pkl
│   ├── predictions_PubMedCLIP.pkl
│   ├── test.csv
│   ├── test_age.csv
│   └── test_race.csv
└── NIH
    ├── predictions_BiomedCLIP.pkl
    ├── predictions_CLIP.pkl
    ├── predictions_MedCLIP.pkl
    ├── predictions_PubMedCLIP.pkl
    └── test_meta_FM.csv
```
4. Run python gradio/gradio_FM.py
## Reference
```
@article{jin2024fairmedfm,
  title={FairMedFM: Fairness Benchmarking for Medical Imaging Foundation Models},
  author={Jin, Ruinan and Xu, Zikang and Zhong, Yuan and Yao, Qiongsong and Dou, Qi and Zhou, S Kevin and Li, Xiaoxiao},
  journal={arXiv preprint arXiv:2407.00983},
  year={2024}
}
```

