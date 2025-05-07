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
## How to evaluate model utility and fairness
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
4. Run metric_evaluation.ipynb
## Reference
```
@article{jin2024fairmedfm,
  title={FairMedFM: Fairness Benchmarking for Medical Imaging Foundation Models},
  author={Jin, Ruinan and Xu, Zikang and Zhong, Yuan and Yao, Qiongsong and Dou, Qi and Zhou, S Kevin and Li, Xiaoxiao},
  journal={arXiv preprint arXiv:2407.00983},
  year={2024}
}
```

