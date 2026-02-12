# Age-invariant brain MRI representations
This repository contains code and resources for the paper _Alzheimer’s Disease Brain Phenotypes are Age-dependent_ (2026) by Fermin Travi, Anushree Mehta, Eduardo Castro, Hongyang Li, Jenna Reinen, Amit Dhurandhar, Pablo Meyer Rojas, Diego Fernández Slezak, Guillermo A. Cecchi, and Pablo Polosecki.

![Age-invariant brain MRI architecture](figures/age_invariant_vae_diagram.png#gh-light-mode-only)
![Age-invariant brain MRI architecture - dark mode](figures/age_invariant_vae_diagram_dark.png#gh-dark-mode-only)
## Data structure
```aiignore
.
├── cfg/
│   └── <config_name>.yaml          # model architecture and training hyperparameters
├── checkpoints/
│   └── general/
│       └── <config_name>/          
│           └── <run_name>/
│               └── last.ckpt
├── datasets/
│   └── {general, diseased}/
│       └── <dataset_name>/         # dataset1 ... datasetN
│           ├── metadata/
│           │   └── <dataset_name>_image_baseline_metadata.csv
│           └── splits/
│               ├── test.csv
│               ├── train.csv
│               └── val.csv
└── evaluation/
    └── {general, diseased}/
        └── <split_name>/
            └── <config_name>/
                └── <run_name>/
                    └── subjects_embeddings.pkl
```

Metadata files `<dataset_name>_image_baseline_metadata.csv` must contain columns: `subject_id`, `age_at_scan`, `gender`, `bmi`, `dataset` (dataset source name), `image_path` (path to processed NifTI image relative to `datasets/{general, diseased}`).

Those from diseased and healthy controls datasets must also contain a `dx` column (e.g., Healthy Control, HC; Mild Cognitive Impairment, MCI; Alzheimer\'s Disease, AD) and `dvp`, `dvh`, `hvp` columns (indicating rows corresponding to either AD or MCI, AD or HC, and HC or MCI, respectively).

## How to run
### Setup
Python 3.10+ is required. It is recommended to create a virtual environment and install the dependencies using:
```bash
pip install -r requirements.txt
```
### Training
To train the variational autoencoders on the general population datasets, use:
```bash
python train.py --cfg <config_name> --epochs <n_epochs>
```
For example, to train the age-invariant VAE defined in the paper:
```bash
python train.py --cfg age_invariant --epochs 100
```

### Testing
Testing is done on the embeddings extracted from the general population and diseased and healthy controls datasets, training a shallow neural network to evaluate the information they codify:
```bash
python test.py <ckpt_path> --dataset <dataset_name> --cfg <config_name> --set <split_name> --label <target_label> --test_size <test_size> --epochs <n_epochs> --batch_size <batch_size>
```
Where `<ckpt_path>` is the relative path to the trained model checkpoint (e.g., `e100/last`), `<dataset_name>` is the name of the aggregated dataset to evaluate on (e.g., `general`), `<split_name>` is one of `val` or `test`, `<target_label>` is one of `age_at_scan`, `gender`, `bmi`, `dvp`, `dvh`, or `hvp`, and `<test_size>` is the proportion of the training set to use for evaluation. In the case of diseased and healthy controls datasets, add `--balance` to balance the classes for classification tasks. Add `--use_last` to use the hyperparameters from the last run.

For example, to evaluate age prediction on the test set of the general population datasets using the age-invariant VAE embeddings:
```bash
python test.py e100/last --cfg age_invariant --label age_at_scan --test_size 0.3 --use_last
```

To evaluate disease prediction (AD vs. HC) on the test set of the diseased and healthy controls datasets using the age-invariant VAE embeddings:
```bash
python test.py e100/last --dataset diseased --balance --cfg age_invariant --label dvh --use_last
```
### Plotting
To plot the results of the evaluations on phenotype estimation by the age-invariant, age-agnostic, and age-aware VAE embeddings in the test set of general populations datasets, use:
```bash
python plot.py -b -t age_at_scan bmi gender -d general -c age_invariant/e100 age_agnostic/e100 age_aware/e100 baseline
```

To plot the AUC-ROC results of the evaluations on balanced disease estimation by the age-invariant, age-agnostic, and age-aware VAEs embeddings in the test set of diseased and healthy controls datasets, use:
```bash
python plot.py -d diseased_balanced -c age_invariant/e100 age_agnostic/e100 age_aware/e100
```

To perform dimensionality reduction on the embeddings and plot all classes (e.g., AD, MCI, HC) matched by age and sex in 2D space, use:
```bash
python test.py e100/last --cfg <cfg_name> --dataset diseased_balanced --label dx --manifold pca
```
