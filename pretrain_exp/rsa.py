from rsatoolbox.data import Dataset
from rsatoolbox.rdm.calc import calc_rdm
import matplotlib.pyplot as plt
from numpy import max, random, triu, corrcoef, sqrt, squeeze, ones_like
from torch import load
from scripts.t1_dataset import T1Dataset
from scripts.data_handler import balance_dataset
from pandas import read_csv, concat, DataFrame
from scripts import constants
from tqdm import tqdm
from numpy import inf, zeros
from pretrain_exp.sfcn import SFCN, SFCN_TL
from collections import OrderedDict
from torchvision.models.feature_extraction import create_feature_extractor
from torch import cat, no_grad
import seaborn as sns
import pickle
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os


def plot_maps(model_features, model_name, class_labels, layers=None, colorbar=False):
    """
    Plots representational dissimilarity matrices (RDMs) across different layers of a model.

    Inputs:
    - model_features (dict): a dictionary where keys are layer names and values are numpy arrays representing RDMs for each layer.
    - model_name (str): the name of the model being visualized.
    """
    fig = plt.figure(figsize=(18, 4))
    fig.suptitle(f"RDMs across layers for {model_name}")
    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    if layers is not None:
        model_features = {k: model_features[k] for k in layers if k in model_features}
    gs = fig.add_gridspec(1, len(model_features))
    
    for l in range(len(model_features)):

        layer = list(model_features.keys())[l]
        map_ = squeeze(model_features[layer])

        if len(map_.shape) < 2:
            map_ = map_.reshape((int(sqrt(map_.shape[0])), int(sqrt(map_.shape[0]))))

        map_ = map_ / max(map_)

        ax = plt.subplot(gs[0,l])
        ax_ = ax.imshow(map_, cmap='magma_r')
        ax.set_title(f'{layer}')
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        yticks = build_class_ticks(class_labels, map_.shape[0], len(class_labels))
        ax.set_xticks(range(len(yticks)))
        ax.set_xticklabels(yticks)
        if l == 0:
            ax.set_yticks(range(len(yticks)))
            ax.set_yticklabels(yticks)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

    fig.subplots_adjust(right=0.9)
    if colorbar:
        cbar_ax = fig.add_axes([0.92, 0.18, 0.01, 0.53])
        cbar = fig.colorbar(ax_, cax=cbar_ax, ticks=[])
        cbar.set_label('Dissimilarity', rotation=270, labelpad=15)
    plt.tight_layout()
    plt.show()

    return fig


def build_class_ticks(class_labels, total_ticks, num_classes):
    ticks = ['' for _ in range(total_ticks)]
    ticks_pos = total_ticks // num_classes
    for i in range(num_classes):
        ticks[i*ticks_pos + ticks_pos // 2] = class_labels[i]
    ticks[total_ticks // 2] = '----'
    return ticks


def plot_correlations(correlations, title):
    correlations_lst = []
    for layer, values in correlations.items():
        for value in values:
            correlations_lst.append({'Layer': layer, 'Correlation': value})
    df = DataFrame(correlations_lst)
    plt.figure(figsize=(8, 4))
    sns.stripplot(x='Layer', y='Correlation', data=df, 
                    jitter=True, size=4, hue='Layer', legend=False, alpha=0.4)
    sns.pointplot(x='Layer', y='Correlation', data=df, 
                  linestyle='none', errorbar=('ci', 95), color='black', markers='D')
    plt.title(title)
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Correlation', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def load_task_rdms(rdms_path, task, filename='rdms.pkl'):
    task_path = rdms_path / task
    rdms_filepath = task_path / filename
    task_df = read_csv(task_path / 'task_df.csv', index_col=0)
    if not rdms_filepath.exists():
        raise FileNotFoundError(f"Missing files for task {task}")
    with open(rdms_filepath, 'rb') as f:
        rdms = pickle.load(f)
    return rdms, task_df


def calc_baseline_rdms(task_df, datapath):
    task_dataset = T1Dataset(constants.BRAIN_MASK, datapath, task_df, 250, 1, [20, 90], [inf, -inf], testing=True)
    indices = list(range(len(task_dataset)))
    imgs = load_imgs(task_dataset, indices).squeeze()
    zero_layer = {'input': imgs}
    _, baseline_rdms = calc_rdms(zero_layer)
    return baseline_rdms


def compare_rdms(rdms_dict1, rdms_dict2, layers_dict):
    for layer in layers_dict:
        rdm1 = rdms_dict1[layer]
        rdm2 = rdms_dict2[layer]
        mask = triu(ones_like(rdm1, dtype=bool), k=1)
        rdm1_flat = rdm1[mask]
        rdm2_flat = rdm2[mask]
        correlation = corrcoef(rdm1_flat, rdm2_flat)[0, 1]
        layers_dict[layer].append(correlation)


def _correlate_rdms(rdms_dict1, rdms_dict2, layers):
    """Compute per-layer correlations between two RDM dicts (non-mutating)."""
    correlations = {}
    for layer in layers:
        rdm1 = rdms_dict1[layer]
        rdm2 = rdms_dict2[layer]
        mask = triu(ones_like(rdm1, dtype=bool), k=1)
        correlations[layer] = corrcoef(rdm1[mask], rdm2[mask])[0, 1]
    return correlations


def _process_seed(seed, base_model_rdm, ft_rdm, pretrained_rdm, compare_with_pretrained, dataset_size, layers):
    """Process a single bootstrap seed: resample RDMs and compute correlations."""
    indices_seed = random_indices(dataset_size, seed)
    resampled_base_model_rdm = resample_rdm(base_model_rdm, indices_seed)

    result = {}
    if compare_with_pretrained:
        resampled_pt_rdm = resample_rdm(pretrained_rdm, indices_seed)
        result['pretrained'] = _correlate_rdms(resampled_base_model_rdm, resampled_pt_rdm, layers)

    resampled_ft_rdm = resample_rdm(ft_rdm, indices_seed)
    result['tl'] = _correlate_rdms(resampled_base_model_rdm, resampled_ft_rdm, layers)

    return result


def compare_models(tasks, models, layers, rdms_path, n_iters, random_state, n_jobs=None):
    if n_jobs is None:
        n_jobs = os.cpu_count()
    models_comparison = {}
    training_modes = ['pretrained', 'tl']
    other_models = models + ['baseline']
    random_seeds = [random_state + i for i in range(n_iters)]
    chunksize = max([1, n_iters // n_jobs])

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for task in tasks:
            task_rdms, task_df = load_task_rdms(rdms_path, task)
            models_comparison[task] = {}
            for training_mode in training_modes:
                for model in models:
                    if training_mode == 'pretrained' and model == 'none':
                        continue
                    model_rdms = task_rdms[model]
                    base_model_rdm = model_rdms[training_mode]
                    models_comparison[task][f'{model}_{training_mode}'] = {}
                    for other_model in other_models:
                        compare_with_pretrained = training_mode == 'pretrained' and other_model != 'baseline'
                        other_model_rdms = task_rdms[other_model]
                        training_mode_dct = models_comparison[task][f'{model}_{training_mode}']
                        if compare_with_pretrained:
                            training_mode_dct[f'{other_model}_pretrained'] = {layer: [] for layer in layers}
                            pretrained_rdm = other_model_rdms[training_modes[0]]
                        else:
                            pretrained_rdm = None
                        training_mode_dct[f'{other_model}_tl'] = {layer: [] for layer in layers}
                        ft_rdm = other_model_rdms[training_modes[1]] if other_model != 'baseline' else other_model_rdms

                        process_fn = partial(
                            _process_seed,
                            base_model_rdm=base_model_rdm,
                            ft_rdm=ft_rdm,
                            pretrained_rdm=pretrained_rdm,
                            compare_with_pretrained=compare_with_pretrained,
                            dataset_size=len(task_df),
                            layers=layers,
                        )
                        results = list(executor.map(process_fn, random_seeds, chunksize=chunksize))

                        for result in results:
                            if compare_with_pretrained and 'pretrained' in result:
                                for layer, corr in result['pretrained'].items():
                                    training_mode_dct[f'{other_model}_pretrained'][layer].append(corr)
                            for layer, corr in result['tl'].items():
                                training_mode_dct[f'{other_model}_tl'][layer].append(corr)

    return models_comparison


def random_indices(dataset_size, random_state):
    rng = random.default_rng(random_state)
    indices = list(range(dataset_size))
    random_indices = rng.choice(indices, size=len(indices), replace=True)
    return random_indices


def resample_rdm(rdms, indices):
    resampled_rdms = {layer: rdms[layer][indices][:, indices] for layer in rdms}
    return resampled_rdms


def calc_rdms(model_features, method='correlation'):
    """
    Calculates representational dissimilarity matrices (RDMs) for model features.

    Inputs:
    - model_features (dict): A dictionary where keys are layer names and values are features of the layers.
    - method (str): The method to calculate RDMs, e.g., 'correlation'. Default is 'correlation'.

    Outputs:
    - rdms (pyrsa.rdm.RDMs): RDMs object containing dissimilarity matrices.
    - rdms_dict (dict): A dictionary with layer names as keys and their corresponding RDMs as values.
    """
    ds_list = []
    for l in range(len(model_features)):
        layer = list(model_features.keys())[l]
        feats = model_features[layer]

        if type(feats) is list:
            feats = feats[-1]

        feats = feats.cpu()

        if len(feats.shape) > 2:
            feats = feats.flatten(1)

        feats = feats.detach().numpy()
        ds = Dataset(feats, descriptors=dict(layer=layer))
        ds_list.append(ds)

    rdms = calc_rdm(ds_list, method=method)
    rdms_dict = {list(model_features.keys())[i]: rdms.get_matrices()[i] for i in range(len(model_features))}

    return rdms, rdms_dict


def load_pretrained_model(pt_path, ckpt_filename, model_name, device):
    model_path = pt_path / model_name / ckpt_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Pretrained model not found at {model_path}")
    checkpoint = load(model_path, map_location=device, weights_only=False)
    if model_name == 'age':
        output_dim = 75
    elif model_name == 'bmi':
        output_dim = 50
    else:
        output_dim = 1
    ckpt_model_state = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:]
        ckpt_model_state[name] = v
    model = SFCN(output_dim=output_dim)
    model.load_state_dict(ckpt_model_state, strict=False)
    model = model.to(device)
    model.eval()
    
    return model


def load_tl_model(models_path, model_name, pretrained_model, task, suffix, device):
    if task == 'ad_vs_mci':
        prefix = 'dvp'
    elif task == 'mci_vs_hc':
        prefix = 'hvp'
    elif task == 'ad_vs_hc':
        prefix = 'dvh'
    else:
        raise ValueError(f"Unknown task: {task}")
    model_path = models_path / model_name / task / f'{prefix}_train{suffix}.pth.tar'
    if not model_path.exists():
        raise FileNotFoundError(f"TL model not found at {model_path}")
    checkpoint = load(model_path, map_location=device, weights_only=False)
    ckpt_model_state = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:]
        ckpt_model_state[name] = v
    model = SFCN_TL(pretrained_model, num_classes=1)
    model.load_state_dict(ckpt_model_state, strict=False)
    model = model.to(device)
    model.eval()

    return model


def sample_dataset(dataset_df, n_subjects, random_state=42):
    if n_subjects > len(dataset_df):
        n_subjects = len(dataset_df)
    subset_df = dataset_df.groupby('dx').apply(lambda x: x.sample(n_subjects // 2, random_state=random_state),
                                               include_groups=False).reset_index().drop(columns=['level_1'])
    dataset_df = dataset_df[~dataset_df['image_id'].isin(subset_df['image_id'])]
    return subset_df, dataset_df


def sample_dataset_sequentially(dataset_df, n_subjects):
    if n_subjects > len(dataset_df):
        n_subjects = len(dataset_df)
    subset_df = dataset_df.iloc[:n_subjects]
    dataset_df = dataset_df.iloc[n_subjects:]
    return subset_df, dataset_df


def load_dataset(datasets_path, task, splits_path, split, n_subjects=None, random_state=42):
    balanced_csv = datasets_path / task / f'balanced_{split}.csv'
    if balanced_csv.exists():
        balanced_df = read_csv(balanced_csv)
    else:
        split_csv = read_csv(datasets_path / task / splits_path / f'{split}.csv')
        balanced_df = balance_dataset(split_csv, 'dx')
        balanced_df.to_csv(balanced_csv, index=False)
    if n_subjects is not None:
        balanced_df, _ = sample_dataset(balanced_df, n_subjects, random_state=random_state)
    return balanced_df


def load_imgs(dataset, indices):
    imgs_tensor = []
    for i in indices:
        img, *_ = dataset[i]
        imgs_tensor.append(img)
    imgs_tensor = cat(imgs_tensor, dim=0).unsqueeze(1)
    return imgs_tensor


def load_precomputed_rdms(rdms_file, models, training_modes, layers, dataset_size):
    if rdms_file.exists():
        with open(rdms_file, 'rb') as f:
            task_rdms_dict = pickle.load(f)
        for model in models:
            if model not in task_rdms_dict:
                task_rdms_dict[model] = {}
            for mode in training_modes:
                if mode not in task_rdms_dict[model]:
                    task_rdms_dict[model][mode] = {}
                for layer in layers:
                    if layer not in task_rdms_dict[model][mode]:
                        task_rdms_dict[model][mode][layer] = zeros((dataset_size, dataset_size))
    else:
        task_rdms_dict = {model: {mode: {layer: zeros((dataset_size, dataset_size)) for layer in layers} for mode in training_modes} for model in models}
    return task_rdms_dict


def task_rdms(task, models, training_modes, layers, pretrained_layers, datasets_path, datapath, splits_path, split, batch_size, save_path):
    task_df = load_dataset(datasets_path, task, splits_path, split)
    task_df = task_df.reset_index(names='idx')
    task_df.to_csv(save_path / 'task_df.csv', index=False)
    dataset_size = len(task_df)
    num_batches = dataset_size // batch_size
    if dataset_size % batch_size != 0:
        num_batches += 1
    print(f"Task: {task}, Models: {models}, Pretrained layers: {pretrained_layers}")
    print(f"Dataset size: {dataset_size}, Batch size: {batch_size}, Number of batches: {num_batches}")

    save_path.mkdir(parents=True, exist_ok=True)
    batches_dfs = []
    # assuming batch_size == half of what fits into memory
    for _ in range(num_batches):
        batch_df, task_df = sample_dataset_sequentially(task_df, batch_size)
        batches_dfs.append(batch_df)
    with open(save_path / 'batches.pkl', 'wb') as f:
        pickle.dump(batches_dfs, f)

    pairwise_batch_rdms(batches_dfs, models, datapath, dataset_size, batch_size, training_modes, layers, pretrained_layers, task, save_path)


def pairwise_batch_rdms(batches, models, datapath, dataset_size, batch_size, training_modes, layers, pretrained_layers, task_name, save_path):
    rdms_file = save_path / f'rdms.pkl'
    task_rdms_dict = load_precomputed_rdms(rdms_file, models, training_modes, layers, dataset_size)
    for first_half_idx, first_batch_df in enumerate(tqdm(batches)):
        for snd_half_idx in range(first_half_idx + 1, len(batches)):
            snd_batch_df = batches[snd_half_idx]
            batch_df = concat([first_batch_df, snd_batch_df], ignore_index=True)
            batch_dataset = T1Dataset(constants.BRAIN_MASK, datapath, batch_df, 250, 1, [20, 90], [inf, -inf], testing=True)
            batch_imgs_tensor = load_imgs(batch_dataset, range(len(batch_dataset)))
            for model_name in models:
                if model_name == 'baseline':
                    continue
                for training_mode in training_modes:
                    training_mode_dict = training_modes[training_mode]
                    model_path, model_filename = training_mode_dict['path'], training_mode_dict['ckpt_filename']
                    return_nodes = training_mode_dict['return_nodes']
                    model_dict = task_rdms_dict[model_name][training_mode]
                    if training_mode == 'pretrained':
                        model = load_pretrained_model(model_path, model_filename, model_name, 'cpu')
                    else:
                        pt_model = load_pretrained_model(training_modes['pretrained']['path'], training_modes['pretrained']['ckpt_filename'], model_name, 'cpu')
                        model = load_tl_model(model_path, model_name, pt_model, task_name, pretrained_layers, 'cpu')
                    for layer in return_nodes.items():
                        layer_name = layer[1]
                        method = 'correlation'
                        if layer_name == 'fc6' and (training_mode == 'tl' or model_name == 'sex'):
                            method = 'euclidean'
                        rdms_dict = compute_rdms(model, batch_imgs_tensor, {layer[0]: layer[1]}, method)
                        model_dict[layer_name][first_half_idx * batch_size:(first_half_idx + 1) * batch_size,
                                               first_half_idx * batch_size:(first_half_idx + 1) * batch_size] = rdms_dict[layer_name][:batch_size, :batch_size]
                        end_index = (snd_half_idx + 1) * len(snd_batch_df)
                        if len(snd_batch_df) < batch_size:
                            end_index = dataset_size
                        model_dict[layer_name][snd_half_idx * batch_size:end_index, snd_half_idx * batch_size:end_index] = rdms_dict[layer_name][batch_size:, batch_size:]
                        model_dict[layer_name][first_half_idx * batch_size:(first_half_idx + 1) * batch_size,
                                               snd_half_idx * batch_size:end_index] = rdms_dict[layer_name][:batch_size, batch_size:] 
                        model_dict[layer_name][snd_half_idx * batch_size:end_index,
                                               first_half_idx * batch_size:(first_half_idx + 1) * batch_size] = rdms_dict[layer_name][batch_size:, :batch_size]
                    with open(rdms_file, 'wb') as f:
                        pickle.dump(task_rdms_dict, f)


def compute_rdms(model, imgs, return_nodes, method='correlation'):
    activations = get_activations(model, imgs, return_nodes)
    _, rdms_dict = calc_rdms(activations, method)
    return rdms_dict


def get_activations(model, imgs, return_nodes):
    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
    with no_grad():
        activations = feature_extractor(imgs)
    activations = {k: v.cpu() for k, v in activations.items()}
    return activations
