from pathlib import Path
from scripts.constants import CFG_PATH, CHECKPOINT_PATH, EVALUATION_PATH
from scripts.data_handler import (get_datapath, load_set, create_test_splits, target_mapping,
                                  balance_dataset, save_predictions)
from scripts.embedding_dataset import EmbeddingDataset
from scripts.embedding_eval import test_on_val, test_classifier, train_classifier, compute_metrics, report_results, \
    add_baseline_results
from scripts.t1_dataset import T1Dataset
from scripts.utils import (load_yaml, reconstruction_comparison_grid, init_embedding, subjects_embeddings, load_model,
                           load_hyperparameters)
from lightning.pytorch import seed_everything
from models.utils import get_latent_representation
from tqdm import tqdm
from numpy import array, random
from pandas import DataFrame
from seaborn import scatterplot, color_palette, set_style, jointplot
from PIL import ImageDraw, ImageFont
import matplotlib.pyplot as plt
from torch import cat, device as dev, cuda
from wandb import Image
import multiprocessing as mp
import argparse


def _init_bootstrap_worker(classifier_, test_df_, target_label_, transform_fn_, binary_classification_,
                           bin_centers_, use_age_, device_, metrics_):
    # Globals used by workers
    global _BOOTSTRAP_CTX
    _BOOTSTRAP_CTX = {
        'classifier': classifier_,
        'test_df': test_df_,
        'target_label': target_label_,
        'transform_fn': transform_fn_,
        'binary_classification': binary_classification_,
        'bin_centers': bin_centers_,
        'use_age': use_age_,
        'device': device_,
        'metrics': metrics_,
    }


def _bootstrap_one(seed):
    ctx = _BOOTSTRAP_CTX
    test_resampled = ctx['test_df'].sample(frac=1, replace=True, random_state=int(seed))
    test_dataset = EmbeddingDataset(test_resampled,
                                    target=ctx['target_label'],
                                    transform_fn=ctx['transform_fn'])
    predictions, labels = test_classifier(ctx['classifier'], test_dataset,
                                          ctx['binary_classification'],
                                          ctx['bin_centers'],
                                          ctx['use_age'],
                                          ctx['device'])
    model_results = {m: [] for m in ctx['metrics']}
    baseline_results = {m: [] for m in ctx['metrics']}
    compute_metrics(predictions, labels, ctx['binary_classification'], model_results)
    rnd_gen = random.default_rng(seed=seed)
    add_baseline_results(labels, ctx['binary_classification'], baseline_results, rnd_gen)
    return model_results, baseline_results


def predict_from_embeddings(embeddings_df, cfg_name, dataset, ukbb_size, test_size, latent_dim, age_range, bmi_range,
                            target_label, target_dataset, batch_size, n_layers, epochs, lr, dp, n_iters, n_workers,
                            use_age, split_val, save_path, device):
    train, test = create_test_splits(embeddings_df, dataset, test_size, ukbb_size, target_dataset, n_upsampled=180)
    transform_fn, output_dim, bin_centers = target_mapping(embeddings_df, target_label, age_range, bmi_range)
    if split_val:
        _ = test_on_val(train, test_size, cfg_name, latent_dim, target_label, transform_fn, output_dim, bin_centers,
                        use_age, device, lr, dp, n_layers, batch_size, epochs)
    else:
        train_dataset = EmbeddingDataset(train, target=target_label, transform_fn=transform_fn)
        classifier = train_classifier(train_dataset, DataFrame(), cfg_name, latent_dim, output_dim, n_layers,
                                      bin_centers, use_age, batch_size, epochs, lr, dp, device,
                                      log=False, seed=42)
        rnd_gen = random.default_rng(seed=42)
        random_seeds = [rnd_gen.integers(1, 100000) for _ in range(n_iters)]
        binary_classification = output_dim == 1
        metrics = ['Accuracy', 'Precision', 'Recall'] if binary_classification else ['MAE', 'Corr', 'p_value']
        metrics.extend(['Predictions', 'Labels'])
        if n_workers > 1:
            with mp.get_context('spawn').Pool(
                    processes=n_workers,
                    initializer=_init_bootstrap_worker,
                    initargs=(classifier, test, target_label, transform_fn, binary_classification,
                              bin_centers, use_age, device, metrics)) as pool:
                results = []
                for result in tqdm(pool.imap(_bootstrap_one, random_seeds),
                                   total=len(random_seeds),
                                   desc='Bootstrapping test'):
                    results.append(result)
            model_results = {m: [] for m in metrics}
            baseline_results = {m: [] for m in metrics}

            for model_res, baseline_res in results:
                for metric in metrics:
                    model_results[metric].extend(model_res[metric])
                    baseline_results[metric].extend(baseline_res[metric])
        else:
            model_results = {m: [] for m in metrics}
            baseline_results = {m: [] for m in metrics}
            for seed in tqdm(random_seeds, desc='Bootstrapping test'):
                test_resampled = test.sample(frac=1, replace=True, random_state=seed)
                test_dataset = EmbeddingDataset(test_resampled, target=target_label, transform_fn=transform_fn)
                predictions, labels = test_classifier(classifier, test_dataset, binary_classification,
                                                      bin_centers, use_age, device)
                compute_metrics(predictions, labels, binary_classification, model_results)
                add_baseline_results(labels, binary_classification, baseline_results,
                                     random.default_rng(seed))
        params = {'cfg': cfg_name, 'dataset': dataset, 'target': target_label, 'n_iters': n_iters,
                  'batch_size': batch_size, 'n_layers': n_layers, 'epochs': epochs, 'dropout': dp, 'lr': lr}
        baseline_preds, _ = report_results(baseline_results, target_label, name='baseline')
        model_preds, labels = report_results(model_results, target_label, name=cfg_name)
        baseline_savepath = save_path.parents[1] / 'baseline' / 'random'
        save_predictions(test, model_preds, labels, target_label, params, save_path)
        save_predictions(test, baseline_preds, labels, target_label, params, baseline_savepath)


def sample(model, dataset, age, subject_id, device, save_path):
    seed_everything(42, workers=True)
    save_path = save_path / 'samples'
    save_path.mkdir(parents=True, exist_ok=True)
    sample = dataset.get_subject(subject_id)
    t1_img, _ = dataset.load_and_process_img(sample)
    t1_img = t1_img.unsqueeze(dim=0).to(device)
    z = get_latent_representation(t1_img, model.encoder)
    if age > 0.0:
        sample['age_at_scan'] = age
    age = dataset.age_mapping(sample['age_at_scan']).unsqueeze(dim=0)
    reconstructed = model.decoder(z, age.to(device))
    axes_comparisons, _ = reconstruction_comparison_grid(t1_img, reconstructed, 1, 50, 0)
    comparison = cat(axes_comparisons, dim=2)
    comparison_img = Image(comparison).image
    draw, font = ImageDraw.Draw(comparison_img), ImageFont.truetype("LiberationSans-Regular.ttf", 25)
    draw.text((225, 183), f'Age: {int(sample["age_at_scan"])}', (255, 255, 255), font=font)
    comparison_img_name = f'{subject_id}_age_{int(sample["age_at_scan"])}.png'
    comparison_img.save(save_path / comparison_img_name)
    plt.imshow(comparison_img)
    plt.axis('off'), plt.xticks([]), plt.yticks([])
    plt.show()
    print(f'Reconstructed MRI saved at {save_path / comparison_img_name}')


def plot_embeddings(subjects_df, method, label, save_path, color_by=None, annotate_ids=False):
    if label not in subjects_df:
        raise ValueError(f'{label} not found in the dataframe')
    seed_everything(42, workers=True)
    save_path.mkdir(parents=True, exist_ok=True)
    components = init_embedding(method).fit_transform(array(subjects_df['embedding'].to_list()))

    subjects_df['emb_x'], subjects_df['emb_y'] = components[:, 0], components[:, 1]
    subjects_df = subjects_df.sample(frac=0.3, random_state=42)
    set_style('white')
    if color_by:
        fig, ax = plt.subplots(figsize=(7, 7))
        color_by_is_float = subjects_df[color_by].dtype == 'float64'
        if color_by_is_float:
            subjects_df[color_by] = subjects_df[color_by].astype(int)
            palette = 'viridis_r'
        else:
            palette = color_palette()[2:4]
        scatter = scatterplot(data=subjects_df, x='emb_x', y='emb_y', hue=color_by, ax=ax, alpha=0.8,
                              size=.8, palette=palette, edgecolor='grey', linewidth=0.5, legend=False,
                              style=label)
        handles_scatter, labels_scatter = scatter.get_legend_handles_labels()
        if color_by_is_float:
            norm = plt.Normalize(subjects_df[color_by].min(), subjects_df[color_by].max())
            sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(color_by)
            handles_scatter, labels_scatter = handles_scatter[-2:], labels_scatter[-2:]
            labels_kde = labels_scatter
        else:
            handles_scatter = handles_scatter[1:3] + handles_scatter[-2:]
            labels_scatter = labels_scatter[1:3] + labels_scatter[-2:]
            labels_kde = labels_scatter[-2:]
        handles_kde = [plt.Line2D([0], [0], color=color_palette()[0]),
                       plt.Line2D([0], [0], color=color_palette()[1])]
        ax.legend(handles_scatter + handles_kde, labels_scatter + labels_kde,
                  loc='lower center', ncol=2)
        if annotate_ids:
            for i, subject_id in enumerate(subjects_df.index):
                ax.annotate(subject_id, (components[i, 0], components[i, 1]), alpha=0.6)
        ax.set_title(f'Latent representations {method.upper()} embeddings')
        ax.axes.xaxis.set_visible(False), ax.axes.yaxis.set_visible(False)
    else:
        unique_labels = subjects_df[label].unique()
        colors = color_palette('Set2', n_colors=len(unique_labels))
        label_to_color = dict(zip(unique_labels, colors))
        min_age = subjects_df['age_at_scan'].min()
        max_age = subjects_df['age_at_scan'].max()
        normalized_ages = (subjects_df['age_at_scan'] - min_age) / (max_age - min_age) + 0.15

        rgba_colors = []
        for idx, row in subjects_df.iterrows():
            rgb = label_to_color[row[label]]
            alpha = min(normalized_ages[idx], 1.0)
            rgba_colors.append((*rgb, alpha))
        joint_ax = jointplot(data=subjects_df, x='emb_x', y='emb_y', hue=label, alpha=0.9,
                             edgecolor='black', linewidth=.5, legend=False, palette=colors, height=7,
                             marginal_kws={'fill': True, 'linewidth': 2})
        for point, color in zip(joint_ax.ax_joint.collections[0].get_offsets(), rgba_colors):
            joint_ax.ax_joint.scatter(point[0], point[1], c=[color], linewidth=0.1, edgecolor='black', s=75)
        x_range = subjects_df['emb_x'].max() - subjects_df['emb_x'].min()
        y_range = subjects_df['emb_y'].max() - subjects_df['emb_y'].min()
        x_margin = x_range * 0.05
        y_margin = y_range * 0.05

        joint_ax.ax_marg_x.set_xlim(subjects_df['emb_x'].min() - x_margin,
                                    subjects_df['emb_x'].max() + x_margin)
        joint_ax.ax_marg_y.set_ylim(subjects_df['emb_y'].min() - y_margin,
                                    subjects_df['emb_y'].max() + y_margin)
        joint_ax.ax_joint.collections[0].remove()
        joint_ax.ax_joint.axes.xaxis.set_visible(False)
        joint_ax.ax_joint.axes.yaxis.set_visible(False)
        joint_ax.ax_marg_x.axes.xaxis.set_visible(False)
        joint_ax.ax_marg_y.axes.yaxis.set_visible(False)

    plt.tight_layout()
    filename = save_path / f'latents_{method}_{label}.png'
    plt.savefig(filename, dpi=300, transparent=True, bbox_inches='tight')
    plt.show()
    print(f'Figure saved at {filename}')
    return plt.gcf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('weights', type=str, help='checkpoint file')
    parser.add_argument('--ckpt_dataset', type=str, default='general',
                        help='dataset on which the checkpoint file was trained')
    parser.add_argument('--dataset', type=str, default='general', help='dataset on which to train the predictor')
    parser.add_argument('--splits_path', type=str, default='splits', help='path to the data splits')
    parser.add_argument('--target', type=str, default='general', help='target dataset for predicting features')
    parser.add_argument('--cfg', type=str, default='age_agnostic', help='config file used for the trained model')
    parser.add_argument('--device', type=str, default='gpu', help='device used for training and evaluation')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size used for training the age classifier')
    parser.add_argument('--epochs', type=int, default=50,
                        help='max number of epochs used for training the embedding classifier')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate used for training the embedding classifier')
    parser.add_argument('--dp', type=float, default=0.0,
                        help='dropout probability used for training the embedding classifier')
    parser.add_argument('--n_iters', type=int, default=1000,
                        help='number of iterations (with different seeds) to evaluate the classifier')
    parser.add_argument('--n_layers', type=int, default=2, help='number of layers in the classifier')
    parser.add_argument('--n_workers', type=int, default=12, help='number of workers for bootstrapping')
    parser.add_argument('--use_last', action='store_true', help='use the hyperparameters from the last run')
    parser.add_argument('--use_age', action='store_true', help='add age as a feature to the classifier')
    parser.add_argument('--sample', type=int, default=0, help='subject id from which to reconstruct MRI data')
    parser.add_argument('--age', type=float, default=0.0, help='age of the subject to resample to, if using ICVAE')
    parser.add_argument('--manifold', type=str, default=None,
                        help='Method to use for manifold learning (PCA, MDS, tSNE, Isomap)')
    parser.add_argument('--label', type=str, default='age_at_scan',
                        help='label used for prediction and plotting latent representations'),
    parser.add_argument('--color_label', type=str, default=None,
                        help='label used for coloring the embeddings when doing manifold learning')
    parser.add_argument('--set', type=str, default='test', help='set to evaluate (val or test)')
    parser.add_argument('--balance', action='store_true', help='balance the dataset by age and sex')
    parser.add_argument('--ukbb_size', type=float, default=0.15,
                        help='size of the validation split constructed from the ukbb set to evaluate')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='size of the test split constructed from the training set')
    parser.add_argument('--random_state', type=int, default=42, help='random state for reproducibility')
    parser.add_argument('--val', action='store_true',
                        help='split train to perform hyperparameter tuning')
    args = parser.parse_args()

    config = load_yaml(Path(CFG_PATH, f'{args.cfg}.yaml'))
    weights_path = Path(CHECKPOINT_PATH, args.ckpt_dataset, args.cfg, args.weights)
    run_name = weights_path.parent.name
    save_path = Path(EVALUATION_PATH, args.dataset, args.set, args.cfg) / run_name
    save_path.mkdir(parents=True, exist_ok=True)

    datapath = get_datapath(args.dataset)
    if args.sample > 0:
        data, age_range, bmi_range = load_set(args.dataset, args.set, args.splits_path, args.random_state)
        if args.age > 0 and not age_range[0] < args.age < age_range[1]:
            print(f'age {args.age} is not within the training range of {age_range[0]} and {age_range[1]}')
        dataset = T1Dataset(config['input_shape'], datapath, data, config['latent_dim'], config['age_dim'],
                            age_range, bmi_range, testing=True)
        device = dev('cuda' if args.device == 'gpu' and cuda.is_available() else 'cpu')
        model = load_model(weights_path, config, device)
        sample(model, dataset, args.age, args.sample, device, save_path)
    else:
        embeddings_df = subjects_embeddings(weights_path, args.cfg, args.dataset, config, args.set, datapath,
                                            args.splits_path, args.random_state, save_path)
        if args.label not in embeddings_df:
            raise ValueError(f'Label {args.label} not found in the embeddings dataframe')
        embeddings_df = embeddings_df[~embeddings_df[args.label].isna()]
        if embeddings_df.empty:
            raise ValueError(f'No embeddings found for label {args.label} in the dataset {args.dataset}')
        if args.use_age:
            run_name += '_with_age'
        if args.balance:
            embeddings_df = balance_dataset(embeddings_df, args.label)
            print(embeddings_df.groupby(args.label)['age_at_scan'].describe().iloc[:, :7])
            print(embeddings_df.groupby(args.label)['gender'].describe())
            save_path = Path(EVALUATION_PATH, args.dataset + '_balanced', args.set, args.cfg) / run_name
            save_path.mkdir(parents=True, exist_ok=True)

        if not args.manifold:
            if args.use_last:
                load_hyperparameters(save_path, args)
            data, age_range, bmi_range = load_set(args.dataset, args.set, args.splits_path, args.random_state)
            predict_from_embeddings(embeddings_df, args.cfg, args.dataset, args.ukbb_size, args.test_size,
                                    config['latent_dim'], age_range, bmi_range, args.label, args.target,
                                    args.batch_size, args.n_layers, args.epochs, args.lr, args.dp, args.n_iters,
                                    args.n_workers, args.use_age, args.val, save_path, args.device)
        else:
            plot_embeddings(embeddings_df, args.manifold.lower(), args.label, save_path, color_by=args.color_label)
