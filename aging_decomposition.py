#!/usr/bin/env python
"""
Aging Decomposition Analysis Script

This script performs either PCA or NMF decomposition on brain imaging data
to analyze aging-related changes.

Usage:
    python aging_decomposition.py --pca
    python aging_decomposition.py --nmf
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from scipy import stats
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler
from ants import image_read, image_list_to_matrix, matrix_to_images


def keep_negative_values(imgs_list):
    neg_imgs_list = []
    for img in imgs_list:
        neg_img_mtx = img.numpy()
        neg_img_mtx[neg_img_mtx > 0] = 0
        neg_img_mtx = abs(neg_img_mtx)
        neg_img = img.new_image_like(neg_img_mtx)
        neg_imgs_list.append(neg_img)
    return neg_imgs_list


def load_images(mask_path, age_changes_dir):
    mask_img = image_read(mask_path)
    aged_diffs = list((age_changes_dir / 'aged').glob('*.nii.gz'))
    rejuvenated_diffs = list((age_changes_dir / 'rejuvenated').glob('*.nii.gz'))
    diffs = aged_diffs + rejuvenated_diffs
    imgs_list = [image_read(str(img)) for img in diffs]
    return imgs_list, mask_img


def load_age_changes(path, mask_path='MNI152_T1_1mm_brain_mask.nii.gz'):
    age_changes_dir = Path(path)
    diffs = list(age_changes_dir.glob('*.nii.gz'))
    mask_img = image_read(mask_path)
    imgs_list = [image_read(str(img)) for img in diffs]
    return imgs_list, mask_img


def run_pca_analysis(imgs_list, mask_img, save_path, n_components=30):
    print("Converting images to matrix...")
    imgs_matrix = image_list_to_matrix(imgs_list, mask_img)
    scaler = StandardScaler(with_std=False)
    centered_data = scaler.fit_transform(imgs_matrix)
    del imgs_matrix
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(centered_data)
    explained_variance = pca.explained_variance_ratio_
    plot_explained_variance(explained_variance, n_components, save_path)
    
    return pca, principal_components, explained_variance


def plot_explained_variance(explained_variance, n_components, save_path):
    sns.set_theme()
    sns.set_style('white')
    fig, ax = plt.subplots(figsize=(6, 7))
    n_components_to_plot = n_components

    x_positions = np.arange(1, n_components_to_plot + 1)
    variances_to_plot = explained_variance[:n_components_to_plot]
    bar_color = sns.color_palette("deep")[0]
    line_color = "#143A80"

    ax.bar(x_positions, variances_to_plot, color=bar_color, width=1.0, alpha=0.9,
           edgecolor='white', linewidth=0.5)
    ax.plot(x_positions, variances_to_plot, color=line_color, alpha=0.9, linewidth=2,
            marker='o', markersize=3, markerfacecolor=line_color)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlim(0.5, n_components_to_plot + 0.5)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel('Principal Component', fontsize=15, fontweight='bold', color='#333333')

    max_variance = max(variances_to_plot)
    if max_variance > 0.08:
        y_ticks = [0, 0.02, 0.04, 0.06, 0.10]
        y_labels = ['0%', '2%', '4%', '6%', '10%']
    else:
        y_ticks = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        y_labels = ['0%', '1%', '2%', '3%', '4%', '5%', '6%']

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylim(0, max_variance)
    ax.set_ylabel('Explained Variance Ratio', fontsize=15, fontweight='bold', color='#333333')
    ax.set_axisbelow(True)

    ax.tick_params(axis='both', which='major', labelsize=17, colors='#333333')
    ax.tick_params(axis='x', which='major', length=4, width=1.0)
    ax.tick_params(axis='y', which='major', length=4, width=1.0)

    cumulative_variance = np.cumsum(explained_variance[:30])
    ax.text(0.52, 0.98, f'First 30 PCs: {cumulative_variance[-1]:.1%}',
            transform=ax.transAxes, fontsize=18, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.3',
            facecolor='white', alpha=1.0, edgecolor="#8b8b8b"))

    fig.patch.set_facecolor('white')
    plt.tight_layout()

    save_path.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path / 'explained_variance.png', dpi=300, bbox_inches='tight',
                transparent=True)
    print(f"Saved explained variance plot to {save_path / 'explained_variance.png'}")

    plt.show()
    print(f'Explained variance ratio: {explained_variance}')
    print(f'Total explained variance: {sum(explained_variance):.3f}')
    return fig


def save_3d_components(H, mask_img, out_path, filename):
    for i in range(H.shape[0]):
        component_3d = matrix_to_images(H[i][None, :], mask_img)[0]
        component_3d.to_filename(str(out_path / f'{filename}{i + 1}_aging.nii.gz'))


def nmf_decomposition(mask_path, save_path, n_components=3, filename='nmf_neg'):
    out_path = save_path / f'{n_components}_components'
    out_path.mkdir(exist_ok=True, parents=True)
    w_filepath = out_path / f'{filename}_weights.npy'
    h_filepath = out_path / f'{filename}_components.npy'
    if w_filepath.exists() and h_filepath.exists():
        print(f'Loading existing NMF components from {out_path}...')
        H = np.load(h_filepath)
        W = np.load(w_filepath)
        nmf = NMF(n_components=n_components, init='custom', random_state=0)
        nmf.components_ = H
        nmf.n_components_ = n_components
    else:
        age_changes_dir = Path('evaluation') / 'general' / 'test' / 'age_invariant' / 'e100' / 'age_changes'
        imgs_list, mask_img = load_images(mask_path, age_changes_dir)
        imgs_list = keep_negative_values(imgs_list)
        neg_diffs = image_list_to_matrix(imgs_list, mask_img)
        del imgs_list
        print(f'Performing NMF decomposition with {n_components} components...')
        nmf = NMF(n_components=n_components, init='random', random_state=0)
        W = nmf.fit_transform(neg_diffs)
        H = nmf.components_
        print(f'Saving NMF components to {out_path}')
        save_3d_components(H, mask_img, out_path, filename)
        del neg_diffs
        np.save(h_filepath, H)
        np.save(w_filepath, W)
    
    return nmf, W, H


def project_onto_components(imgs_matrix, components):
    n_components = components.shape[0]
    nmf_model = NMF(n_components=n_components, init='custom', random_state=0)
    nmf_model.components_ = components
    nmf_model.n_components_ = n_components
    W_new = nmf_model.transform(imgs_matrix)
    return W_new


def compare_nmf_groups(weights_group1, weights_group2, 
                       group1_name='Group 1', group2_name='Group 2',
                       save_path=None):

    W1_norm = weights_group1 / weights_group1.sum(axis=1, keepdims=True)
    W2_norm = weights_group2 / weights_group2.sum(axis=1, keepdims=True)

    data_list = []
    for i in range(3):
        for val in W1_norm[:, i]:
            data_list.append({'Component': f'Component {i+1}', 'Normalized Weight': val, 'Group': group1_name})
        for val in W2_norm[:, i]:
            data_list.append({'Component': f'Component {i+1}', 'Normalized Weight': val, 'Group': group2_name})
    
    df = DataFrame(data_list)

    _, ax = plt.subplots(figsize=(7, 6))
    
    fig = sns.violinplot(data=df, x='Component', y='Normalized Weight', hue='Group',
                   split=True, palette=['steelblue', 'coral'],
                   cut=0, inner='quartile', width=0.5,
                   linecolor='black', linewidth=1.0,
                   ax=ax)

    ax.set_ylabel('Normalized Weight', fontsize=11)
    ax.set_xlabel('')
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)

    components = ['Component 1', 'Component 2', 'Component 3']
    for i, _ in enumerate(components):
        stat, p_value = stats.mannwhitneyu(W1_norm[:, i], W2_norm[:, i], alternative='two-sided')
        y_max = max(W1_norm[:, i].max(), W2_norm[:, i].max())

        if p_value < 0.001:
            sig = '***'
        elif p_value < 0.01:
            sig = '**'
        elif p_value < 0.05:
            sig = '*'
        else:
            sig = 'ns'

        ax.text(i, y_max * 1.02, sig, ha='center', fontsize=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path / 'nmf_violinplot_comparison.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

    print("\n" + "="*70)
    print("STATISTICAL COMPARISON SUMMARY")
    print("="*70)

    for i in range(3):
        print(f"\nComponent {i+1}:")
        print(f"  {group1_name}: mean={W1_norm[:, i].mean():.3f}, std={W1_norm[:, i].std():.3f}")
        print(f"  {group2_name}: mean={W2_norm[:, i].mean():.3f}, std={W2_norm[:, i].std():.3f}")

        stat, p_value = stats.mannwhitneyu(W1_norm[:, i], W2_norm[:, i], alternative='two-sided')
        cohens_d = (W1_norm[:, i].mean() - W2_norm[:, i].mean()) / \
                   np.sqrt((W1_norm[:, i].std()**2 + W2_norm[:, i].std()**2) / 2)
        print(f"  Mann-Whitney U test: U={stat:.2f}, p={p_value:.4f}")
        print(f"  Effect size (Cohen's d): {cohens_d:.3f}")

    print("="*70 + "\n")
    return fig


def run_nmf_analysis(save_path):
    mask_path = 'MNI152_T1_1mm_brain_mask.nii.gz'
    ad_age_changes = Path('evaluation', 'diseased', 'test', 'age_invariant', 'e100', 'ad_rejuvenated')
    hc_age_changes = Path('evaluation', 'diseased', 'test', 'age_invariant', 'e100', 'hc_rejuvenated')
    aging_components = np.load(save_path / '3_components' / 'nmf_neg_components.npy')

    print('Reconstructing AD and HC age changes using NMF components...')
    ad_imgs, ad_mask = load_age_changes(ad_age_changes, mask_path)
    ad_imgs = keep_negative_values(ad_imgs)
    ad_imgs = image_list_to_matrix(ad_imgs, ad_mask)
    hc_imgs, hc_mask = load_age_changes(hc_age_changes, mask_path)
    hc_imgs = keep_negative_values(hc_imgs)
    hc_imgs = image_list_to_matrix(hc_imgs, hc_mask)
    ad_projected = project_onto_components(ad_imgs, aging_components)
    hc_projected = project_onto_components(hc_imgs, aging_components)
    del ad_imgs, hc_imgs
    fig = compare_nmf_groups(ad_projected, hc_projected,
                             group1_name="Alzheimer's Disease",
                             group2_name='Healthy Controls',
                             save_path=save_path)
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Aging Decomposition Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python aging_decomposition.py --pca
    python aging_decomposition.py --nmf
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pca', action='store_true',
                       help='Perform PCA decomposition and plot variance explained')
    group.add_argument('--nmf', action='store_true',
                       help='Perform NMF decomposition and reconstruct HC/AD patients')
    
    parser.add_argument('--n-components', type=int, default=None,
                        help='Number of components (default: 30 for PCA, 3 for NMF)')
    parser.add_argument('--output-dir', type=str, default='aging_decomposition',
                        help='Output directory for results (default: aging_decomposition)')
    
    args = parser.parse_args()
    
    save_path = Path(args.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if args.pca:
        n_components = args.n_components if args.n_components else 30
        print(f"Running PCA analysis with {n_components} components...")
        
        mask_path = 'MNI152_T1_1mm_brain_mask.nii.gz'
        age_changes_dir = Path('evaluation') / 'general' / 'test' / 'age_invariant' / 'e100' / 'age_changes'
        imgs_list, mask_img = load_images(mask_path, age_changes_dir)
        run_pca_analysis(imgs_list, mask_img, save_path, n_components=n_components)
        
    elif args.nmf:
        n_components = args.n_components if args.n_components else 3
        print(f"Running NMF analysis with {n_components} components...")
        run_nmf_analysis(save_path, n_components=n_components)


if __name__ == '__main__':
    main()
