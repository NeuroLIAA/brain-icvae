"""
RSA Analysis Script
Computes models' activations comparisons based on the pre-computed "Representational Dissimilarity Matrices" (RDMs) located in pretrain_exp/results/{task}/rdms.pkl. These matrices are the result of comparing the activations from each layer of each model on a subset of the diseased and healthy controls dataset.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path
from numpy import array, mean, abs, std
from pretrain_exp.rsa import compare_models


def base_name(model_name):
    basename = model_name.split('_')[0]
    if basename == 'bmi':
        return 'BMI'
    elif basename == 'baseline':
        return 'Voxel Representation'
    return basename.capitalize()


def parse_task_name(task):
    if task == 'ad_vs_hc':
        return 'AD vs HC'
    elif task == 'ad_vs_mci':
        return 'AD vs MCI'
    elif task == 'mci_vs_hc':
        return 'MCI vs HC'
    return task


def plot_model_comparisons(comparisons_dict, model_type, layers='all', x_label='Layers', group_by='layer', legend=True, title=None,
                           fig_size=(10, 8), only_baseline=False, colors=None):
    models = [key for key in comparisons_dict.keys() if key.endswith(f'_{model_type}')]
    if model_type == 'tl' and only_baseline:
        models += [f'baseline_{model_type}']
        model_pairs = [(model, f'baseline_{model_type}') for model in models if model != f'baseline_{model_type}']
    else:
        model_pairs = list(combinations(models, 2))
    model_pairs = sorted(model_pairs, key=lambda x: ('none' in x[0].lower(), 'none' in x[1].lower(), x[0], x[1]))
    
    if layers == 'all':
        first_comparison = next(iter(comparisons_dict.values()))
        first_layer_dict = next(iter(first_comparison.values()))
        layers = list(first_layer_dict.keys())
    if not isinstance(layers, list):
        raise ValueError("Layers should be a list of layer names or 'all' to use all available layers.")
    
    data_rows = []
    for layer in layers:
        for model1, model2 in model_pairs:            
            if model1 in comparisons_dict and model2 in comparisons_dict[model1]:
                correlations = comparisons_dict[model1][model2][layer]
            elif model2 in comparisons_dict and model1 in comparisons_dict[model2]:
                correlations = comparisons_dict[model2][model1][layer]
            else:
                continue
            
            comparison = f"{base_name(model1)} vs {base_name(model2)}"
            if 'none' in model1.lower() or 'none' in model2.lower():
                group = 'non-pretrained'
            elif 'baseline' in model1.lower() or 'baseline' in model2.lower():
                group = 'voxel model'
            else:
                group = 'pretrained'
            for corr in correlations:
                data_rows.append({
                    'layer': layer,
                    'comparison': comparison,
                    'correlation': corr,
                    'group': group
                })
    
    df = pd.DataFrame(data_rows)
    if colors:
        color_map = colors
    else:
        color_map = sns.color_palette("Set2")
    sns.set_style('whitegrid')
    plt.rcParams['font.family'] = 'Roboto'
    fig = plt.figure(figsize=fig_size)
    
    if group_by == 'layer':
        x = 'layer'
    else:
        x = 'group'

    ax = sns.violinplot(data=df, x=x, y='correlation', hue='comparison', palette=color_map, legend=legend, cut=0, alpha=1.0,
                        density_norm='width', width=0.8, bw_adjust=2)

    if title:
        ax.set_title(title, fontsize=16)
    
    min_value = round(df['correlation'].min() - 0.05, 1)
    plt.yticks(list(np.arange(min_value, 1.0, 0.1)) + [1.0])
    # yticks = list(np.arange(0.0, 0.25, 0.1))
    # plt.yticks(yticks)
    # ax.set_ylim(top=max(yticks))

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        n_comparisons = len(model_pairs)
        ax.legend(handles[:n_comparisons], labels[:n_comparisons], ncol=n_comparisons,
                title='Comparisons', loc='lower center', # bbox_to_anchor=(1.05, 1)
                )
        ax.set_ylim(bottom=0.0)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel('Correlation distance', fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_linewidth(.8)
    ax.spines['right'].set_linewidth(.8)
    ax.spines['bottom'].set_linewidth(.8)
    ax.spines['left'].set_linewidth(.8)
    if legend:
        ax.legend(title='Models')
    fig.patch.set_alpha(0.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig


def plot_representational_shifts(representational_shifts, legend=False):
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'Roboto'
    models = [m.lower() for m in representational_shifts.keys()]
    tasks = list(next(iter(representational_shifts.values())).keys())
    colors = sns.color_palette('deep', n_colors=len(models))

    # Build long-form DataFrame for seaborn
    plot_data = []
    for model in models:
        model_label = f'{model.capitalize()} finetuned' if model != 'bmi' else 'BMI finetuned'
        for task in tasks:
            values = representational_shifts[model][task]
            for v in values:
                plot_data.append({
                    'Task': parse_task_name(task),
                    'Model': model_label,
                    'Value': v,
                })
    df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(6, 5))
    model_labels = [f'{m.capitalize()} finetuned' if m != 'bmi' else 'BMI finetuned' for m in models]
    palette = dict(zip(model_labels, colors))

    sns.violinplot(data=df, x='Task', y='Value', hue='Model', hue_order=model_labels,
                   palette=palette, ax=ax, inner='box', linewidth=0.8, alpha=1.0, cut=0, width=0.5, legend=legend)

    ax.set_ylim(bottom=0.0)
    ax.set_xlabel('')
    ax.set_ylabel('Distance to pretrained representation', fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_linewidth(.8)
    ax.spines['right'].set_linewidth(.8)
    ax.spines['bottom'].set_linewidth(.8)
    ax.spines['left'].set_linewidth(.8)
    if legend:
        ax.legend(title='Models')
    fig.patch.set_alpha(0.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_rsa_histogram_grid(compared_rdms, layer, tasks, only_diagonal=False):
    sns.set_theme()
    pretrained_color = sns.color_palette()[0]
    finetuned_color = sns.color_palette()[1]

    if only_diagonal:
        first_task_rdms = compared_rdms[tasks[0]]
        models = [model for model in first_task_rdms.keys() if 'pretrained' in model]
        n_models = len(models)
        n_tasks = len(tasks)
        fig, axes = plt.subplots(n_models, n_tasks, figsize=(2 * n_tasks, 1.431 * n_models), squeeze=False, sharex=True)

        all_values = []
        for model in models:
            for task in tasks:
                task_rdms = compared_rdms[task]
                y_pretrained = f"{model.split('_')[0]}_pretrained"
                y_finetuned = f"{model.split('_')[0]}_tl"
                all_values.extend(task_rdms[model][y_pretrained][layer])
                all_values.extend(task_rdms[model][y_finetuned][layer])
        global_min = min(all_values)
        global_max = max(all_values)
        padding = (global_max - global_min) * 0.05
        global_min -= padding
        global_max += padding

        for row, model in enumerate(models):
            model_name = model.split('_')[0].capitalize()
            if model_name == 'Bmi':
                model_name = 'BMI'
            for col, task in enumerate(tasks):
                ax = axes[row, col]
                task_rdms = compared_rdms[task]
                y_pretrained = f"{model.split('_')[0]}_pretrained"
                y_finetuned = f"{model.split('_')[0]}_tl"
                finetuned_data = task_rdms[model][y_finetuned][layer]
                ax.axvline(x=1, color=pretrained_color, lw=2, alpha=0.7)
                sns.kdeplot(finetuned_data, ax=ax, fill=True, alpha=0.5, color=finetuned_color,
                            label=f'vs {y_finetuned}', linewidth=1, bw_adjust=0.8, legend=False)
                ax.set_xlim(global_min, global_max)
                ax.grid(alpha=0.5)
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(f'{model_name} pretrained', fontsize=10)
                else:
                    ax.set_ylabel('')
                    ax.set_yticklabels([])
                if row == n_models - 1:
                    ax.tick_params(axis='x', labelsize=9)
                if row == 0:
                    parsed_task_title = task.replace('ad_vs_hc', 'AD vs HC').replace('ad_vs_mci', 'AD vs MCI').replace('mci_vs_hc', 'MCI vs HC')
                    ax.set_title(parsed_task_title, fontsize=10)
                else:
                    ax.set_title('')

        handles = [
            plt.Line2D([0], [0], color=pretrained_color, lw=4, alpha=0.7),
            plt.Line2D([0], [0], color=finetuned_color, lw=4, alpha=0.7)
        ]
        labels = ['Pretrained', 'Fine-tuned']
        fig.legend(handles, labels, fontsize=9, frameon=True, loc='lower right', ncol=2, columnspacing=0.8, handlelength=.1)
        fig.supxlabel('Correlation value', fontsize=10)
        plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95], pad=0)
        return fig

    if isinstance(tasks, str):
        task = tasks
    else:
        task = tasks[0]
    compared_rdms = compared_rdms[task]
    x_models = [model for model in compared_rdms.keys() if 'pretrained' in model]
    y_model_pairs = [(f'{y_model.split("_")[0]}_pretrained', f'{y_model.split("_")[0]}_tl') for y_model in x_models]

    fig, axes = plt.subplots(len(y_model_pairs), len(x_models), figsize=(10, 8))
    fig.tight_layout(pad=3.0)
    plt.subplots_adjust(top=0.9)
    parsed_task_title = task.replace('ad_vs_hc', 'AD and HC')
    parsed_task_title = parsed_task_title.replace('ad_vs_mci', 'AD and MCI')
    parsed_task_title = parsed_task_title.replace('mci_vs_hc', 'MCI and HC')
    fig.suptitle(f'RSA Analysis on {parsed_task_title} subjects for layer {layer}', fontsize=14)

    all_values = []
    for x_model in x_models:
        for y_pretrained, y_finetuned in y_model_pairs:
            all_values.extend(compared_rdms[x_model][y_pretrained][layer])
            all_values.extend(compared_rdms[x_model][y_finetuned][layer])
    global_min = min(all_values)
    global_max = max(all_values)
    padding = (global_max - global_min) * 0.05
    global_min -= padding
    global_max += padding

    for col, x_model in enumerate(x_models):
        model_name = x_model.split('_')[0]
        if model_name == 'bmi':
            model_name = 'BMI'
        col_title = f'Similarity to {model_name} prediction model'
        axes[0, col].set_title(col_title, fontsize=11)

    for row, (pretrained, _) in enumerate(y_model_pairs):
        model_name = pretrained.split('_')[0].capitalize()
        if model_name == 'Bmi':
            model_name = 'BMI'
        ylabel = f'{model_name} prediction\n before and after fine-tuning'
        axes[row, 0].set_ylabel(ylabel, fontsize=10)

    fig.text(0.5, 0.02, 'Correlation value', ha='center', fontsize=14)
    fig.text(0.02, 0.5, 'Similarity density', va='center', rotation='vertical', fontsize=14)

    for row, (y_pretrained, y_finetuned) in enumerate(y_model_pairs):
        for col, x_model in enumerate(x_models):
            ax = axes[row, col]
            ax.set_yticks([])
            if col > 0:
                ax.set_yticklabels([])
            if row < len(y_model_pairs) - 1:
                ax.set_xticks([])
            else:
                ax.tick_params(axis='x', labelsize=10)

            pretrained_data = compared_rdms[x_model][y_pretrained][layer]
            finetuned_data = compared_rdms[x_model][y_finetuned][layer]
            if row != col:
                sns.kdeplot(pretrained_data, ax=ax, fill=True, alpha=0.5, color=pretrained_color,
                            label=f'vs {y_pretrained}', linewidth=1, bw_adjust=0.8, legend=False)
            else:
                ax.axvline(x=1, color=pretrained_color, lw=2, alpha=0.7)
            sns.kdeplot(finetuned_data, ax=ax, fill=True, alpha=0.5, color=finetuned_color,
                        label=f'vs {y_finetuned}', linewidth=1, bw_adjust=0.8, legend=False)

            ax.set_xlim(global_min, global_max)
            ax.grid(alpha=0.3)

    handles = [
        plt.Line2D([0], [0], color=pretrained_color, lw=4, alpha=0.7),
        plt.Line2D([0], [0], color=finetuned_color, lw=4, alpha=0.7)
    ]
    labels = ['Pretrained', 'Fine-tuned']
    fig.legend(handles, labels, fontsize=10, frameon=True, loc='upper left')
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])
    return fig


def main():
    base_path = Path('pretrain_exp')
    rdms_path = base_path / 'results'
    tasks = ['ad_vs_hc', 'ad_vs_mci', 'mci_vs_hc']
    models = ['age', 'sex', 'bmi', 'none']
    layers = ['conv0', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5']
    layer = 'conv5'
    n_iters = 1000
    random_state = 42
    print('Computing model comparisons...')
    comparisons_dict = compare_models(tasks, models, layers, rdms_path, n_iters, random_state)
    save_path = base_path / 'figures'
    save_path.mkdir(exist_ok=True, parents=True)
    for task in tasks:
        print(f"  - {task}")
        fig = plot_model_comparisons(comparisons_dict[task], model_type='pretrained', legend=False, 
                                      title=f'Pretrained Comparisons - {parse_task_name(task)} (layer {layer})')
        fig.savefig(save_path / f'{task}_pretrained_comparisons_{layer}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved pretrained comparison figures for {task}')

    for task in tasks:
        print(f"  - {task}")
        fig = plot_model_comparisons(comparisons_dict[task], model_type='tl', 
                                      layers=['conv0', 'conv1'],
                                      title=f'Finetuned Comparisons - {parse_task_name(task)} (layer {layer})')
        fig.savefig(save_path / f'{task}_finetuned_comparisons_{layer}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved fine-tuned comparison figures for {task}')

    representational_shifts = {}
    for task in tasks:
        print(f"  Task: {task}")
        for model in comparisons_dict[task]:
            if model.endswith('_tl'):
                continue
            model_name = model.split('_')[0]
            if model_name not in representational_shifts:
                representational_shifts[model_name] = {}
            finetuned_model = model_name + '_tl'
            comparison_with_self = array(comparisons_dict[task][model][model][layer])
            comparison_with_finetuned = array(comparisons_dict[task][model][finetuned_model][layer])
            representational_shift = abs(comparison_with_self - comparison_with_finetuned)
            representational_shifts[model_name][task] = [mean(representational_shift), std(representational_shift)]
            print(f'    Model: {model_name}, Representational Shift: {mean(representational_shift):.6f} ± {std(representational_shift):.6f}')

    fig = plot_representational_shifts(representational_shifts)
    fig.patch.set_alpha(0)
    fig.savefig(save_path / 'representational_shifts.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved representational shifts figure at {save_path / "representational_shifts.png"}')

    fig = plot_rsa_histogram_grid(comparisons_dict, layer=layer, tasks=tasks, only_diagonal=True)
    fig.patch.set_alpha(0)
    fig.savefig(save_path / f'rsa_histogram_grid_{layer}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved histogram grid at {save_path / f"rsa_histogram_grid_{layer}.png"}')

    for task in tasks:
        fig = plot_rsa_histogram_grid(comparisons_dict, layer=layer, tasks=task)
        fig.savefig(save_path / f'rsa_histogram_{task}_{layer}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved histogram for {task} at {save_path / f"rsa_histogram_{task}_{layer}.png"}')


if __name__ == '__main__':
    main()
