"""Plot recorded training losses and tabulate measured runtime/adapter details."""
import argparse
import csv
import json
from pathlib import Path


def make_training_figures(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = Path(root)
    config = json.loads((root / 'config.json').read_text())
    seeds = config['seeds']
    fig, axes = plt.subplots(1, len(seeds), figsize=(4.4 * len(seeds), 3.8),
                             squeeze=False, constrained_layout=True)
    rows = []
    for ax, seed in zip(axes[0], seeds):
        for method, color in [('adamw', '#43566e'), ('ivon', '#108078')]:
            folder = root / f'seed_{seed}' / method
            info = json.loads((folder / 'complete.json').read_text())
            history = json.loads((folder / 'training_history.json').read_text())
            ax.plot([h['step'] for h in history], [h['loss'] for h in history],
                    label=method, color=color, linewidth=1.3, alpha=.85)
            rows.append({'seed': seed, 'optimizer': method, 'steps': info['steps'],
                         'train_seconds': info['train_seconds'],
                         'seconds_per_step': info['train_seconds'] / info['steps'],
                         'peak_train_memory_gib': info['peak_train_memory_gib'],
                         'trainable_parameters': info['trainable_parameters'],
                         'total_parameters': info['total_parameters'],
                         'first_batch_ce': history[0]['loss'],
                         'last_batch_ce': history[-1]['loss']})
        ax.set(title=f'Seed {seed}', xlabel='Optimizer step', ylabel='Training batch cross-entropy')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend()
    fig.suptitle('Recorded training losses\nIVON evaluates a sampled adapter; these curves exclude KL regularization.', fontsize=11)
    dest = root / 'summary'
    dest.mkdir(exist_ok=True)
    fig.savefig(dest / 'training_curves.png', dpi=180)
    plt.close(fig)
    with (dest / 'training_details.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Saved:', dest / 'training_curves.png')
    print('Saved:', dest / 'training_details.csv')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    make_training_figures(parser.parse_args().run_dir)
