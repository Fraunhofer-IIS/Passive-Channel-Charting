import json
from pathlib import Path
import random
from lightning import Trainer
import argparse
import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness
import torch
import matplotlib.pyplot as plt
from passive_sensing.utils import continuity, kruskal_stress, least_squares_transform, plot_channel_chart

from passive_sensing.dataset import load_dataset

from passive_sensing.model.lightning_model import PositioningModel

def plot_hexplot(ax, positions, distances, gridsize=20, vmax = 2):
    import matplotlib.pyplot as plt
    import numpy as np


    hb = ax.hexbin(
        positions[:, 0],
        positions[:, 1],
        C=distances,
        gridsize=20,
        vmin=0,
        vmax=vmax,
    )

    cb = plt.colorbar(hb, ax=ax)
    cb.set_label("Avg. Accuracy [m]")

    ax.set_title("Accuracy")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    return ax

def plot_cdf(ax, distances, max_points=1000, percentiles=[50, 90, 95]):

    distances = np.sort(np.array(distances))
    cdf = np.arange(1, len(distances) + 1) / len(distances)

    if len(distances) > max_points:
        indices = np.linspace(0, len(distances) - 1, num=max_points, dtype=int)
        distances_plot = distances[indices]
        cdf_plot = cdf[indices]
    else:
        distances_plot = distances
        cdf_plot = cdf

    ax.plot(distances_plot, cdf_plot, label='CDF')

    colors = plt.cm.tab10.colors
    for i, ce in enumerate(percentiles):
        ce_value = np.quantile(distances, ce / 100)
        ax.plot(
            [ce_value, ce_value, 0],
            [0, ce / 100, ce / 100],
            color=colors[i], linestyle='dashed',
            label=f'CE{ce}={ce_value:.2f}'
        )

    ax.set_title('CDF of Error')
    ax.set_xlabel('Error [m]')
    ax.set_ylabel('CDF')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()


def eval(data_path, model_path, cc, output_folder, filter_links=None):
    test_datset, _, in_dim = load_dataset(data_path, force_fp=True, filter_links=filter_links)
    dl_test = torch.utils.data.DataLoader(test_datset, batch_size=512,
                                            shuffle=False, persistent_workers=True, num_workers=8)

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    model = PositioningModel.load_from_checkpoint(model_path)

    trainer = Trainer()

    prediciton = trainer.predict(model, dl_test)

    prediciton = np.array(torch.cat(prediciton, dim=0))
    
    ground_truth = test_datset.fp_data["reference"]

    if cc:
        prediciton = least_squares_transform(prediciton, ground_truth)(prediciton)

    error = np.linalg.norm(ground_truth - prediciton, axis=1)

    subset_indices = random.sample(range(len(ground_truth)), len(ground_truth) // 10)
    labels_subset = ground_truth[subset_indices]
    pred_subset = prediciton[subset_indices]

    ct = continuity(labels_subset, pred_subset, n_neighbors=int(0.05 * len(labels_subset)))
    tw = trustworthiness(labels_subset, pred_subset, n_neighbors=int(0.05 * len(labels_subset)))
    ks = kruskal_stress(labels_subset, pred_subset)

    metrics = {
        "MAE": float(error.mean()),
        "CEP": float(np.percentile(error, 50)),
        "CE90": float(np.percentile(error, 90)),
        "R95": float(np.percentile(error, 95)),
        "Continuity": float(ct),
        "Trustworthiness": float(tw),
        "Krukal Stress": float(ks),
    }

    with open(output_folder / "metrics.json", "w") as f:
        json.dump(metrics, f)

    fig, ax = plt.subplots(figsize=(12, 8))
    plot_cdf(ax, error)
    fig.savefig(output_folder / "cdf.pdf", bbox_inches='tight')


    fig, ax = plt.subplots(figsize=(12, 8))
    plot_hexplot(ax, ground_truth, error)
    fig.savefig(output_folder / "hexplot.pdf", bbox_inches='tight')


    if cc:
        plot_channel_chart(ground_truth[:,0], ground_truth[:,1], prediciton[:,0], prediciton[:,1], path=output_folder / "cc_plot.pdf")

    np.save(output_folder / "prediction.npy", prediciton)
    np.save(output_folder / "error.npy", error)
    np.save(output_folder / "ground_truth.npy", ground_truth)


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description='Evaluation script for the demo pipeline')
    ap.add_argument("--test-data-path", default=None, required=True, help="Path to the dataset")
    ap.add_argument("--model-path", default=None, required=True, help="Path to the model")
    ap.add_argument("--output-folder", default=None, required=False, help="Path to the folder, where the results should be stored")
    ap.add_argument("--cc", action="store_true", help="Evaluate CC Model")

    args = ap.parse_args()

    data_path: str = args.test_data_path
    model_path: str = args.model_path
    output_folder: str = args.output_folder
    cc: bool = args.cc

    eval(data_path=data_path, model_path=model_path, cc=cc, output_folder=output_folder)