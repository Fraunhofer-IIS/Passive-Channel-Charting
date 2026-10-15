import lightning as L
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

class TrainPredictionsCallback(L.Callback):
    """
    Args
    ----
    train_dl : torch.utils.data.DataLoader
        Dataloader des Trainings-Sets (OHNE Shuffle, damit die Reihenfolge
        reproduzierbar bleibt, falls gewünscht).
    freq : int
        Alle wie viele Epochen ausgewertet werden soll.
    plot_fn : Callable[[np.ndarray, np.ndarray, Path], None]
        Eigene Plot-Funktion:  plot_fn(targets, preds, save_path)
    outdir : str | Path
        Verzeichnis für Plots und .npy-Dateien.
    """

    def __init__(self, train_dl, freq: int, plot_fn, outdir: str = "train_preds"):
        self.train_dl = train_dl
        self.freq = freq
        self.plot_fn = plot_fn
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        epoch = trainer.current_epoch + 1
        if epoch % self.freq:
            return

        pl_module.eval()
        preds, targets = [], []

        for batch in tqdm(self.train_dl, desc="Running predicitons"):
            x, y = batch
            x = x.to(pl_module.device, non_blocking=True)
            preds.append(pl_module(x).cpu())
            targets.append(y.cpu())

        preds = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()

        # ----- Speichern -----
        np.save(self.outdir / f"epoch_{epoch:04d}_targets.npy", targets)
        np.save(self.outdir / f"epoch_{epoch:04d}_preds.npy",   preds)

        # ----- Plotten (eigenes Skript) -----
        plot_path = self.outdir / f"epoch_{epoch:04d}.png"
        self.plot_fn(targets, preds, plot_path)

        pl_module.train()
