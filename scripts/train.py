from datetime import datetime
import os
import lightning as L
from passive_sensing.callbacks import TrainPredictionsCallback
from passive_sensing.model.lightning_model import PositioningModel
import argparse
import torch
from lightning.pytorch.loggers import TensorBoardLogger
import csv
from evaluation import eval
from passive_sensing.utils import plot_channel_chart


from passive_sensing.dataset import load_dataset

if __name__ == "__main__":

    ap = argparse.ArgumentParser(description='Training script for the demo pipeline')
    ap.add_argument("--val-data-path", required=False, default=None, help="Path to the preprocessed validation dataset")
    ap.add_argument("--test-data-path", required=False, default=None, help="Path to the preprocessed test dataset. Test will be running automatically if given")
    ap.add_argument("--train-data-path", required=False, help="Path to the preprocessed training dataset")
    ap.add_argument("--results-path", required=False, help="Path to the preprocessed training dataset")
    ap.add_argument("--load-from-checkpoint", required=False, help="Path to the model")
    ap.add_argument("-b", "--batch-size", default=256, type=int, help="Batch size")
    ap.add_argument("--epochs", required=False, default=1000, help="Amount of epochs to train")
    ap.add_argument("--model-dir", required=True, help="Path where to store the models")
    ap.add_argument("--force-fp", required=False, action="store_true", help="Train fingerprinting, even if the dataset is a cc dataset")
    ap.add_argument("--filter-links-path", required=False, default=None)
    ap.add_argument("--log-dir", required=False, default="logs")
    ap.add_argument("--cc-training-plot", action="store_true", help="Enable plotting channel charts during training. This can be used to create an animation of the training process")
    ap.add_argument("--cc-training-plot-path", required=False, default="cc-plots", help="Path to the folder where plots during the training should be stored")
    ap.add_argument("--cc-training-plot-freq", type=int, required=False, default=1, help="Frequency of the plots")


    PositioningModel.add_cmd_args(ap)

    args = ap.parse_args()

    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    test_data_path = args.test_data_path
    epochs = args.epochs
    batch_size = args.batch_size
    model_type = args.model_type
    force_fp = args.force_fp
    filter_links_path = args.filter_links_path
    results_path = args.results_path

    cc_training_plot = args.cc_training_plot
    cc_training_plot_path = args.cc_training_plot_path
    cc_training_plot_freq = args.cc_training_plot_freq

    cc = False

    filter_links = None
    if filter_links_path is not None:
        with open(filter_links_path) as f:
            filter_links = csv.reader(filter(lambda row: row[0]!='#', f), delimiter=",")
            filter_links = list(filter_links)

    
    assert train_data_path is not None, "Training data must be supplied!"

    training_data, cc, in_dim = load_dataset(train_data_path, cc_epoch=10000, force_fp=force_fp, filter_links=filter_links)
    print("\nTraining with shape ", training_data[0][0].shape, " with mask ", training_data.mask, "\n\n")
        
    dl_train = torch.utils.data.DataLoader(training_data, batch_size=batch_size,
                                            shuffle=True, persistent_workers=True, num_workers=8)

    dl_valid = None
    if val_data_path:
        validation_data, _, val_in_dim = load_dataset(val_data_path, cc_limit=5000, force_fp=force_fp, filter_links=filter_links)
        assert val_in_dim == in_dim, "Validation dataset must have the same input dimension as the train dataset"
        dl_valid = torch.utils.data.DataLoader(validation_data, batch_size=batch_size,
                                            shuffle=False, persistent_workers=True, num_workers=8)
        


    metric = "validation/loss" if dl_valid else "training/loss"

    if args.load_from_checkpoint:
        model = PositioningModel.load_from_checkpoint(args.load_from_checkpoint, scheduler_monitor_metric=metric)
    else:
        model = PositioningModel(scheduler_monitor_metric=metric, in_channels=in_dim, **vars(args))
    
    date = datetime.now().isoformat()

    # Save top 2 models
    mc = L.pytorch.callbacks.ModelCheckpoint(args.model_dir, save_top_k=1, 
                                            monitor=metric, mode="min",
                                            every_n_train_steps=None if dl_valid else 50,
                                            auto_insert_metric_name=False,
                                            filename=date + "epoch={epoch:02d}_vloss={" + metric + ":.2f}")

    # Monitor learning rate
    lr_monitor = L.pytorch.callbacks.LearningRateMonitor()

    # Stop if val loss is not improving for 20 epochs
    early_stopping = L.pytorch.callbacks.EarlyStopping(monitor=metric, mode="min", patience=100)

    ## Instantiate Training Process
    callbacks = [
        # Save checkpoint when validation loss decreases
        mc,
        lr_monitor,
        # early_stopping
    ]

    if cc_training_plot is True:
        training_data_plots, _, _ = load_dataset(
        train_data_path, cc_epoch=10000, force_fp=True, filter_links=filter_links)
        dl_train_plots = torch.utils.data.DataLoader(training_data_plots, batch_size=batch_size,
                                            shuffle=False, persistent_workers=True, num_workers=8)
        plot_callback = TrainPredictionsCallback(
            train_dl=dl_train_plots, freq=cc_training_plot_freq, plot_fn=lambda target, pred, path: plot_channel_chart(target[:, 0], target[:, 1], pred[:, 0], pred[:, 1], path=path), outdir=cc_training_plot_path)
        callbacks.append(plot_callback)        

    trainer = L.Trainer(
        max_epochs=int(epochs), 
        callbacks=callbacks,
        accelerator='gpu', 
        #strategy='ddp',
        logger=TensorBoardLogger(args.log_dir)
    )
    
    try:
        trainer.fit(model, dl_train, dl_valid)
    except KeyboardInterrupt:
        # Allow manual stopping with Ctrl + C
        pass

    print(f"Best model: {mc.best_model_path}")


    if results_path is not None:
        eval(data_path=train_data_path, model_path=mc.best_model_path, cc=cc, output_folder=results_path + "/train", filter_links=filter_links)
        if val_data_path is not None:
            eval(data_path=val_data_path, model_path=mc.best_model_path, cc=cc, output_folder=results_path + "/valid", filter_links=filter_links)

        if test_data_path is not None:
            print("Running tests..")
            eval(data_path=test_data_path, model_path=mc.best_model_path, cc=cc, output_folder=results_path + "/test", filter_links=filter_links)
