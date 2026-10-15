# Passive Sensing

## Installation

Pull the repo, create a new python environment (e.g. `python -m venv .venv`), activate it (e.g. `source .venv/bin/activate`) and install this package with `pip install .`

## Dataset Download

Download the dataset from [here](https://iis.fraunhofer.de/passive-uwb)

## Dataset Preprocessing

Preprocess the dataset to the format for this repository:

```
python scripts/preprocess.py --data-path <training-file>.h5 --output-file <out-file-training>.h5 --mag --validation-split 2000
python scripts/preprocess.py --data-path <test-file>.h5 --output-file <out-file-test>.h5 --mag 
```
The paper used take01 for training and for testing take02 is P1 and take03 is P2
This dataset can already used for fingerprinting


## Adding IMU data

Adding IMU data requires installation of the ```gaitmap_mad``` package.

To add channel charting data, use the dist_estimation script:
```
python scripts/dist_estimation.py --imu-data <imu-data-path>.csv --data-path <dataset-path>.h5 --output-path <out-file>.h5 --window-length <window-length-seconds>
```

## Training

Train using the train.py script

```
python scripts/train.py --epochs 500 -r 0.01 -we 10 -m ResNet --model-dir models --train-data-path <train-data-path> --val-data-path <val-data-path> --test-data-path <test-data-path>
```

To train a fingerprinting model, either provide a dataset before adding imu data (it will be automatically detected) or by proiding the ```--force-fp``` flag.

If you want to use only a subset of links, you can filter them using the ```--filter-links-path <config_file>``` parameter. Subsets used in the paper can be found in the `config` subdirectory.

## Evaluation

You can evluate your checkpoints using the evaluate.py script:

```
python scripts/evaluation.py --test-data-path <test-dataset-path>.h5 --model-path <model-checkpoint>.ckpt --output-folder <results-folder> --cc
```

The ```--cc``` will apply optimal affine transform for the prediction, so use this flag for channel charting experiments

## Citation

If you use this code for your work, please consider citing

```
@inproceedings{poeggel2025passive,
  title={Passive Channel Charting: Locating Passive Targets using a UWB Mesh},
  author={Poeggel, Raffael and Stahlke, Maximilian and Pirkl, Jonas and Ott, Jonathan and Yammine, George and Feigl, Tobias and Mutschler, Christopher},
  booktitle={2025 International Conference on Indoor Positioning and Indoor Navigation (IPIN)},
  pages={1--6},
  year={2025},
}
```

## License

This repository is licensed under the MIT License, with the exception of
`scripts/dist_estimation.py`, which depends on the AGPLv3-licensed
`gaitmap_mad` package and is therefore itself licensed under the
GNU Affero General Public License v3.0 (AGPLv3).

See [LICENSE](LICENSE) for the MIT license text and
[LICENSE-AGPL](LICENSE-AGPL) for the full AGPLv3 text.