# This script is licensed under the GNU Affero General Public License v3.0 (AGPLv3),
# as it depends on the AGPLv3-licensed `gaitmap_mad` package.
# See LICENSE-AGPL for details.

from pathlib import Path
import argparse
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import h5py
from gaitmap.utils.consts import SF_GYR
from gaitmap.preprocessing import align_dataset_to_gravity
import math
from gaitmap.preprocessing.sensor_alignment import PcaAlignment
from gaitmap.preprocessing.sensor_alignment import ForwardDirectionSignAlignment
from gaitmap.trajectory_reconstruction.orientation_methods import MadgwickAHRS
from gaitmap.trajectory_reconstruction.position_methods import PieceWiseLinearDedriftedIntegration
from gaitmap.utils.consts import GRAV_VEC
from gaitmap.zupt_detection import NormZuptDetector
from gaitmap.trajectory_reconstruction import MadgwickRtsKalman


def preprocess(imu_data, sampling_rate):
    # Gravity alignment
    gravity_aligned_data = align_dataset_to_gravity(
        imu_data, sampling_rate_hz=sampling_rate, window_length_s=0.1, static_signal_th=15)

    # PCA alignment
    pca_alignment = PcaAlignment(target_axis="y", pca_plane_axis=("gyr_x", "gyr_y"))
    pca_alignment = pca_alignment.align(gravity_aligned_data)
    pca_aligned_data = pca_alignment.aligned_data_

    # Direction sign alignment
    fdsa = ForwardDirectionSignAlignment(
        forward_direction="x",
        rotation_axis="z",
        baseline_velocity_threshold=0.2,
        ori_method=MadgwickAHRS(beta=0.1),
        zupt_detector_orientation_init=NormZuptDetector(
            sensor="acc", window_length_s=0.15, inactive_signal_threshold=0.01, metric="variance"
        ),
        pos_method=PieceWiseLinearDedriftedIntegration(
            NormZuptDetector(sensor="gyr", window_length_s=0.15, inactive_signal_threshold=15.0, metric="mean"),
            level_assumption=False,
            gravity=GRAV_VEC,
        ))

    fdsa = fdsa.align(pca_aligned_data, sampling_rate_hz=sampling_rate)
    forward_aligned_data = fdsa.aligned_data_

    return forward_aligned_data


def least_squares_transform(est_pos, ref_pos):

    # Pad the data with ones (homogeneous coordinates), so that our transformation can do translations too
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:,:-1]
    X = pad(est_pos)
    Y = pad(ref_pos)

    A, _, _, _ = np.linalg.lstsq(X, Y)

    transform = lambda x: unpad(np.dot(pad(x), A))

    return transform


def estimate_traj(bf_data, sampling_rate, beta):

    default_rts = MadgwickRtsKalman(madgwick_beta=beta, 
                                    velocity_error_variance=1, 
                                    orientation_error_variance=1).estimate(data=bf_data, sampling_rate_hz=sampling_rate)

    est_pos = np.array(default_rts.position_[['pos_x', 'pos_y']])[1:]

    # Perform a linear transformation to the absolute coordinate frame with the fiest 10% of the estimated trajectory and its ground truth
    if 'ref_x' in bf_data.columns:
        ref_pos = np.array(bf_data[['ref_x', 'ref_y']])
        est_pos = least_squares_transform(est_pos[:int(len(est_pos)*0.1)], ref_pos[:int(len(est_pos)*0.1)])(est_pos)
    
    return est_pos


def trajectory_estimation(imu_data_path, sampling_rate, ref_data_path= None, use_magnetometer=True, beta=0.08):
    
    # Load data
    imu_data_path = Path(imu_data_path)
    imu_data = pd.read_csv(imu_data_path)
    
    # Remove free acceleration data as it confuses the library
    imu_data = imu_data[list(imu_data.columns)[:-3]]
    imu_data[SF_GYR] = imu_data[SF_GYR].map(math.degrees)
    
    if ref_data_path is not None:
        ref_data = pd.read_csv(ref_data_path, delimiter=" ")

        ref_cols = list(ref_data.columns)
        ref_data = ref_data[ref_data[ref_cols[0]] == "POS"]
        ref_data = ref_data[[ref_cols[4], ref_cols[5], ref_cols[6], ref_cols[7]]]
        ref_data.columns = ['time', 'ref_x', 'ref_y', 'ref_z']
        
        # Calculate the reference positions at the times of the imu position estimations via interpolation
        imu_data['ref_x'] = np.interp(imu_data['utc_time'], ref_data.time, ref_data.ref_x)
        imu_data['ref_y'] = np.interp(imu_data['utc_time'], ref_data.time, ref_data.ref_y)

    # Preprocess the data, i.e. perform a gravity alignment and movement direction alignment
    forward_aligned_data = preprocess(imu_data, sampling_rate)
    bf_data = forward_aligned_data

    data_columns = ['gyr_x', 'gyr_y', 'gyr_z', 'acc_x', 'acc_y', 'acc_z']
    if use_magnetometer:
        data_columns += ['mag_x', 'mag_y', 'mag_z']

    return imu_data.utc_time, estimate_traj(bf_data[data_columns], sampling_rate, beta), bf_data


parser = argparse.ArgumentParser()
parser.add_argument('--data-path', required=True, default="passive_sensing/scripts/measurements/01042025/pdr/preprocessed/max_pdr.pkl", type=str, metavar='PATH',
						help='Path to input data')

parser.add_argument('--output-path', required=True, default="passive_sensing/scripts/measurements/01042025/pdr/preprocessed/max_pdr_dist.pkl", type=str,   
						help='Name of the file, the distances have to be stored.')

parser.add_argument('--window-length', required=False, default=10, type=int,   
						help='Length of the window in [s], the distances are calculated along.')

parser.add_argument('--stride-length', required=False, default=1, type=int,   
						help='Stride length within the window in indices. You have to consider the sampling rate of your signal to get the exact number of samples within a window.')

parser.add_argument('--imu-data', default="passive_sensing/scripts/measurements/01042025/pdr/pdr_max.csv", required=True, type=str, metavar='PATH',
						help='Data with IMU sensor values.')

if __name__ == "__main__":
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    
    # Limit for train/valid
    l1 = None #-2000
    l2 = None

    with h5py.File(data_path, "r") as f:
        data = (f["fp"]["CIR"][:][l1:l2], f["fp"]["reference"][:][l1:l2], f["fp"]["timestamp"][:][l1:l2])
    
    # We only have one IMU for recording and it has 400Hz.
    imu_sampling_rate = 400
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdr_time, pdr_data, _ = trajectory_estimation(args.imu_data, imu_sampling_rate, beta=0.025)

    pdr_time = pdr_time*1e-6 # Convert to microseconds

    data_timestamps = data[-1]

    x_steps = np.concatenate(([0], np.diff(np.interp(data_timestamps, pdr_time, pdr_data[:,0]))))
    y_steps = np.concatenate(([0], np.diff(np.interp(data_timestamps, pdr_time, pdr_data[:,1]))))
    
    with PdfPages(output_path.parent / Path("report_pdr.pdf")) as pdf:
        fig = plt.figure()
        plt.plot(np.cumsum(x_steps), np.cumsum(y_steps))
        pdf.savefig(fig)
        fig = plt.figure()
        plt.plot(pdr_time)
        plt.plot(np.array(data_timestamps))
        pdf.savefig(fig)
    
    dist_indices = []
    distances = []
    weights = []
    
    stride = args.stride_length
    
    time_diffs = np.diff(data_timestamps)
    
    for time_index, time_step in enumerate(tqdm(time_diffs)):
        
        window_time = 0
        window_index = 0
        
        window_weights = 0
        
        while window_time < args.window_length:
            if time_index + window_index > len(x_steps)-1:
                break
            
            if window_index % stride == 0:
                dist = np.sqrt(np.sum(x_steps[time_index:time_index+window_index])**2 + np.sum(y_steps[time_index:time_index+window_index])**2)

                if not np.isnan(dist).any():
                    dist_indices.append([time_index, time_index + window_index])
                    distances.append(dist)
                
                window_weights += 1
            
            window_time += time_diffs[window_index]
            window_index += 1
            
        weights.extend(np.array(range(window_weights,0, -1)) / window_weights)
    
    print(np.max(distances))
    print(np.min(distances))

    distances = np.array(distances, dtype=np.float32)
    weights = np.array(weights, dtype=np.float16)
    
    dist_dtype = np.dtype([
        ('indices', np.int32, (2,)),
        ('distance', np.int32),
        ('weight', np.float32)
    ])

    dist_data_array = np.zeros(len(dist_indices), dtype=dist_dtype)
    dist_data_array["indices"] = np.array(dist_indices)
    dist_data_array["distance"] = np.array(distances)
    dist_data_array["weight"] = np.array(weights)

    # Store distances with the corresponding CSI input tensors
    with h5py.File(output_path, "w") as output_file:
       with h5py.File(data_path, "r") as f:
           f.copy('fp', output_file) # Copy fp data
       output_file.create_dataset("cc", data=dist_data_array)

    print(np.array(dist_indices).shape)