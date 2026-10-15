import numpy as np
import h5py
import pandas as pd

from scipy.fft import fft, ifft
import cmath
import math

from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description='Preprocessing script for the demo pipeline')
parser.add_argument('--data-path', default="mpp-data/passive_uwb/01042025/slow/max_short.hdf5", type=str, metavar='PATH', required=False,
						help='Path to the folder the data should be stored')
parser.add_argument('--output-file', default="mpp-data/passive_uwb/01042025/pdr/test_prepro/max_pdr.hdf5", type=str, metavar='PATH', required=False,
						help='Path to the output file')
parser.add_argument('--validation-split', type=int, default=None, required=False, help="Number of samples that should be used for a validation split. File will be saved like output file but with _valid. Leave for no validation")
parser.add_argument('--iq', default=False, action="store_true", help="Include IQ data")
parser.add_argument('--mag', default=True, action="store_true", help="Include magnitude data")
parser.add_argument('--interpolate', default=None, type=int, help="Interpolate X missing cirs. Leave out for no interpolation")


def calc_mag(row):
    cir = np.vstack((list(row.CIR_R), list(row.CIR_I)))
    complex_cir = np.apply_along_axis(lambda args: [complex(*args)], 0, cir).flatten()

    return np.abs(complex_cir)

def delay_sub_sample(input_signal, sub_sample):
    
    # Apply fourier transform shift theorem
    input_fft = fft(input_signal)
    result = np.multiply(input_fft, [cmath.exp(-1j*2*math.pi*item*sub_sample) for item in np.fft.fftfreq(len(input_signal))])
    
    return ifft(result)

def align_cir(row):
    
    if 'fpi_frac' in row:
        fpi = row['fpi_frac']
    else:
        fpi = row['fpi']
        
    complex_cir = row.CIR_R + row.CIR_I*1j
    shifted_cir = delay_sub_sample(complex_cir, 5-fpi)
    
    row['CIR_R'] = np.array([item.real for item in shifted_cir])
    row['CIR_I'] = np.array([item.imag for item in shifted_cir])
    
    return row

def preprocess_data(data_path, interpolate = None):
    file = h5py.File(data_path, "r")

    cir_data = file['CIR']
    data = []

    fpi_col = "fpi_frac" if "fpi_frac" in cir_data.dtype.names else "fpi"
    columns = ["CIR_R", "CIR_I", "Time_stamp", fpi_col, "B_ID", "REC_ID", "TRANS_ID"]

    for item in cir_data:
        data_row = {}
        for col in columns:
            data_row[col] = item[col]
        
        data.append(data_row)
    
    # Align CIR by the first path indexy
    data = pd.DataFrame(data)
    data = data.apply(align_cir, axis=1)

    data['A_ID'] = data['TRANS_ID'].astype(str) + "-" + data['REC_ID'].astype(str)

    if interpolate:
        raise NotImplementedError("Check data before, currently multiple bursts with double A_IDs can occur")
        print("Interpolating up to ", interpolate, " missing samples")
        unique_a_ids = data['A_ID'].unique()
        unique_b_ids = data['B_ID'].unique()

        # Make sure, a value for each A_ID and B_ID exists
        complete_index = pd.MultiIndex.from_product([unique_a_ids, unique_b_ids], names=['A_ID', 'B_ID'])

        # Reindex the DataFrame to include all combinations
        df_full = data.set_index(['A_ID', 'B_ID']).reindex(complete_index).sort_index()

        # Function to interpolate only if missing values are within a threshold
        def conditional_interpolation(group):
            group = group.sort_index()
            mask = group.isna().all(axis=1)
            
            consecutive_missing = mask.groupby((mask != mask.shift()).cumsum()).transform('size')

            consecutive_missing = consecutive_missing | ~mask
            
            # Interpolate where consecutive missing values are less than or equal to threshold
            group.loc[consecutive_missing <= interpolate] = group.loc[consecutive_missing <= interpolate].interpolate(method='linear', axis=0)
            
            return group

        data = df_full.groupby('A_ID').apply(conditional_interpolation).reset_index()

    data['CIR_MAG'] = data.apply(calc_mag, axis=1)
    
    ref_pos = np.vstack((file['POS']['Time_stamp'], file['POS']['POS_X'], file['POS']['POS_Y'])).T
    
    return ref_pos, data

if __name__ == "__main__":

    args = parser.parse_args()
    data_path = Path(args.data_path)

    iq = args.iq
    mag = args.mag
    interpolate = args.interpolate
    validation_split = args.validation_split

    assert iq or mag, "At least one of IQ data (--iq) or magnitude (--mag) must be included"

    ref_pos, cir_data = preprocess_data(data_path, interpolate=interpolate)
    
    burst_groups = cir_data.groupby(by=["B_ID"], as_index=False)
    valid_groups = burst_groups.filter(lambda x: len(x) == 14)
    
    cir_times = []
    cir_bursts = []
    
    a_ids = None

    for _, burst in valid_groups.groupby(by=["B_ID"], as_index=False):
        burst = burst.sort_values("A_ID")

        current_a_ids = burst.A_ID.to_list()
        if a_ids is None:
            a_ids = current_a_ids
        else:
            if a_ids != current_a_ids:
                print(f"[WARNING] Burst with different a ids: {a_ids} vs {current_a_ids}, skipping")
                continue

        channels = []
        if iq:
            channels.append(np.array(list(burst.CIR_R), dtype=np.float32) / 25000.0)
            channels.append(np.array(list(burst.CIR_I), dtype=np.float32) / 25000.0)

        if mag:
            channels.append(np.array(list(burst.CIR_MAG), dtype=np.float32) / 25000.0)

        cir_bursts.append(np.stack(channels, axis=0))
        cir_times.append(burst[burst.Time_stamp > 0].Time_stamp.mean())
    
    cir_bursts = np.array(cir_bursts)
    
    cir_pos = np.vstack((np.interp(cir_times, ref_pos[:,0], ref_pos[:, 1]), np.interp(cir_times, ref_pos[:,0], ref_pos[:, 2]))).T

    output_file_path = Path(args.output_file)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)


    def write_file(filename, cir_bursts, cir_times, cir_pos):
        dtype = np.dtype([
                ('CIR', np.float64, cir_bursts.shape[1:]),
                ('timestamp', np.float64),
                ('reference', np.float64, cir_pos.shape[1:])
        ])

        # Create data array
        data_array = np.zeros(len(cir_bursts), dtype=dtype)
        data_array['CIR'] = cir_bursts
        data_array['timestamp'] = cir_times
        data_array['reference'] = cir_pos

        with h5py.File(Path(filename), 'w') as data_file:
            data_file.create_dataset("fp", data=data_array)
            data_file["fp"].attrs["a_ids"] = a_ids
        
        print(filename, cir_bursts.shape)

    cir_times = np.array(cir_times) / 1000.0

    if validation_split is not None:
        val_cirs = cir_bursts[-validation_split:]
        val_times = cir_times[-validation_split:]
        val_pos = cir_pos[-validation_split:]
        val_filename = args.output_file.replace(".h5", "")
        val_filename = val_filename + "_valid.h5"

        write_file(val_filename, val_cirs, val_times, val_pos)

        train_cirs = cir_bursts[:-validation_split]
        train_times = cir_times[:-validation_split]
        train_pos = cir_pos[:-validation_split]
        train_filename = args.output_file

        write_file(train_filename, train_cirs, train_times, train_pos)
    
    else:
        train_cirs = cir_bursts
        train_times = cir_times
        train_pos = cir_pos
        train_filename = args.output_file

        write_file(train_filename, train_cirs, train_times, train_pos)