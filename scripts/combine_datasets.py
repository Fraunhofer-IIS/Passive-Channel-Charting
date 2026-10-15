import h5py
import numpy as np
import sys

def combine_datasets(input_paths, output_path):
    total_fp_size = 0
    combined_fp = []
    combined_cc = []

    a_ids = None

    for input_path in input_paths:
        with h5py.File(input_path, 'r') as file:
            print(input_path, "...")
            fp_data = file['fp'][:]
            combined_fp.append(fp_data)

            current_a_ids = file["fp"].attrs["a_ids"]

            if a_ids is None:
                a_ids = current_a_ids
            else:
                if list(a_ids) != list(current_a_ids):
                    print(f"[Error] Different A-IDs: {a_ids} vs {current_a_ids}, aborting")
                    exit(0)

            if 'cc' in file:
                cc_data = file['cc'][:]
                cc_data['indices'] += total_fp_size
                combined_cc.append(cc_data)
                
            total_fp_size += fp_data.shape[0]

    combined_fp = np.concatenate(combined_fp, axis=0)
    if len(combined_cc) > 0:
        combined_cc = np.concatenate(combined_cc, axis=0)

    with h5py.File(output_path, 'w') as output_file:
        output_file.create_dataset('fp', data=combined_fp)
        output_file["fp"].attrs["a_ids"] = a_ids
        if len(combined_cc) > 0:
            output_file.create_dataset('cc', data=combined_cc)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python combine_datasets.py <input_path1> <input_path2> ... <output_path>")
        sys.exit(1)

    input_paths = sys.argv[1:-1]
    output_path = sys.argv[-1]
    combine_datasets(input_paths, output_path)