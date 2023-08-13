import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pandas as pd

import tensorflow as tf
from kingfisher.inference.run_model import Ensemble
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
from pathlib import Path
OUTPUT_PATH = Path("E:/Eisvogel/results")


def windowsify_path(path):
    return path.replace("/media/julian/TOSHIBA EXT/", "F:/")

def set_up_ensemble():
    global ensemble
    import os
    os.environ["PATH"] += ";E:\\CUDNN\\v8.1\\bin;"
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1500)])
    ensemble = Ensemble(glob.glob("E:/Eisvogel/kingfisher_bt/ensemble_weights/*/*.hdf5"))

def get_dataframe(csv_path):
    df_folder = pd.read_csv(csv_path)
    df_folder = df_folder.rename(columns={"Unnamed: 0": "timestamp"})
    df_folder["images"] = df_folder["images"].apply(windowsify_path)
    return df_folder


def join_results(df_data, result):
    df_result = pd.DataFrame(result).T
    df_result["mean"] = df_result.mean(axis=1)
    df_result = df_data.join(df_result, on="images")
    return df_result

def write_result(df_result, folder_path):
    output = get_output_path(folder_path)
    df_result.to_csv(output, index=False)


def get_output_path(folder_path):
    folder_path = Path(folder_path)
    output = OUTPUT_PATH / folder_path.name
    return output

def run_data(folder_path):
    global ensemble
    df_data = get_dataframe(folder_path)
    result = ensemble.predict(df_data["images"])
    df_result = join_results(df_data, result)
    return folder_path, df_result


def filter_existing(list_of_folders):
    to_do = []
    for folder in list_of_folders:
        if not os.path.isfile(get_output_path(folder)):
            to_do.append(folder)
    n_tot = len(list_of_folders)
    n_done = n_tot - len(to_do)
    print(f"Found {n_done} existing outputs. Continuing with {len(to_do)} tasks.")
    return to_do, n_tot, n_done


if __name__ == "__main__":
    all_folders = glob.glob("F:/frame_times/*.csv")

    all_folders, n_total, n_done = filter_existing(all_folders)

    with ProcessPoolExecutor(max_workers=2, initializer=set_up_ensemble) as ppe, \
          tqdm.tqdm(total=n_total, initial=n_done, desc="Processing folders") as pbar:
       futs = [ppe.submit(run_data, folder) for folder in all_folders]
       for this_fut in as_completed(futs):
           folder_path, df_result = this_fut.result()
           write_result(df_result, folder_path)
           pbar.update(1)