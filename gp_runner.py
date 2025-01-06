import gc
import os
import shutil
import time
import warnings
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from gpytorch.utils.warnings import NumericalWarning

from src import config as cfg
from src.batt_data import batt_data, data_utils
from src.batt_data.batt_data import SegmentCriteria
from src.batt_models.battgp_full import BattGP_Full
from src.batt_models.battgp_spatiotemporal import BattGP_SpatioTemporal
from src.batt_models.fault_probabilities import calc_fault_probabilities
from src.batt_models.plotting import plot_cell_r0_predictions, plot_fault_probabilities
from src.batt_models.ref_strategy import RefStrategy
from src.operating_point import Op
from src.path_setup import setup_paths
from src.plot_utils import merge_pdfs

warnings.filterwarnings("ignore", category=NumericalWarning)

MODE = "full_gp"  # "full_gp"  # "spatio_temporal"
FAULT_PROBS = 10 ** (-3) * np.array([0.55])  # np.linspace(0.50, 0.70, 41)
OP = Op(-15, 90, 25)


def battgp_analysis(
    batt_id: List[str],
    cell_characterstics: dict,
    fault_prob: bool,
    save_path: str,
    device,
    segment_criteria,
):
    for id in batt_id:
        print(f"Training model for battery with ID: {id}")

        battdata = batt_data.BattData(
            id,
            cell_characterstics,
            segment_selection=True,
            segment_criteria=segment_criteria,
            gap_removal=cfg.GAP_REMOVAL,
            min_data_threshold=2000,
        )

        if MODE == "spatio_temporal":
            battmodel = BattGP_SpatioTemporal(
                battdata,
                max_age=None,
                device=device,
                sampling_time_sec=3600,
                max_batch_size=1000,
                save_path=save_path,
                ref_strategy=RefStrategy(OP),
                # basis_vectors=cfg.BASIS_VECTORS_TEST,
                basis_vector_strategy="kmeans",
                nbasis=[60],
            )
        elif MODE == "full_gp":
            battmodel = BattGP_Full(
                battdata,
                ref_strategy=RefStrategy(OP),
                max_training_data=cfg.GP_SETTINGS["nb_data_points"],
                max_age=None,
                device=device,
                save_path=save_path,
            )
        else:
            raise ValueError(f"Unknown mode: {MODE}")

        if cfg.HYPER_OPT_PARAMS["optimize"]:
            print("Optimizing hyperparameters, this may take a while...")
            print(f"Data points: {cfg.GP_SETTINGS['nb_data_points']}")
            battmodel.train_hyperparameters(
                parallelize=cfg.HYPER_OPT_PARAMS["parallelize"], messages=True
            )
            if cfg.HYPER_OPT_PARAMS["save"]:
                battmodel.save_hyperparameters(save_path)

        if MODE == "spatio_temporal":
            # start = time.time()
            gp_res = battmodel.predict_cell_r0_op(smooth=True)
            # end = time.time()
            # print(f"Batt_id {id} took {end-start} seconds, STGP")
        elif MODE == "full_gp":
            # start = time.time()
            gp_res = battmodel.predict_cell_r0_op()
            # end = time.time()
            # sprint(f"Batt_id {id} took {end-start} seconds, full GP")
        else:
            raise ValueError(f"Unknown mode: {MODE}")

        plot_cell_r0_predictions(
            gp_res,
            single_plot=True,
            save=True,
            save_path=save_path,
            y_lim=[9, 15],
        )

        plt.close()
        if fault_prob:
            for i in FAULT_PROBS:
                df_faults = calc_fault_probabilities(
                    gp_res, causal=False, r0_band=i, r0_upper_threshold=2.0e-3
                )
                plot_fault_probabilities(
                    df_faults,
                    gp_res,
                    r0_band=i,
                    save=True,
                    save_path=save_path,
                    causal=False,
                )
                if MODE == "spatio_temporal":
                    df_faults = calc_fault_probabilities(
                        gp_res, causal=True, r0_band=i, r0_upper_threshold=2.0e-3
                    )
                    plot_fault_probabilities(
                        df_faults,
                        gp_res,
                        r0_band=i,
                        save=True,
                        save_path=save_path,
                        causal=True,
                    )
                plt.close("all")

        del battdata
        del battmodel
        del gp_res
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(
                f"Error occured during garbage collection or while clearing the cuda cache: {e}"
            )


def worker(
    batt_id,
    cell_characterstics,
    fault_prob,
    save_path,
    device,
    segment_criteria,
):
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    torch.cuda.set_device(device)

    cuda_device = torch.cuda.current_device()
    print(f"Here's process: {os.getpid()} running on cuda: {cuda_device}")
    battgp_analysis(
        batt_id,
        cell_characterstics,
        fault_prob,
        save_path,
        cuda_device,
        segment_criteria,
    )


def main():
    # Depending on the actual hardware setup it might be faster to use the cpu even if
    # a cuda gpu is available.
    use_always_cpu = False
    if not use_always_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        n_devices = torch.cuda.device_count()
        if n_devices > 1:
            print("Planning to run on {} GPUs.".format(n_devices))
    else:
        device = torch.device("cpu")
        n_devices = 0
    if n_devices > 1:
        try:
            mp.set_start_method("spawn")
        except Exception as e:
            print(f"{e}, continuing!")
    print(f"    on device: {device}")
    
    plt.style.use("seaborn-v0_8-white")
    setup_paths()
    data_utils.build_data_cache()
    
    single_system = False
    fault_prob = True
    ocv_path = "data/ocv_linear_approx.csv"
    cell_characterstics = data_utils.read_cell_characteristics(path=ocv_path)
    
    
    save_path_base = cfg.PATH_RESULTS
    SOC_UCO = cfg.SOC_UPPER_LIMIT
    SOC_LCO = cfg.SOC_LOWER_LIMIT
    ICL = cfg.Ibat_LOWER_LIMIT
    I_CU = cfg.Ibat_UPPER_LIMIT

    run_date = time.strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(save_path_base, "batt_gp_run_" + run_date + "/")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    shutil.copyfile("src/config.py", os.path.join(save_path, "config.py"))
    shutil.copyfile("gp_runner.py", os.path.join(save_path, "gp_runner.py"))

    cs_string = f"SOC_{SOC_UCO}_{SOC_LCO}_I_{I_CU}_{ICL}"
    cs_string += f"_{ocv_path.split('ocv_')[1].split('.')[0]}"
    print(f"Running CS: {cs_string}")
    Path(save_path).mkdir(parents=True, exist_ok=True)

    segment_criteria: SegmentCriteria = SegmentCriteria(
        soc_upper_limit=SOC_UCO,
        soc_lower_limit=SOC_LCO,
        ibat_upper_limit=I_CU,
        ibat_lower_limit=ICL,
        t_upper_limit=cfg.T_UPPER_LIMIT,
        t_lower_limit=cfg.T_LOWER_LIMIT,
    )

    df_info = pd.DataFrame(
        {
            "ocv_curve": ocv_path,
            "SOC_UCO": SOC_UCO,
            "SOC_LCO": SOC_LCO,
            "I_CU": I_CU,
        },
        index=[0],
    )
    df_info.to_csv(os.path.join(save_path, "case_study_info.csv"))
    
    if single_system:
        batt_id = ["4"]
    else:
        # batt_id = [str(i) for i in range(1, 29)]
        # For basis point studies, take:
        batt_id = ["6", "8", "9", "10", "18", "21"]
    if n_devices > 1:
        print("ID of main process: {}".format(os.getpid()))
        print("Number of processes: {}".format(n_devices))
        if n_devices == 4:
            batt_id_partitions = [
                [str(i) for i in range(1, 8)],
                [str(i) for i in range(8, 12)],
                [str(i) for i in range(12, 17)],
                [str(i) for i in range(17, 30)],
            ]
        elif n_devices == 3:
            batt_id_partitions = [
                [str(i) for i in range(1, 9)],
                [str(i) for i in range(9, 12)],
                [str(i) for i in range(12, 30)],
            ]
            # For hyperpramerter studies, take only these!
            # batt_id_partitions = [
            #     ["6"],
            #     ["8"],
            #     ["21"],
            # ]
        elif n_devices == 2:
            batt_id_partitions = [
                [str(i) for i in range(1, 11)],
                [str(i) for i in range(11, 30)],
            ]
        else:
            batt_id_partitions = np.array_split(batt_id, n_devices)
        cuda_ids = list(np.arange(n_devices))

        processes = []
        for batt_id_partition, cuda_id in zip(batt_id_partitions, cuda_ids):
            print(batt_id_partition, cuda_id)
            device = torch.device(f"cuda:{cuda_id}")
            process = mp.Process(
                target=worker,
                args=(
                    batt_id_partition,
                    cell_characterstics,
                    fault_prob,
                    save_path,
                    device,
                    segment_criteria,
                ),
            )
            processes.append(process)

        for process in processes:
            process.start()

        for process in processes:
            process.join()
    else:
        battgp_analysis(
            batt_id,
            cell_characterstics,
            fault_prob,
            save_path,
            device,
            segment_criteria,
        )
    if not single_system:
        merge_pdfs(save_path, "single_axis", cs_string=cs_string)
        for i in FAULT_PROBS:
            i_ = i * 1e3
            merge_pdfs(
                save_path, "fault_prob_acausal", cs_string=cs_string, fault_prob=i_
            )
            if MODE == "spatio_temporal":
                merge_pdfs(
                    save_path, "forward_prob_causal", cs_string=cs_string, fault_prob=i_
                )

if __name__ == "__main__":
    main()
    