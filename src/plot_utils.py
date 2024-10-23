import datetime
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import PyPDF2
import torch
from PyPDF2 import PdfReader, PdfWriter, Transformation

from . import config as cfg
from .batt_data.batt_data import BattData
from .batt_data.cell_characteristics import CellCharacteristics


def ocv_lookup_plot(
    cellchar: CellCharacteristics,
    save: bool = True,
    save_path: Optional[str] = None,
    ylim: Optional[tuple[float, float]] = (2.5, 3.8),
):
    """Plot ocv-soc lookup and confidence evaluation."""

    if save_path is None:
        save_path = cfg.PATH_FIGURES_DATA_VIS

    test_soc = np.arange(0, 100, 0.1)
    # make subplots for ocv-soc lookup and confidence evaluation
    fig, ax = plt.subplots(figsize=(10 / 1.3, 6 / 1.3))
    ax.plot(test_soc, cellchar.ocv_lookup(test_soc, extrapolate=True))
    ax.set_xlabel("SOC (%)")
    ax.set_ylabel("OCV (V)")
    ax.set_title("OCV-SOC relationship for LFP cells")
    ax.grid()
    # tciks every 0.1 in y and every 5 in x
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.set_ylim(ylim)
    if save:
        # Make directory if it does not exist
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/OCV-SOC_uncertainty.pdf", bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(figsize=(10 / 1.6, 6 / 1.6))
    # second axis, plot soc-ocv in range of SOC limits
    ax.plot(test_soc, cellchar.ocv_lookup(test_soc), label="OCV-SOC relationship")
    ax.set_xlabel("SOC (%)", fontsize=12)
    ax.set_ylabel("OCV (V)", fontsize=12)
    ax.set_title("Approximated OCV Curve", fontsize=14)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.set_ylim(ylim)

    ax.set_xlim(cfg.SOC_LOWER_LIMIT, cfg.SOC_UPPER_LIMIT)
    ax.grid()
    if save:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/OCV-SOC.pdf", bbox_inches="tight")
    plt.show()


def add_second_datetime_xaxis_below(fig: plt.Figure, axs: plt.Axes, start_dt: datetime):
    """Add a second x-axis with the datetime format below the last axis in axs"""
    # Create a second x-axis with the datetime format
    ax2: plt.Axes = axs[-1].twiny()
    if fig is not None:
        fig.subplots_adjust(bottom=0.2)
    ax2.spines["bottom"].set_position(("outward", 40))
    ax2.set_frame_on(True)
    ax2.patch.set_visible(False)

    # Ticks and labels
    age_ticks = axs[-1].get_xticks()
    age_xaxis_limits = axs[-1].get_xlim()
    # Filter out ticks that are outside the data range
    age_ticks_f = age_ticks[
        (age_ticks >= age_xaxis_limits[0]) & (age_ticks <= age_xaxis_limits[1])
    ]
    ax2.set_xlim(
        start_dt + datetime.timedelta(axs[-1].get_xlim()[0]),
        start_dt + datetime.timedelta(axs[-1].get_xlim()[1]),
    )
    new_ticks = [
        start_dt + datetime.timedelta(age_ticks_f[i]) for i in range(len(age_ticks_f))
    ]
    # fortmat the ticks
    new_ticks = [dt.strftime("%Y-%m-%d") for dt in new_ticks]
    ax2.set_xticks(new_ticks)
    ax2.set_xticklabels(new_ticks)
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.set_xlabel("Date")
    # Rotate the tick labels and set their alignment such that they are centered on the tick
    ax2.tick_params(axis="x", rotation=30)
    return ax2


def merge_pdfs(
    save_path: str,
    file: str,
    cs_string: Optional[str] = None,
    cols: int = 3,
    fault_prob: Optional[float] = None,
):
    # Now go through all the plotted pdfs and assemble them into a single single page pdf
    pdf_pages: List[PyPDF2.PageObject] = []
    # Get all subfolder in the save path
    analyzed_batts = os.listdir(save_path)
    analyzed_batts = [i for i in analyzed_batts if not i.endswith(".py")]
    analyzed_batts = [i for i in analyzed_batts if not i.endswith(".csv")]
    analyzed_batts = [i for i in analyzed_batts if not i.startswith(".")]
    analyzed_batts = [i for i in analyzed_batts if not i[:3] == "all"]
    analyzed_batts = [i for i in analyzed_batts if not i.endswith(".pdf")]

    # Sort the folders, but make sure 10 comes after 9
    analyzed_batts.sort(key=int)

    def reader(file):
        return PdfReader(open(file, "rb"))

    for batt_folder in analyzed_batts:
        # Read the pdf file in the respective folder
        if file == "single_axis":
            pdf_file_path = os.path.join(
                save_path, batt_folder, f"Batt{batt_folder}_R_single_axis.pdf"
            )
        elif file == "multiaxis":
            pdf_file_path = os.path.join(
                save_path, batt_folder, f"Batt{batt_folder}_R_multi_axis.pdf"
            )
        elif file == "fault_prob_acausal":
            pdf_file_path = os.path.join(
                save_path,
                batt_folder,
                f"Batt{batt_folder}_fault_acausal_band{fault_prob:.2f}_prob.pdf",
            )
        elif file == "forward_prob_causal":
            pdf_file_path = os.path.join(
                save_path,
                batt_folder,
                f"Batt{batt_folder}_fault_causal_band{fault_prob:.2f}_prob.pdf",
            )
        else:
            raise ValueError(
                "Unknown file string, file must be single_axis, multiaxis or fault_prob."
            )
        if os.path.exists(pdf_file_path):
            pdf = reader(pdf_file_path)
            pdf_pages.append(pdf.pages[0])

    if file == "fault_prob":
        cs_string = f"{cs_string}_{fault_prob:.2f}"
    elif file == "fault_forward_prob":
        cs_string = f"{cs_string}_forward_{fault_prob:.2f}"

    width = float(pdf_pages[0].mediabox.width)
    height = float(pdf_pages[0].mediabox.height)

    rows = int(np.ceil(len(analyzed_batts) / cols))

    total_height = rows * height
    total_width = cols * width
    translated_page = PyPDF2.PageObject.create_blank_page(
        width=total_width,
        height=total_height,
    )

    for i in range(rows):
        for j in range(cols):
            if i * cols + j < len(pdf_pages):
                print(f"Adding page {i * cols + j}")
                page2 = pdf_pages[i * cols + j]
                # Increase the canvas of the page to fit the new page
                page2.mediabox.upper_right = (
                    (j + 1.2) * width,
                    (rows - 1 - i + 1.2) * height,
                )
                page2.add_transformation(
                    Transformation().translate(
                        tx=1.05 * j * width,
                        ty=(total_height - 0.8 * height) - (1.01 * i * height),
                    )
                )
                translated_page.merge_page(page2, expand=True)

    pdf_writer = PdfWriter()
    pdf_writer.add_page(translated_page)
    if cs_string is None:
        with open(os.path.join(save_path, f"all_batts_{file}.pdf"), "wb") as out:
            pdf_writer.write(out)
    else:
        save_path_ = save_path.split("case_study")[0]
        with open(
            os.path.join(save_path_, f"{cs_string}_all_batts_{file}.pdf"), "wb"
        ) as out:
            pdf_writer.write(out)
    print("All results assembled into a single pdf")


def get_basis_vectors(
    batt_data: BattData,
    ref_op,
    nbasis: Tuple[int, int, int] = [4, 4, 5],
) -> torch.Tensor:
    """
    Set basis vectors for the spatiotemporal GP model
    TODO: Adapt battmodels/basis_vector_selection.py to avoid code duplication
    """

    # "Linearly spaced basis vectors"
    # Issues: Assymetric, the op/mean is not part of the basis vectors!

    spacing_multiplier = np.array([1, 1, 1])
    # Center the basis vectors around the mean

    first_dim_points = np.linspace(
        ref_op[0] - spacing_multiplier[0] * cfg.LENGTHSCALE_RBF[0],
        np.min(
            [
                ref_op[0] + spacing_multiplier[0] * cfg.LENGTHSCALE_RBF[0],
                batt_data.segment_criteria.ibat_upper_limit,
            ]
        ),
        nbasis[0],
    )
    second_dim_points = np.linspace(
        np.max(
            [
                ref_op[1] - spacing_multiplier[1] * cfg.LENGTHSCALE_RBF[1],
                batt_data.segment_criteria.soc_lower_limit,
            ]
        ),
        np.min(
            [
                ref_op[1] + spacing_multiplier[1] * cfg.LENGTHSCALE_RBF[1],
                batt_data.segment_criteria.soc_upper_limit,
            ]
        ),
        nbasis[1],
    )

    third_dim_points = np.linspace(
        np.max(
            [
                ref_op[2] - spacing_multiplier[2] * cfg.LENGTHSCALE_RBF[2],
                batt_data.segment_criteria.t_lower_limit,
            ]
        ),
        np.min(
            [
                ref_op[2] + spacing_multiplier[2] * cfg.LENGTHSCALE_RBF[2],
                batt_data.segment_criteria.t_upper_limit,
            ]
        ),
        nbasis[2],
    )
    basis_vectors = np.zeros((nbasis[0] * nbasis[1] * nbasis[2], 3))
    for i, p1 in enumerate(first_dim_points):
        for j, p2 in enumerate(second_dim_points):
            for k, p3 in enumerate(third_dim_points):
                basis_vectors[i * nbasis[1] * nbasis[2] + j * nbasis[2] + k] = [
                    p1,
                    p2,
                    p3,
                ]

    # Check whether ref_op is part of the basis vectors
    if not np.any(np.all(basis_vectors == ref_op, axis=1)):
        basis_vectors = np.vstack(
            [
                basis_vectors,
                ref_op,
            ]
        )
    return basis_vectors
