import os
import warnings

import colorcet as cc
import datashader as ds
import holoviews as hv
import holoviews.operation.datashader as hd
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import panel as pn
from holoviews.operation.datashader import datashade

from .. import config as cfg
from .batt_data import BattData

hv.extension("bokeh")


def calculate_voltage_statistics(df: pd.DataFrame, voltage_cols: list[str]) -> None:
    """Calculate voltage statistics, mean subtracted voltage, voltage stdv."""
    voltage_data = df[voltage_cols].values
    n_cells = len(voltage_cols)

    voltage_sum = np.sum(voltage_data, axis=1)

    for c in range(n_cells):
        df[f"Ucell_WO{c + 1}_mean"] = (voltage_sum - voltage_data[:, c]) / (n_cells - 1)

    df["CellVoltageVar"] = np.var(voltage_data, axis=1, ddof=1)
    df["CellVoltageStdv"] = np.sqrt(df["CellVoltageVar"])
    return None


def add_nan_rows(df: pd.DataFrame, *, max_gap: int = 1) -> pd.DataFrame:
    """
    Add a nan row to the dataframe when the time difference between the rows is larger than max_gap days.
    Do not delete or owerwrite any rows, add a new row instead with a time index that is max_gap days later.
    """
    # Remove duplicated indices for plotting this
    df = df[~df.index.duplicated(keep="first")].copy()
    rows_prior = len(df)
    duplicated_indices = df.index[df.index.duplicated(keep="last")]
    if len(duplicated_indices) > 0:
        print(
            f"Warning: {duplicated_indices} duplicated indices found, adding 0.01 to duplicated indices"
        )
        for index in duplicated_indices:
            df.loc[index] = df.loc[index] + 0.01
    else:
        print("No duplicated indices found")

    # Add nan rows
    df.loc[:, "TimeDiff"] = df.index.to_series().diff()
    df.loc[:, "TimeDiff"] = df["TimeDiff"].dt.total_seconds()
    df.loc[:, "TimeDiff"] = df["TimeDiff"] / 3600 / 24
    df.loc[:, "TimeDiff"] = df["TimeDiff"].fillna(0)
    df.loc[:, "TimeDiff"] = df["TimeDiff"].astype(int)
    i_ind = np.where(df["TimeDiff"] > max_gap)[0]
    i_ind_prev = i_ind - 1
    dt_ind_prev = df.iloc[i_ind_prev].index.copy()
    for i in dt_ind_prev:
        df.loc[i + pd.Timedelta(days=max_gap)] = np.nan
    df = df.sort_index(
        kind="stable"
    )  # this is needed to put the nan row at the right location in the new nan rows
    df.drop(columns=["TimeDiff"], inplace=True)
    rows_after = len(df)
    print(f"{rows_after - rows_prior} nan rows added, to avoid lines between gaps")
    return df


def diagnostic_plot_datashader(
    battdata: BattData,
    *,
    dynamic: bool = False,
    save: bool = False,
    zoomed_in_plot: bool = False,
    scale: float = 0.4,
) -> None:
    """Plot all time series data using datashader, which allows to plot millions of data points

    Parameters
    ----------
    battdata : BattData
        Battery data object
    dynamic : bool, optional
        If True, the plot is interactive, by default False.
    save : bool, optional
        If True, the plot is saved, by default False.
    zoomed_in_plot : bool, optional
        If True, a zoomed in plot of the voltage standard deviation is added, by default False.
    scale : float, optional
        Scale of the plot, by default 0.4, scale 1 will result in a plot of 1000x6500 pixels.
        only used when dynamic is False.
    """
    batt_id = battdata.id
    battdata.df = add_nan_rows(battdata.df)
    if dynamic:
        # https://github.com/holoviz/holoviews/issues/5746
        warnings.warn(
            """Warning: Dynamic plot is not working, see issue #5746:
            https://github.com/holoviz/holoviews/issues/5746 \n
            This needs more investigation"""
        )
        datashade.dynamic = True
        height = 200
        width = 1200
        lwl = 0.2
        lwm = 0.4
        lwh = 0.6
        fontscale = 0.5
        labelfontsize = "12pt"

        def hook(plot, element):
            # print('plot.state:   ', plot.state)
            # print('plot.handles: ', sorted(plot.handles.keys()))
            plot.handles["plot"].legend.label_text_font_size = labelfontsize
            plot.handles["plot"].legend.spacing = 0
            plot.handles["plot"].legend.glyph_height = 20
            plot.handles["plot"].legend.glyph_width = 20

    else:
        datashade.dynamic = False
        height = int(1000 * scale)
        width = int(6500 * scale)
        lwl = 1.4 * scale
        lwm = 3 * scale
        lwh = 5 * scale
        fontscale = 4.7 * scale
        labelfontsize = f"{40*scale}pt"

        def hook(plot, element):
            # print('plot.state:   ', plot.state)
            # print('plot.handles: ', sorted(plot.handles.keys()))
            plot.handles["plot"].legend.label_text_font_size = labelfontsize
            plot.handles["plot"].legend.spacing = int(15 * scale)
            plot.handles["plot"].legend.glyph_height = int(60 * scale)
            plot.handles["plot"].legend.glyph_width = int(60 * scale)

    start_time = battdata.df.index[0]
    end_time = battdata.df.index[-1]
    time_range = end_time - start_time
    padding = 0.01 * time_range
    x_lim = (start_time - padding, end_time + padding)

    # Alphabetical title prefix list
    title_prefix_list = [
        "a) ",
        "b) ",
        "c) ",
        "d) ",
        "e) ",
        "f) ",
        "g) ",
        "h) ",
        "i) ",
        "j) ",
        "k) ",
    ]

    opts_plotting = {
        "width": width,
        "height": height,
        "xlabel": "Time (Date)",
        "fontscale": 1.25 * fontscale,
        "xticks": 8,
        "yticks": 8,
        "legend_position": "right",
        # "legend_offset": (0, height / 2.465),
        "xlim": x_lim,
    }

    opts_plotting_upper = opts_plotting.copy()
    opts_plotting_upper["xlabel"] = ""

    datashade_opts = {
        "width": width,
        "height": height,
    }

    def melted_df(
        df: pd.DataFrame,
        cols: list[str],
        *,
        new_col_names: list[str] = None,
        y_dim_name: str = "value",
        x_dim_name: str = "Timestamp",
    ):
        # add a nan row to the end of the dataframe to make sure all curves are plotted until the end

        if isinstance(df.index, pd.core.indexes.datetimes.DatetimeIndex):
            df = df.reset_index()
        df.loc[len(df)] = pd.Series()
        df_melted = pd.melt(df, id_vars=x_dim_name, value_vars=cols).copy()
        df_melted["variable"] = df_melted["variable"].astype("category")
        # Rename "variable" to "cat" to be consistent with the example below
        df_melted.rename(columns={"variable": "cat"}, inplace=True)
        # Switch order of columns
        df_melted = df_melted[[x_dim_name, "value", "cat"]]
        # rename columns
        df_melted.rename(
            columns={"value": y_dim_name, "Date": x_dim_name}, inplace=True
        )
        if new_col_names is not None:
            df_melted["cat"] = df_melted["cat"].cat.rename_categories(new_col_names)
        return df_melted

    plot_shades_list = []
    subplot_index = 0
    # Cell Voltage
    title_str = f"Battery {batt_id} \n {title_prefix_list[subplot_index]} Cell voltage"
    voltage_cell_cols = [
        f"U_Cell_{i}" for i in range(1, len(battdata.cell_voltage_cols) + 1)
    ]
    # Calculate the mean subtracted voltage and the voltage standard deviation
    calculate_voltage_statistics(battdata.df, voltage_cell_cols)

    new_voltage_col_names = [
        f"Cell {i}" for i in range(1, len(battdata.cell_voltage_cols) + 1)
    ]
    df_melted_voltage = melted_df(
        battdata.df,
        voltage_cell_cols,
        new_col_names=new_voltage_col_names,
        y_dim_name="Voltage (V)",
    )
    # colors = {
    #     key: mcolors.rgb2hex(cc.glasbey_bw[i])
    #     for i, key in enumerate(new_voltage_col_names)
    # }
    curve_cell_voltage = hv.Curve(df_melted_voltage)
    ds_cell_voltage = hd.datashade(
        curve_cell_voltage,
        aggregator=ds.count_cat("cat"),
        # color_key=colors,
        line_width=lwl,
        **datashade_opts,
    ).opts(**opts_plotting_upper, ylabel="Voltage (V)", title=title_str, hooks=[hook])
    plot_shades_list.append(ds_cell_voltage)
    subplot_index += 1

    # Mean Subtracted Voltage
    title_str = f"{title_prefix_list[subplot_index]} Cell voltage minus other cell's mean voltage"
    mean_WOx_voltage_cell_cols = [
        f"Ucell_WO{i}_mean" for i in range(1, len(battdata.cell_voltage_cols) + 1)
    ]
    df_mean_sub = battdata.df[voltage_cell_cols + mean_WOx_voltage_cell_cols].copy()
    df_mean_sub[voltage_cell_cols] = (
        df_mean_sub[voltage_cell_cols].values
        - df_mean_sub[mean_WOx_voltage_cell_cols].values
    )
    df_mean_sub.drop(columns=mean_WOx_voltage_cell_cols, inplace=True)
    # Tell me, what is x in the rename function?
    #
    df_mean_sub.rename(
        columns=lambda x: "\u0168" + " C" + x[3:-2] + " " + x[-1], inplace=True
    )
    mean_sub_cell_cols = [
        "\u0168" + f" Cell {i}" for i in range(1, len(battdata.cell_voltage_cols) + 1)
    ]
    df_melted_mean_sub_voltage = melted_df(
        df_mean_sub, mean_sub_cell_cols, y_dim_name="Delta Voltage (V)"
    )

    curve_mean_sub_voltage = hv.Curve(df_melted_mean_sub_voltage)
    ds_mean_sub_voltage = hd.datashade(
        curve_mean_sub_voltage,
        aggregator=ds.count_cat("cat"),
        # color_key=colors,
        line_width=lwl,
        **datashade_opts,
    ).opts(
        **opts_plotting_upper,
        ylabel="\u0394" + " Voltage (V)",
        title=title_str,
        hooks=[hook],
    )
    plot_shades_list.append(ds_mean_sub_voltage)
    subplot_index += 1

    # Voltage Standard Deviation
    title_str = (
        f"{title_prefix_list[subplot_index]} Standard deviation of cell voltages"
    )
    df_voltage_std = pd.DataFrame(battdata.df["CellVoltageStdv"].copy(deep=True))
    # get min max of voltage std
    y_max = df_voltage_std["CellVoltageStdv"].max()
    # y_min = df_voltage_std["CellVoltageStdv"].min()
    curve_cell_voltage_std = hv.Curve(df_voltage_std, label="Cell Voltage Std")
    ds_std_voltage = hd.datashade(
        curve_cell_voltage_std,
        cmap=mcolors.rgb2hex(cc.glasbey_bw[15]),
        line_width=1.3 * lwh,
        **datashade_opts,
    ).opts(
        **opts_plotting_upper,
        ylabel="\u03C3" + " Voltage (V)",
        title=title_str,
        ylim=(-0.05 * y_max, y_max),
    )
    plot_shades_list.append(ds_std_voltage)
    subplot_index += 1

    # Zoomed in Voltage Standard Deviation
    # Get the max value of the voltage std
    if zoomed_in_plot:
        voltage_std_max = df_voltage_std["CellVoltageStdv"].max()
        if voltage_std_max > 3 * 0.15:
            y_max_zoom = 0.15
        else:
            y_max_zoom = voltage_std_max / 3
        title_str = f"{title_prefix_list[subplot_index]} Zoomed view: Standard deviation of cell voltages"
        df_voltage_std_z = pd.DataFrame(battdata.df["CellVoltageStdv"].copy(deep=True))
        # Delete all values above y_max_zoom
        df_voltage_std_z.loc[df_voltage_std_z["CellVoltageStdv"] > y_max_zoom] = (
            1.01 * y_max_zoom
        )
        # Add zoom to column name
        df_voltage_std_z.rename(
            columns={"CellVoltageStdv": "CellVoltageStdvZoom"}, inplace=True
        )
        curve_cell_voltage_std_z = hv.Curve(
            df_voltage_std_z, label="Cell Voltage Std"
        ).opts(ylim=(-y_max_zoom / 20, y_max_zoom))
        ds_std_voltage_zoom = hd.datashade(
            curve_cell_voltage_std_z,
            cmap=mcolors.rgb2hex(cc.glasbey_bw[15]),
            line_width=1.3 * lwh,
            **datashade_opts,
        ).opts(
            **opts_plotting_upper,
            ylabel="\u03C3" + " Voltage (V)",
            title=title_str,
            ylim=(-0.05 * y_max_zoom, y_max_zoom),
        )
        plot_shades_list.append(ds_std_voltage_zoom)
        subplot_index += 1

    # Battery Current
    title_str = f"{title_prefix_list[subplot_index]} Battery current"
    black_color = "#000000"
    df_voltage_std_z = pd.DataFrame(battdata.df["I_Batt"].copy(deep=True))
    curve_current = hv.Curve(df_voltage_std_z, label="Battery Current")
    ds_current = hd.datashade(
        curve_current,
        cmap=black_color,
        line_width=lwm,
        **datashade_opts,
    ).opts(**opts_plotting_upper, ylabel="Current (A)", title=title_str)
    plot_shades_list.append(ds_current)
    subplot_index += 1

    # For all batteries plot Icnv
    title_str = f"{title_prefix_list[subplot_index]} Cell balancing converter current"
    icnv_cols = [
        f"I_CNV_Cell_{i}" for i in range(1, len(battdata.cell_voltage_cols) + 1)
    ]
    new_icnv_col_names = [
        f"I Cnv. Cell {i}" for i in range(1, len(battdata.cell_voltage_cols) + 1)
    ]
    df_melted_icnv = melted_df(
        battdata.df,
        icnv_cols,
        new_col_names=new_icnv_col_names,
        y_dim_name="Current (A)",
    )
    curve_cell_icnv = hv.Curve(df_melted_icnv)
    ds_cell_icnv = hd.datashade(
        curve_cell_icnv,
        aggregator=ds.count_cat("cat"),
        # color_key=colors,
        line_width=lwl,
        **datashade_opts,
    ).opts(
        **opts_plotting_upper, ylabel="I Converter (A)", title=title_str, hooks=[hook]
    )
    plot_shades_list.append(ds_cell_icnv)
    subplot_index += 1

    # State of Charge
    title_str = f"{title_prefix_list[subplot_index]} State of charge"
    # blue_color = "#1f77b4"
    df_soc = pd.DataFrame(battdata.df["SOC_Batt"].copy(deep=True))
    curve_soc = hv.Curve(df_soc, label="State of Charge")
    ds_soc = hd.datashade(
        curve_soc,
        cmap=mcolors.rgb2hex(cc.glasbey_bw[8]),
        line_width=lwm,
        **datashade_opts,
    ).opts(**opts_plotting_upper, ylabel="SOC (%)", title=title_str, ylim=(-3, 103))
    plot_shades_list.append(ds_soc)
    subplot_index += 1

    # Temperatures plot
    title_str = f"{title_prefix_list[subplot_index]} Temperature sensors"
    names = ["T Sensor 1", "T Sensor 2", "T Sensor 3", "T Sensor 4"]
    col_names = ["T_Cell_1_2", "T_Cell_3_4", "T_Cell_5_6", "T_Cell_7_8"]
    # colors_temp = cfg.COLORS[1:]

    df_melted_temp = melted_df(
        battdata.df,
        col_names,
        new_col_names=names,
        y_dim_name="Temperature (°C)",
    )
    curve_cell_temp = hv.Curve(df_melted_temp)
    ds_cell_temp = hd.datashade(
        curve_cell_temp,
        aggregator=ds.count_cat("cat"),
        # color_key=colors_temp,
        line_width=lwl,
        **datashade_opts,
    ).opts(
        **opts_plotting_upper, ylabel="Temperature (°C)", title=title_str, hooks=[hook]
    )
    plot_shades_list.append(ds_cell_temp)
    subplot_index += 1

    # Create a plot to show where data is missing
    # Format the length of the dataframe to be displayed in the title of the plot to be more readable
    if len(battdata.df) > 1000000:
        len_df_str = f"{len(battdata.df)/1000000:.1f} M"
    elif len(battdata.df) > 1000:
        len_df_str = f"{len(battdata.df)/1000:.1f} k"
    else:
        len_df_str = f"{len(battdata.df)}"
    title_str = f"{title_prefix_list[7]} Data availability, #rows: {len_df_str}"
    df_ones = pd.DataFrame(index=battdata.df.index, data=np.ones(len(battdata.df)))
    df_ones.columns = ["DataAvail"]
    df_ones["DataAvail"] = df_ones["DataAvail"] + np.random.uniform(
        0, 0.1, len(df_ones)
    )
    curve_data_avail = hv.Scatter(df_ones)
    opts_data_avail = opts_plotting.copy()
    opts_data_avail["height"] = int(opts_plotting["height"] / 2.5)
    if opts_data_avail["height"] / opts_data_avail["width"] < 1 / 20:
        print(
            "Warning: height/width ratio is less than 1/20 for data availability plot, this may cause issues with the title and ticks."
        )
    opts_data_avail["yticks"] = 0
    ds_data_avail = hd.datashade(curve_data_avail).opts(
        **opts_data_avail, title=title_str, ylabel=""
    )
    plot_shades_list.append(ds_data_avail)

    layout = hv.Layout(plot_shades_list).cols(1)

    if save:
        os.makedirs(cfg.PATH_FIGURES_DATA_VIS, exist_ok=True)
        f_name = os.path.join(
            cfg.PATH_FIGURES_DATA_VIS, f"Batt{battdata.id}DiagnosticDatashader.png"
        )
        pn.pane.HoloViews(layout).save(f_name)
        # Clear the plot to avoid memory issues
        layout = None
    else:
        return layout
