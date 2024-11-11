"""Helper functions for sparse analysis.
author: @gergelyturi
date: 11/10/2024
"""

from os.path import join

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ----------------------------- calculation functions -----------------------------      

def xcorr_data_calc(dataframe: pd.DataFrame, cond1: str, cond2: str, sima_folder: str, cell_id: int, save_data: bool = True) -> pd.DataFrame:
    # TODO: the correlation calculation and the filtering based on the column names looks fishy!!!!
    df = dataframe.copy()
    # Split the DataFrame based on mobility status
    cond1_df = df[df[cond1] == True].drop(
        columns=["mobility", "mobile", "immobile", "awake", "NREM", "REM", "other"]
    )
    cond2_df = df[df[cond2] == True].drop(
        columns=["mobility", "mobile", "immobile", "awake", "NREM", "REM", "other"]
    )

    # Calculate Pearson cross-correlation matrices for both conditions
    cond1_corr = cond1_df.corr()
    cond2_corr = cond2_df.corr()

    # Extract columns containing "plane" for both correlation matrices
    plane_columns = [col for col in cond1_corr.columns if "plane" in col]

    # Extract the correlations for these columns from both matrices
    cond1_plane_corr = cond1_corr.loc[plane_columns, plane_columns]
    cond2_plane_corr = cond2_corr.loc[plane_columns, plane_columns]

    # Flatten the correlation matrices and prepare data for scatter plot
    cond1_corr_values = cond1_plane_corr.values.flatten()
    cond2_corr_values = cond2_plane_corr.values.flatten()

    # Create a DataFrame for plotting
    scatter_data = pd.DataFrame(
        {
            f"{cond1} Correlation": cond1_corr_values,
            f"{cond2} Correlation": cond2_corr_values,
        }
    )
    if save_data:
        cell_num = str(cell_id)
        scatter_data.to_csv(join(sima_folder, f"xcorr_mob_immob_{cell_num}.csv"), index=False)
    
    return scatter_data    

# ----------------------------- plotting functions -----------------------------

def plot_soma_denrite_traces(dataframe: pd.DataFrame, sima_folder: str, cell_id: int, savefig: bool = True):
    """
    Plot the traces of the soma and dendrite of a neuron. This will try to plot everything in the dataframe.
    """
    df = dataframe.copy()
    # Downsample the data by a factor of 10
    downsampled_df = df.iloc[::10, :]

    # Set up a figure with multiple rows and one column
    num_cells = downsampled_df.shape[1]  # Include the 'mobility' column
    fig, axes = plt.subplots(num_cells, 1, figsize=(12, num_cells * 2), sharex=True)

    # Plot each cell's activity with color-coded mobility states
    for idx, cell in enumerate(
        downsampled_df.columns
    ):  # Iterate through all columns including 'mobility'
        ax = axes[idx]

        # If it's the mobility column, plot it differently
        if cell == "mobility":
            ax.plot(
                downsampled_df.index,
                downsampled_df[cell],
                color="black",
                label="Mobility",
                linewidth=1.5,
            )
        else:
            # Plot the entire trace with segments color-coded by mobility state
            ax.plot(
                downsampled_df.index,
                downsampled_df[cell],
                color="blue",
                label="Immobile (False)",
                linewidth=1.5,
            )
            ax.plot(
                downsampled_df.index,
                downsampled_df[cell].where(downsampled_df["mobility"]),
                color="red",
                label="Mobile (True)",
                linewidth=1.5,
            )

        # Set titles and labels
        ax.set_title(f"{cell} Activity" if cell != "mobility" else "Mobility State")
        if idx == num_cells - 1:
            ax.set_xlabel("Time")
        ax.set_ylabel("Activity Level" if cell != "mobility" else "Mobility")

        # Only add legend to the first subplot to reduce clutter
        if idx == 0:
            ax.legend(loc="upper right")

    # Adjust layout to avoid overlap
    plt.tight_layout()
    if savefig:
        cell_num = str(cell_id)
        plt.savefig(
            join(sima_folder, f"soma_dendrite_traces_mob_immob_{cell_num}.png"), dpi=300
        )
        plt.savefig(
            join(sima_folder, f"soma_dendrite_traces_mob_immob_{cell_num}.svg"),
            format="svg",
            dpi=300,
        )
    plt.show()

def plot_xcorr_mob_immob_scatter(xcorr_dataframe: pd.DataFrame, cond1:str, cond2: str, sima_folder: str, cell_id: int, savefig: bool = True):
    scatter_data = xcorr_dataframe.copy()
    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=scatter_data, x=f"{cond1} Correlation", y=f"{cond2} Correlation", color="blue"
    )
    plt.title(f"Scatter Plot of Pearson Correlations ({cond2} vs. {cond1})")
    plt.xlabel(f"{cond1} Correlation")
    plt.ylabel(f"{cond2} Correlation")
    plt.axline((0, 0), slope=1, linestyle="--", color="red", label="y = x")

    # Calculate and plot the mean of the correlations
    cond1_mean_corr = scatter_data[f'{cond1} Correlation'].mean()
    cond2_mobile_corr = scatter_data[f'{cond2} Correlation'].mean()
    plt.scatter(
        cond1_mean_corr,
        cond2_mobile_corr,
        color="green",
        s=200,
        edgecolor="gray",
        linewidth=2,
        label="Mean Correlation",
    )

    plt.legend()
    plt.tight_layout()
    if savefig:
        cell_num = str(cell_id)
        plt.savefig(join(sima_folder, f"xcorr_scatter_{cond1}_{cond2}_{cell_num}.png"), dpi=300)
        plt.savefig(
            join(sima_folder, f"xcorr_scatter_{cond1}_{cond2}_{cell_num}.svg"),
            format="svg",
            dpi=300,
        )
    plt.show()