"""Helper functions for sparse analysis.
author: @gergelyturi
date: 11/10/2024
"""

from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ----------------------------- Dataframe filtering functions -----------------------------

def filter_dataframe(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    # Filter rows based on the given condition and drop unwanted columns
    filtered_df = df[df[condition] == True]
    # Dynamically drop columns that are not needed for correlation calculation
    columns_to_drop = [condition] + [
        col for col in df.columns if col in ["mobility", "mobile", "immobile",
                                              "awake", "NREM", "REM", "other"]
                                              ]
    retained_columns = [col for col in df.columns if col not in columns_to_drop]
    return filtered_df.drop(columns=columns_to_drop, errors='ignore'), retained_columns


# ----------------------------- calculation functions -----------------------------      

def data_calculation(dataframe: pd.DataFrame, cond1: str, cond2: str,
                    sima_folder: str, cell_id: int, calculation: str,
                    save_data: bool = True) -> pd.DataFrame:    
    # Make a copy of the original DataFrame to avoid modifying it directly
    df = dataframe.copy()

    # Filter the DataFrame for both conditions using the helper function
    cond1_df, cond1_retained_columns = filter_dataframe(df, cond1)
    cond2_df, cond2_retained_columns = filter_dataframe(df, cond2)

    # Calculate Pearson cross-correlation matrices for both conditions
    if calculation == "pearson":
        cond1_result = cond1_df.corr()
        cond2_result = cond2_df.corr()

        # Flatten the correlation matrices for scatter plot
        cond1_values = cond1_result.values.flatten()
        cond2_values = cond2_result.values.flatten()

        # Use retained column pairs as labels for the correlation matrix
        retained_labels = [
            f"{col1}-{col2}" for col1 in cond1_retained_columns for col2 in cond2_retained_columns
        ]

    elif calculation == "stdev":
        cond1_result = cond1_df.std()
        cond2_result = cond2_df.std()

        # Since standard deviation is per column, we use the retained column names directly
        cond1_values = cond1_result.values
        cond2_values = cond2_result.values

        # Use retained column names as labels for standard deviation
        retained_labels = cond1_retained_columns

    else:
        raise ValueError("Invalid calculation method. Please choose 'pearson' or 'stdev'.")

    # Create a DataFrame for plotting
    calculated_data = pd.DataFrame(
        {
            f"{cond1} {calculation}": cond1_values,
            f"{cond2} {calculation}": cond2_values,
            "Retained Labels": retained_labels
        }
    )
    
    # Save the DataFrame to a CSV file if needed
    if save_data:
        cell_num = str(cell_id)
        calculated_data.to_csv(join(sima_folder, f"{calculation}_{cond1}_{cond2}_{cell_num}.csv"), index=False)
    
    return calculated_data

def calculate_mean_correlations_triangle(dataframe: pd.DataFrame, cond1: str = "awake pearson", cond2: str = "NREM pearson") -> dict:
    # Extract retained labels and correlation values for both conditions
    retained_labels = dataframe["Retained Labels"].values
    cond1_values = dataframe[cond1].values
    cond2_values = dataframe[cond2].values

    # Split retained labels to get unique labels and identify the index of "soma"
    split_labels = [label.split('-') for label in retained_labels]
    unique_labels = sorted(set([label[0] for label in split_labels] + [label[1] for label in split_labels]))
    
    if "soma" not in unique_labels:
        raise ValueError("Label 'soma' not found in the unique labels.")
    
    # Fix the "soma" label to be first and sort the rest
    unique_labels.remove("soma")
    sorted_labels = ["soma"] + sorted(unique_labels)

    # Determine matrix size and reshape to form the correlation matrices for both conditions
    size = len(sorted_labels)
    cond1_matrix = cond1_values.reshape(size, size)
    cond2_matrix = cond2_values.reshape(size, size)

    # Extract the lower triangle of the matrix excluding the diagonal
    lower_triangle_indices = np.tril_indices(size, k=-1)
    
    # Find the index of the "soma" row/column
    soma_index = sorted_labels.index("soma")

    # Extract values for the "soma" row/column in the lower triangle for both conditions
    cond1_soma_values = np.concatenate((
        cond1_matrix[soma_index, :soma_index],  # Values left of the diagonal in the "soma" row
        cond1_matrix[(soma_index+1):, soma_index]  # Values below the diagonal in the "soma" column
    ))
    cond2_soma_values = np.concatenate((
        cond2_matrix[soma_index, :soma_index],  # Values left of the diagonal in the "soma" row
        cond2_matrix[(soma_index+1):, soma_index]  # Values below the diagonal in the "soma" column
    ))

    # Extract the rest of the values in the lower triangle excluding the "soma" row/column for both conditions
    cond1_non_soma_values = [
        cond1_matrix[i, j]
        for i, j in zip(lower_triangle_indices[0], lower_triangle_indices[1])
        if i != soma_index and j != soma_index
    ]
    cond2_non_soma_values = [
        cond2_matrix[i, j]
        for i, j in zip(lower_triangle_indices[0], lower_triangle_indices[1])
        if i != soma_index and j != soma_index
    ]
    
    # Calculate the mean for "soma" and non-"soma" for both conditions
    soma_mean_cond1 = np.mean(cond1_soma_values)
    non_soma_mean_cond1 = np.mean(cond1_non_soma_values)

    soma_mean_cond2 = np.mean(cond2_soma_values)
    non_soma_mean_cond2 = np.mean(cond2_non_soma_values)

    # Return the results as a dictionary
    results = {
        f"soma_mean_{cond1}": soma_mean_cond1,
        f"non_soma_mean_{cond1}": non_soma_mean_cond1,
        f"soma_mean_{cond2}": soma_mean_cond2,
        f"non_soma_mean_{cond2}": non_soma_mean_cond2,
    }

    return results

# ----------------------------- plotting functions -----------------------------

def plot_soma_denrite_traces(dataframe: pd.DataFrame, sima_folder: str,
                              cell_id: int, savefig: bool = True):
    """
    Plot the traces of the soma and dendrite of a neuron. This will try to plot everything in the dataframe.
    """
    df = dataframe.copy()
    # Downsample the data by a factor of 10
    downsampled_df = df.iloc[::10, :]

    # Set up a figure with multiple rows and one column
    num_cells = downsampled_df.shape[1]  # Include the cond1 column
    fig, axes = plt.subplots(num_cells, 1, figsize=(12, num_cells * 2), sharex=True)

    # Plot each cell's activity with color-codedcond1states
    for idx, cell in enumerate(
        downsampled_df.columns
    ):  # Iterate through all columns including cond1
        ax = axes[idx]

        # If it's thecond1column, plot it differently
        if cell == "mobility":
            ax.plot(
                downsampled_df.index,
                downsampled_df[cell],
                color="black",
                label="Mobility",
                linewidth=1.5,
            )
        else:
            # Plot the entire trace with segments color-coded bycond1state
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

def plot_xcorr_scatter(xcorr_dataframe: pd.DataFrame, cond1:str, cond2: str,
                                  sima_folder: str, cell_id: int, savefig: bool = True):
    calculated_data = xcorr_dataframe.copy()
    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=calculated_data, x=f"{cond1} pearson", y=f"{cond2} pearson", color="blue"
    )
    plt.title(f"Scatter Plot of Pearson Correlations ({cond2} vs. {cond1})")
    plt.xlabel(f"{cond1} Correlation")
    plt.ylabel(f"{cond2} Correlation")
    plt.axline((0, 0), slope=1, linestyle="--", color="red", label="y = x")

    # Calculate and plot the mean of the correlations
    cond1_mean_corr = calculated_data[f'{cond1} pearson'].mean()
    cond2_mobile_corr = calculated_data[f'{cond2} pearson'].mean()
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

def plot_std_dev_bars(std_dev_dataframe: pd.DataFrame, cond1: str, cond2: str,
                      sima_folder: str, cell_id: int, savefig: bool = True):
    # Extract the labels for the x-axis from the 'Retained Labels' column
    retained_labels = std_dev_dataframe["Retained Labels"].values

    # Plot the standard deviations for each condition
    ax = std_dev_dataframe[[f"{cond1} stdev", f"{cond2} stdev"]].plot(
        kind="bar", figsize=(14, 6), colormap="viridis", legend=True
    )

    # Set custom x-tick labels to the retained labels
    ax.set_xticks(range(len(retained_labels)))
    ax.set_xticklabels(retained_labels, rotation=45, ha="right")

    # Set labels and title
    plt.ylabel("Standard Deviation")
    plt.title(f"Comparison of Standard Deviation ({cond1} by {cond2})")
    plt.tight_layout()

    # Save the figure if required
    if savefig:
        cell_num = str(cell_id)
        plt.savefig(
            join(sima_folder, f"zscored_traces_standard_dev_{cond1}_{cond2}_{cell_num}.png"),
            dpi=300,
        )
        plt.savefig(
            join(sima_folder, f"zscored_traces_standard_dev_{cond1}_{cond2}_{cell_num}.svg"),
            format="svg",
            dpi=300,
        )

    # Display the plot
    plt.show()

def plot_correlation_heatmap(dataframe: pd.DataFrame, cond1: str, cond2: str,
                             sima_folder: str, cell_id: int, savefig: bool = True):
    # Extract the retained labels and correlation values
    retained_labels = dataframe["Retained Labels"].values
    cond1_values = dataframe[f"{cond1} pearson"].values
    cond2_values = dataframe[f"{cond2} pearson"].values

    # Split the retained labels to get unique row and column labels
    split_labels = [label.split('-') for label in retained_labels]
    unique_labels = sorted(set([label[0] for label in split_labels] + [label[1] for label in split_labels]))

    # Fix the "soma" label to be first and sort the rest
    if "soma" in unique_labels:
        unique_labels.remove("soma")
        sorted_labels = ["soma"] + sorted(unique_labels)
    else:
        sorted_labels = sorted(unique_labels)

    # Calculate the matrix size
    size = len(sorted_labels)

    # Reshape the correlation values to form matrices
    cond1_matrix = cond1_values.reshape(size, size)
    cond2_matrix = cond2_values.reshape(size, size)

    # Set up the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Set common color range for both heatmaps
    vmin = min(cond1_values.min(), cond2_values.min())
    vmax = max(cond1_values.max(), cond2_values.max())

    # Plot the heatmaps for cond1 and cond2
    sns.heatmap(cond1_matrix, annot=True, cmap="Greens", vmin=vmin, vmax=vmax, ax=axes[0], cbar=False, 
                xticklabels=False, yticklabels=False, annot_kws={"ha": "center", "va": "center"})
    axes[0].set_title(f"Cross-Correlation Heatmap ({cond1})")

    sns.heatmap(cond2_matrix, annot=True, cmap="Greens", vmin=vmin, vmax=vmax, ax=axes[1], cbar=False, 
                xticklabels=False, yticklabels=False, annot_kws={"ha": "center", "va": "center"})
    axes[1].set_title(f"Cross-Correlation Heatmap ({cond2})")

    # Manually add the labels in the center without tick marks
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Set the labels manually in the middle of each cell
        ax.set_xticks([i + 0.5 for i in range(size)], minor=True)
        ax.set_yticks([i + 0.5 for i in range(size)], minor=True)
        ax.set_xticklabels(sorted_labels, minor=True, rotation=45, ha="center")
        ax.set_yticklabels(sorted_labels, minor=True, rotation=0, va="center")

    plt.tight_layout()

    # Save the heatmap figures if needed
    if savefig:
        cell_num = str(cell_id)
        plt.savefig(
            join(sima_folder, f"correlation_heatmap_{cond1}_{cond2}_{cell_num}.png"), dpi=300
        )
        plt.savefig(
            join(sima_folder, f"correlation_heatmap_{cond1}_{cond2}_{cell_num}.svg"),
            format="svg",
            dpi=300,
        )

    # Display the plot
    plt.show()

def plot_mean_correlations_line(results: dict, cond1: str = "awake pearson", cond2: str = "NREM pearson"):
    # Extract values for plotting
    soma_mean_cond1 = results[f"soma_mean_{cond1}"]
    non_soma_mean_cond1 = results[f"non_soma_mean_{cond1}"]

    soma_mean_cond2 = results[f"soma_mean_{cond2}"]
    non_soma_mean_cond2 = results[f"non_soma_mean_{cond2}"]

    # Determine the y-axis limits based on the minimum and maximum values in the results dictionary
    all_values = [soma_mean_cond1, non_soma_mean_cond1, soma_mean_cond2, non_soma_mean_cond2]
    y_min = min(all_values) * 0.9  # Add a bit of margin for visibility
    y_max = max(all_values) * 1.1  # Add a bit of margin for visibility

    # Create a 1 row, 2 column figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

    # Labels for the x-axis
    x_labels = ['Soma', 'Non-Soma']
    x_positions = range(len(x_labels))

    # Plot for cond1 (awake pearson)
    axes[0].plot(x_positions, [soma_mean_cond1, non_soma_mean_cond1], marker='o', linestyle='-', color='#66c2a5', linewidth=2)
    axes[0].set_title(f"Mean Correlations ({cond1})")
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(x_labels)
    axes[0].set_ylabel("Mean Correlation")
    axes[0].set_ylim([y_min, y_max])

    # Plot for cond2 (NREM pearson)
    axes[1].plot(x_positions, [soma_mean_cond2, non_soma_mean_cond2], marker='o', linestyle='-', color='#fc8d62', linewidth=2)
    axes[1].set_title(f"Mean Correlations ({cond2})")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(x_labels)
    axes[1].set_ylim([y_min, y_max])

    # Tight layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()
