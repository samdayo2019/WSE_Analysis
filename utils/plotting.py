import numpy as np
import matplotlib.pyplot as plt
from utils.calculations import (
    calculate_total_params,
    calculate_total_KV_cache_size,
    calculate_activations,
    calculate_total_flops
)

def plot_model_chip_requirements(models, weight_density, weight_tiers, kv_density, act_density, tmacs_per_mm2, w_res, act_res):
    """
    Plots the number of chips required for LLama 3.5T and 405B models (for now) based on compute and storage requirements. 
    Input contex length for prefill is assumed to be 1/2 of the model context length.

    Parameters:
        models (list): List of LLMModel instances to analyze (for now LLama 3.5T and 405B).
        weight_density (float): Weight storage density in GB/mm².
        weight tiers (int): Number of tiers (layers) for weight memory.
        kv_density (float): KV$ storage density in GB/mm².
        act_density (float): Activation storage density in GB/mm².
        tmacs_per_mm2 (float): Compute density in TMACs/mm².

    Returns:
        None: Displays the plot.
    """
    # Chip sizes in mm²
    chip_sizes = {"reticle": 800, "mobile": 80}
    
    # User and model size ranges
    user_range = [2 ** (i) for i in range(11) if 2 ** i <= 1024]
    model_sizes = []

    # Prepare data for plotting
    data_points = []
    
    for model in models:
        total_params, _, _ = calculate_total_params(model)
        model_size = total_params
        model_sizes.append(model_size)

        for users in user_range:
            # Calculate storage requirements
            for weight_res, activation_res in zip(w_res, act_res):
                kv_cache, _ = calculate_total_KV_cache_size(model, users=users)
                act_storage, _, _ = calculate_activations(model, input_len=model.context_len * 0.5, users=users)
                
                weight_storage = (weight_res / 8) * total_params * (1 / weight_density) / weight_tiers  # mm²
                kv_storage = (activation_res / 8) * kv_cache * (1 / kv_density)  # mm²
                act_storage = (activation_res / 8) * act_storage * (1 / act_density)  # mm²

                total_SRAM_storage_area = kv_storage + act_storage  # mm²
                
                # Calculate compute area
                peak_flops, _, _ = calculate_total_flops(model, input_len=model.context_len * 0.5, users=users)
                compute_area = 2 * peak_flops / 1000 / tmacs_per_mm2  # mm²

                # Total chip area required. We assume LtRAM for weights is stacked on top of compute/SRAM which are layed in 2D.
                total_area = max(total_SRAM_storage_area + compute_area, weight_storage)

                # Calculate number of chips
                num_reticle_chips = np.ceil(total_area / chip_sizes["reticle"])
                #num_mobile_chips = np.ceil(total_area / chip_sizes["mobile"])
                
                # Append data for plotting
                data_points.append({
                    "model_size": model_size,
                    "users": users,
                    "total_area": total_area,
                    "num_reticle_chips": num_reticle_chips
                })

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # custom colormap from cool to warm (blue to red)
    cmap = plt.cm.get_cmap("coolwarm")
    norm = plt.Normalize(1, max(dp['num_reticle_chips'] for dp in data_points))
    
    # Plot data points
    scatter = None
    for dp in data_points:
        scatter = ax.scatter(dp['model_size'], dp['users'], c=[cmap(norm(dp['num_reticle_chips']))], s=100, edgecolors='k', alpha=0.8)
        ax.text(dp['model_size'], dp['users'], f"{int(dp['num_reticle_chips'])}", fontsize=8, ha='center', va='center', color='black')
    
    # Colorbar setup
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Number of Reticle Chips Required", rotation=270, labelpad=15)

    # X-axis settings
    x_ticks = sorted(set(dp['model_size'] for dp in data_points))
    x_labels = [f"{size: .1f} GB" for size in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    # Plot settings
    ax.set_xlabel("Model Size (Number of Parameters in Billions)")
    ax.set_ylabel("Number of Users")
    ax.set_title("Chip Requirements for LLama3 Models")
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("data/chip_requirements.png")