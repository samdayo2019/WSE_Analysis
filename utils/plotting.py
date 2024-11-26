import numpy as np
import matplotlib.pyplot as plt
from utils.calculations import (
    calculate_total_params,
    calculate_total_KV_cache_size,
    calculate_activations,
    calculate_total_flops
)

def plot_model_chip_requirements(models, weight_density, kv_density, act_density, tmacs_per_mm2):
    """
    Plots the number of chips required for LLama 3.5T and 405B models based on compute and storage requirements.

    Parameters:
        models (list): List of LLMModel instances to analyze (LLama 3.5T and 405B).
        weight_density (float): Weight storage density in GB/mm².
        kv_density (float): KV$ storage density in GB/mm².
        act_density (float): Activation storage density in GB/mm².
        tmacs_per_mm2 (float): Compute density in TMACs/mm².

    Returns:
        None: Displays the plot.
    """
    # Chip sizes in mm²
    chip_sizes = {"reticle": 800, "mobile": 80}
    
    # User and model size ranges
    user_range = np.arange(1, 21, 1)  # Number of users from 1 to 20
    model_sizes = []

    # Prepare data for plotting
    data_points = []
    
    for model in models:
        total_params, _, _ = calculate_total_params(model)
        for users in user_range:
            # Calculate storage requirements
            kv_cache, _ = calculate_total_KV_cache_size(model, context_len=512, users=users)
            act_storage, _, _ = calculate_activations(model, context_len=512, input_len=128, users=users)
            
            weight_storage = total_params * (1 / weight_density)  # mm²
            kv_storage = kv_cache * (1 / kv_density)  # mm²
            act_storage = act_storage * (1 / act_density)  # mm²

            total_storage_area = weight_storage + kv_storage + act_storage  # mm²
            
            # Calculate compute area
            _, _, peak_flops = calculate_total_flops(model, context_len=512, input_len=128, users=users)
            compute_area = peak_flops / tmacs_per_mm2  # mm²

            # Total chip area required
            total_area = total_storage_area + compute_area

            # Calculate number of chips
            num_reticle_chips = np.ceil(total_area / chip_sizes["reticle"])
            num_mobile_chips = np.ceil(total_area / chip_sizes["mobile"])
            
            # Append data for plotting
            data_points.append((model.name, users, total_area, num_reticle_chips, num_mobile_chips))
            if model.name not in model_sizes:
                model_sizes.append(model.name)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_name in model_sizes:
        model_data = [point for point in data_points if point[0] == model_name]
        x = [point[2] for point in model_data]  # Total area
        y = [point[1] for point in model_data]  # Users
        colors = [point[3] for point in model_data]  # Chips required

        scatter = ax.scatter(x, y, c=colors, cmap="coolwarm", label=model_name, s=80, edgecolors='k', alpha=0.8)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Number of Reticle Chips Required", rotation=270, labelpad=15)

    # Plot settings
    ax.set_title("Chip Requirements for LLM Models")
    ax.set_xlabel("Total Model Size (mm²)")
    ax.set_ylabel("Number of Users")
    ax.legend(title="Model")
    plt.tight_layout()
    plt.show()
