from models.model import LLMModel, load_llm_model, parse_models_by_type
from utils import calculations
from utils import plotting
CONTEXT_LEN = 2048
INPUT_LEN = 1024
USERS = 256

if __name__ == "__main__":
    model_data = load_llm_model("data/models.hjson")

    for model_type in model_data["model_families"]:
        if model_type["family_name"] == "LLama3.1":
            llama3_1_models = [LLMModel(**model) for model in model_type["models"]]
        if model_type["family_name"] == "LLama3":
            llama3_models = [LLMModel(**model) for model in model_type["models"]]
        if model_type["family_name"] == "LLama2":
            llama2_models = [LLMModel(**model) for model in model_type["models"]]

    # Parameters
    weight_density = 0.02  # GB/mm²
    weight_tiers = 64      # JUNGVI: What does this represents
    kv_density = 0.034     # GB/mm²
    act_density = 0.034    # GB/mm²
    tmacs_per_mm2 = 1.352   # TMACs/mm²
    w_res = [8]
    act_res = [8]

    plotting.plot_model_chip_requirements(llama3_models, weight_density, weight_tiers, kv_density, act_density, tmacs_per_mm2, w_res, act_res)
    """
    models = parse_models_by_type(model_config)
    
    for model_type, model in models.items():
        print(f"Model: {model_type}")
        for model_name in model:
            model_name.display()
            print(f"  Number of Parameters: {calculations.calculate_total_params(model_name)} GB")
            print(f"  KV Memory: {calculations.calculate_total_KV_cache_size(model_name, USERS)} GB")
            print(f"  Number of FLOPs: {calculations.calculate_total_flops(model_name, model_name.context_len * 0.5, USERS)} GFLOPs")
            print(f"  Total Memory Transfer: {calculations.calculate_total_mem_transfer(model_name,  model_name.context_len * 0.5, USERS)} GB")
            print(f"  Memory Storage Requirement: {calculations.calculate_storage(model_name, 8, 8,  model_name.context_len * 0.5, USERS)} GB")
            # print(f"  Name: {model_name.name}")
    """
    

