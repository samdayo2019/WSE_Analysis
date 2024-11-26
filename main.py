from models.model import LLMModel, load_llm_model, parse_models_by_type
from utils import calculations
CONTEXT_LEN = 2048
INPUT_LEN = 1024
USERS = 256

if __name__ == "__main__":
    model_path = 'data/models.json'
    model_config = load_llm_model(model_path)
    models = parse_models_by_type(model_config)
    
    for model_type, model in models.items():
        print(f"Model: {model_type}")
        for model_name in model:
            print(f"  Name: {model_name.name}")
            print(f"  Vocab Size: {model_name.vocab_size}")
            print(f"  Layers: {model_name.layers}")
            print(f"  Embedding Dimension: {model_name.emb_dim}")
            print(f"  Number of Attention Heads: {model_name.num_attention_heads}")
            print(f"  Number of Key-Value Heads: {model_name.num_kv_heads}")
            print(f"  Head Dimension: {model_name.head_dim}")
            print(f"  Feed Forward Network Dimension: {model_name.ffn_dim}")
            print(f"  Feed Forward Network Layers: {model_name.ffn_layers}")
            print(f"  Number of Parameters: {calculations.calculate_total_params(model_name)} GB")
            print(f"  KV Memory: {calculations.calculate_total_KV_cache_size(model_name, CONTEXT_LEN,USERS)} GB")
            print(f"  Number of FLOPs: {calculations.calculate_total_flops(model_name, CONTEXT_LEN, INPUT_LEN, USERS)} GFLOPs")
            print(f"  Total Memory Transfer: {calculations.calculate_total_mem_transfer(model_name, CONTEXT_LEN, INPUT_LEN, USERS)} GB")
            print(f"  Memory Storage Requirement: {calculations.calculate_storage(model_name, 8, 8, CONTEXT_LEN, INPUT_LEN, USERS)} GB")
            # print(f"  Name: {model_name.name}")

