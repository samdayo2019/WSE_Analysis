import json

class LLMModel:
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'llm_model')
        self.vocab_size = kwargs.get('vocab_size', 0)
        self.layers = kwargs.get('layers', 0)
        self.emb_dim = kwargs.get('emb_dim', 0)
        self.num_attention_heads = kwargs.get('num_attention_heads', 0)
        self.num_kv_heads = kwargs.get('num_kv_heads', 0)
        self.head_dim = kwargs.get('head_dim', 0)
        self.ffn_dim = kwargs.get('ffn_dim', 0)
        self.ffn_layers = kwargs.get('ffn_layers', 0)

def load_llm_model(model_path):
    with open(model_path, 'r') as f:
        model_config = json.load(f)
    return model_config

def parse_models_by_type(json_data):
    set_of_models = {}
    for model_types in json_data["model_types"]:
        model_type = model_types["model_name"]
        models = [LLMModel(**model) for model in model_types["models"]]
        set_of_models[model_type] = models
    return set_of_models


            