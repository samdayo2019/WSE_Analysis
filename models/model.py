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
        self.context_len = kwargs.get('context_size', 0)
        self.max_context_len = kwargs.get('max_context_size', 0)
    
    def display(self):
        print(f"  Model: {self.name}")
        print(f"  Vocab Size: {self.vocab_size}")
        print(f"  Layers: {self.layers}")
        print(f"  Embedding Dimension: {self.emb_dim}")
        print(f"  Attention Heads: {self.num_attention_heads}")
        print(f"  KV Heads: {self.num_kv_heads}")
        print(f"  Head Dimension: {self.head_dim}")
        print(f"  Feedforward Dimension: {self.ffn_dim}")
        print(f"  Feedforward Layers: {self.ffn_layers}")
        print(f"  Context Length: {self.context_len}")
        print(f"  Max Context Length: {self.max_context_len}")

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


            