def calculate_total_params(model):
    # calculate total number of parameters (weights) in the model 
    
    # per layer Wo / Wq
    w_o = (model.num_attention_heads * model.head_dim)**2
    w_q = (model.num_attention_heads * model.head_dim)**2

    # per layer Wk / Wv
    w_k = (model.num_attention_heads * model.head_dim) * model.head_dim * model.num_kv_heads
    w_v = (model.num_attention_heads * model.head_dim) * model.head_dim * model.num_kv_heads
    
    # per layer Wffn
    w_ffn = model.ffn_layers * model.ffn_dim * model.emb_dim

    # per layer total weights
    w_tot_layer = w_o + w_q + w_k + w_v + w_ffn

    # linear layer weights
    w_ll = model.emb_dim * model.vocab_size

    # total weights for all layers
    total_weights = w_tot_layer * model.layers + w_ll

    per_layer_weights = [w_ll, w_o, w_q, w_k, w_v, w_ffn]
    return total_weights/1000**3, w_tot_layer/1000**3, [value/1000**3 for value in per_layer_weights]

def calculate_total_KV_cache_size(model, context_len, users):
    # calculate total size of key-value cache (G)
    # context_len: length of the context; users: number of users
    
    # per layer K/V cache
    kv_cache = 2 * model.num_kv_heads * model.head_dim * context_len * users

    # total K/V cache for all layers
    total_kv_cache = kv_cache * model.layers

    return total_kv_cache/1000**3, kv_cache/1000**3

def calculate_activations(model, context_len, input_len, users):
    # calculate total number of activations (G). We assume each layer processes max 1 user
    # max_seq_len: maximum sequence length; users: number of users
    # prefill: batch processing prompts with I input tokens to pre-fill I K/V pair; AR: serially producing O output tokens using I cached K/V pairs
    # Context_len = I + O
    # since each layer processes max 1 user at a time, the maximum number of concurrent layers processing is min(model.layers, users)
    
    # per input/output activations; in is the input prompt/token, out is the output prompt/token
    in_out_prefill = 2 * input_len* model.emb_dim
    in_out_AR = 2 * 1 * model.emb_dim 

    # per layer query activations; Query (Q) = XWq
    query_prefill = input_len * model.num_attention_heads * model.head_dim
    query_AR = 1 * model.num_attention_heads * model.head_dim

    # per layer attention matrix activations; qkT = QK^T. We exclude the softmax operation for activation calculation
    qkT_prefill = input_len * input_len * model.num_attention_heads
    qkT_AR = 1 * context_len * model.num_attention_heads

    # qkTV = (QK^T)V.
    qkTV_prefill = input_len * model.head_dim * model.num_attention_heads
    qkTV_AR = 1 * model.head_dim * model.num_attention_heads

    # O = Concat(qkTV)Wo. Concatenate the qkTVs from each attention head and multiply by Wo
    o_prefill = input_len * model.emb_dim
    o_AR = 1 * model.emb_dim

    # FFN activations; FFN = (Swish(OW1) + OW2)W3 for LLama Models
    ffn_prefill = (model.ffn_layers - 1) * input_len * model.ffn_dim + input_len * model.emb_dim
    ffn_AR = (model.ffn_layers - 1) * 1 * model.ffn_dim + model.emb_dim

    # FLL activations
    fll_prefill = input_len * model.vocab_size
    fll_AR = 1 * model.vocab_size 

    # maximum total activations for all layers
    max_total_activations = max(in_out_prefill, in_out_AR) + max(query_prefill, query_AR) + max(qkT_prefill, qkT_AR) + max(qkTV_prefill, qkTV_AR) + max(o_prefill, o_AR) + max(ffn_prefill, ffn_AR)
    max_total_activations = max_total_activations * min(model.layers, users) + max(fll_prefill, fll_AR)

    total_activations_prefill = in_out_prefill + query_prefill + qkT_prefill + qkTV_prefill + o_prefill + ffn_prefill
    total_activations_prefill = total_activations_prefill * min(model.layers, users) + fll_prefill

    total_activations_AR = in_out_AR + query_AR + qkT_AR + qkTV_AR + o_AR + ffn_AR
    total_activations_AR = total_activations_AR * min(model.layers, users) + fll_AR

    return max_total_activations/1000**3, total_activations_prefill/1000**3, total_activations_AR/1000**3

def calculate_total_flops(model, context_len, input_len, users):
    # calculate total number of FLOPs. We assume each layer processes max 1 user. 
    # context_len: length of the context; users: number of users
    # compute flops count for OI calculations:

    # store Gflops per parameter storage type (weighht LtRAM, KV$ StRAM, activations StRAM) for prefill and AR
    prefill_flops_breakdown = []
    AR_flops_breakdown = []

    # Q = XWq; for AR, x = 1 x emb_dim; for prefill, x = I x emb_dim; Wq: emb_dim x num_attention_heads x head_dim;
    q_flops_prefill = 2 * model.emb_dim * model.num_attention_heads * model.head_dim * input_len
    q_flops_AR = 2 * model.emb_dim * model.num_attention_heads * model.head_dim

    # K/V = XWk/XWv; for AR, x = 1 x emb_dim; for prefill, x = I x emb_dim; Wk/Wv: emb_dim x head_dim x num_kv_heads;
    kv_flops_prefill = 2 * 2 * model.emb_dim * model.head_dim * model.num_kv_heads * input_len  
    kv_flops_AR = 2 * 2 * model.emb_dim * model.head_dim * model.num_kv_heads

    # QK^T = QK^T; for prefill Q: I x num_attention_heads x head_dim; K: I x head_dim x num_kv_heads; for AR, Q: 1 x num_attention_heads x head_dim; K: (I  + O) x head_dim x num_kv_heads
    qkT_flops_prefill = 2 * model.head_dim * input_len * input_len * model.num_attention_heads
    qkT_flops_AR = 2 * model.head_dim * context_len * model.num_attention_heads

    # QK^TV = QK^TV; for prefill QKT = I x I x num_attention_heads; V: I x head_dim x num_kv_heads; for AR, Q: 1 x (I + O) * num_attention_heads; V: (I + O) x head_dim x num_kv_heads
    qkTV_flops_prefill = 2 * input_len * model.head_dim * input_len * model.num_attention_heads
    qkTV_flops_AR = 2 *  context_len * model.head_dim * 1 * model.num_attention_heads

    # O = Concat(qkTV)Wo; Wo: emb_dim x num_attention_heads x head_dim; for prefill qkTV: I x head_dim x num_attention_heads; for AR qkTV: 1 x head_dim x num_attention_heads
    o_flops_prefill = 2 * model.emb_dim * model.emb_dim * input_len
    o_flops_AR = 2 * model.emb_dim * model.emb_dim

    # FFN = (Swish(OW1) + OW2)W3; W1: emb_dim x ffn_dim; W2: emb_dim x nffn_dim; W3: ffn_dim x emb_dim; for refill O: I x emb_dim, for AR, O = 1 x emb_dim
    ffn_flops_prefill = 2 * model.ffn_layers * model.emb_dim * input_len * model.ffn_dim
    ffn_flops_AR = 2 * model.ffn_layers * model.emb_dim * model.ffn_dim 

    # FLL = O x WLL; WLL: emb_dim x vocab_size; for prefill O: I x emb_dim; for AR, O = 1 x emb_dim
    fll_flops_prefill = 2 * model.emb_dim * model.vocab_size * input_len
    fll_flops_AR = 2 * model.emb_dim * model.vocab_size

    # total FLOPs for all layers
    total_flops_prefill = min(model.layers, users) * (q_flops_prefill + kv_flops_prefill + qkT_flops_prefill + qkTV_flops_prefill + o_flops_prefill + ffn_flops_prefill) + fll_flops_prefill
    total_flops_AR = min(model.layers, users) * (q_flops_AR + kv_flops_AR + qkT_flops_AR + qkTV_flops_AR + o_flops_AR + ffn_flops_AR) + fll_flops_AR

    prefill_flops_breakdown.append(min(model.layers, users) * (q_flops_prefill + kv_flops_prefill + o_flops_prefill + ffn_flops_prefill) + fll_flops_prefill)
    prefill_flops_breakdown.append(min(model.layers, users) * (qkT_flops_prefill + qkTV_flops_prefill + kv_flops_prefill))
    prefill_flops_breakdown.append(total_flops_prefill)

    AR_flops_breakdown.append(min(model.layers, users) * (q_flops_AR + kv_flops_AR + o_flops_AR + ffn_flops_AR) + fll_flops_AR)
    AR_flops_breakdown.append(min(model.layers, users) * (qkT_flops_AR + qkTV_flops_AR + kv_flops_AR))
    AR_flops_breakdown.append(total_flops_AR)

    return [value /1000**3 for value in prefill_flops_breakdown], [value/1000**3 for value in AR_flops_breakdown]

def calculate_total_mem_transfer(model, context_len, input_len, users):
    # calculate total number of memory transfers. We assume each layer processes max 1 user. 
    # context_len: length of the context; users: number of users
    # compute memory transfer count for OI calculations:

    # store memory transfer (G) per parameter storage type (weighht LtRAM, KV$ StRAM, activations StRAM) for prefill and AR
    prefill_mem_transfer_breakdown = []
    AR_mem_transfer_breakdown = []

    weight_mem_transfer = calculate_total_params(model)[0]*1000**3    
    prefill_mem_transfer_breakdown.append(weight_mem_transfer)
    AR_mem_transfer_breakdown.append(weight_mem_transfer)

    # Q = XWq; read X once from act. mem and write each Q to act. mem. 
    q_act_mem_transfer_prefill = input_len* model.emb_dim + (input_len * model.head_dim) * model.num_attention_heads
    q_act_mem_transfer_AR = 1 * model.emb_dim + (1 * model.head_dim) * model.num_attention_heads

    # K/V = XWk/XWv; write K/V to KV$. X reaad once alrady for Q above. 
    kv_cache_mem_transfer_prefill = 2 * (input_len * model.head_dim) * model.num_kv_heads

    kv_cache_mem_transfer_AR = 2 * (1 * model.head_dim) * model.num_kv_heads # write K/V to KV$

    # QK^T = QK^T; read Q from act. mem and K from KV$ and write result to act. mem. For GQA, each KV$ is reach G times per group, for n_kv groups with n_h/n_kv heads per group
    qkT_act_mem_transfer_prefill = (input_len * model.head_dim + input_len * input_len) * model.num_attention_heads
    qkT_cache_mem_transfer_prefill = (input_len * model.head_dim * model.num_kv_heads / model.num_attention_heads) * model.num_attention_heads
  
    qkT_act_mem_transfer_AR = (1 * model.head_dim + 1 * context_len) * model.num_attention_heads
    qkT_cache_mem_transfer_AR = (context_len * model.head_dim * model.num_kv_heads / model.num_attention_heads) * model.num_attention_heads

    # QK^TV = QK^TV; read QK^T from act. mem and V from KV$ and write result to act. mem. For GQA, each KV$ is reach G times per group, for n_kv groups with n_h/n_kv heads per group
    qkTV_act_mem_transfer_prefill = (input_len * input_len + input_len * model.head_dim) * model.num_attention_heads
    qkTV_cache_mem_transfer_prefill = (input_len * model.head_dim * model.num_kv_heads / model.num_attention_heads) * model.num_attention_heads

    qkTV_act_mem_transfer_AR = (1 * context_len + 1 * model.head_dim) * model.num_attention_heads
    qkTV_cache_mem_transfer_AR = (context_len * model.head_dim * model.num_kv_heads / model.num_attention_heads) * model.num_attention_heads

    # O = Concat(qkTV)Wo; read qkTV from act. mem and Wo from weight mem and write result to act. mem.
    o_act_mem_transfer_prefill = input_len * model.head_dim * model.num_attention_heads + input_len * model.emb_dim
    o_act_mem_transfer_AR = 1 * model.head_dim * model.num_attention_heads + model.emb_dim

    # FFN = (Swish(OW1) + OW2)W3; read O from act. mem and write swish(OW1), OW2, and final product result to act. mem. Reread Swish(OW1) and OW2.
    ffn_act_mem_transfer_prefill = input_len * model.emb_dim + 2 * (model.ffn_layers - 1) * input_len * model.ffn_dim + input_len * model.emb_dim
    ffn_act_mem_transfer_AR = 1 * model.emb_dim + 2 * (model.ffn_layers - 1) * 1 * model.ffn_dim + model.emb_dim

    # FLL = O x WLL; read O from act. mem and write result to act. mem.
    fll_act_mem_transfer_prefill = input_len * model.emb_dim + input_len * model.vocab_size
    fll_act_mem_transfer_AR = 1 * model.emb_dim + 1 * model.vocab_size

    # total memory transfer for all layers
    total_act_mem_transfer_prefill = (q_act_mem_transfer_prefill  + qkT_act_mem_transfer_prefill + qkTV_act_mem_transfer_prefill + o_act_mem_transfer_prefill + 
                                      ffn_act_mem_transfer_prefill) * min(model.layers, users) + fll_act_mem_transfer_prefill

    total_act_mem_transfer_AR = (q_act_mem_transfer_AR + qkT_act_mem_transfer_AR + qkTV_act_mem_transfer_AR + o_act_mem_transfer_AR + 
                                 ffn_act_mem_transfer_AR) * min(model.layers, users) + fll_act_mem_transfer_AR
    
    total_kv_mem_transfer_prefill = (kv_cache_mem_transfer_prefill + qkT_cache_mem_transfer_prefill + qkTV_cache_mem_transfer_prefill) * min(model.layers, users)
    total_kv_mem_transfer_AR = (kv_cache_mem_transfer_AR + qkT_cache_mem_transfer_AR + qkTV_cache_mem_transfer_AR) * min(model.layers, users)

    prefill_mem_transfer_breakdown.append(total_act_mem_transfer_prefill)
    prefill_mem_transfer_breakdown.append(total_kv_mem_transfer_prefill)

    AR_mem_transfer_breakdown.append(total_act_mem_transfer_AR)
    AR_mem_transfer_breakdown.append(total_kv_mem_transfer_AR)

    return [value/1000**3 for value in prefill_mem_transfer_breakdown], [value/1000**3 for value in AR_mem_transfer_breakdown]

def calculate_storage(model, w_res, act_res, context_len, input_len, users):
    # calculate weight storage, KV$, and activation storage in GB
    # w_res: weight resolution in bits; act_res: activation resolution in bits
    total_weights, total_kv_cache, total_activations = calculate_total_params(model)[0], calculate_total_KV_cache_size(model, context_len, users)[0], calculate_activations(model, context_len, input_len, users)[0]
    weight_storage = total_weights * (w_res / 8)
    
    kv_storage = total_kv_cache * (act_res / 8)

    act_storage = total_activations * (act_res / 8)

    return weight_storage, kv_storage, act_storage



