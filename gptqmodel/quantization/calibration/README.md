# 校准集生成

## 面向dense的校准集生成
使用说明：以Qwen3-8B为例，
```
cd gptqmodel/quantization/calibration/Dense_IVSG

python qwen3_dense_gen.py \ 
    --model_path /moe/model/path \  # 模型地址
    --num_beams 4 \ # 束宽，一般4足够
    --seqlen 512 \  # 句子长度
    --nsamples 100 \  # 生成多少条句子 
    --gen_dir gen_data # 生成目录名
```
当前qwen3系列的dense模型均可直接使用上述脚本生成

### 开发方法
以Qwen3-8B为例，环境为transformers==4.57.0

以monkey-patch的方式将原来库里的utils中beam search方法替换:
```
from custom_ivsg import _beam_search_ivsg
GenerationMixin._beam_search = _beam_search_ivsg
```
在生成时会自动进入_beam_search_ivsg进行生成

#### 生成脚本中hook
对加载的模型，进行线性层的维度统计，同时记录每个linear的输入x
```
    def save_input_hook(name):
        def hook_fn(module, input, output):
            x = input[0]
            module.input = x
        return hook_fn

    hook_handles = []
    hidden_dim = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handle = module.register_forward_hook(save_input_hook(name))
            hook_handles.append(handle)
            hidden_dim = hidden_dim+module.weight.shape[1]
    builtins.HIDDEN_DIM = hidden_dim
```
#### beam search改造，生成过程加入IVSG影响

```
        ...
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # a. Forward current tokens, obtain the logits
            flat_running_sequences = self._flatten_beam_dim(running_sequences[:, :, :cur_len])
            model_inputs = self.prepare_inputs_for_generation(flat_running_sequences, **model_kwargs)

            model_outputs = self(**model_inputs, return_dict=True)
            # IVSG
            _useless_output= self(flat_running_sequences,return_dict = True)
            # diag_scores = torch.zeros(num_beams).to('cuda')
            # r_l2 = torch.zeros(num_beams).to('cuda')
            scores = []
            # stretch_scores = []
            if flat_running_sequences.shape[1]>8:
                for name, module in self.named_modules():
                    if isinstance(module, nn.Linear):  
                        x = module.input
                        x = x - x.mean(dim=1, keepdim=True)
                        # 计算多样性指标（trace）
                        H = torch.bmm(x.transpose(1,2), x)
                        diversity = torch.mean(x.pow(2),dim=(1,2))
                        # 计算稳定性惩罚（norm）
                        trace_H = H.diagonal(dim1=-2, dim2=-1).sum(-1)
                        norm_term = torch.norm(x, p='fro',dim=(1,2))
                        # 归一化得分
                        raw_penalty = norm_term / (trace_H + 1e-6)
                        div_norm = (diversity - diversity.mean()) / (diversity.std() + 1e-6)
                        pen_norm = (raw_penalty - raw_penalty.mean()) / (raw_penalty.std() + 1e-6)
                        # 核心公式: Normalized Diversity - Beta * Normalized Penalty
                        layer_weight = module.weight.shape[1] / HIDDEN_DIM
                        combined_score =layer_weight*( div_norm - 1.2 * pen_norm)
                        scores.append(combined_score.tolist())
                        # scores.append((module.weight.shape[1]/hidden_dim*(diversity - 0.5* norm_term)).tolist())
                final_score = torch.tensor(scores).sum(dim=0)
                del _useless_output
                ...
```

对最后的得分进行干预：

```
...
log_probs = log_probs + running_beam_scores[:, :, None]
if flat_running_sequences.shape[1]>8: # 这里和前面的随机生成的短语长度相关，大于6就行
    
    # final_score =2* (final_score-final_score.min())/(final_score.max()-final_score.min())-1
    # combine
    log_probs = log_probs + final_score.unsqueeze(0).unsqueeze(-1).to(log_probs.device)
log_probs = torch.reshape(log_probs, (batch_size, num_beams * vocab_size))
...
```
由于每次都是对整句话从新计算，所以不需要维护什么状态矩阵，也就不需要像kvcache一样和束搜索的路径对齐（后续应该可以优化）


## 面向MOE的校准集生成

### 当前主路径：仓库内原生 EBSS 集成

当前仓库已经支持通过量化配置直接启用 EBSS 校准集生成，并优先走进程内原生路径：

- 不再以 subprocess 脚本调用作为主路径
- 运行时优先直接返回内存中的 calibration rows
- 如配置了 output_path，仍可额外导出 JSONL 用于复现与调试
- beam search 采用实例级绑定方式，避免全局 monkey patch transformers.GenerationMixin

推荐入口：

- QuantizeConfig.calibration_generation.enabled=True
- QuantizeConfig.calibration_generation.provider="moe_ebss"

### 兼容路径：旧脚本

下面的 qwen3_moe_gen.py / qwen3_next_gen.py 保留作为兼容与调试工具，便于单独分析生成质量或导出外部 JSONL；但仓库内量化流程不再依赖它们作为主执行路径。
使用说明：以Qwen3-30B-A3B为例,

```
cd gptqmodel/quantization/calibration/MoE_EBSS

python qwen3_moe_gen.py \ 
    --model_path /moe/model/path \  # 模型地址
    --num_beams 4 \ # 束宽，一般4足够
    --seqlen 512 \  # 句子长度
    --nsamples 100 \  # 生成多少条句子 
    --gen_dir gen_data # 生成目录名

```
当前qwen3-moe的一系列it模型都可以直接使用上述脚本生成

### 开发方法

以Qwen3-30B-A3B为例，环境为transformers==4.57.0

以monkey-patch的方式将原来库里的utils中beam search方法替换:

```
from custom_ebss import _beam_search_ebss
GenerationMixin._beam_search = _beam_search_ebss
```
在生成时会自动进入_beam_search_ebss进行生成

#### model改造，对moe模型加入专家统计的属性

在EBSS_models中，将moe模型的sparseblock部分重新定制
```
class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )
        # experts状态初始化
        self.save_expert_cnt = False
        self.top_cnt = torch.zeros(NUM_BEAMS,self.num_experts) 

    # 清空当前专家统计
    def clear_expert_cnt(self):
        self.top_cnt = torch.zeros(NUM_BEAMS,self.num_experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        # 进行当前token的专家频次统计
        if self.save_expert_cnt:
            for m in range(selected_experts.shape[0]):
                idx = m//sequence_length
                for n in range(self.top_k):
                    self.top_cnt[idx,selected_experts[m,n]] = self.top_cnt[idx,selected_experts[m,n]]+1
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        ...
```
#### beam search改造，生成过程加入专家得分影响

在beam search进行前打开统计状态，在预测完一次后，立刻更新当前统计矩阵
```
# init expert cnt
num_layers = self.model.config.num_hidden_layers
num_experts = self.model.config.num_experts
expert_cnt = torch.zeros(num_layers,num_beams,num_experts)
for layer_num in range(0,num_layers):
self.model.layers[layer_num].mlp.save_expert_cnt=True
# 4. run the generation loop
while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
# a. Forward current tokens, obtain the logits
flat_running_sequences = self._flatten_beam_dim(running_sequences[:, :, :cur_len])
model_inputs = self.prepare_inputs_for_generation(flat_running_sequences, **model_kwargs)
model_outputs = self(**model_inputs, return_dict=True)
for layer_num in range(0,num_layers):
    expert_dist = self.model.layers[layer_num].mlp.top_cnt
    # expert_dist = top_cnt.sum(dim=0)#,keepdim=True)
    expert_cnt[layer_num] += expert_dist
layer_vars = torch.var(expert_cnt, dim=-1)
expert_scores = torch.mean(layer_vars, dim=0)
expert_scores = expert_scores/(expert_scores.sum())

# synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
model_kwargs = self._update_model_kwargs_for_generation(
    model_outputs,
    model_kwargs,
    is_encoder_decoder=self.config.is_encoder_decoder,
)
if synced_gpus and this_peer_finished:
    continue
    ...
```

在专家得分确认后，融入当前beam的分中，同时记得清空当前模型中当前token对应的专家统计，保留总的专家统计expert_cnt
```
...
log_probs = self._unflatten_beam_dim(log_probs, batch_size, num_beams)
running_beam_scores[:, :, None] = running_beam_scores[:, :, None] * torch.exp(-expert_scores).unsqueeze(0).unsqueeze(-1).to(log_probs.device)
log_probs = log_probs + running_beam_scores[:, :, None]
# log_probs = log_probs * torch.exp(-expert_scores).unsqueeze(0).unsqueeze(-1).to(log_probs.device)
log_probs = torch.reshape(log_probs, (batch_size, num_beams * vocab_size))

for layer_num in range(0,num_layers):
    self.model.layers[layer_num].mlp.clear_expert_cnt()
...
```

在最后，因为beam search会进行句子重排，对应累积的expert_cnt也需要跟着kv cache一起根据beam_idx进行重排：
```
if model_kwargs.get("past_key_values", None) is not None:
    beam_idx = self._flatten_beam_dim(running_beam_indices[..., cur_len - decoder_prompt_len])
    if hasattr(self, "_reorder_cache"):
        model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)
    else:
        model_kwargs["past_key_values"].reorder_cache(beam_idx)
    expert_cnt = expert_cnt.index_select(1, beam_idx.cpu())
```
整条句子预测后，关掉统计状态：
```
sequences = self._flatten_beam_dim(sequences[:, :num_return_sequences, :])
beam_scores = self._flatten_beam_dim(beam_scores[:, :num_return_sequences])
beam_indices = self._flatten_beam_dim(beam_indices[:, :num_return_sequences, :])
# Crop the static-shaped tensors to the actual size.
# `beam_indices` is initialized with -1s, and is updated with the beam index of the generated token at each
# step. We can use it to detect the generated length, which may be != `cur_len`  (e.g. selected beam is from a
# previous decoding iteration)
for layer_num in range(0,num_layers):
    self.model.layers[layer_num].mlp.save_expert_cnt=False
max_generated_length = ((beam_indices + 1).bool()).sum(dim=1).max()
output_length = decoder_prompt_len + max_generated_length
sequences = sequences[:, :output_length]
```


## 生成性能说明（资源占用情况）

1. dense模型以Qwen3-8B为例，单卡占50G左右，生成128条句子的时间为8-10小时左右（待优化）

2. moe模型以Qwen3-30B-A3B为例，双卡各占60G左右，生成512条句子的时间为24-26小时左右

