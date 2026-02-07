#!/bin/bash

# 检查是否提供了8个参数：GPU编号
if [ "$#" -ne 7 ]; then
    echo "Usage: $0 <GPU_ID> <MODEL> <W_BITS> <A_BITS> <KV_BITS> <R2_PATH> <R1_PATH>"
    exit 1
fi

# 获取传入的参数
GPU_ID=$1
MODEL=$2
R1_PATH=$7
R2_PATH=$6
W_BITS=$3
A_BITS=$4
KV_BITS=$5

# 提取“/”后面的字符串
MODEL_NAME=${MODEL##*/}

# 执行命令并使用传入的参数
CUDA_VISIBLE_DEVICES=${GPU_ID} python main_for_test.py \
    --model ${MODEL} \
    --fuse_norm \
    --use_r1 \
    --r1_path ${R1_PATH} \
    --use_r2 offline \
    --r2_path ${R2_PATH} \
    --use_r3 \
    --use_r4 \
    --w_groupsize 128 \
    --w_clip \
    --a_asym \
    --a_clip_ratio 0.9 \
    --w_bits ${W_BITS} \
    --a_bits ${A_BITS} \
    --k_bits ${KV_BITS} \
    --v_bits ${KV_BITS} \
    --k_groupsize 128 \
    --v_groupsize 128 \
    --k_asym \
    --v_asym \
    --o_per_head \
    --percdamp 0.1 \
    --no-w_ft \
    --ft_percdamp 0.0 \
    --save_name dart_w${W_BITS}a${A_BITS}kv${KV_BITS}_128nsamples \
    --ppl_eval \
    --ppl_eval_batch_size 1 \
    --ppl_eval_dataset wikitext2 ptb c4 \
    --lm_eval \
    --lm_eval_batch_size 2 \
    --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada_openai social_iqa openbookqa mmlu \
    --w_rtn
    # --a_lac \
    # --s_channel_scale \
    # --s_lr 1e-4 \
    # --k_lac \
    # --v_lac \
    # --kron_lac_layerwise \
    # --lac_lr 1e-4 \
    # --kron_lr 8e-6 \
    # --lac_epochs 20 \
    # --lac_cali_bsz 8 \
    # --use_rkron \
    # --rkron_path /data02/cheny/dartquant/QSR-geoopt/trained_rotation/wikitext2_1024samples/llama3-8b-hf/rkron_r1/RAdam.0.001.0.9.20.64.0.3.1.cos/RAdam.0.001.0.9.20.64.0.3.1.cos.best.pt \
    # --use_rkron_o \
    # --rkron_o_path /data02/cheny/dartquant/QSR-geoopt/trained_rotation/wikitext2_1024samples/llama3-8b-hf/rkron_o_proj/RAdam.0.001.0.9.20.64.0.3.1.cos/RAdam.0.001.0.9.20.64.0.3.1.cos.pt \
    # --use_rkron_down \
    # --rkron_down_path /data02/cheny/dartquant/QSR-geoopt/trained_rotation/wikitext2_1024samples/llama3-8b-hf/rkron_down_proj/RAdam.0.001.0.9.20.32.0.3.1.cos/RAdam.0.001.0.9.20.32.0.3.1.cos.pt \
    # --kron_inv_eps 1e-7 \
    # --kron_lac_debug \
    # --deactive_amp \
    # --warmup \
    # --no-kron_orthogonal \
    # --layerwise_loss kl --layerwise_kl_temp 1.0
    # --kron_lac_grad_clip 1.0 \
    # --a_lac \
    # --k_lac \
    # --v_lac \
    # --lac_layerwise \
    # --lac_lr 1e-2 \
    # --lac_epochs 1 \
    # --lac_cali_bsz 1 \
    # --load_qmodel_path ${LOAD_QMODEL_PATH} \
