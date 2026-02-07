### 目前最好结果 

2026-01-22 18:28:43,451 - INFO - Arguments: 
2026-01-22 18:28:43,454 - INFO - {'a_asym': True,
 'a_bits': 4,
 'a_bits_down_proj': None,
 'a_clip_ratio': 0.9,
 'a_groupsize': -1,
 'a_lac': True,
 'a_residual': False,
 'act_order': False,
 'cal_dataset': 'wikitext2',
 'capture_layer_io': False,
 'distribute': False,
 'fp32_had': False,
 'ft_percdamp': 0.0,
 'fuse_norm': True,
 'hf_token': None,
 'k_asym': True,
 'k_bits': 4,
 'k_clip_ratio': 1.0,
 'k_groupsize': 128,
 'k_lac': True,
 'k_pre_rope': False,
 'kron_cali_bsz': None,
 'kron_epochs': None,
 'kron_fp32': True,
 'kron_inv_eps': 0.0,
 'kron_lac_layerwise': False,
 'kron_lr': None,
 'kron_orthogonal': True,
 'kron_proj_every': 1,
 'kron_wd': 0.0,
 'lac_cali_bsz': 4,
 'lac_epochs': 10,
 'lac_layerwise': True,
 'lac_log_interval': 10,
 'lac_lr': 0.005,
 'lac_wd': 0.0,
 'layer_idx': 10,
 'lm_eval': True,
 'lm_eval_batch_size': '2',
 'load_qmodel_path': None,
 'log_to_console': True,
 'model': '/data01/datasets/llama3-8b-hf',
 'nsamples': 512,
 'o_per_head': True,
 'percdamp': 0.1,
 'ppl_eval': True,
 'ppl_eval_batch_size': 1,
 'ppl_eval_dataset': ['wikitext2', 'ptb', 'c4'],
 'r1_path': '/data02/cheny/dartquant/QSR-geoopt/trained_rotation/wikitext2_1024samples/llama3-8b-hf/r1/RAdam.0.1.0.9.40.128.0.3.1.cos',
 'r2_path': '/data02/cheny/dartquant/QSR-geoopt/trained_rotation/wikitext2_1024samples/llama3-8b-hf/r2/RAdam.0.05.0.9.20.64.2.cos',
 'rkron_down_n1': None,
 'rkron_down_n2': None,
 'rkron_down_path': None,
 'rkron_inv_eps': 0.0,
 'rkron_mode': None,
 'rkron_n1': None,
 'rkron_n2': None,
 'rkron_o_n1': None,
 'rkron_o_n2': None,
 'rkron_o_path': None,
 'rkron_path': None,
 'rotate_mode': 'hadamard',
 'save_name': 'dart_w4a4kv4_128nsamples',
 'save_path': './experiments/llama3-8b-hf/dart_w4a4kv4_128nsamples',
 'save_qmodel_path': None,
 'seed': 0,
 'smooth': None,
 'tasks': ['piqa',
           'hellaswag',
           'arc_easy',
           'arc_challenge',
           'winogrande',
           'lambada_openai',
           'social_iqa',
           'openbookqa',
           'mmlu'],
 'use_r1': True,
 'use_r2': 'offline',
 'use_r3': True,
 'use_r4': True,
 'use_rkron': False,
 'use_rkron_down': False,
 'use_rkron_o': False,
 'v_asym': True,
 'v_bits': 4,
 'v_clip_ratio': 1.0,
 'v_groupsize': 128,
 'v_lac': True,
 'w_asym': False,
 'w_bits': 4,
 'w_bits_down_proj': None,
 'w_clip': True,
 'w_ft': False,
 'w_groupsize': 128,
 'w_rtn': False,
 'w_static_groups': False,
 'wandb': False,
 'wandb_id': None,
 'wandb_project': None}
2026-01-22 18:28:43,454 - INFO - ------------------------------------------------------------
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  3.92it/s]
2026-01-22 18:28:44,595 - INFO - ---> Loading /data01/datasets/llama3-8b-hf Model with seq_len: 2048
2026-01-22 18:28:44,597 - INFO - Fuse LayerNorms
2026-01-22 18:28:44,597 - INFO - Rotate the model use_r1=True, use_r2=offline, use_r4=True, use_r3=True, use_rkron=False, use_rkron_o=False, use_rkron_down=False
/data01/home/cheny/project/DartQuant/fake_quant/rotation_utils.py:375: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  Q = torch.load(args.r1_path, map_location=torch.device(utils.DEV))['R1'].to(dtype=torch.float64)
2026-01-22 18:29:24,015 - INFO - GPU memory (from rotate_model): 8.02 -> 0.19 GB (-7.83 GB)
Rotating:   0%|                                                                                    | 0/32 [00:00<?, ?layer/s]/data01/home/cheny/project/DartQuant/fake_quant/rotation_utils.py:319: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  r2s = torch.load(r2_path, map_location='cpu')
Rotating: 100%|███████████████████████████████████████████████████████████████████████████| 32/32 [02:14<00:00,  4.21s/layer]
2026-01-22 18:31:39,012 - INFO - GPU memory (from main): 1.53 -> 0.06 GB (-1.47 GB)
/data01/home/cheny/miniconda3/envs/dartquant/lib/python3.11/site-packages/transformers/modeling_utils.py:4713: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
  warnings.warn(
2026-01-22 18:31:39,073 - INFO - Add v quantization: v_bits=4, v_groupsize=128, v_sym=False, v_clip_ratio=1.0
2026-01-22 18:31:39,081 - INFO - Add k quantization: k_bits=4, k_groupsize=128, k_sym=False, k_clip_ratio=1.0
2026-01-22 18:31:39,086 - INFO - Add weight quantization: w_rtn = False, w_bits = 4, w_groupsize = 128, w_sym = True, w_clip = True
2026-01-22 18:32:02,293 - INFO - -----GPTQ Quantization-----
cuda:0
(GPTQ Quant.) Layers:   0%|                                                                           | 0/32 [00:00<?, ?it/s]The attention layers in this model are transitioning from computing the RoPE embeddings internally through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed `position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be removed and `position_embeddings` will be mandatory.
(GPTQ Quant.) Layers: 100%|██████████████████████████████████████████████████████████████████| 32/32 [44:56<00:00, 84.28s/it]
2026-01-22 19:17:01,216 - INFO - GPU memory (from gptq_fwrd): 16.07 -> 16.07 GB (0.00 GB)
2026-01-22 19:17:01,216 - INFO - -----GPTQ Quantization Done-----
2026-01-22 19:17:19,919 - INFO - ========= LAC Layer 0 =========
2026-01-22 19:17:45,448 - INFO - LAC layer 0 epoch 1/10: mse=0.000432
2026-01-22 19:20:53,302 - INFO - LAC layer 0 epoch 10/10: mse=0.000401
2026-01-22 19:20:53,628 - INFO - ========= LAC Layer 1 =========
2026-01-22 19:21:19,659 - INFO - LAC layer 1 epoch 1/10: mse=0.036773
2026-01-22 19:24:27,407 - INFO - LAC layer 1 epoch 10/10: mse=0.030236
2026-01-22 19:24:27,960 - INFO - ========= LAC Layer 2 =========
2026-01-22 19:24:54,286 - INFO - LAC layer 2 epoch 1/10: mse=0.002962
2026-01-22 19:28:03,026 - INFO - LAC layer 2 epoch 10/10: mse=0.002559
2026-01-22 19:28:03,423 - INFO - ========= LAC Layer 3 =========
2026-01-22 19:28:29,832 - INFO - LAC layer 3 epoch 1/10: mse=0.005539
2026-01-22 19:31:38,376 - INFO - LAC layer 3 epoch 10/10: mse=0.004581
2026-01-22 19:31:38,766 - INFO - ========= LAC Layer 4 =========
2026-01-22 19:32:04,714 - INFO - LAC layer 4 epoch 1/10: mse=0.007483
2026-01-22 19:35:12,092 - INFO - LAC layer 4 epoch 10/10: mse=0.006058
2026-01-22 19:35:12,488 - INFO - ========= LAC Layer 5 =========
2026-01-22 19:35:38,707 - INFO - LAC layer 5 epoch 1/10: mse=0.009485
2026-01-22 19:38:47,070 - INFO - LAC layer 5 epoch 10/10: mse=0.007626
2026-01-22 19:38:47,475 - INFO - ========= LAC Layer 6 =========
2026-01-22 19:39:13,709 - INFO - LAC layer 6 epoch 1/10: mse=0.011300
2026-01-22 19:42:21,647 - INFO - LAC layer 6 epoch 10/10: mse=0.009077
2026-01-22 19:42:22,017 - INFO - ========= LAC Layer 7 =========
2026-01-22 19:42:48,249 - INFO - LAC layer 7 epoch 1/10: mse=0.013529
2026-01-22 19:45:55,889 - INFO - LAC layer 7 epoch 10/10: mse=0.011055
2026-01-22 19:45:56,279 - INFO - ========= LAC Layer 8 =========
2026-01-22 19:46:22,343 - INFO - LAC layer 8 epoch 1/10: mse=0.013561
2026-01-22 19:49:29,455 - INFO - LAC layer 8 epoch 10/10: mse=0.011133
2026-01-22 19:49:29,878 - INFO - ========= LAC Layer 9 =========
2026-01-22 19:49:55,882 - INFO - LAC layer 9 epoch 1/10: mse=0.015507
2026-01-22 19:53:03,064 - INFO - LAC layer 9 epoch 10/10: mse=0.012601
2026-01-22 19:53:03,612 - INFO - ========= LAC Layer 10 =========
2026-01-22 19:53:29,716 - INFO - LAC layer 10 epoch 1/10: mse=0.016256
2026-01-22 19:56:36,721 - INFO - LAC layer 10 epoch 10/10: mse=0.013342
2026-01-22 19:56:37,108 - INFO - ========= LAC Layer 11 =========
2026-01-22 19:57:03,123 - INFO - LAC layer 11 epoch 1/10: mse=0.016232
2026-01-22 20:00:10,056 - INFO - LAC layer 11 epoch 10/10: mse=0.013260
2026-01-22 20:00:10,602 - INFO - ========= LAC Layer 12 =========
2026-01-22 20:00:36,793 - INFO - LAC layer 12 epoch 1/10: mse=0.019194
2026-01-22 20:03:44,084 - INFO - LAC layer 12 epoch 10/10: mse=0.015797
2026-01-22 20:03:44,515 - INFO - ========= LAC Layer 13 =========
2026-01-22 20:04:10,533 - INFO - LAC layer 13 epoch 1/10: mse=0.020585
2026-01-22 20:07:17,567 - INFO - LAC layer 13 epoch 10/10: mse=0.016704
2026-01-22 20:07:18,083 - INFO - ========= LAC Layer 14 =========
2026-01-22 20:07:44,228 - INFO - LAC layer 14 epoch 1/10: mse=0.025036
2026-01-22 20:10:51,254 - INFO - LAC layer 14 epoch 10/10: mse=0.020546
2026-01-22 20:10:51,710 - INFO - ========= LAC Layer 15 =========
2026-01-22 20:11:17,747 - INFO - LAC layer 15 epoch 1/10: mse=0.027773
2026-01-22 20:14:24,754 - INFO - LAC layer 15 epoch 10/10: mse=0.022454
2026-01-22 20:14:25,317 - INFO - ========= LAC Layer 16 =========
2026-01-22 20:14:51,393 - INFO - LAC layer 16 epoch 1/10: mse=0.028235
2026-01-22 20:17:58,636 - INFO - LAC layer 16 epoch 10/10: mse=0.022811
2026-01-22 20:17:59,165 - INFO - ========= LAC Layer 17 =========
2026-01-22 20:18:25,319 - INFO - LAC layer 17 epoch 1/10: mse=0.030395
2026-01-22 20:21:32,438 - INFO - LAC layer 17 epoch 10/10: mse=0.024544
2026-01-22 20:21:32,979 - INFO - ========= LAC Layer 18 =========
2026-01-22 20:21:59,045 - INFO - LAC layer 18 epoch 1/10: mse=0.029982
2026-01-22 20:25:06,584 - INFO - LAC layer 18 epoch 10/10: mse=0.024060
2026-01-22 20:25:07,060 - INFO - ========= LAC Layer 19 =========
2026-01-22 20:25:33,119 - INFO - LAC layer 19 epoch 1/10: mse=0.031523
2026-01-22 20:28:40,566 - INFO - LAC layer 19 epoch 10/10: mse=0.024987
2026-01-22 20:28:40,968 - INFO - ========= LAC Layer 20 =========
2026-01-22 20:29:07,288 - INFO - LAC layer 20 epoch 1/10: mse=0.033710
2026-01-22 20:32:15,082 - INFO - LAC layer 20 epoch 10/10: mse=0.026883
2026-01-22 20:32:15,480 - INFO - ========= LAC Layer 21 =========
2026-01-22 20:32:41,571 - INFO - LAC layer 21 epoch 1/10: mse=0.041330
2026-01-22 20:35:49,453 - INFO - LAC layer 21 epoch 10/10: mse=0.033155
2026-01-22 20:35:49,845 - INFO - ========= LAC Layer 22 =========
2026-01-22 20:36:15,978 - INFO - LAC layer 22 epoch 1/10: mse=0.042124
2026-01-22 20:39:23,901 - INFO - LAC layer 22 epoch 10/10: mse=0.033697
2026-01-22 20:39:24,295 - INFO - ========= LAC Layer 23 =========
2026-01-22 20:39:50,321 - INFO - LAC layer 23 epoch 1/10: mse=0.046974
2026-01-22 20:42:58,121 - INFO - LAC layer 23 epoch 10/10: mse=0.037754
2026-01-22 20:42:58,549 - INFO - ========= LAC Layer 24 =========
2026-01-22 20:43:24,590 - INFO - LAC layer 24 epoch 1/10: mse=0.055368
2026-01-22 20:46:32,755 - INFO - LAC layer 24 epoch 10/10: mse=0.044817
2026-01-22 20:46:33,170 - INFO - ========= LAC Layer 25 =========
2026-01-22 20:46:59,328 - INFO - LAC layer 25 epoch 1/10: mse=0.064182
2026-01-22 20:50:07,381 - INFO - LAC layer 25 epoch 10/10: mse=0.052290
2026-01-22 20:50:07,870 - INFO - ========= LAC Layer 26 =========
2026-01-22 20:50:34,010 - INFO - LAC layer 26 epoch 1/10: mse=0.071640
2026-01-22 20:53:41,872 - INFO - LAC layer 26 epoch 10/10: mse=0.058172
2026-01-22 20:53:42,424 - INFO - ========= LAC Layer 27 =========
2026-01-22 20:54:08,704 - INFO - LAC layer 27 epoch 1/10: mse=0.095367
2026-01-22 20:57:16,100 - INFO - LAC layer 27 epoch 10/10: mse=0.077816
2026-01-22 20:57:16,703 - INFO - ========= LAC Layer 28 =========
2026-01-22 20:57:42,928 - INFO - LAC layer 28 epoch 1/10: mse=0.120815
2026-01-22 21:00:50,905 - INFO - LAC layer 28 epoch 10/10: mse=0.097691
2026-01-22 21:00:51,316 - INFO - ========= LAC Layer 29 =========
2026-01-22 21:01:17,634 - INFO - LAC layer 29 epoch 1/10: mse=0.176598
2026-01-22 21:04:25,483 - INFO - LAC layer 29 epoch 10/10: mse=0.142799
2026-01-22 21:04:25,931 - INFO - ========= LAC Layer 30 =========
2026-01-22 21:04:52,006 - INFO - LAC layer 30 epoch 1/10: mse=0.316711
2026-01-22 21:07:59,646 - INFO - LAC layer 30 epoch 10/10: mse=0.251450
2026-01-22 21:08:00,191 - INFO - ========= LAC Layer 31 =========
2026-01-22 21:08:26,588 - INFO - LAC layer 31 epoch 1/10: mse=1.663269
2026-01-22 21:11:34,568 - INFO - LAC layer 31 epoch 10/10: mse=1.366516
2026-01-22 21:11:42,070 - INFO - Evaluating PPL on datasets: ['wikitext2', 'ptb', 'c4']
PPL: 7.20: 100%|███████████████████████████████████████████████████████████████████████████| 141/141 [01:11<00:00,  1.97it/s]
2026-01-22 21:13:03,860 - INFO - WIKITEXT2 PPL: 7.20
Using the latest cached version of the module from /data01/home/cheny/.cache/huggingface/modules/datasets_modules/datasets/ptb_text_only/8d1b97746fb9765d140e569ec5ddd35e20af4d37761f5e1bf357ea0b081f2c1f (last modified on Sun Jan  4 13:43:43 2026) since it couldn't be found locally at ptb_text_only, or remotely on the Hugging Face Hub.
PPL: 12.62: 100%|████████████████████████████████████████████████████████████████████████████| 46/46 [00:23<00:00,  1.97it/s]
2026-01-22 21:13:35,431 - INFO - PTB PPL: 12.62