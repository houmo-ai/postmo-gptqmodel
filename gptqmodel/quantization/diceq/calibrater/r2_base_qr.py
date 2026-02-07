import sys

from zmq import device
import wandb
from torch.utils.data import DataLoader
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import torch
from pathlib import Path
import argparse
import os
import transformers
import torch.nn.functional as F
import numpy as np
import math

from dataloader import R2Dataset

import geoopt

sys.path.append('../fake_quant')

from rotation_utils import get_orthogonal_matrix
# from utils import supported_models, supported_datasets
from model_utils import get_model


os.environ['WANDB_API_KEY'] = "XXXXXXXXXXXXXXXXX"
os.environ['WANDB_MODE'] = "offline"

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'


default_path = '../data/'


class R2_Per_Head(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 head_num: int,
                 kv_head: int,
                 use_geoopt: bool = False):
        super(R2_Per_Head, self).__init__()
        assert hidden_size % head_num == 0, "hidden_size must be divisible by head_num"
        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_dim = hidden_size // head_num
        self.kv_head = kv_head
        self.use_geoopt = use_geoopt

        if use_geoopt:
            # Stiefel manifold for square matrices => orthogonal matrix (batched over kv_head)
            self.matrix = geoopt.ManifoldParameter(
                torch.eye(self.head_dim).repeat(self.kv_head, 1, 1),
                manifold=geoopt.manifolds.Stiefel(),
            )
        else:
            self.matrix = nn.Parameter(torch.eye(self.head_dim).repeat(self.kv_head, 1, 1))

    def forward(self, x):
        x_shape = x.shape   # [batch_size, seqlen, hidden_size]
        x = x.reshape(-1, x_shape[-1])
        x = x.reshape(-1, self.head_num, self.head_dim)  # 分头分块
        x = x.transpose(0, 1)
        # Keep the effective rotation orthogonal:
        # - geoopt optimizers keep ManifoldParameter on the manifold
        # - torch SGD/Adam do not, so we project via QR each forward
        if self.use_geoopt:
            self.rotate = self.matrix
        else:
            self.rotate, _ = torch.linalg.qr(self.matrix)
        rotate_exp = self.rotate[:, None, :, :].expand(
            self.kv_head, self.head_num // self.kv_head,
            self.head_dim, self.head_dim)
        rotate_exp = rotate_exp.reshape(self.head_num, self.head_dim,
                                        self.head_dim)
        r_x = torch.matmul(x, rotate_exp)
        r_x = r_x.transpose(0, 1)
        r_x = r_x.reshape(x_shape)

        return r_x


def get_mutli_head_init(hidden_size, head_num, kv_head, mode, device):
    org = get_orthogonal_matrix(hidden_size // head_num, mode, device)
    return org.unsqueeze(0).repeat(kv_head, 1, 1)


def train_r2(dataloader,
             layer_id,
             device,
             args,):

    use_geoopt = args.optim in ['RSGD', 'RAdam']
    R2 = R2_Per_Head(hidden_size=args.hidden_size,
                     head_num=args.head_num,
                     kv_head=args.kv_head,
                     use_geoopt=use_geoopt).to(device)
    init_r = get_mutli_head_init(args.hidden_size,
                                 args.head_num,
                                 args.kv_head,
                                 'hadamard',
                                 device).float()
    R2.matrix.data.copy_(init_r)

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(R2.parameters(),
                                    lr=args.lr, momentum=args.mom)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(R2.parameters(),
                                     lr=args.lr)
    elif args.optim == 'RSGD':
        optimizer = geoopt.optim.RiemannianSGD(
            R2.parameters(), weight_decay=0, lr=args.lr, stabilize=10
        )
    elif args.optim == 'RAdam':
        optimizer = geoopt.optim.RiemannianAdam(
            R2.parameters(), weight_decay=0, lr=args.lr, stabilize=10
        )
    else:
        raise NotImplementedError

    if args.cos_lr:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.ep, eta_min=0)

    if args.wandb:  # 初始化 wandb
        wandb_ = wandb.init(project="learnable balance rotate 2 search params layer by layer",
                            name=f"{args.run_target_time} layer ID: [{layer_id}] lr: [{args.lr}]",
                            config=vars(args)
                            )
        loss_table = wandb.Table(columns=['Losses'])
        for loss in args.loss_list:
            loss_table.add_data(loss)
        wandb_.log({"loss list": loss_table})

        param_table = wandb.Table(columns=['Parameter', 'Value'])
        for key, value in vars(args).items():
            param_table.add_data(key, str(value))
        wandb_.log({"parameter table": param_table})

    R2.train()

    print(f"---> start training R2 of layer {layer_id} ")
    for epoch in range(args.ep):
        loss_log = []

        for batch_idx, batch_samples in enumerate(dataloader):
            batch_samples = batch_samples.to(device).float()
            outputs = R2(batch_samples)
            # loss = torch.sum(torch.exp((-outputs.abs())),
            #                  dim=-1, keepdim=True).mean() / args.accumulation_steps
            # LRC (Log-Range Contraction) loss:
            # u_i = log(|o_i| + eps)
            # L = (1/a) log( (1/n) sum_i exp(a u_i) ) + (1/b) log( (1/n) sum_i exp(-b u_i) )
            # u = torch.log(outputs.abs() + args.lrc_eps)
            # log_n = math.log(u.shape[-1])
            # term_max = (torch.logsumexp(args.lrc_alpha * u, dim=-1) - log_n) / args.lrc_alpha
            # term_min = (torch.logsumexp(-args.lrc_beta * u, dim=-1) - log_n) / args.lrc_beta
            # loss = (term_max + term_min).mean()
            # loss = loss / args.accumulation_steps
            # loss.backward()

            # args.gme_eps = 1e-3
            # rms = outputs.pow(2).mean(dim=-1, keepdim=True).sqrt()  # per-token RMS
            # eps = args.gme_eps * rms.detach()                       # 相对eps（detach很重要）
            # u = torch.log(outputs.abs() + eps)
            # loss = (-u.mean(dim=-1)).mean()
            # loss = loss / args.accumulation_steps
            # loss.backward()

            # # ---- UE-KL (Uniform-Energy KL) loss ----
            # args.uekl_eps = 1e-3   # 推荐起步：1e-3~1e-4，越大越平滑

            # n = outputs.shape[-1]

            # # per-token RMS (用于相对 eps，和你现有风格一致)
            # rms = outputs.pow(2).mean(dim=-1, keepdim=True).sqrt()          # [B,1]
            # eps2 = (args.uekl_eps * rms.detach()).pow(2)                    # [B,1]  因为下面用的是 o^2

            # e = outputs.pow(2)                                              # [B,n]
            # den = e.sum(dim=-1, keepdim=True) + n * eps2                    # [B,1]
            # p = (e + eps2) / den                                            # [B,n]  energy distribution

            # # KL(p || uniform) = sum p * (log p - log(1/n)) = sum p*(log p + log n)
            # log_n = math.log(n)
            # loss = (p * (torch.log(p) + log_n)).sum(dim=-1).mean()

            # =========================================================
            # Q-SR Loss (Quantization-Aware Soft-Range Loss)
            # 理论来源：MSE Upper Bound Minimization
            # 机制：
            # 1. 外压 (Outer Pressure): 最小化 Scale (tilde_A)，压制离群值。
            # 2. 内推 (Inner Push): 最小化下溢比例 (tilde_U)，将死区的数值推回有效区。
            # =========================================================
            
            # --- 1. 超参数设定 (建议放入 args) ---
            q_max = 7.0   # 量化最大整数 (INT8=127)
            # alpha: 控制 Softmax 逼近 Max 的程度。建议 20.0 ~ 50.0。
            #        越大越接近真实 Max，但梯度越陡峭。
            alpha = getattr(args, 'qsr_alpha', 100) 
            # tau:   Sigmoid 温度系数。建议 0.05 ~ 0.1。
            #        越小越像阶跃函数(计数准)，越大梯度传得越远(利于优化)。
            tau = getattr(args, 'qsr_tau', 0.1)     
            eps = 1e-6      # 数值稳定项

            # --- 2. 预计算：平滑归一化幅度 (Normalized Smoothed Magnitude) ---
            d = outputs.shape[-1]
            # 平滑绝对值 sqrt(x^2 + e^2)，保证在 0 点处可微
            smooth_abs = (outputs.pow(2) + eps**2).sqrt() 
            # L2 范数
            rho = outputs.norm(p=2, dim=-1, keepdim=True) + eps
            # 归一化幅度 s \in (0, 1]
            s = smooth_abs / rho

            # --- 3. 构造 Scale 代理 (Soft-Max Proxy) ---
            # m_alpha = (1/alpha) * log( mean( exp(alpha * s) ) )
            # 使用 LogSumExp 技巧防止溢出
            log_d = math.log(d)
            m_alpha = (torch.logsumexp(alpha * s, dim=-1, keepdim=True) - log_d) / alpha
            
            # 恢复到真实幅度的 Scale 估计: tilde_A = rho * m_alpha
            tilde_A = rho * m_alpha

            # --- 4. 构造下溢比例代理 (Soft Underflow Ratio) ---
            # 动态阈值 t: 随着 Scale (tilde_A) 变小，阈值也会自动变小
            # t = tilde_A / (2 * q_max * rho) = m_alpha / (2 * q_max)
            t = m_alpha / (2 * q_max)
            
            # Sigmoid 门控: u_i = sigmoid( (t - s_i) / tau )
            # 当 s_i < t (下溢) 时，gate -> 1；当 s_i > t (正常) 时，gate -> 0
            u_gate = torch.sigmoid((t - s) / tau)
            
            # 计算软下溢比例 (Near-Zero Ratio)
            tilde_U = u_gate.mean(dim=-1, keepdim=True)

            # --- 5. 最终 Loss 计算 ---
            # Loss ~ Scale^2 * (Uniform_Noise + Dead_Zone_Error)
            # 系数 1/12 是均匀分布舍入噪声，1/4 是死区最大误差
            
            # scale_term = (tilde_A / q_max).pow(2)
            scale_term = tilde_A.pow(2)
            distortion_term = (1.0/12.0) + 0.25 * tilde_U
            
            loss = (scale_term * distortion_term).mean() * 100

            loss = loss / args.accumulation_steps
            loss.backward()

            # 如果达到指定的累计步数，更新梯度并清零
            if (batch_idx + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_log.append(loss.detach())

        if args.cos_lr:
            scheduler.step()

        # 计算平均损失 并打印日志
        mean_loss = torch.stack(loss_log).mean()
        log_message = f'Epoch [{epoch+1}/{args.ep}], Loss: {mean_loss.item():.8f}, '  # quant: {quant.item():.4e}
        if args.cos_lr:
            log_message += f', LR: {scheduler.get_last_lr()[0]:.4e}'
        print(log_message)

        if args.wandb:  # 记录到 wandb
            log_data = {
                "loss": mean_loss.item() if mean_loss else None,
            }
            wandb_.log(log_data)

    print(f"---> R2 of layer {layer_id} training done ")
    if args.wandb:
        wandb_.finish()

    return R2.rotate.data


def parser_gen():
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument('--model', type=str, 
                        help='model name.')
    parser.add_argument('--calib_dataset', type=str, default='wikitext2',
                        # choices=supported_datasets,
                        help='Dataset for Evaluation (default: wikitext2)',)
    parser.add_argument('--nsamples', type=int, default=1024, help='Number of train sample.')
    parser.add_argument('--calib_sample', type=int, default=1024,
                        help='Number of sample.')
    parser.add_argument('--hf_token', type=str, default=None)

    parser.add_argument('--save_model', action=argparse.BooleanOptionalAction, default=True,
                        help='Whether save learned r1 (default: True).')
    parser.add_argument('--save_file_name', type=str, default='',
                        help="Saving folder name for trained model (default: '')")

    parser.add_argument('--ep', type=int, default=5,
                        help='Number of epochs for training (default: 500)')
    parser.add_argument('--bsz', type=int, default=128,
                        help='Batch-size for training (default: 128)')
    parser.add_argument('--accumulation_steps', type=int, default=2,
                        help='Number of accumulation steps (default: 2)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate for training (default: 2.)")
    parser.add_argument('--mom', type=float, default=0.9,
                        help='Momentum for training (default: 0.9)')
    parser.add_argument('--optim', type=str, default='RAdam', choices=['sgd', 'adam', 'adamw', 'RSGD', 'RAdam'],
                        help='Optimization method (default: sgd)')
    parser.add_argument('--cos_lr', action=argparse.BooleanOptionalAction, default=True,
                        help='Whether to use cosine learning rate scheduler (default: True)')

    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False,
                        help='Whether to use wandb for logging (default: False)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random Seed for HuggingFace and PyTorch')
    parser.add_argument('--lrc_alpha', type=float, default=3.0,
                        help='Alpha (temperature) for LRC soft-max on log(|o|+eps). Larger => closer to max.')
    parser.add_argument('--lrc_beta', type=float, default=3.0,
                        help='Beta (temperature) for LRC soft-min on log(|o|+eps). Larger => closer to min.')
    parser.add_argument('--lrc_eps', type=float, default=1e-3,
                        help='Epsilon for numerical stability in log(|o|+eps).')

    args = parser.parse_args()

    now = datetime.now()
    run_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    setattr(args, 'run_target_time', run_time)
    # args.save_file_name += f'{run_target_time}.'
    args.save_file_name += f'{args.optim}.{args.lr}.{args.mom}.{args.ep}.{args.bsz}.{args.accumulation_steps}'
    args.save_file_name += '.cos' if args.cos_lr else ''

    data_path = f"/data02/cheny/dartquant/{args.calib_dataset}_{args.calib_sample}samples/{args.model.split('/')[-1]}/r2_train"
    setattr(args, 'data_path', data_path)
    print(f'---> data path: {data_path}')

    save_path = f"/data02/cheny/dartquant/QSR-geoopt/trained_rotation/{args.calib_dataset}_{args.nsamples}samples/{args.model.split('/')[-1]}/r2/{args.save_file_name}"
    setattr(args, 'save_path', save_path)
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    return args, now


def mian():
    args, run_time = parser_gen()
    print(f'---> start time: {run_time}')

    transformers.set_seed(args.seed)  # 初始化随机种子
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda')

    model = get_model(args.model, args.hf_token)
    setattr(args, 'hidden_size', model.config.hidden_size)
    setattr(args, 'kv_head', model.config.num_key_value_heads)
    setattr(args, 'head_num', model.config.num_attention_heads)
    setattr(args, 'hidden_layers', model.config.num_hidden_layers)
    del model

    save_dict = {}

    for layer_id in range(args.hidden_layers):
        # 加载数据集
        data_dire = os.path.join(args.data_path,
                                 f'layer_{layer_id}_self_attn_o_proj.pt')
        r2_dataset = R2Dataset(data_dire, args.nsamples, 'cuda')
        # 创建数据加载器
        dataloader = DataLoader(r2_dataset,
                                batch_size=args.bsz,
                                shuffle=True,
                                # pin_memory=True,
                                # num_workers=4
                                )

        r2 = train_r2(dataloader=dataloader,
                      layer_id=layer_id,
                      device=device,
                      args=args)
        save_dict[f"model.layers.{layer_id}.self_attn.R2"] = r2.detach()  # .cpu()
        print(f"---> layer {layer_id} 's r2 has been trained.")
        print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

    if args.save_model:  # 保存模型
        save_name = os.path.join(args.save_path, f"{args.save_file_name}.pt")
        torch.save(save_dict, save_name)

    finish_time = datetime.now()
    train_time = finish_time - run_time
    print(f"---> Training time: {train_time}")
    print(f'---> Save in: {save_name}')

    os.makedirs(f'./logs/', exist_ok=True)
    with open(f'./logs/r2_time_log.txt', 'a') as f:
        f.write(f'{args.model.split("/")[-1]} train r2 time: {train_time} {args.nsamples}sample \n')


if __name__ == "__main__":
    mian()
