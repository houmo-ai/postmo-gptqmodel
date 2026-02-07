import sys
import wandb
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
from datetime import datetime
import torch.nn as nn
import torch
from pathlib import Path
import argparse
import os
import transformers
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import math

from dataloader import R1Dataset

import geoopt

sys.path.append('../fake_quant')

from rotation_utils import get_orthogonal_matrix
# from utils import supported_models, supported_datasets
from model_utils import get_model

os.environ['WANDB_API_KEY'] = "XXXXXXXXXXXXXXX"
os.environ['WANDB_MODE'] = "offline"

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'


default_path = '../data/'


class R1_QR(nn.Module):
    def __init__(self, hidden_size: int, use_geoopt: bool = False):
        super(R1_QR, self).__init__()
        self.hidden_size = hidden_size
        self.use_geoopt = use_geoopt
        if use_geoopt:
            # Stiefel manifold for square matrices => orthogonal matrix
            self.matrix = geoopt.ManifoldParameter(
                torch.eye(hidden_size),
                manifold=geoopt.manifolds.Stiefel(),
            )
        else:
            self.matrix = nn.Parameter(torch.eye(hidden_size))

    def forward(self, x):
        # Keep the effective rotation orthogonal:
        # - geoopt optimizers keep ManifoldParameter on the manifold
        # - torch SGD/Adam do not, so we project via QR each forward
        if self.use_geoopt:
            self.rotate = self.matrix
        else:
            self.rotate, _ = torch.linalg.qr(self.matrix, mode='complete')
        o_x = torch.matmul(x, self.rotate)
        return o_x


def train_R1(dataset, args):
    # transformers.set_seed(args.seed)  # 初始化随机种子
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    use_geoopt = args.optim in ['RSGD', 'RAdam']
    R1 = R1_QR(hidden_size=args.hidden_size, use_geoopt=use_geoopt).to(device)
    init_r = get_orthogonal_matrix(args.hidden_size, args.init_mode, device).float()
    R1.matrix.data.copy_(init_r)

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(R1.parameters(),
                                    lr=args.lr, momentum=args.mom)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(R1.parameters(),
                                     lr=args.lr,)
    elif args.optim == 'RSGD':
        optimizer = geoopt.optim.RiemannianSGD(
            R1.parameters(), weight_decay=0, lr=args.lr, stabilize=10
        )
    elif args.optim == 'RAdam':
        optimizer = geoopt.optim.RiemannianAdam(
            R1.parameters(), weight_decay=0, lr=args.lr, stabilize=10
        )
    else:
        raise NotImplementedError

    if args.cos_lr:  # 设置余弦退火学习率调度器
        scheduler = CosineAnnealingLR(optimizer, T_max=args.ep, eta_min=0)

    if args.wandb:
        wandb.init(project="learnable_balance_rotate",
                   name=f"{args.run_time}_{args.st}",
                   config=vars(args)
                   )
        loss_table = wandb.Table(columns=['Losses'])
        for loss in args.loss_list:
            loss_table.add_data(loss)
        wandb.log({"loss list": loss_table})

        param_table = wandb.Table(columns=['Parameter', 'Value'])
        for key, value in vars(args).items():
            param_table.add_data(key, str(value))
        wandb.log({"parameter table": param_table})

    # 每个epoch训练的样本比例
    num_samples = int(len(dataset) * args.train_subset_size)
    R1.train()
    print("---> start training R1 ")
    for epoch in range(args.ep):
        loss_log = []
        # 创建 RandomSampler，每次从数据集中随机选择一部分样本
        indices = np.random.choice(len(dataset), size=num_samples, replace=False)
        # sampler = RandomSampler(indices)
        sampler = SubsetRandomSampler(indices)
        dataloader = DataLoader(dataset,        # 创建数据加载器
                                sampler=sampler,
                                batch_size=args.bsz,
                                num_workers=8,
                                prefetch_factor=3,
                                persistent_workers=True,
                                pin_memory=True)

        for batch_idx, batch_samples in enumerate(dataloader):
            batch_samples = batch_samples.to(device).float().reshape(-1, args.hidden_size)
            outputs = R1(batch_samples)
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
            # Geometric-Mean Equalization loss
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
            q_max = 7.0   # 量化最大整数 (INT4=7)
            # alpha: 控制 Softmax 逼近 Max 的程度。建议 20.0 ~ 50.0。
            #        越大越接近真实 Max，但梯度越陡峭。
            alpha = getattr(args, 'qsr_alpha', 80) 
            # tau:   Sigmoid 温度系数。建议 0.05 ~ 0.1。
            #        越小越像阶跃函数(计数准)，越大梯度传得越远(利于优化)。
            tau = getattr(args, 'qsr_tau', 0.05)     
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
            
            loss = (scale_term * distortion_term).mean() * 10

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
        mean_inv_loss = torch.stack(loss_log).mean()
        log_message = f'Epoch [{epoch+1}/{args.ep}], LRC Loss: {mean_inv_loss.item():.8f}'
        if args.cos_lr:
            log_message += f', LR: {scheduler.get_last_lr()[0]:.4f}'
        print(log_message)

        # 记录到 wandb
        if args.wandb:
            log_data = {
                "inv_loss": mean_inv_loss.item()
            }
            wandb.log(log_data)

        # if args.save_model:  # 保存训练好的旋转矩阵 每一epoch
        #     save_path = f"{args.save_path}/{args.save_folder}/"
        #     Path(save_path).mkdir(parents=True, exist_ok=True)
        #     save_path = Path(save_path)
        #     save_name = f"{save_path}/{args.save_folder}_{epoch}.pt"
        #     torch.save({'R1': R1.rotate.data}, save_name)

    print("---> R1 training done ")

    return R1.rotate.data


def parser_gen():
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument('--model', type=str, 
                        help='model name.')
    parser.add_argument('--calib_dataset', type=str, default='wikitext2',
                        # choices=supported_datasets,
                        help='Dataset for Evaluation (default: wikitext2)',)
    parser.add_argument('--nsamples', type=int, default=1024, help='Number of train sample.')
    parser.add_argument('--train_subset_size', type=float, default=0.1, help='')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Number of accumulation steps (default: 1)')
    parser.add_argument('--calib_sample', type=int, default=1024,
                        help='Number of sample.')
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--ep', type=int, default=10,
                        help='Number of epochs for training (default: 500)')
    parser.add_argument('--bsz', type=int, default=64,
                        help='Batch-size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate for training (default: 1e-3)")
    parser.add_argument('--mom', type=float, default=0.9,
                        help='Momentum for training (default: 0.9)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random Seed for HuggingFace and PyTorch')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False,
                        help='Whether to use wandb for logging (default: False)')
    parser.add_argument('--optim', type=str, default='RAdam', choices=['sgd', 'adam', 'adamw', 'RSGD', 'RAdam'],
                        help='Optimization method (default: sgdg)')
    parser.add_argument('--cos_lr', action=argparse.BooleanOptionalAction, default=True,
                        help='Whether to use cosine learning rate scheduler (default: True)')
    parser.add_argument('--init_mode', type=str, default='hadamard',
                        choices=['hadamard', 'random', 'identity'],
                        help='Initialization mode (default: hadamard)')
    parser.add_argument('--save_model', action=argparse.BooleanOptionalAction, default=True,
                        help='Whether save learned r1 (default: True).')
    parser.add_argument('--lrc_alpha', type=float, default=20.0,
                        help='Alpha (temperature) for LRC soft-max on log(|o|+eps). Larger => closer to max.')
    parser.add_argument('--lrc_beta', type=float, default=20.0,
                        help='Beta (temperature) for LRC soft-min on log(|o|+eps). Larger => closer to min.')
    parser.add_argument('--lrc_eps', type=float, default=1e-3,
                        help='Epsilon for numerical stability in log(|o|+eps).')

    args = parser.parse_args()

    now = datetime.now()
    run_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    setattr(args, 'run_time', run_time)

    # {run_time}.
    save_folder = f'{args.optim}.{args.lr}.{args.mom}.{args.ep}.{args.bsz}.{args.train_subset_size}.{args.accumulation_steps}'
    save_folder = f'{save_folder}.cos' if args.cos_lr else save_folder
    setattr(args, 'save_folder', save_folder)

    data_path = f"/data02/cheny/dartquant/{args.calib_dataset}_{args.calib_sample}samples/{args.model.split('/')[-1]}/r1_train"
    setattr(args, 'data_path', data_path)
    print(f'---> data path: {data_path}')
    save_path = f"/data02/cheny/dartquant/QSR-geoopt/trained_rotation/{args.calib_dataset}_{args.nsamples}samples/{args.model.split('/')[-1]}/r1"
    setattr(args, 'save_path', save_path)

    return args, now


if __name__ == "__main__":
    args, run_time = parser_gen()

    model = get_model(args.model, args.hf_token)
    setattr(args, 'hidden_size', model.config.hidden_size)
    del model

    # 加载数据集
    dataset = R1Dataset(args.data_path, nsamples=args.nsamples)
    print(f'---> start time: {run_time}')
    r1 = train_R1(dataset, args)
    finish_time = datetime.now()
    train_time = finish_time - run_time
    print(f"---> Training time: {train_time}")

    os.makedirs(f'./logs/', exist_ok=True)
    with open(f'./logs/time_log.txt', 'a') as f:
        f.write(
            f'{args.model.split("/")[-1]} train r1 time: {train_time} {args.nsamples}sample {args.train_subset_size}% \n')

    if args.save_model:  # 保存训练好的旋转矩阵
        save_path = f"{args.save_path}/{args.save_folder}/"
        Path(save_path).mkdir(parents=True, exist_ok=True)
        save_path = Path(save_path)
        save_name = f"{save_path}/{args.save_folder}.pt"
        torch.save({'R1': r1}, save_name)

        print(f"---> R1 has been saved in: {save_path}")
