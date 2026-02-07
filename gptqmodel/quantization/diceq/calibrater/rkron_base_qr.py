import sys
import wandb
from torch.utils.data import DataLoader, SubsetRandomSampler
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

from dataloader import R1LayerDataset, R2Dataset, R4Dataset

import geoopt

sys.path.append('../fake_quant')

from rotation_utils import get_orthogonal_matrix, apply_kron_right, get_decompose_dim
import model_utils
from model_utils import get_model

os.environ['WANDB_API_KEY'] = "XXXXXXXXXXXXXXX"
os.environ['WANDB_MODE'] = "offline"

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'


def _maybe_hadamard_init(size, mode, device):
    if mode != 'hadamard':
        return get_orthogonal_matrix(size, mode, device)
    try:
        return get_orthogonal_matrix(size, mode, device)
    except Exception:
        return get_orthogonal_matrix(size, 'random', device)


def _near_identity_orthogonal(size: int, device, noise_std: float):
    noise_std = float(noise_std)
    if noise_std <= 0:
        return torch.eye(size, dtype=torch.float32, device=device)
    m = torch.eye(size, dtype=torch.float32, device=device) + noise_std * torch.randn(
        size, size, dtype=torch.float32, device=device
    )
    q, r = torch.linalg.qr(m)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


@torch.no_grad()
def _rkron_init_search(rkron, dataset, rg, args, r2=None):
    device = next(rkron.parameters()).device
    trials = int(getattr(args, 'init_search_trials', 8))
    batches = int(getattr(args, 'init_search_batches', 2))
    subset_size = float(getattr(args, 'init_search_subset_size', 0.02))
    noise_std = float(getattr(args, 'init_noise_std', 0.02))

    num_items = len(dataset)
    if num_items == 0:
        raise RuntimeError('Empty dataset; cannot run init search.')
    num_subset = max(1, min(num_items, int(round(num_items * subset_size))))
    indices = np.random.choice(num_items, size=num_subset, replace=False)
    sampler = SubsetRandomSampler(indices)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.bsz, num_workers=0)

    if r2 is not None:
        r2 = r2.to(device=device, dtype=torch.float32)

    def score_current():
        scores = []
        for j, batch_samples in enumerate(dataloader):
            if j >= batches:
                break
            batch_samples = batch_samples.to(device=device, dtype=torch.float32).reshape(-1, args.target_dim)
            if rg is not None:
                batch_samples = torch.matmul(batch_samples, rg)
            if r2 is not None:
                batch_samples = apply_r2_to_tokens(batch_samples, r2, args.head_num, args.kv_head)
            outputs = rkron(batch_samples)
            scores.append(whip_loss(outputs, args))
        if len(scores) == 0:
            return float('inf')
        return torch.stack(scores).mean().item()

    n1 = args.rkron_n1
    n2 = args.rkron_n2

    candidate_specs = [
        ('identity', 'identity'),
        ('random', 'random'),
        ('hadamard', 'hadamard'),
    ]
    while len(candidate_specs) < trials:
        candidate_specs.append(('near_identity', 'near_identity'))
        if len(candidate_specs) < trials:
            candidate_specs.append(('random', 'near_identity'))
        if len(candidate_specs) < trials:
            candidate_specs.append(('near_identity', 'random'))

    best_score = float('inf')
    best_r1 = None
    best_r2 = None

    for mode1, mode2 in candidate_specs[:trials]:
        if mode1 == 'near_identity':
            r1 = _near_identity_orthogonal(n1, device, noise_std)
        else:
            r1 = _maybe_hadamard_init(n1, mode1, device)
        if mode2 == 'near_identity':
            r2_ = _near_identity_orthogonal(n2, device, noise_std)
        else:
            r2_ = _maybe_hadamard_init(n2, mode2, device)

        rkron.matrix1.data.copy_(r1.float())
        rkron.matrix2.data.copy_(r2_.float())
        score = score_current()
        if score < best_score:
            best_score = score
            best_r1 = r1
            best_r2 = r2_

    if best_r1 is None or best_r2 is None:
        raise RuntimeError('Init search failed to produce any candidate.')
    rkron.matrix1.data.copy_(best_r1.float())
    rkron.matrix2.data.copy_(best_r2.float())
    print(f'---> init search picked score={best_score:.6f}')


class RKron_QR(nn.Module):
    def __init__(self, n1: int, n2: int, use_geoopt: bool = False):
        super(RKron_QR, self).__init__()
        self.n1 = n1
        self.n2 = n2
        self.use_geoopt = use_geoopt
        if use_geoopt:
            self.matrix1 = geoopt.ManifoldParameter(
                torch.eye(n1),
                manifold=geoopt.manifolds.Stiefel(),
            )
            self.matrix2 = geoopt.ManifoldParameter(
                torch.eye(n2),
                manifold=geoopt.manifolds.Stiefel(),
            )
        else:
            self.matrix1 = nn.Parameter(torch.eye(n1))
            self.matrix2 = nn.Parameter(torch.eye(n2))

    def forward(self, x):
        if self.use_geoopt:
            r1 = self.matrix1
            r2 = self.matrix2
        else:
            r1, _ = torch.linalg.qr(self.matrix1, mode='complete')
            r2, _ = torch.linalg.qr(self.matrix2, mode='complete')
        r1 = r1.to(dtype=x.dtype, device=x.device)
        r2 = r2.to(dtype=x.dtype, device=x.device)
        self.rotate1 = r1
        self.rotate2 = r2
        return apply_kron_right(x, r1, r2)

def whip_loss(outputs, args):
    loss = torch.sum(torch.exp((-outputs.abs())), dim=-1, keepdim=True).mean()
    return loss

def qsr_loss(outputs, args):
    q_max = 7.0
    alpha = getattr(args, 'qsr_alpha', 100)
    tau = getattr(args, 'qsr_tau', 0.05)
    eps = 1e-6

    d = outputs.shape[-1]
    smooth_abs = (outputs.pow(2) + eps**2).sqrt()
    rho = outputs.norm(p=2, dim=-1, keepdim=True) + eps
    s = smooth_abs / rho

    log_d = math.log(d)
    m_alpha = (torch.logsumexp(alpha * s, dim=-1, keepdim=True) - log_d) / alpha
    tilde_A = rho * m_alpha

    t = m_alpha / (2 * q_max)
    u_gate = torch.sigmoid((t - s) / tau)
    tilde_U = u_gate.mean(dim=-1, keepdim=True)

    scale_term = tilde_A.pow(2)
    distortion_term = (1.0 / 12.0) + 0.25 * tilde_U
    return (scale_term * distortion_term).mean() * 10


def apply_r2_to_tokens(x, r2, head_num, kv_head):
    head_dim = x.shape[-1] // head_num
    x = x.reshape(-1, head_num, head_dim)
    if r2.dim() == 2:
        r2 = r2.unsqueeze(0).expand(head_num, head_dim, head_dim)
    elif r2.dim() == 3 and r2.shape[0] != head_num:
        n_rep = head_num // kv_head
        r2 = r2[:, None, :, :].expand(kv_head, n_rep, head_dim, head_dim)
        r2 = r2.reshape(head_num, head_dim, head_dim)
    x = torch.einsum('bhd,hdf->bhf', x, r2)
    return x.reshape(-1, head_num * head_dim)


def train_rkron(dataset, rg, args, r2=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    use_geoopt = args.optim in ['RSGD', 'RAdam']
    rkron = RKron_QR(args.rkron_n1, args.rkron_n2, use_geoopt=use_geoopt).to(device)
    if args.init_mode == 'search':
        _rkron_init_search(rkron=rkron, dataset=dataset, rg=rg, args=args, r2=r2)
    else:
        init_r1 = _maybe_hadamard_init(args.rkron_n1, args.init_mode, device).float()
        init_r2 = _maybe_hadamard_init(args.rkron_n2, args.init_mode, device).float()
        rkron.matrix1.data.copy_(init_r1)
        rkron.matrix2.data.copy_(init_r2)

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(rkron.parameters(), lr=args.lr, momentum=args.mom)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(rkron.parameters(), lr=args.lr)
    elif args.optim == 'RSGD':
        optimizer = geoopt.optim.RiemannianSGD(
            rkron.parameters(), weight_decay=0, lr=args.lr, stabilize=10
        )
    elif args.optim == 'RAdam':
        optimizer = geoopt.optim.RiemannianAdam(
            rkron.parameters(), weight_decay=0, lr=args.lr, stabilize=10
        )
    else:
        raise NotImplementedError

    if args.cos_lr:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.ep, eta_min=0)

    num_samples = int(len(dataset) * args.train_subset_size)
    num_samples = max(1, num_samples)
    rkron.train()
    if r2 is not None:
        r2 = r2.to(device=device, dtype=torch.float32)

    for epoch in range(args.ep):
        loss_log = []
        indices = np.random.choice(len(dataset), size=num_samples, replace=False)
        sampler = SubsetRandomSampler(indices)
        dataloader = DataLoader(dataset,
                                sampler=sampler,
                                batch_size=args.bsz,
                                num_workers=4,
                                prefetch_factor=2,
                                persistent_workers=True,
                                pin_memory=True)

        for batch_idx, batch_samples in enumerate(dataloader):
            batch_samples = batch_samples.to(device=device, dtype=torch.float32).reshape(-1, args.target_dim)
            if rg is not None:
                batch_samples = torch.matmul(batch_samples, rg)
            if r2 is not None:
                batch_samples = apply_r2_to_tokens(batch_samples, r2, args.head_num, args.kv_head)
            outputs = rkron(batch_samples)
            loss = whip_loss(outputs, args)
            loss = loss / args.accumulation_steps
            loss.backward()

            if (batch_idx + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            loss_log.append(loss.detach())

        if args.cos_lr:
            scheduler.step()

        mean_loss = torch.stack(loss_log).mean()
        log_message = f'Epoch [{epoch+1}/{args.ep}], Loss: {mean_loss.item():.8f}'
        if args.cos_lr:
            log_message += f', LR: {scheduler.get_last_lr()[0]:.4e}'
        print(log_message)

    if use_geoopt:
        r1 = rkron.matrix1.detach()
        r2 = rkron.matrix2.detach()
    else:
        r1, _ = torch.linalg.qr(rkron.matrix1, mode='complete')
        r2, _ = torch.linalg.qr(rkron.matrix2, mode='complete')
    return r1.data, r2.data


def parser_gen():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, help='model name.')
    parser.add_argument('--calib_dataset', type=str, default='wikitext2',
                        help='Dataset for Evaluation (default: wikitext2)')
    parser.add_argument('--nsamples', type=int, default=1024, help='Number of train sample.')
    parser.add_argument('--train_subset_size', type=float, default=0.3, help='')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Number of accumulation steps (default: 1)')
    parser.add_argument('--calib_sample', type=int, default=1024,
                        help='Number of sample.')
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--ep', type=int, default=10,
                        help='Number of epochs for training (default: 10)')
    parser.add_argument('--bsz', type=int, default=64,
                        help='Batch-size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate for training (default: 1e-3)")
    parser.add_argument('--mom', type=float, default=0.9,
                        help='Momentum for training (default: 0.9)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random Seed for HuggingFace and PyTorch')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False,
                        help='Whether to use wandb for logging (default: False)')
    parser.add_argument('--optim', type=str, default='RAdam', choices=['sgd', 'adam', 'adamw', 'RSGD', 'RAdam'],
                        help='Optimization method (default: sgd)')
    parser.add_argument('--cos_lr', action=argparse.BooleanOptionalAction, default=True,
                        help='Whether to use cosine learning rate scheduler (default: True)')
    parser.add_argument('--init_mode', type=str, default='identity',
                        choices=['hadamard', 'random', 'identity', 'search'],
                        help='Initialization mode (default: hadamard)')
    parser.add_argument('--init_search_trials', type=int, default=8,
                        help='Number of candidate inits to try when init_mode=search.')
    parser.add_argument('--init_search_batches', type=int, default=2,
                        help='Number of batches used to score each init candidate.')
    parser.add_argument('--init_search_subset_size', type=float, default=0.02,
                        help='Fraction of dataset items used for init search scoring.')
    parser.add_argument('--init_noise_std', type=float, default=0.02,
                        help='Noise std for near-identity candidates during init search.')
    parser.add_argument('--save_model', action=argparse.BooleanOptionalAction, default=True,
                        help='Whether save learned rkron (default: True).')
    parser.add_argument('--save_file_name', type=str, default='',
                        help="Saving folder name for trained model (default: '')")
    parser.add_argument('--r1_path', type=str, default=None,
                        help='Path to global rotation R1 (R_g).')
    parser.add_argument('--r2_path', type=str, default=None,
                        help='Path to R2 rotation (used for o_proj).')
    parser.add_argument('--target', type=str, default='r1',
                        choices=['r1', 'o_proj', 'down_proj'],
                        help='Stage2 target module.')
    parser.add_argument('--rkron_n1', type=int, default=None,
                        help='First Kronecker factor size (default: get_decompose_dim).')
    parser.add_argument('--rkron_n2', type=int, default=None,
                        help='Second Kronecker factor size (default: get_decompose_dim).')

    args = parser.parse_args()

    now = datetime.now()
    run_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    setattr(args, 'run_time', run_time)

    args.save_file_name += f'{args.optim}.{args.lr}.{args.mom}.{args.ep}.{args.bsz}.{args.train_subset_size}.{args.accumulation_steps}'
    args.save_file_name += '.cos' if args.cos_lr else ''

    if args.target == 'r1':
        data_dir = 'r1_train'
    elif args.target == 'o_proj':
        data_dir = 'r2_train'
    else:
        data_dir = 'r4_train'
    data_path = f"/data02/cheny/dartquant/{args.calib_dataset}_{args.calib_sample}samples/{args.model.split('/')[-1]}/{data_dir}"
    setattr(args, 'data_path', data_path)
    print(f'---> data path: {data_path}')

    save_path = f"/data02/cheny/dartquant/whip-geoopt/trained_rotation/{args.calib_dataset}_{args.nsamples}samples/{args.model.split('/')[-1]}/rkron_{args.target}/{args.save_file_name}"
    setattr(args, 'save_path', save_path)
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    return args, now


def main():
    args, run_time = parser_gen()
    print(f'---> start time: {run_time}')

    transformers.set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(args.model, args.hf_token)
    setattr(args, 'hidden_size', model.config.hidden_size)
    setattr(args, 'head_num', model.config.num_attention_heads)
    setattr(args, 'kv_head', model.config.num_key_value_heads)
    setattr(args, 'hidden_layers', model.config.num_hidden_layers)
    setattr(args, 'intermediate_size', model_utils.get_mlp_bottleneck_size(model))
    del model

    if args.target == 'down_proj':
        args.target_dim = args.intermediate_size
    else:
        args.target_dim = args.hidden_size

    if args.rkron_n1 is None and args.rkron_n2 is None:
        args.rkron_n1, args.rkron_n2 = get_decompose_dim(args.target_dim)
    elif args.rkron_n1 is None:
        if args.target_dim % args.rkron_n2 != 0:
            raise ValueError(f'Invalid rkron_n2: {args.rkron_n2} does not divide {args.target_dim}')
        args.rkron_n1 = args.target_dim // args.rkron_n2
    elif args.rkron_n2 is None:
        if args.target_dim % args.rkron_n1 != 0:
            raise ValueError(f'Invalid rkron_n1: {args.rkron_n1} does not divide {args.target_dim}')
        args.rkron_n2 = args.target_dim // args.rkron_n1
    if args.rkron_n1 * args.rkron_n2 != args.target_dim:
        raise ValueError(f'Invalid rkron shape: {args.rkron_n1}*{args.rkron_n2} != {args.target_dim}')

    rg = None
    if args.r1_path is not None:
        rg = torch.load(args.r1_path, map_location=device)['R1'].to(dtype=torch.float32)

    if args.target == 'r1':
        save_dict = {'rkron_shape': (args.rkron_n1, args.rkron_n2)}
        module_list = None
    elif args.target == 'o_proj':
        save_dict = {'rkron_o_shape': (args.rkron_n1, args.rkron_n2)}
        module_list = None
    else:
        save_dict = {'rkron_down_shape': (args.rkron_n1, args.rkron_n2)}
        module_list = None

    r2_state = None
    if args.target == 'o_proj' and args.r2_path is not None:
        r2_state = torch.load(args.r2_path, map_location='cpu')

    for layer_id in range(args.hidden_layers):
        if args.target == 'r1':
            dataset_q = R1LayerDataset(args.data_path, layer_id, args.nsamples, modules=['self_attn_q_proj'])
            dataset_up = R1LayerDataset(args.data_path, layer_id, args.nsamples, modules=['mlp_up_proj'])
            if len(dataset_q) == 0 or len(dataset_up) == 0:
                raise RuntimeError(f'No data found for layer {layer_id} in {args.data_path}')

            r1, r2 = train_rkron(dataset=dataset_q, rg=rg, args=args)
            save_dict[f"model.layers.{layer_id}.rkron_attn1"] = r1.detach().cpu()
            save_dict[f"model.layers.{layer_id}.rkron_attn2"] = r2.detach().cpu()

            r1, r2 = train_rkron(dataset=dataset_up, rg=rg, args=args)
            save_dict[f"model.layers.{layer_id}.rkron_mlp1"] = r1.detach().cpu()
            save_dict[f"model.layers.{layer_id}.rkron_mlp2"] = r2.detach().cpu()
        elif args.target == 'o_proj':
            data_path = os.path.join(args.data_path, f'layer_{layer_id}_self_attn_o_proj.pt')
            if not os.path.exists(data_path):
                raise RuntimeError(f'No data found for layer {layer_id} in {data_path}')
            dataset = R2Dataset(data_path, args.nsamples, dev='cpu')
            r2_layer = None
            if r2_state is not None:
                r2_layer = r2_state.get(f"model.layers.{layer_id}.self_attn.R2", None)
                if r2_layer is not None:
                    r2_layer = r2_layer.to(dtype=torch.float32)
            r1, r2 = train_rkron(dataset=dataset, rg=rg, args=args, r2=r2_layer)
            save_dict[f"model.layers.{layer_id}.rkron_o1"] = r1.detach().cpu()
            save_dict[f"model.layers.{layer_id}.rkron_o2"] = r2.detach().cpu()
        else:
            data_path = os.path.join(args.data_path, f'layer_{layer_id}_mlp_down_proj.pt')
            if not os.path.exists(data_path):
                raise RuntimeError(f'No data found for layer {layer_id} in {data_path}')
            dataset = R4Dataset(data_path, args.nsamples, dev='cpu')
            r1, r2 = train_rkron(dataset=dataset, rg=rg, args=args)
            save_dict[f"model.layers.{layer_id}.rkron_down1"] = r1.detach().cpu()
            save_dict[f"model.layers.{layer_id}.rkron_down2"] = r2.detach().cpu()
        print(f"---> layer {layer_id} rkron trained.")

    if args.save_model:
        save_name = os.path.join(args.save_path, f"{args.save_file_name}.pt")
        torch.save(save_dict, save_name)

    finish_time = datetime.now()
    train_time = finish_time - run_time
    print(f"---> Training time: {train_time}")
    if args.save_model:
        print(f'---> Save in: {save_name}')


if __name__ == "__main__":
    main()
