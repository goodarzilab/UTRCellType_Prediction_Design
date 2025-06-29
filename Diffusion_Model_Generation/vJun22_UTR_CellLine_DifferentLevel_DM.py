'''
python vJun22_UTR_CellLine_DifferentLevel_DM.py --file_name /home/yanyichu/1_UTR_Cell_Type/UTR_celltype_github/Dffusion_Model_Generation/Data/vJun26_5UTR_4or5Level_CellType.csv --category_col combined_5_category_with_cell_type_code --prefix vJun26_5UTR_DM --cell_type ''

python vJun22_UTR_CellLine_DifferentLevel_DM.py --file_name /home/yanyichu/1_UTR_Cell_Type/UTR_celltype_github/Dffusion_Model_Generation/Data/vJun26_5UTR_4or5Level_CellType.csv --category_col combined_5_category --prefix vJun26_5UTR_DM --cell_type K562
'''
import argparse
import copy
import gc
import glob
import math
import os
import random
from inspect import isfunction
from typing import List, Union

import matplotlib
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
import statsmodels.api as sm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as save_image
from einops import rearrange
from functools import partial
from IPython.core.debugger import set_trace
from IPython.display import display, Image
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from PIL import Image
from scipy.optimize import fsolve, minimize
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr, kl_div
from scipy.misc import derivative
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from torch import einsum
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.nn.modules.activation import ReLU
from torchmetrics.functional import kl_divergence
from tqdm import tqdm, tqdm_notebook
import itertools
import RNA
from polyleven import levenshtein
from scipy.stats import ks_2samp,kstest,ttest_ind, mannwhitneyu, norm
from cliffs_delta import cliffs_delta
import logomaker
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
warnings.filterwarnings('ignore')

# Set up any necessary specific library configurations
matplotlib.use('Agg')  # For matplotlib to work in headless mode
sns.set(style="whitegrid")  # Setting the seaborn style

# Argument parser
parser = argparse.ArgumentParser(description="Training configuration for UNET with time warping.")
parser.add_argument('--device', type=str, default='0', help='GPU device')
parser.add_argument('--file_name', type=str,
                    default='./Data/vJun25_5UTR_4or5LevelM.csv', help='input_file')
parser.add_argument('--cell_type', type=str, default='', help='Cell type to train on')
parser.add_argument('--category_col', type=str, default='combined_4_category', help='Cell type to train on')
parser.add_argument('--prefix', type=str, default='vJun25_5UTR_DM', help='Prefix for the run')

parser.add_argument('--epochs', type=int, default=1608, help='Number of epochs to train for')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
parser.add_argument('--image_size', type=int, default=50, help='UTR sequence length')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')

parser.add_argument('--timesteps', type=int, default=100, help='Timesteps for diffusion')
parser.add_argument('--beta_scheduler', type=str, default='linear', help='Beta scheduler type')
parser.add_argument('--global_seed', type=int, default=42, help='Global seed for randomness')
parser.add_argument('--ema_beta', type=float, default=0.995, help='EMA beta')
parser.add_argument('--n_steps', type=int, default=10, help='Number of steps before time warping')
parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples for evaluation')
parser.add_argument('--save_and_sample_every', type=int, default=1, help='Epoch interval for saving')
parser.add_argument('--epochs_loss_show', type=int, default=5, help='Epoch interval to show loss')
parser.add_argument('--time_warping', type=bool, default=True, help='Use time warping')
parser.add_argument('--gen_num', type=int, default=5000, help='Number of sequences to generate')

args = parser.parse_args()
print(args)

# Constants
NUCLEOTIDES = ['A', 'C', 'T', 'G']
CHANNELS = 1
RESNET_BLOCK_GROUPS = 4

IMAGE_SIZE = args.image_size
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
GLOBAL_SEED = args.global_seed
N_STEPS = args.n_steps
TIMESTEPS = args.timesteps
N_SAMPLES = args.n_samples
SAVE_AND_SAMPLE_EVERY = args.save_and_sample_every
TIME_WARPING = args.time_warping
EPOCHS_LOSS_SHOW = args.epochs_loss_show

device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

prefix = f"{args.prefix}_{args.cell_type}__{args.category_col}__Epoch{EPOCHS}_L{IMAGE_SIZE}_Batch{BATCH_SIZE}_TimeSteps{TIMESTEPS}_{args.beta_scheduler}BETA_lr{str(LEARNING_RATE)}"
print(f"Experiment prefix: {prefix}")

def seed_everything(seed=GLOBAL_SEED):
    """Seed everything for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

def calculate_class_weight(labels):
    """Calculate class weights based on frequency."""
    class_counts = torch.bincount(labels)
    if class_counts.numel() == 1:
        return torch.tensor([1.0])
    
    class_counts = class_counts.float()
    total_samples = labels.numel()
    class_freq = class_counts / total_samples
    class_weight = 1.0 / class_freq
    class_weight = class_weight * class_weight.numel() / class_weight.sum()
    
    return class_weight

# Load and process data
df = pd.read_csv(args.file_name, index_col=0)
if args.cell_type: df = df[df.cell_type == args.cell_type]

# Get unique categories and calculate class weights
category = list(df[args.category_col].unique())
cell_type_codes = df['cell_type_code']
TOTAL_CLASS_NUMBER = len(category)
TOTAL_CELL_LINE_NUMBER = len(cell_type_codes)

# Create train/val split
df_train = df[df['fold'] == 'train']
df_val = df[df['fold'] == 'val']

# Calculate class weights based on training data
class_weight = calculate_class_weight(torch.LongTensor(df_train[args.category_col].values))
print(f'Number of classes: {TOTAL_CLASS_NUMBER}, Class weights: {class_weight}')

train_count = len(df_train)
val_count = len(df_val)
print(f"Training samples: {train_count} ({train_count/len(df)*100:.2f}%)")
print(f"Validation samples: {val_count} ({val_count/len(df)*100:.2f}%)")

class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

class BetaScheduler:
    """Beta scheduler for diffusion process."""
    def __init__(self, timesteps, scheduler_type, **kwargs):
        self.timesteps = timesteps
        self.scheduler_type = scheduler_type
        self.kwargs = kwargs

    def cosine_beta_schedule(self):
        s = self.kwargs.get('s', 0.008)
        steps = self.timesteps + 1
        x = torch.linspace(0, self.timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def linear_beta_schedule(self):
        beta_end = self.kwargs.get('beta_end', 0.02)
        beta_start = 0.0001
        return torch.linspace(beta_start, beta_end, self.timesteps)

    def quadratic_beta_schedule(self):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start**0.5, beta_end**0.5, self.timesteps) ** 2

    def sigmoid_beta_schedule(self):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, self.timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

    def get_betas(self):
        if self.scheduler_type == 'cosine':
            return self.cosine_beta_schedule()
        elif self.scheduler_type == 'linear':
            return self.linear_beta_schedule()
        elif self.scheduler_type == 'quadratic':
            return self.quadratic_beta_schedule()
        elif self.scheduler_type == 'sigmoid':
            return self.sigmoid_beta_schedule()
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

# Initialize beta scheduler
beta_scheduler = BetaScheduler(timesteps=TIMESTEPS, scheduler_type=args.beta_scheduler, beta_end=0.02)
betas = beta_scheduler.get_betas()

# Define diffusion parameters
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

def extract(a, t, x_shape):
    """Extract values from tensor a at indices t."""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

@torch.no_grad()
def p_sample(model, x, t, t_index):
    """Sample from p(x_{t-1} | x_t)."""
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    model_output = model(x, time=t, classes=None)
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t)
    
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_guided(model, x, classes, t, t_index, context_mask, cond_weight=0.0):
    """Classifier-free guidance sampling."""
    batch_size = x.shape[0]
    t_double = t.repeat(2)
    x_double = x.repeat(2, 1, 1, 1)
    betas_t = extract(betas, t_double, x_double.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t_double, x_double.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t_double, x_double.shape)

    classes_masked = classes * context_mask
    classes_masked = classes_masked.type(torch.long)
    
    preds = model(x_double, time=t_double, classes=classes_masked)
    eps1 = (1 + cond_weight) * preds[:batch_size]
    eps2 = cond_weight * preds[batch_size:]
    x_t = eps1 - eps2

    model_mean = sqrt_recip_alphas_t[:batch_size] * (
        x - betas_t[:batch_size] * x_t / sqrt_one_minus_alphas_cumprod_t[:batch_size]
    )
    
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, classes, shape, cond_weight):
    """Sample from the model."""
    device = next(model.parameters()).device
    b = shape[0]
    img = torch.randn(shape, device=device)
    imgs = []

    if classes is not None:
        n_sample = classes.shape[0]
        context_mask = torch.ones_like(classes).to(device)
        classes = classes.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 0.0
        sampling_fn = partial(p_sample_guided, classes=classes, cond_weight=cond_weight, context_mask=context_mask)
    else:
        sampling_fn = partial(p_sample)

    for i in tqdm(reversed(range(0, TIMESTEPS)), desc='sampling loop time step', total=TIMESTEPS):
        img = sampling_fn(model, x=img, t=torch.full((b,), i, device=device, dtype=torch.long), t_index=i)
        imgs.append(img.cpu().numpy())
        
    return imgs

@torch.no_grad()
def sample(model, image_size, classes=None, batch_size=16, channels=1, cond_weight=0):
    """Generate samples from the model."""
    return p_sample_loop(model, classes=classes, shape=(batch_size, channels, len(NUCLEOTIDES), image_size), cond_weight=cond_weight)

def sampling_to_metric(model_best, number_of_samples=20, specific_group=False, group_number=None, cond_weight_to_metric=0):
    """Generate sequences and convert to strings."""
    seq_final = []
    total_count = 0
    
    for n_a in range(number_of_samples):
        if specific_group and group_number is not None:
            sampled = torch.from_numpy(np.array([group_number] * BATCH_SIZE))
            print(f'Generating for class {group_number}')
        else:
            # Random sampling from all classes
            sampled = torch.from_numpy(np.random.choice(TOTAL_CLASS_NUMBER, BATCH_SIZE))

        random_classes = sampled.float().to(device)
        
        sampled_images = sample(
            model_best,
            classes=random_classes,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            channels=1,
            cond_weight=cond_weight_to_metric,
        )
        
        for n_b, x in enumerate(sampled_images[-1]):
            # Convert one-hot to sequence，输入形状是(1, 4, 50)
            seq = ''.join([NUCLEOTIDES[s] for s in np.argmax(x.reshape(4, IMAGE_SIZE), axis=0)])
            total_count += 1
            
            # Check if valid sequence
            if len(seq) == IMAGE_SIZE and all(n in NUCLEOTIDES for n in seq):
                seq_final.append(seq)
                    
    return list(set(seq_final)), total_count

def q_sample(x_start, t, noise=None):
    """Forward diffusion process."""
    if noise is None:
        noise = torch.randn_like(x_start)
        
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def predict_xstart_from_eps(x_t, t, eps):
    """Predict x_0 from noise prediction."""
    assert x_t.shape == eps.shape
    return (
        extract(sqrt_recip_alphas, t, x_t.shape) * x_t
        - extract(sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
    )

def calculate_weighted_nll_loss(x_start, pred_x_start, sample_weight=None):
    """Calculate negative log-likelihood loss."""
    pred_x_start = pred_x_start.squeeze(1)
    x_start = x_start.argmax(dim=2).squeeze(1)
    
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    nll_loss = loss_fct(pred_x_start, x_start)
    
    if sample_weight is not None:
        sample_weight = sample_weight.unsqueeze(1).expand_as(nll_loss)
        nll_loss = nll_loss * sample_weight
        total_loss = nll_loss.sum() / sample_weight.sum()
    else:
        total_loss = nll_loss.mean()
    
    return total_loss

def p_losses(denoise_model, x_start, t, classes, noise=None, loss_type="huber", 
             p_uncond=0.1, sample_weight=None):
    """Calculate training losses."""
    if noise is None:
        noise = torch.randn_like(x_start)
    
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    
    # Classifier-free guidance training
    context_mask = torch.bernoulli(torch.zeros(classes.shape[0]) + (1 - p_uncond)).to(x_start.device)
    classes = classes * context_mask
    classes = classes.type(torch.long)
    
    predicted_noise = denoise_model(x_noisy, t, classes)
    
    # Diffusion loss
    if loss_type == 'l1':
        dif_loss = F.l1_loss(predicted_noise, noise)
    elif loss_type == 'l2':
        dif_loss = F.mse_loss(predicted_noise, noise)
    elif loss_type == "huber":
        dif_loss = F.smooth_l1_loss(predicted_noise, noise)
    else:
        raise NotImplementedError("Unsupported loss type provided.")
    
    # NLL loss for reconstruction
    pred_x_start = predict_xstart_from_eps(x_noisy, t, predicted_noise)
    nll_loss = calculate_weighted_nll_loss(x_start, pred_x_start, sample_weight)
    
    return nll_loss, dif_loss

class SequenceDataset(Dataset):
    """Dataset for UTR sequences."""
    def __init__(self, sequences, labels, class_weight=None, transform=None):
        self.labels = labels
        self.nucleotides = NUCLEOTIDES  
        self.max_seq_len = IMAGE_SIZE   
        self.transform = transform
        self.images = []
        self.class_weight = class_weight

        for seq in sequences:
            encoded_img = self.one_hot_encode(seq)
            self.images.append(encoded_img)

        self.images = np.array([x.T.tolist() for x in self.images])

    def one_hot_encode(self, seq):
        """One-hot encode DNA sequence."""
        seq_array = np.zeros((len(seq), 4))
        
        for i, nucleotide in enumerate(seq):
            if nucleotide in self.nucleotides:
                seq_array[i, self.nucleotides.index(nucleotide)] = 1
        
        return seq_array

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        
        if self.transform:
            image = self.transform(image)
        
        if self.class_weight is not None: 
            sample_weight = self.class_weight[label]
        else:
            sample_weight = torch.tensor([1.0])
        
        return image, label, sample_weight

def seq_to_fasta(seq_list, outfilename):
    """Write sequences to FASTA file."""
    with open(outfilename, 'w') as file:
        for index, seq in enumerate(seq_list):
            file.write(f">seq_{index}_{len(seq)}\n{seq}\n")

def process_and_evaluate_sequences(model_best, class_=0):
    """Evaluate model and generate sequences."""
    model_best.eval()
    number_of_batches = math.ceil(args.gen_num / BATCH_SIZE)
    gen_seqs, total_count = sampling_to_metric(model_best, number_of_samples=number_of_batches,  
                                              specific_group=True, group_number=class_)
    
    valid_count = len(gen_seqs)
    print(f'Generated {valid_count}/{total_count} valid sequences for class {class_}')
    
    if valid_count > 0:
        os.makedirs('./results', exist_ok=True)
        seq_to_fasta(gen_seqs, f'./results/{prefix}_Class{class_}_num{valid_count}.fasta')
    
    return gen_seqs

# Helper functions for model architecture
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def l2norm(t):
    return F.normalize(t, dim=-1)

# Model architecture components
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        layers = [nn.Linear(input_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), LayerNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)

def Downsample(dim, dim_out=None):
    # 对于50nt序列，使用标准的2x2下采样，但padding调整
    return nn.Conv2d(dim, default(dim_out, dim), kernel_size=4, stride=2, padding=1)

def Upsample(dim, dim_out=None):
    # 标准的2x上采样
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'), 
        nn.Conv2d(dim, default(dim_out, dim), kernel_size=3, padding=1)
    )

class Unet(nn.Module):
    """U-Net architecture for diffusion model."""
    def __init__(
        self,
        dim,
        init_dim=None,
        dim_mults=(1, 2, 4),
        channels=1,
        resnet_block_groups=8,
        learned_sinusoidal_dim=18,
        num_classes=None,
    ):
        super().__init__()

        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, kernel_size=7, padding=3)
        
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        time_dim = dim * 4

        # Time embedding
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1
        self.time_mlp = nn.Sequential(
            sinu_pos_emb, 
            nn.Linear(fourier_dim, time_dim), 
            nn.GELU(), 
            nn.Linear(time_dim, time_dim)
        )

        # Class embedding
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)
        else:
            self.label_emb = None

        # Downsampling layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList([
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                ])
            )

        # Middle layers
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Upsampling layers
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList([
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                ])
            )

        # Final layers
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, 1, 1)  # 输出1个通道，因为输入已经是one-hot编码

    def forward(self, x, time, classes):
        # Debug: print input shape
        # print(f"Input shape: {x.shape}")
        
        x = self.init_conv(x)
        r = x.clone()
        
        t = self.time_mlp(time)
        
        if self.label_emb is not None and classes is not None:
            t += self.label_emb(classes)
        
        h = []
        
        # Downsampling
        for i, (block1, block2, attn, downsample) in enumerate(self.downs):
            x = block1(x, t)
            h.append(x)
            # print(f"Down {i} after block1: {x.shape}")
            
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            # print(f"Down {i} after block2+attn: {x.shape}")
            
            x = downsample(x)
            # print(f"Down {i} after downsample: {x.shape}")
        
        # Middle
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        # print(f"After middle: {x.shape}")
        
        # Upsampling - 这里需要确保维度匹配
        for i, (block1, block2, attn, upsample) in enumerate(self.ups):
            skip_connection = h.pop()
            # print(f"Up {i} - x: {x.shape}, skip: {skip_connection.shape}")
            
            # 如果维度不匹配，使用插值调整
            if x.shape[2:] != skip_connection.shape[2:]:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='nearest')
            
            x = torch.cat((x, skip_connection), dim=1)
            x = block1(x, t)
            
            skip_connection = h.pop()
            # print(f"Up {i} - x: {x.shape}, skip2: {skip_connection.shape}")
            
            # 再次检查维度
            if x.shape[2:] != skip_connection.shape[2:]:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='nearest')
                
            x = torch.cat((x, skip_connection), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
            # print(f"Up {i} after upsample: {x.shape}")
        
        # Final - 确保与残差连接的维度匹配
        # print(f"Before final - x: {x.shape}, r: {r.shape}")
        if x.shape[2:] != r.shape[2:]:
            x = F.interpolate(x, size=r.shape[2:], mode='nearest')
            
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        
        return x

@torch.no_grad()
def validate(model, val_loader):
    """Validate model on validation set."""
    model.eval()
    val_losses = []
    val_nll_losses = []
    val_dif_losses = []
    
    for batch in tqdm(val_loader, desc='Validation'):
        x, y, sample_weight = batch
        x = x.type(torch.float32).to(device)
        y = y.type(torch.long).to(device)
        sample_weight = sample_weight.to(device)
        
        t = torch.randint(0, TIMESTEPS, (x.shape[0],)).long().to(device)
        
        nll_loss, dif_loss = p_losses(model, x, t, y, loss_type="huber", sample_weight=sample_weight)
        
        val_nll_losses.append(nll_loss.item())
        val_dif_losses.append(dif_loss.item())
        
        loss = nll_loss + dif_loss
        val_losses.append(loss.item())
    
    return {
        'total_loss': np.mean(val_losses),
        'nll_loss': np.mean(val_nll_losses),
        'dif_loss': np.mean(val_dif_losses),
    }

# Initialize model with proper dimensions for 50nt sequences
model = Unet(
    dim=32,  # 减小维度以适应较小的序列长度
    channels=CHANNELS,
    dim_mults=(1, 2),  # 减少层数，因为50nt序列较短
    resnet_block_groups=RESNET_BLOCK_GROUPS,
    num_classes=TOTAL_CLASS_NUMBER,
).to(device)

# Initialize EMA
ema = EMA(args.ema_beta)
ema_model = copy.deepcopy(model).eval().requires_grad_(False)

# Initialize optimizer
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# Create datasets
tf = T.Compose([T.ToTensor()])

train_dataset = SequenceDataset(
    sequences=df_train.sequence.values,
    labels=df_train[args.category_col].values,
    class_weight=class_weight,
    transform=tf
)

val_dataset = SequenceDataset(
    sequences=df_val.sequence.values,
    labels=df_val[args.category_col].values,
    class_weight=class_weight,
    transform=tf
)

train_dl = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Training loop
loss_list, nll_loss_list, dif_loss_list = [], [], []
val_losses, val_nll_losses, val_dif_losses, val_epochs = [], [], [], []

patience = 50
epochs_without_improvement = 0
best_val_loss = np.inf
best_ep = 0

print(f"Starting training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    epoch_losses, nll_epoch_losses, dif_epoch_losses = [], [], []
    batch_t = []
    model.train()

    for step, batch in tqdm(enumerate(train_dl), desc=f'Epoch {epoch+1}/{EPOCHS}'):
        x, y, sample_weight = batch
        x = x.type(torch.float32).to(device)
        y = y.type(torch.long).to(device)
        sample_weight = sample_weight.to(device)
        
        t = torch.randint(0, TIMESTEPS, (x.shape[0],)).long().to(device)
        
        # Time warping logic
        if TIME_WARPING and len(dif_epoch_losses) >= 5 and step >= N_STEPS:
            sort_val = np.argsort(dif_epoch_losses[-len(batch_t):])
            sorted_t = [batch_t[i] for i in sort_val if i < len(batch_t)]
            
            if len(sorted_t) >= 5:
                last_n_t = sorted_t[-5:]
                unnested_last_n_t = [item for sublist in last_n_t for item in sublist]
                t_not_random = torch.tensor(np.random.choice(unnested_last_n_t, size=x.shape[0]), device=device)
                t = t if torch.rand(1) < 0.5 else t_not_random
        
        batch_t.append(list(t.cpu().detach().numpy()))

        # Calculate loss
        nll_loss, dif_loss = p_losses(model, x, t, y, loss_type="huber", sample_weight=sample_weight)
        
        loss = nll_loss + dif_loss
        
        # Record losses
        nll_epoch_losses.append(nll_loss.item())
        dif_epoch_losses.append(dif_loss.item())
        epoch_losses.append(loss.item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        ema.step_ema(ema_model, model)

    # Validation
    if epoch % 5 == 0:
        val_results = validate(model, val_dl)
        
        val_losses.append(val_results['total_loss'])
        val_nll_losses.append(val_results['nll_loss'])
        val_dif_losses.append(val_results['dif_loss'])
        val_epochs.append(epoch)
        
        print(f"Epoch {epoch+1} - Train Loss: {np.mean(epoch_losses):.4f}, Val Loss: {val_results['total_loss']:.4f}")
        print(f"  Val NLL: {val_results['nll_loss']:.4f}, Val Dif: {val_results['dif_loss']:.4f}")
        
        # Save best model
        if val_results['total_loss'] < best_val_loss:
            best_val_loss = val_results['total_loss']
            best_ep = epoch
            epochs_without_improvement = 0
            os.makedirs('./models', exist_ok=True)
            torch.save(model.state_dict(), f"./models/{prefix}_best_model.pt")
            print(f"New best model saved at epoch {epoch+1}")
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement > patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Record training losses
    loss_list.extend(epoch_losses)
    nll_loss_list.extend(nll_epoch_losses)
    dif_loss_list.extend(dif_epoch_losses)
    
    if (epoch + 1) % EPOCHS_LOSS_SHOW == 0:
        print(f"Epoch {epoch+1} | NLL: {np.mean(nll_epoch_losses):.4f} | Dif: {np.mean(dif_epoch_losses):.4f}")

# Generate training plots
def plot_training_curves():
    """Plot training and validation curves."""
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12))
    
    # Calculate epoch averages
    samples_per_epoch = len(train_dl)
    epochs_completed = len(loss_list) // samples_per_epoch
    
    epoch_avg_losses = {'nll': [], 'dif': [], 'total': []}
    for i in range(epochs_completed):
        start_idx = i * samples_per_epoch
        end_idx = (i + 1) * samples_per_epoch
        epoch_avg_losses['nll'].append(np.mean(nll_loss_list[start_idx:end_idx]))
        epoch_avg_losses['dif'].append(np.mean(dif_loss_list[start_idx:end_idx]))
        epoch_avg_losses['total'].append(np.mean(loss_list[start_idx:end_idx]))
    
    # Plot NLL loss
    if len(epoch_avg_losses['nll']) > 0:
        axes[0].plot(range(len(epoch_avg_losses['nll'])), epoch_avg_losses['nll'], 'b-', label='Train NLL')
    if len(val_epochs) > 0:
        axes[0].plot(val_epochs, val_nll_losses, 'ro-', label='Val NLL')
    axes[0].set_title('NLL Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot Diffusion loss
    if len(epoch_avg_losses['dif']) > 0:
        axes[1].plot(range(len(epoch_avg_losses['dif'])), epoch_avg_losses['dif'], 'g-', label='Train Diffusion')
    if len(val_epochs) > 0:
        axes[1].plot(val_epochs, val_dif_losses, 'mo-', label='Val Diffusion')
    axes[1].set_title('Diffusion Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot Total loss
    if len(epoch_avg_losses['total']) > 0:
        axes[2].plot(range(len(epoch_avg_losses['total'])), epoch_avg_losses['total'], 'k-', label='Train Total')
    if len(val_epochs) > 0:
        axes[2].plot(val_epochs, val_losses, 'co-', label='Val Total')
    axes[2].set_title('Total Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('./results', exist_ok=True)
    plt.savefig(f'./results/{prefix}_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

plot_training_curves()

# Load best model and generate sequences
print("Loading best model for evaluation...")
model.load_state_dict(torch.load(f"./models/{prefix}_best_model.pt"))

# Generate sequences for each class
print("Generating sequences for each class...")
for class_idx in range(TOTAL_CLASS_NUMBER):
    print(f"\nGenerating sequences for class {class_idx}...")
    gen_seqs = process_and_evaluate_sequences(model, class_=class_idx)
    print(f"Generated {len(gen_seqs)} unique sequences for class {class_idx}")

print("Training and evaluation completed!")