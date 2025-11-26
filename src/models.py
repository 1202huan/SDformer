import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import dgl.nn
from dgl.nn.pytorch import GATConv
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
from src.gltcn import *
from src.dlutils import *
from src.constants import *
from src.parser import args
import math
if args.dataset.lower() in ['swat']:
    torch.manual_seed(3)
    torch.cuda.manual_seed(3)
else:
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)



## DAGMM Model (ICLR 18)
class DAGMM(nn.Module):
    def __init__(self, feats):
        super(DAGMM, self).__init__()
        self.name = 'DAGMM'
        self.lr = 0.0001
        self.beta = 0.01
        self.n_feats = feats
        self.n_hidden = 16 #16
        self.n_latent = 8 #8
        self.n_window = 5 #5  # DAGMM w_size = 5
        self.n = self.n_feats * self.n_window
        self.n_gmm = self.n_feats * self.n_window
        self.encoder = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )
        self.estimate = nn.Sequential(
            nn.Linear(self.n_latent + 2, self.n_hidden), nn.Tanh(), nn.Dropout(0.5),
            nn.Linear(self.n_hidden, self.n_gmm), nn.Softmax(dim=1),
        )

    def compute_reconstruction(self, x, x_hat):
        relative_euclidean_distance = (x - x_hat).norm(2, dim=1) / x.norm(2, dim=1)
        cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
        return relative_euclidean_distance, cosine_similarity

    def forward(self, x):
        ## Encode Decoder
        x = x.view(1, -1)
        z_c = self.encoder(x)
        x_hat = self.decoder(z_c)
        ## Compute Reconstructoin
        rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
        z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
        ## Estimate
        gamma = self.estimate(z)
        return z_c, x_hat.view(-1), z, gamma.view(-1)


## OmniAnomaly Model (KDD 19)
class OmniAnomaly(nn.Module):
    def __init__(self, feats):
        super(OmniAnomaly, self).__init__()
        self.name = 'OmniAnomaly'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_hidden = 32
        self.n_latent = 8
        self.lstm = nn.GRU(feats, self.n_hidden, 2)
        self.encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Flatten(),
            nn.Linear(self.n_hidden, 2 * self.n_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
        )

    def forward(self, x, hidden=None):
        # 确保输入和隐藏状态在同一设备上
        device = x.device
        if hidden is None:
            hidden = torch.rand(2, 1, self.n_hidden, dtype=torch.float64).to(device)
        else:
            # 如果hidden已提供，确保它与x在相同设备上
            hidden = hidden.to(device)
        
        # 创建输入的副本，避免原地操作
        x_input = x.clone().view(1, 1, -1)
        out, hidden_new = self.lstm(x_input, hidden)
        
        # 编码器处理
        encoded = self.encoder(out)
        mu, logvar = torch.split(encoded, [self.n_latent, self.n_latent], dim=-1)
        
        # 重参数化技巧
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # 解码器处理
        decoded = self.decoder(z)
        
        return decoded.view(-1), mu.view(-1), logvar.view(-1), hidden_new


## USAD Model (KDD 20)
class USAD(nn.Module):
    def __init__(self, feats):
        super(USAD, self).__init__()
        self.name = 'USAD'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_hidden = 16
        self.n_latent = 5
        self.n_window = 5  # USAD w_size = 5
        self.n = self.n_feats * self.n_window
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_latent), nn.ReLU(True),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )

    def forward(self, g):
        ## Encode
        z = self.encoder(g.view(1, -1))
        ## Decoders (Phase 1)
        ae1 = self.decoder1(z)
        ae2 = self.decoder2(z)
        ## Encode-Decode (Phase 2)
        ae2ae1 = self.decoder2(self.encoder(ae1))
        return ae1.view(-1), ae2.view(-1), ae2ae1.view(-1)


## MSCRED Model (AAAI 19)
class MSCRED(nn.Module):
    def __init__(self, feats):
        super(MSCRED, self).__init__()
        self.name = 'MSCRED'
        self.lr = 0.00008
        self.n_feats = feats
        self.n_window = feats
        self.encoder = nn.ModuleList([
            ConvLSTM(1, 32, (3, 3), 1, True, True, False),
            ConvLSTM(32, 64, (3, 3), 1, True, True, False),
            ConvLSTM(64, 128, (3, 3), 1, True, True, False),
        ]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (3, 3), 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, (3, 3), 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, (3, 3), 1, 1), nn.Sigmoid(),
        )

    def forward(self, g):
        ## Encode
        z = g.view(1, 1, self.n_feats, self.n_window)
        for cell in self.encoder:
            _, z = cell(z.view(1, *z.shape))
            z = z[0][0]
        ## Decode
        x = self.decoder(z)
        return x.view(-1)


## CAE-M Model (TKDE 21)
class CAE_M(nn.Module):
    def __init__(self, feats):
        super(CAE_M, self).__init__()
        self.name = 'CAE_M'
        self.lr = 0.001
        self.n_feats = feats
        self.n_window = feats
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3), 1, 1), nn.Sigmoid(),
            nn.Conv2d(8, 16, (3, 3), 1, 1), nn.Sigmoid(),
            nn.Conv2d(16, 32, (3, 3), 1, 1), nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 4, (3, 3), 1, 1), nn.Sigmoid(),
            nn.ConvTranspose2d(4, 4, (3, 3), 1, 1), nn.Sigmoid(),
            nn.ConvTranspose2d(4, 1, (3, 3), 1, 1), nn.Sigmoid(),
        )

    def forward(self, g):
        ## Encode
        z = g.view(1, 1, self.n_feats, self.n_window)
        z = self.encoder(z)
        ## Decode
        x = self.decoder(z)
        return x.view(-1)


## MTAD_GAT Model (ICDM 20)
class MTAD_GAT(nn.Module):
    def __init__(self, feats):
        super(MTAD_GAT, self).__init__()
        self.name = 'MTAD_GAT'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_window = feats
        # 减小隐藏层大小，原来是 feats * feats，现在减小为 feats * 4 或更小
        self.n_hidden = feats * 4
        # 添加批处理大小
        self.batch = 64
        # 不在初始化时创建图，而是在每次forward调用时创建
        # 这样可以避免图结构的共享状态问题
        self.feature_gat = GATConv(feats, 1, feats)
        self.time_gat = GATConv(feats, 1, feats)
        self.gru = nn.GRU((feats + 1) * feats * 3, feats * 4, 1)  # 修改输出大小

    def forward(self, data, hidden):
        hidden = torch.rand(1, 1, self.n_hidden, dtype=torch.float64) if hidden is None else hidden
        data = data.view(self.n_window, self.n_feats)
        
        # 获取数据的设备
        device = data.device
        
        # 确保零张量在正确的设备上
        zeros = torch.zeros(1, self.n_feats, device=device)
        
        data_r = torch.cat((zeros, data))
        
        # 每次前向传播时创建新的图，避免共享状态
        src_nodes = torch.tensor(list(range(1, self.n_feats + 1)), device=device)
        dst_nodes = torch.tensor([0] * self.n_feats, device=device)
        g = dgl.graph((src_nodes, dst_nodes), device=device)
        g = dgl.add_self_loop(g)
        
        # 使用新创建的图而不是实例变量
        feat_r = self.feature_gat(g, data_r)
        data_t = torch.cat((zeros, data.t()))
        time_r = self.time_gat(g, data_t)
        data = torch.cat((zeros, data))
        data = data.view(self.n_window + 1, self.n_feats, 1)
        x = torch.cat((data, feat_r, time_r), dim=2).view(1, 1, -1)
        
        # 确保隐藏状态在正确的设备上
        if hidden is not None and hidden.device != device:
            hidden = hidden.to(device)
            
        x, h = self.gru(x, hidden)
        
        # 确保输出大小与输入匹配
        # 原始输入大小是self.n_window * self.n_feats
        output_size = self.n_window * self.n_feats
        x_view = x.view(-1)
        
        # 如果输出大小不匹配，进行调整
        if x_view.shape[0] != output_size:
            if x_view.shape[0] < output_size:
                # 如果输出太小，用零填充
                x_padded = torch.zeros(output_size, device=device, dtype=x_view.dtype)
                x_padded[:x_view.shape[0]] = x_view
                return x_padded, h
            else:
                # 如果输出太大，截断它
                return x_view[:output_size], h
        
        return x_view, h


## GDN Model (AAAI 21)
class GDN(nn.Module):
    def __init__(self, feats):
        super(GDN, self).__init__()
        self.name = 'GDN'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_window = 5
        self.n_hidden = 16
        self.n = self.n_window * self.n_feats
        src_ids = np.repeat(np.array(list(range(feats))), feats)
        dst_ids = np.array(list(range(feats)) * feats)
        self.g = dgl.graph((torch.tensor(src_ids), torch.tensor(dst_ids)))
        self.g = dgl.add_self_loop(self.g)
        self.feature_gat = GATConv(1, 1, feats)
        self.attention = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_window), nn.Softmax(dim=0),
        )
        self.fcn = nn.Sequential(
            nn.Linear(self.n_feats, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_window), nn.Sigmoid(),
        )

    def forward(self, data):
        # 获取数据的设备
        device = data.device
        
        # Bahdanau style attention
        att_score = self.attention(data).view(self.n_window, 1)
        data = data.view(self.n_window, self.n_feats)
        data_r = torch.matmul(data.permute(1, 0), att_score)
        
        # 确保图和特征在同一设备上
        self.g = self.g.to(device)
            
        # GAT convolution on complete graph
        feat_r = self.feature_gat(self.g, data_r)
        feat_r = feat_r.view(self.n_feats, self.n_feats)
        # Pass through a FCN
        x = self.fcn(feat_r)
        x=abs(x)
        return x.view(-1)


# MAD_GAN (ICANN 19)
class MAD_GAN(nn.Module):
    def __init__(self, feats):
        super(MAD_GAN, self).__init__()
        self.name = 'MAD_GAN'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_hidden = 8
        self.n_window = 5# MAD_GAN w_size = 5
        self.n = self.n_feats * self.n_window
        self.generator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )
        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, 1), nn.Sigmoid(),
        )

    def forward(self, g):
        ## Generate
        z = self.generator(g.view(1, -1))
        ## Discriminator
        real_score = self.discriminator(g.view(1, -1))
        fake_score = self.discriminator(z.view(1, -1))
        return z.view(-1), real_score.view(-1), fake_score.view(-1)

    
# Proposed Model + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD(nn.Module):
    def __init__(self, feats):
        super(TranAD, self).__init__()
        self.name = 'TranAD'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n = self.n_feats * self.n_window 
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window) 
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)  
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2  # 广播机制
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2  

    
# Proposed Model + Tcn_Local + Tcn_Global + Callback + Transformer + MAML
class DTAAD(nn.Module):
    def __init__(self, feats):
        super(DTAAD, self).__init__()
        self.name = 'DTAAD'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.l_tcn = Tcn_Local(num_outputs=feats, kernel_size=4, dropout=0.2)  # K=3&4 (Batch, output_channel, seq_len)
        self.g_tcn = Tcn_Global(num_inputs=self.n_window, num_outputs=feats, kernel_size=4, dropout=0.2)
        self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
        dim_ff = 8 if args.dataset.lower() in [ 'smd',] else 16
        print(f"使用数据集: {args.dataset}, dim_feedforward设置为: {dim_ff}")
        encoder_layers1 = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16,
                                                  dropout=0.1)  # (seq_len, Batch, output_channel)
        encoder_layers2 = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16,
                                                  dropout=0.1)
        self.transformer_encoder1 = TransformerEncoder(encoder_layers1, num_layers=1)  # only one layer
        self.transformer_encoder2 = TransformerEncoder(encoder_layers2, num_layers=1)
        self.fcn = nn.Linear(feats, feats)
        self.decoder1 = nn.Sequential(nn.Linear(self.n_window, 1), nn.Sigmoid())
        self.decoder2 = nn.Sequential(nn.Linear(self.n_window, 1), nn.Sigmoid())

    def callback(self, src, c):
        src2 = src + c
        g_atts = self.g_tcn(src2)
        src2 = g_atts.permute(2, 0, 1) * math.sqrt(self.n_feats)
        src2 = self.pos_encoder(src2)
        memory = self.transformer_encoder2(src2)
        #memory = torch.abs(memory)
       
        return memory

    def forward(self, src):
        l_atts = self.l_tcn(src)
        src1 = l_atts.permute(2, 0, 1) * math.sqrt(self.n_feats)
        src1 = self.pos_encoder(src1)
        z1 = self.transformer_encoder1(src1)
        #z1 = torch.abs(z1)
        
        c1 = z1 + self.fcn(z1)
        x1 = self.decoder1(c1.permute(1, 2, 0))
        z2 = self.fcn(self.callback(src, x1))
        c2 = z2 + self.fcn(z2)
        x2 = self.decoder2(c2.permute(1, 2, 0))
        return x1.permute(0, 2, 1), x2.permute(0, 2, 1)  # (Batch, 1, output_channel)


# 差分注意力相关的辅助类和函数 - SDfomer模型需要使用
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len: int, *, device: torch.device):
        seq = torch.arange(max_seq_len, device=device)
        freqs = torch.einsum("i,j->ij", seq, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.silu = nn.SiLU()
    
    def forward(self, x):
        return self.w2(self.silu(self.w1(x)) * self.w3(x))

class DifferentialAttention(nn.Module):
    def __init__(self, dim_model: int, head_nums: int, depth: int):
        super().__init__()
        
        # 修改后: 确保维度足够大
        self.head_dim = max(4, dim_model // head_nums)
        
        # 确保Q、K、V矩阵的输出维度合理
        qk_dim = 2 * self.head_dim
        self.Q = nn.Linear(dim_model, qk_dim, bias=False)
        self.K = nn.Linear(dim_model, qk_dim, bias=False)
        self.V = nn.Linear(dim_model, self.head_dim, bias=False)  # V只需要一个head_dim
        self.scale = self.head_dim ** -0.5
        self.depth = depth
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.rotary_emb = RotaryEmbedding(self.head_dim * 2)

    def forward(self, x):
        # 直接计算lambda_init，而不是调用函数
        lambda_init = 0.8 - 0.6 * math.exp(-0.3 * self.depth)
        Q = self.Q(x)
        K = self.K(x)

        seq_len = x.shape[1]
        cos, sin = self.rotary_emb(seq_len, device=x.device)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
    
        Q1, Q2 = Q.chunk(2, dim=-1)
        K1, K2 = K.chunk(2, dim=-1)
        V = self.V(x)
        A1 = Q1 @ K1.transpose(-2, -1) * self.scale
        A2 = Q2 @ K2.transpose(-2, -1) * self.scale
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(Q1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(Q2)
        lambda_ = lambda_1 - lambda_2 + lambda_init
        
        # 计算差分注意力权重
        attn1 = F.softmax(A1, dim=-1)
        attn2 = F.softmax(A2, dim=-1)
        diff_attn = attn1 - lambda_ * attn2
        
        # 返回(输出, 差分注意力权重)
        return diff_attn @ V, diff_attn

class MultiHeadDifferentialAttention(nn.Module):
    def __init__(self, dim_model: int, head_nums: int, depth: int):
        super().__init__()
        self.heads = nn.ModuleList([DifferentialAttention(dim_model, head_nums, depth) for _ in range(head_nums)])
        self.group_norm = RMSNorm(dim_model)  # 确保使用正确的维度
        
        # 修改后: 确保输出维度正确
        head_dim = max(4, dim_model // head_nums)
        self.output = nn.Linear(head_nums * head_dim, dim_model, bias=False)
        # 直接计算lambda_init，而不是调用函数
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        # 多头数量
        self.head_nums = head_nums
    
    def forward(self, x):
        outputs = []
        attn_weights = []
        for h in self.heads:
            # 修改后: 收集所有头的输出和注意力权重
            output, attn_weight = h(x)
            outputs.append(output)
            attn_weights.append(attn_weight)
        
        # 拼接所有头的输出
        o = torch.cat(outputs, dim=-1)
        # 应用输出映射
        o = self.output(o)
        # 最后进行归一化
        o = self.group_norm(o)
        o = o * (1 - self.lambda_init)
        
        # 返回输出和注意力权重
        return o, attn_weights
    
class DifferentialTransformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int = 8, head_dim: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([
            MultiHeadDifferentialAttention(dim, heads, depth_idx)
            for depth_idx in range(depth)
        ])
        self.ln1 = RMSNorm(dim)
        self.ln2 = RMSNorm(dim)
        self.ffn = FeedForward(dim, (dim // 3) * 8)
        self.depth = depth
        self.heads = heads
    
    def forward(self, x):
        # 存储每一层每个头的注意力权重
        all_attn_weights = []
        
        for attn in self.layers:
            # 得到这一层的注意力权重
            y, layer_attn_weights = attn(self.ln1(x))
            all_attn_weights.append(layer_attn_weights)
            y = y + x
            x = self.ffn(self.ln2(y)) + y
            
        return x, all_attn_weights

       
# SDfomer模型 
# 上分支：TCN捕捉多尺度时间关系 + 稀疏Transformer建模精细的局部依赖关系
# 下分支：差分Transformer捕获全局依赖关系并抑制注意力噪声
class SDfomer(nn.Module):
    def __init__(self, feats):
        super(SDfomer, self).__init__()
        self.name = 'SDfomer'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.block_size = 2
        # 修改标志，启用注意力可视化
        self.no_attention_visualization = True
        # 初始化存储最近注意力权重的属性
        self.last_attn_weights = None
        
        # 检测GPU可用性
        self.has_cuda = torch.cuda.is_available()
        dim_ff = 8 if args.dataset.lower() in ['wadi'] else 16
        if self.has_cuda:
            print(f"检测到可用GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("警告: 未检测到GPU。")
        

        kernel_size = 4 if args.dataset.lower() in ['smd'] else 3
        self.tcn_upper = Tcn_Global(num_inputs=self.n_window, num_outputs=feats, kernel_size=kernel_size, dropout=0.2)
 
        if args.dataset.lower() in ['swat','mba','wadi','smap']:
            self.tcn_local = Tcn_Local(num_outputs=feats, kernel_size=3, dropout=0.2)
        

        self.pos_encoder_upper = PositionalEncoding(feats, 0.1, self.n_window)
        sparse_encoder_layer = SparseTransformerEncoderLayer(
            d_model=feats, 
            nhead=feats, 
            block_size=self.block_size,
            dim_feedforward=dim_ff,
            dropout=0.1
        )
        self.sparse_transformer_upper = TransformerEncoder(sparse_encoder_layer, num_layers=1)
 
        self.pos_encoder_lower = PositionalEncoding(feats, 0.1, self.n_window)
        num_heads = max(1, min(32, feats // 2)) if args.dataset.lower() in ['wadi', 'smd'] else max(1, min(8, feats // 8))
        self.diff_transformer_lower = DifferentialTransformer(
            dim=feats,
            depth=1,
            heads=num_heads,
            head_dim=max(8, feats // num_heads)
        )
        

        self.fcn = nn.Linear(feats, feats)
        self.decoder = nn.Sequential(nn.Linear(self.n_window, 1), nn.Sigmoid())
        self.gate = nn.Sequential(
            nn.Linear(feats * 2, feats),
            nn.Sigmoid()
        )
        

    
    def forward(self, src):
 
        tcn_features = self.tcn_upper(src)
        src_upper = tcn_features.permute(2, 0, 1) * math.sqrt(self.n_feats)
        src_upper = self.pos_encoder_upper(src_upper)

        z_upper = self.sparse_transformer_upper(src_upper)
        c_upper = z_upper + self.fcn(z_upper)
        output_upper = self.decoder(c_upper.permute(1, 2, 0))
   
        src_lower = src.permute(2, 0, 1) * math.sqrt(self.n_feats)
        src_lower = self.pos_encoder_lower(src_lower)
 
        src_diff = src_lower.permute(1, 0, 2)

        z_lower, attn_weights = self.diff_transformer_lower(src_diff)

        self.last_attn_weights = attn_weights

        z_lower_trans = z_lower.permute(1, 0, 2)  # [seq_len, batch, hidden_dim]
        c_lower = z_lower_trans + self.fcn(z_lower_trans)
        output_lower = self.decoder(c_lower.permute(1, 2, 0))
        

        lmd = 0.3 if args.dataset.lower() in ['swat','mba','nab'] else 0.8
        output_final = lmd * output_upper + (1 - lmd) * output_lower
        
        return output_final.permute(0, 2, 1), attn_weights




