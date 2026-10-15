import torch
import torch.nn as nn

LATENT_DIM = 256
POS_DIM = 2
SEQ_LEN = 90

class PositionHead(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 512)
        self.layer2 = nn.Linear(512, output_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    
    def forward(self, x):
        x = self.gelu(self.layer1(x))
        x = self.dropout(x)
        x = self.layer2(x)
        return x

class EncoderCNN(nn.Module):
    def __init__(self, num_input_channels: int, latent_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(latent_dim)
        self.convolution_layer = nn.ModuleList([
            nn.LazyConv1d(latent_dim, kernel_size=3, padding=1),
        ])


    def forward(self, x):
        for module in self.convolution_layer: 
            x = module(x)
        return x 


class LinearProjection(nn.Module):
    def __init__(self, input_size, ouptut_size):
        super().__init__()
        self.projection = nn.LazyLinear(ouptut_size) 
    
    def forward(self, x):
        return self.projection(x) 


class PositionalEncoding(nn.Module):
    def __init__(self, len_sequence, emb_size):
        super().__init__()
        self.pos_enc = nn.Parameter(
            data = torch.randn(len_sequence, emb_size),
            requires_grad = True
        )
    
    def forward(self, x):
        return x + self.pos_enc[None, ...]


class CIREncoder(nn.Module):
    def __init__(self, input_dim):
        
        super().__init__()

        self.latent_encoder = LinearProjection(input_dim, LATENT_DIM) 
        self.pos_enc = PositionalEncoding(SEQ_LEN, LATENT_DIM)

        tf_encoder_layer = nn.TransformerEncoderLayer(
            d_model = LATENT_DIM, 
            nhead = 1, 
            dim_feedforward=LATENT_DIM, 
            activation="gelu", 
            batch_first=True, 
            dropout=0.1,
        )

        self.attn_block = nn.TransformerEncoder(tf_encoder_layer, 1) 

    
    def forward(self, x):
        x = x.reshape(-1, x.shape[1]*x.shape[2], x.shape[3]).permute(0, 2, 1)
        x = self.latent_encoder(x)
        x = self.pos_enc(x)
        x = self.attn_block(x)
        return x


class FingerprintingTransformer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = CIREncoder(input_dim)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.head = PositionHead(LATENT_DIM, POS_DIM)
    

    def forward(self, x): 
        x = self.encoder(x) 
        x = self.avg_pool(x.permute(0, 2, 1)).squeeze()
        return self.head(x)