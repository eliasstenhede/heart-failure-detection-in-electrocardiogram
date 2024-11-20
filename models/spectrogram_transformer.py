import torch
import torch.nn as nn

class SpectrogramTokenizer:
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def extract_patches(self, x):
        B, C, H, W = x.shape
        patch_h, patch_w = self.patch_size
        assert H % patch_h == 0 and W % patch_w == 0, f"Height and Width must be divisible by patch size, got {H} and {W} with patch size {patch_h} and {patch_w}"

        patches = x.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
        patches = patches.permute(0, 2, 3, 1, 4, 5)  # (B, num_patches_h, num_patches_w, C, patch_h, patch_w)
        patches = patches.contiguous()

        return patches

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, emb_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.emb_dim = emb_dim

        self.proj = nn.Linear(in_channels * patch_size[0] * patch_size[1], emb_dim)

    def forward(self, x):
        x = x.flatten(2)  # Flatten patches (batch_size, num_patches, in_channels * patch_size[0] * patch_size[1])
        x = self.proj(x)  # Linear projection to emb_dim
        return x

class Encoder(nn.Module):
    def __init__(self, patch_size, in_channels, emb_dim, num_layers, nhead, dropout, dim_feedforward):
        super(Encoder, self).__init__()
        self.patch_embed = PatchEmbedding(patch_size, in_channels, emb_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dropout=dropout, 
                                                        dim_feedforward=dim_feedforward, activation='gelu', 
                                                        batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer(self.patch_embed(x))
    
class Transformer(nn.Module):
    def __init__(self, patch_size, input_shape, emb_dim_enc, num_layers, nhead, dropout, dim_feedforward, num_classes):
        super(Transformer, self).__init__()

        in_channels = input_shape[0]

        self.encoder = Encoder(patch_size, in_channels, emb_dim_enc, num_layers, nhead, dropout, dim_feedforward)
        self.tokenizer = SpectrogramTokenizer(self.encoder.patch_embed.patch_size)
        self.projection = nn.Linear(emb_dim_enc, num_classes)

    def forward(self, spectrograms):
        patches = self.tokenizer.extract_patches(spectrograms).flatten(1,2)
        encoded = self.encoder(patches)
        return self.projection(encoded.mean(dim=1))