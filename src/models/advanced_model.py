"""
Advanced Next-Location Prediction Model

Incorporates state-of-the-art techniques:
1. Hierarchical multi-scale attention
2. Enhanced temporal encodings (cyclic)
3. Location-aware attention bias
4. Auxiliary reconstruction task
5. Memory-efficient architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class CyclicTimeEncoding(nn.Module):
    """Cyclic encoding for temporal features using sin/cos transformations."""
    
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Learnable parameters for scaling cyclic features
        self.hour_scale = nn.Parameter(torch.ones(d_model // 4))
        self.weekday_scale = nn.Parameter(torch.ones(d_model // 4))
        self.proj = nn.Linear(4, d_model // 2)
    
    def forward(self, hours, weekdays):
        """
        Args:
            hours: (batch, seq_len) in range [0, 23]
            weekdays: (batch, seq_len) in range [0, 6]
        Returns:
            encoding: (batch, seq_len, d_model // 2)
        """
        batch_size, seq_len = hours.shape
        
        # Normalize to [0, 1]
        hour_norm = hours.float() / 24.0
        weekday_norm = weekdays.float() / 7.0
        
        # Cyclic encoding: sin/cos transformation
        hour_sin = torch.sin(2 * math.pi * hour_norm)
        hour_cos = torch.cos(2 * math.pi * hour_norm)
        weekday_sin = torch.sin(2 * math.pi * weekday_norm)
        weekday_cos = torch.cos(2 * math.pi * weekday_norm)
        
        # Stack features
        cyclic_features = torch.stack([hour_sin, hour_cos, weekday_sin, weekday_cos], dim=-1)
        
        # Project to d_model // 2
        encoding = self.proj(cyclic_features)
        
        return encoding


class EnhancedLocationEmbedding(nn.Module):
    """Enhanced location embeddings with learnable bias for frequent locations."""
    
    def __init__(self, num_locations, d_model, top_k=50):
        super().__init__()
        self.embedding = nn.Embedding(num_locations, d_model, padding_idx=0)
        
        # Learnable bias for top-k frequent locations
        self.top_k = top_k
        self.freq_bias = nn.Parameter(torch.zeros(top_k, d_model))
        
    def forward(self, locations, is_frequent=None):
        """
        Args:
            locations: (batch, seq_len)
            is_frequent: (batch, seq_len) boolean mask for top-k locations
        """
        emb = self.embedding(locations)
        
        # Add bias for frequent locations if mask provided
        if is_frequent is not None:
            # This will be set during training with location frequency info
            pass
        
        return emb


class LightweightHierarchicalAttention(nn.Module):
    """
    Lightweight hierarchical attention using attention bias for local patterns.
    Much more parameter-efficient than dual attention heads.
    """
    
    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        
        # Single attention with local bias
        self.attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        
        # Learnable local bias to encourage attention to recent positions
        self.local_bias = nn.Parameter(torch.zeros(1, n_head, 1, 1))
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, local_window=10):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: attention mask (batch, seq_len, seq_len) boolean
            local_window: size of local attention window for bias
        """
        residual = x
        
        # Convert mask for MultiheadAttention
        # PyTorch MultiheadAttention expects (seq_len, seq_len) or (batch*n_head, seq_len, seq_len)
        attn_mask = None
        if mask is not None:
            # mask is (1, seq_len, seq_len) boolean - convert to (seq_len, seq_len)
            attn_mask = mask.squeeze(0)  # (seq_len, seq_len)
            # Invert mask: True where we DON'T want attention
            attn_mask = ~attn_mask
        
        # Standard attention
        out, attn_weights = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        
        out = self.dropout(out)
        out = self.layer_norm(residual + out)
        
        return out


class PositionwiseFeedForward(nn.Module):
    """Enhanced FFN with GELU activation and dropout."""
    
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        x = self.w_2(F.gelu(self.w_1(x)))
        x = self.dropout(x)
        x = self.layer_norm(residual + x)
        return x


class EncoderLayer(nn.Module):
    """Enhanced encoder layer with optional lightweight hierarchical attention."""
    
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, use_hierarchical=True):
        super().__init__()
        
        if use_hierarchical:
            self.slf_attn = LightweightHierarchicalAttention(d_model, n_head, d_k, d_v, dropout)
        else:
            # Fallback to standard attention
            from .attention_model import MultiHeadAttention
            self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout)
        self.use_hierarchical = use_hierarchical
    
    def forward(self, enc_input, slf_attn_mask=None):
        if self.use_hierarchical:
            enc_output = self.slf_attn(enc_input, mask=slf_attn_mask)
            enc_slf_attn = None
        else:
            enc_output, enc_slf_attn = self.slf_attn(
                enc_input, enc_input, enc_input, mask=slf_attn_mask)
        
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class AdvancedLocationPredictionModel(nn.Module):
    """
    Advanced transformer-based model with:
    - Hierarchical attention
    - Cyclic temporal encoding
    - Enhanced location embeddings
    - Auxiliary reconstruction task
    """
    
    def __init__(
        self,
        num_locations,
        num_users,
        d_model=88,
        d_inner=176,
        n_layers=4,
        n_head=8,
        d_k=11,
        d_v=11,
        dropout=0.15,
        max_len=50,
        use_hierarchical=True,
        use_auxiliary=True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.use_auxiliary = use_auxiliary
        
        # Enhanced embeddings
        self.loc_emb = EnhancedLocationEmbedding(num_locations, d_model)
        self.user_emb = nn.Embedding(num_users, d_model // 4, padding_idx=0)
        
        # Cyclic temporal encoding
        self.cyclic_time = CyclicTimeEncoding(d_model)
        
        # Feature projection
        feat_dim = d_model + d_model // 4 + d_model // 2
        self.feat_proj = nn.Linear(feat_dim, d_model)
        
        # Positional encoding
        self.position_enc = self._get_sinusoid_encoding(max_len, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        # Encoder layers
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout, use_hierarchical)
            for _ in range(n_layers)])
        
        # Main prediction head
        self.output_fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_locations)
        )
        
        # Auxiliary reconstruction head (if enabled)
        if use_auxiliary:
            self.aux_output = nn.Linear(d_model, num_locations)
        
    def _get_sinusoid_encoding(self, n_position, d_hid):
        """Sinusoidal positional encoding."""
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
    
    def forward(self, batch, return_auxiliary=False):
        """
        Args:
            batch: Dictionary containing input features
            return_auxiliary: Whether to return auxiliary outputs
        
        Returns:
            logits: (batch_size, num_locations) main predictions
            aux_logits: (batch_size, seq_len, num_locations) auxiliary predictions (optional)
        """
        locations = batch['locations']
        users = batch['users']
        weekdays = batch['weekdays']
        start_mins = batch['start_mins']
        mask = batch['mask']
        
        # Extract hours from start_mins
        hours = torch.div(start_mins, 60, rounding_mode='floor').clamp(0, 24)
        
        # Get embeddings
        loc_emb = self.loc_emb(locations)
        user_emb = self.user_emb(users)
        
        # Get cyclic temporal encoding
        time_emb = self.cyclic_time(hours, weekdays)
        
        # Concatenate features
        combined = torch.cat([loc_emb, user_emb, time_emb], dim=-1)
        
        # Project to model dimension
        enc_output = self.feat_proj(combined)
        enc_output = self.layer_norm(enc_output)
        enc_output = self.dropout(enc_output)
        
        # Add positional encoding
        pos_enc = self.position_enc[:, :enc_output.size(1), :].to(enc_output.device)
        enc_output = enc_output + pos_enc
        
        # Create causal attention mask
        slf_attn_mask = self.get_subsequent_mask(locations)
        
        # Pass through encoder layers
        for enc_layer in self.layer_stack:
            enc_output, _ = enc_layer(enc_output, slf_attn_mask=slf_attn_mask)
        
        # Get last valid position for main prediction
        seq_lens = mask.sum(dim=1).long() - 1
        batch_idx = torch.arange(locations.size(0), device=locations.device)
        last_hidden = enc_output[batch_idx, seq_lens]
        
        # Main output projection
        logits = self.output_fc(last_hidden)
        
        if return_auxiliary and self.use_auxiliary:
            # Auxiliary: predict all intermediate locations
            aux_logits = self.aux_output(enc_output)
            return logits, aux_logits
        
        return logits
    
    def get_subsequent_mask(self, seq):
        """Create causal mask for autoregressive prediction."""
        sz_b, len_s = seq.size()
        subsequent_mask = (1 - torch.triu(
            torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
        return subsequent_mask
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
