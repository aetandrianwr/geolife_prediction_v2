"""
Recurrent Transformer V3 - Simplified

Core principle: Feed same input through transformer multiple times (cycles),
updating hidden state each time. But with:
1. FEWER cycles (3-4 instead of 8)
2. STRONGER regularization (dropout 0.3)
3. NO parameter sharing (each cycle has its own transformer)
4. Simpler architecture

Goal: Prevent overfitting while maintaining recurrent refinement.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block."""
    
    def __init__(self, d_model, n_head, d_ff, dropout=0.3):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN  
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x


class RecurrentTransformerV3(nn.Module):
    """
    Simplified Recurrent Transformer.
    
    Key changes from V2:
    - Fewer cycles (4 instead of 8)
    - Higher dropout (0.3 instead of 0.15)
    - Each cycle has its OWN transformer (no parameter sharing)
    - Simpler residual strategy
    """
    
    def __init__(
        self,
        num_locations,
        num_users,
        d_model=88,
        n_head=8,
        d_ff=176,
        n_cycles=4,  # Fewer cycles
        dropout=0.3,  # Stronger regularization
        max_len=50
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_cycles = n_cycles
        
        # Embeddings
        self.location_embed = nn.Embedding(num_locations, d_model)
        self.user_embed = nn.Embedding(num_users, d_model)
        self.hour_embed = nn.Embedding(24, d_model)
        self.weekday_embed = nn.Embedding(7, d_model)
        
        # No projection needed - all same dimension
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        # SEPARATE transformer for each cycle (no sharing!)
        self.cycle_transformers = nn.ModuleList([
            SimpleTransformerBlock(d_model, n_head, d_ff, dropout)
            for _ in range(n_cycles)
        ])
        
        # Output
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_locations)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, batch):
        locations = batch['locations']
        users = batch['users'][:, 0]
        hours = torch.div(batch['start_mins'], 60, rounding_mode='floor') % 24
        weekdays = batch['weekdays']
        
        batch_size, seq_len = locations.shape
        
        # Embeddings
        loc_embed = self.location_embed(locations)
        user_embed = self.user_embed(users).unsqueeze(1).expand(-1, seq_len, -1)
        hour_embed = self.hour_embed(hours)
        weekday_embed = self.weekday_embed(weekdays)
        
        # Combine all embeddings
        x = loc_embed + user_embed + hour_embed + weekday_embed
        
        # Add positional encoding
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.dropout(x)
        
        # Recurrent cycles - each with its own transformer
        for cycle_idx, transformer in enumerate(self.cycle_transformers):
            x = transformer(x)
        
        # Extract final token
        final_repr = x[:, -1, :]
        
        # Predict
        logits = self.output_proj(final_repr)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_recurrent_transformer_v3(num_locations=1187, num_users=46):
    """Create simplified recurrent transformer."""
    model = RecurrentTransformerV3(
        num_locations=num_locations,
        num_users=num_users,
        d_model=88,
        n_head=8,
        d_ff=176,
        n_cycles=4,
        dropout=0.3,
        max_len=50
    )
    
    num_params = model.count_parameters()
    print(f"Recurrent Transformer V3: {num_params:,} parameters")
    
    if num_params >= 500000:
        print(f"WARNING: Exceeds budget! ({num_params/500000*100:.1f}%)")
    else:
        print(f"✓ Within budget ({num_params/500000*100:.1f}% used)")
    
    return model


if __name__ == '__main__':
    model = create_recurrent_transformer_v3()
    
    batch = {
        'locations': torch.randint(0, 1187, (4, 20)),
        'users': torch.randint(0, 46, (4, 20)),
        'start_mins': torch.randint(0, 1440, (4, 20)),
        'weekdays': torch.randint(0, 7, (4, 20)),
        'target': torch.randint(0, 1187, (4,))
    }
    
    logits = model(batch)
    print(f"\nOutput shape: {logits.shape}")
    print("✓ Model works!")
