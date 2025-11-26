"""
Recurrent Transformer V2 - Improved Architecture

Key improvements over V1:
1. Hidden state is the FULL SEQUENCE (not a single token)
2. Each cycle refines the entire sequence
3. More cycles (6-8) for deeper refinement  
4. Strong residual connections between cycles
5. Shared transformer across cycles (parameter efficient)
6. Better normalization strategy

The idea: Process the same input through a transformer multiple times,
each time refining the representations more.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """Single transformer block (will be reused across cycles)."""
    
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        
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
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x


class RecurrentTransformerV2(nn.Module):
    """
    Recurrent Transformer V2 for Next-Location Prediction.
    
    Core idea:
    - Embed input sequence
    - For N cycles:
        * Apply transformer block to refine sequence
        * Add strong residual from input
    - Use final refined sequence for prediction
    """
    
    def __init__(
        self,
        num_locations,
        num_users,
        d_model=96,
        n_head=8,
        d_ff=192,
        n_cycles=6,  # More cycles
        n_blocks_per_cycle=1,  # Blocks per cycle
        dropout=0.15,
        max_len=50
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_cycles = n_cycles
        
        # Embeddings
        self.location_embed = nn.Embedding(num_locations, d_model)
        self.user_embed = nn.Embedding(num_users, d_model)
        
        # Temporal embeddings
        self.hour_embed = nn.Embedding(24, d_model)
        self.weekday_embed = nn.Embedding(7, d_model)
        
        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        # SHARED transformer blocks (reused across cycles)
        # This is parameter-efficient
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, dropout)
            for _ in range(n_blocks_per_cycle)
        ])
        
        # Cycle-specific scaling parameters (learnable residual weights)
        self.cycle_scales = nn.Parameter(torch.ones(n_cycles))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_locations)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
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
        
        # Embed everything
        loc_embed = self.location_embed(locations)
        user_embed = self.user_embed(users).unsqueeze(1).expand(-1, seq_len, -1)
        hour_embed = self.hour_embed(hours)
        weekday_embed = self.weekday_embed(weekdays)
        
        # Combine embeddings
        x = loc_embed + user_embed + hour_embed + weekday_embed
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.dropout(x)
        
        # Store initial embedding for residual
        initial_x = x
        
        # Recurrent refinement cycles
        for cycle_idx in range(self.n_cycles):
            # Save input to this cycle
            cycle_input = x
            
            # Apply transformer blocks
            for block in self.transformer_blocks:
                x = block(x)
            
            # Strong residual: mix with initial embedding AND cycle input
            # This prevents vanishing gradients in deep recurrence
            cycle_scale = self.cycle_scales[cycle_idx]
            x = cycle_scale * x + (1 - cycle_scale) * cycle_input + 0.1 * initial_x
        
        # Extract final representation (last token)
        final_repr = x[:, -1, :]
        
        # Predict
        logits = self.output_proj(final_repr)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_recurrent_transformer_v2(num_locations=1187, num_users=46):
    """Create improved recurrent transformer."""
    model = RecurrentTransformerV2(
        num_locations=num_locations,
        num_users=num_users,
        d_model=112,  # Increased
        n_head=8,
        d_ff=224,  # Increased
        n_cycles=8,  # Even more cycles!
        n_blocks_per_cycle=1,
        dropout=0.15,
        max_len=50
    )
    
    num_params = model.count_parameters()
    print(f"Recurrent Transformer V2: {num_params:,} parameters")
    
    if num_params >= 500000:
        print(f"WARNING: Exceeds budget! ({num_params/500000*100:.1f}%)")
    else:
        print(f"✓ Within budget ({num_params/500000*100:.1f}% used)")
    
    return model


if __name__ == '__main__':
    model = create_recurrent_transformer_v2()
    
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
