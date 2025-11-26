"""
Recurrent Transformer Architecture

Key Idea: Replace RNN/LSTM recurrence with Transformer-based recurrence.
At each cycle:
1. Input: full sequence (with positional encoding) + hidden state from previous cycle
2. Transformer processes both
3. Output: updated hidden state for next cycle
4. After N cycles, predict next location

This allows iterative refinement of representations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentTransformerLayer(nn.Module):
    """
    Single recurrent transformer layer that processes:
    - Input sequence (with positional encoding)
    - Hidden state from previous cycle
    
    Outputs updated hidden state for next cycle.
    """
    
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        
        # Self-attention for input sequence
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        
        # Cross-attention: query from hidden, keys/values from input
        self.cross_attn_input = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        
        # Cross-attention: query from hidden, keys/values from previous hidden
        self.cross_attn_hidden = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        
        # Gating for combining information
        self.gate = nn.Linear(d_model * 2, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_seq, hidden_state):
        """
        Args:
            input_seq: (batch, seq_len, d_model) - input sequence with positional encoding
            hidden_state: (batch, d_model) - hidden state from previous cycle
        Returns:
            new_hidden: (batch, d_model) - updated hidden state
        """
        batch_size, seq_len, d_model = input_seq.shape
        
        # Self-attention on input sequence
        attn_out, _ = self.self_attn(input_seq, input_seq, input_seq)
        attn_out = self.norm1(attn_out + input_seq)
        
        # Expand hidden state for cross-attention
        hidden_expanded = hidden_state.unsqueeze(1)  # (batch, 1, d_model)
        
        # Cross-attention: hidden queries input sequence
        cross_out1, attn_weights = self.cross_attn_input(
            hidden_expanded, attn_out, attn_out
        )
        cross_out1 = cross_out1.squeeze(1)  # (batch, d_model)
        
        # Cross-attention: hidden queries previous hidden (self-loop)
        cross_out2, _ = self.cross_attn_hidden(
            hidden_expanded, hidden_expanded, hidden_expanded
        )
        cross_out2 = cross_out2.squeeze(1)
        
        # Combine cross-attention outputs with gating
        combined = torch.cat([cross_out1, cross_out2], dim=-1)
        gate_weight = torch.sigmoid(self.gate(combined))
        gated = gate_weight * cross_out1 + (1 - gate_weight) * cross_out2
        
        gated = self.norm2(gated + hidden_state)
        
        # Feed-forward
        ffn_out = self.ffn(gated)
        new_hidden = self.norm3(ffn_out + gated)
        
        return new_hidden


class RecurrentTransformer(nn.Module):
    """
    Recurrent Transformer for Next-Location Prediction.
    
    Architecture:
    1. Embed inputs (locations, users, temporal features)
    2. Add positional encodings
    3. Initialize hidden state
    4. For N cycles:
         - Apply recurrent transformer layer
         - Update hidden state
    5. Use final hidden state to predict next location
    """
    
    def __init__(
        self,
        num_locations,
        num_users,
        d_model=96,
        n_layers=3,
        n_head=8,
        d_ff=192,
        n_cycles=3,  # Number of recurrent cycles
        dropout=0.15,
        max_len=50
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_cycles = n_cycles
        
        # Embeddings
        self.location_embed = nn.Embedding(num_locations, d_model)
        self.user_embed = nn.Embedding(num_users, d_model)
        self.hour_embed = nn.Embedding(24, d_model // 4)
        self.weekday_embed = nn.Embedding(7, d_model // 4)
        
        # Project temporal embeddings to match d_model
        self.temporal_proj = nn.Linear(d_model // 2, d_model)
        
        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        # Initial hidden state (learnable)
        self.init_hidden = nn.Parameter(torch.randn(d_model) * 0.02)
        
        # Recurrent transformer layers (shared across cycles or separate)
        # Using separate layers per cycle for more capacity
        self.recurrent_layers = nn.ModuleList([
            RecurrentTransformerLayer(d_model, n_head, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_locations)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, batch):
        """
        Forward pass with recurrent cycles.
        """
        locations = batch['locations']
        users = batch['users'][:, 0]  # Take first (constant across sequence)
        hours = (batch['start_mins'] // 60) % 24
        weekdays = batch['weekdays']
        
        batch_size, seq_len = locations.shape
        
        # Embed inputs
        loc_embed = self.location_embed(locations)  # (batch, seq_len, d_model)
        user_embed = self.user_embed(users).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Temporal embeddings
        hour_embed = self.hour_embed(hours)
        weekday_embed = self.weekday_embed(weekdays)
        temporal_embed = torch.cat([hour_embed, weekday_embed], dim=-1)
        temporal_embed = self.temporal_proj(temporal_embed)
        
        # Combine all embeddings
        input_embed = loc_embed + user_embed + temporal_embed
        
        # Add positional encoding
        input_embed = input_embed + self.pos_embed[:, :seq_len, :]
        input_embed = self.dropout(input_embed)
        
        # Initialize hidden state
        hidden = self.init_hidden.unsqueeze(0).expand(batch_size, -1)
        
        # Recurrent cycles
        for cycle in range(self.n_cycles):
            # Apply each layer
            for layer in self.recurrent_layers:
                hidden = layer(input_embed, hidden)
        
        # Final prediction
        logits = self.output_proj(hidden)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_recurrent_transformer(num_locations=1187, num_users=46):
    """Create recurrent transformer model."""
    model = RecurrentTransformer(
        num_locations=num_locations,
        num_users=num_users,
        d_model=80,  # Reduced
        n_layers=2,  # Fewer layers
        n_head=8,
        d_ff=160,  # Reduced
        n_cycles=4,  # More cycles to compensate
        dropout=0.15,
        max_len=50
    )
    
    num_params = model.count_parameters()
    print(f"Recurrent Transformer: {num_params:,} parameters")
    
    if num_params >= 500000:
        print(f"WARNING: Exceeds budget! ({num_params/500000*100:.1f}%)")
    else:
        print(f"✓ Within budget ({num_params/500000*100:.1f}% used)")
    
    return model


if __name__ == '__main__':
    # Test model
    model = create_recurrent_transformer()
    
    # Test forward pass
    batch = {
        'locations': torch.randint(0, 1187, (4, 20)),
        'users': torch.randint(0, 46, (4, 20)),
        'start_mins': torch.randint(0, 1440, (4, 20)),
        'weekdays': torch.randint(0, 7, (4, 20)),
        'target': torch.randint(0, 1187, (4,))
    }
    
    logits = model(batch)
    print(f"\nOutput shape: {logits.shape}")
    print(f"Expected: (4, 1187)")
