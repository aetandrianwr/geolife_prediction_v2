"""
Memory-Augmented Graph Attention Model

Completely different approach:
1. External memory module for long-term patterns
2. Graph structure over locations (co-occurrence)
3. Pointer networks for retrieval
4. Attention over memory slots
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocationGraphEmbedding(nn.Module):
    """Learn graph structure from location co-occurrences."""
    
    def __init__(self, num_locations, d_model):
        super().__init__()
        # Base location embeddings
        self.base_embed = nn.Embedding(num_locations, d_model)
        
        # Graph convolution (message passing)
        self.graph_conv1 = nn.Linear(d_model, d_model)
        self.graph_conv2 = nn.Linear(d_model, d_model)
        
        # Learnable adjacency (will be sparse in practice)
        # Use low-rank factorization to save parameters
        rank = min(num_locations //  10, 50)
        self.adj_left = nn.Parameter(torch.randn(num_locations, rank) * 0.01)
        self.adj_right = nn.Parameter(torch.randn(rank, num_locations) * 0.01)
    
    def get_adjacency(self):
        """Compute soft adjacency matrix via low-rank factorization."""
        adj = torch.matmul(self.adj_left, self.adj_right)
        # Normalize
        adj = F.softmax(adj, dim=-1)
        return adj
    
    def forward(self, location_ids):
        """
        Args:
            location_ids: (batch, seq_len)
        Returns:
            (batch, seq_len, d_model) - graph-enhanced embeddings
        """
        # Base embeddings
        x = self.base_embed(location_ids)  # (batch, seq_len, d_model)
        
        # Graph convolution
        # For each location, aggregate information from neighbors
        adj = self.get_adjacency()  # (num_locations, num_locations)
        
        # Gather neighbor embeddings
        all_embeds = self.base_embed.weight  # (num_locations, d_model)
        
        # Message passing layer 1
        neighbor_embeds1 = torch.matmul(adj, all_embeds)  # (num_locations, d_model)
        neighbor_embeds1 = self.graph_conv1(neighbor_embeds1)
        neighbor_embeds1 = F.relu(neighbor_embeds1)
        
        # Message passing layer 2
        neighbor_embeds2 = torch.matmul(adj, neighbor_embeds1)
        neighbor_embeds2 = self.graph_conv2(neighbor_embeds2)
        
        # Retrieve for current locations
        batch, seq_len = location_ids.shape
        flat_ids = location_ids.view(-1)
        graph_features = neighbor_embeds2[flat_ids].view(batch, seq_len, -1)
        
        # Combine base and graph features
        enhanced = x + graph_features
        
        return enhanced


class ExternalMemory(nn.Module):
    """External memory module for storing long-term patterns."""
    
    def __init__(self, num_memory_slots, d_model):
        super().__init__()
        self.num_slots = num_memory_slots
        self.d_model = d_model
        
        # Memory slots (learnable)
        self.memory = nn.Parameter(torch.randn(num_memory_slots, d_model))
        
        # Read/write heads
        self.read_query = nn.Linear(d_model, d_model)
        self.read_key = nn.Linear(d_model, d_model)
        self.write_gate = nn.Linear(d_model, 1)
    
    def read(self, query):
        """
        Read from memory using attention.
        Args:
            query: (batch, d_model)
        Returns:
            (batch, d_model) - read value
        """
        # Query transformation
        q = self.read_query(query)  # (batch, d_model)
        k = self.read_key(self.memory)  # (num_slots, d_model)
        
        # Attention weights
        scores = torch.matmul(q, k.t()) / math.sqrt(self.d_model)
        attn_weights = F.softmax(scores, dim=-1)  # (batch, num_slots)
        
        # Read values
        read_value = torch.matmul(attn_weights, self.memory)  # (batch, d_model)
        
        return read_value, attn_weights
    
    def write(self, value, location_id):
        """
        Write to memory (selective based on gate).
        Args:
            value: (batch, d_model)
            location_id: (batch,) - for addressing
        """
        # Write gate (should we write?)
        gate = torch.sigmoid(self.write_gate(value))  # (batch, 1)
        
        # Address based on location
        # Simple: use location_id % num_slots as address
        addresses = location_id % self.num_slots
        
        # Soft write
        batch_size = value.size(0)
        for i in range(batch_size):
            addr = addresses[i]
            # Exponential moving average update
            self.memory.data[addr] = 0.9 * self.memory.data[addr] + 0.1 * gate[i] * value[i]


class PointerAttention(nn.Module):
    """Pointer network for pointing to relevant history."""
    
    def __init__(self, d_model):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
    
    def forward(self, query, keys, values, mask=None):
        """
        Args:
            query: (batch, d_model) - current state
            keys: (batch, seq_len, d_model) - history
            values: (batch, seq_len, d_model) - history values
        Returns:
            (batch, d_model) - pointed value
        """
        q = self.query_proj(query).unsqueeze(1)  # (batch, 1, d_model)
        k = self.key_proj(keys)  # (batch, seq_len, d_model)
        v = self.value_proj(values)  # (batch, seq_len, d_model)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)  # (batch, 1, seq_len)
        
        # Weighted sum
        output = torch.matmul(attn_weights, v).squeeze(1)  # (batch, d_model)
        
        return output, attn_weights.squeeze(1)


class MemoryAugmentedGraphModel(nn.Module):
    """
    Memory-Augmented Graph Attention Model.
    
    Combines:
    - Location graph structure
    - External memory for long-term patterns
    - Pointer networks for retrieval
    """
    
    def __init__(
        self,
        num_locations,
        num_users,
        d_model=64,  # Reduced
        num_memory_slots=80,  # Fewer slots
        dropout=0.2,
        max_len=50
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_locations = num_locations
        
        # Graph-enhanced location embeddings
        self.location_graph_embed = LocationGraphEmbedding(num_locations, d_model)
        
        # User embeddings
        self.user_embed = nn.Embedding(num_users, d_model)
        
        # Temporal embeddings (simple)
        self.hour_embed = nn.Embedding(24, d_model)
        self.weekday_embed = nn.Embedding(7, d_model)
        
        # External memory
        self.memory = ExternalMemory(num_memory_slots, d_model)
        
        # Pointer attention
        self.pointer = PointerAttention(d_model)
        
        # LSTM for sequential modeling
        self.lstm = nn.LSTM(d_model, d_model, num_layers=2, batch_first=True, dropout=dropout)
        
        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=4, dropout=dropout, batch_first=True)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_locations)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, batch):
        """Forward pass."""
        locations = batch['locations']
        users = batch['users'][:, 0]  # Take first (constant across sequence)
        hours = (batch['start_mins'] // 60) % 24
        weekdays = batch['weekdays']
        
        batch_size, seq_len = locations.shape
        
        # Graph-enhanced location embeddings
        loc_embed = self.location_graph_embed(locations)
        
        # User and temporal embeddings
        user_embed = self.user_embed(users).unsqueeze(1).expand(-1, seq_len, -1)
        hour_embed = self.hour_embed(hours)
        weekday_embed = self.weekday_embed(weekdays)
        
        # Combine
        x = loc_embed + user_embed + hour_embed + weekday_embed
        x = self.dropout(x)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_out = self.norm1(lstm_out + x)
        
        # Self-attention
        attn_out, _ = self.self_attn(lstm_out, lstm_out, lstm_out)
        attn_out = self.norm2(attn_out + lstm_out)
        
        # Current state (last timestep)
        current_state = attn_out[:, -1, :]
        
        # Read from external memory
        memory_read, _ = self.memory.read(current_state)
        memory_enhanced = self.norm3(current_state + memory_read)
        
        # Pointer to relevant history
        history_ptr, _ = self.pointer(current_state, attn_out, attn_out)
        
        # Combine all features
        final_repr = torch.cat([memory_enhanced, history_ptr, current_state], dim=-1)
        
        # Output
        logits = self.output_proj(final_repr)
        
        # Write to memory (for next time)
        self.memory.write(current_state.detach(), batch['target'])
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_memory_model(num_locations=1187, num_users=46):
    """Create memory-augmented model."""
    model = MemoryAugmentedGraphModel(
        num_locations=num_locations,
        num_users=num_users,
        d_model=64,
        num_memory_slots=80,
        dropout=0.2,
        max_len=50
    )
    
    num_params = model.count_parameters()
    print(f"Memory Model: {num_params:,} parameters")
    
    if num_params >= 500000:
        print(f"WARNING: Exceeds budget! ({num_params/500000*100:.1f}%)")
    else:
        print(f"✓ Within budget ({num_params/500000*100:.1f}% used)")
    
    return model


if __name__ == '__main__':
    model = create_memory_model()
    
    # Test
    batch = {
        'location': torch.randint(0, 1187, (4, 20)),
        'user': torch.randint(0, 46, (4,)),
        'hour': torch.randint(0, 24, (4, 20)),
        'weekday': torch.randint(0, 7, (4, 20)),
        'target': torch.randint(0, 1187, (4,))
    }
    
    out = model(batch)
    print(f"Output shape: {out.shape}")
