"""
GNN-based Model for Dynamic Weapon-Target Assignment (DWTA)
Following BHGT framework structure with edge-aware learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .Dynamic_HYPER_PARAMETER import *
from .TORCH_OBJECTS import *


class ResidualBlock(nn.Module):
    """Simple residual MLP block with LayerNorm and Dropout for stability."""
    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = dim if hidden_dim is None else hidden_dim
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = residual + x
        x = self.norm2(x)
        return x


class ReplayMemory:
    """
    Circular buffer for storing and sampling experiences in reinforcement learning.
    """
    def __init__(self, capacity):
        """
        Initialize replay memory.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        """
        Store a new experience in memory.
        
        Args:
            transition: Experience tuple to store
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from memory.
        
        Args:
            batch_size: Number of experiences to sample
        """
        return self.memory[:batch_size]

    def __len__(self):
        return len(self.memory)


class EdgeAwareGNNLayer(nn.Module):
    """
    Heterogeneous GNN layer with edge-aware learning.
    Weapon nodes: 4 features, Target nodes: 4 features, Edges: 1 probability feature
    """
    def __init__(self, weapon_dim=4, target_dim=4, edge_dim=1, hidden_dim=EMBEDDING_DIM):
        super().__init__()
        
        # Node feature processors - handle both initial features and hidden states
        self.weapon_proj = nn.Linear(weapon_dim, hidden_dim)
        self.target_proj = nn.Linear(target_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        
        # For subsequent layers when input is already in hidden dimension
        self.weapon_hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.target_hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Message passing networks
        self.weapon_msg = nn.Linear(hidden_dim * 3, hidden_dim)  # weapon + target + edge
        self.target_msg = nn.Linear(hidden_dim * 3, hidden_dim)  # target + weapon + edge
        self.edge_update = nn.Linear(hidden_dim * 3, hidden_dim)  # weapon + target + old_edge
        
        # Node update networks
        self.weapon_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.target_update = nn.GRUCell(hidden_dim, hidden_dim)
        
    def forward(self, weapon_features, target_features, edge_features):
        """
        Args:
            weapon_features: [batch, para, num_weapons, 4 or hidden_dim]
            target_features: [batch, para, num_targets, 4 or hidden_dim] 
            edge_features: [batch, para, num_weapons, num_targets, 1 or hidden_dim]
        Returns:
            Updated weapon, target, and edge features
        """
        batch_size, para_size = weapon_features.shape[:2]
        num_weapons, num_targets = weapon_features.shape[2], target_features.shape[2]
        
        # Project to hidden dimension - check if already in hidden dimension
        if weapon_features.size(-1) == 4:  # Initial features
            weapon_h = self.weapon_proj(weapon_features)
        else:  # Already in hidden dimension
            weapon_h = self.weapon_hidden_proj(weapon_features)
            
        if target_features.size(-1) == 4:  # Initial features
            target_h = self.target_proj(target_features)
        else:  # Already in hidden dimension
            target_h = self.target_hidden_proj(target_features)
            
        if edge_features.size(-1) == 1:  # Initial features
            edge_h = self.edge_proj(edge_features)
        else:  # Already in hidden dimension
            edge_h = self.edge_hidden_proj(edge_features)
        
        # Expand for broadcasting
        weapon_exp = weapon_h.unsqueeze(3).expand(-1, -1, -1, num_targets, -1)  # [batch, para, W, T, hidden]
        target_exp = target_h.unsqueeze(2).expand(-1, -1, num_weapons, -1, -1)  # [batch, para, W, T, hidden]
        
        # Edge updates (key for edge-aware learning)
        edge_input = torch.cat([weapon_exp, target_exp, edge_h], dim=-1)
        edge_h_new = torch.tanh(self.edge_update(edge_input))
        
        # Message aggregation for weapons (from targets through edges)
        weapon_msgs = torch.cat([weapon_exp, target_exp, edge_h_new], dim=-1)
        weapon_msgs = self.weapon_msg(weapon_msgs).sum(dim=3)  # Sum over targets
        
        # Message aggregation for targets (from weapons through edges)  
        target_msgs = torch.cat([target_exp, weapon_exp, edge_h_new], dim=-1)
        target_msgs = self.target_msg(target_msgs).sum(dim=2)  # Sum over weapons
        
        # Update nodes
        weapon_h_flat = weapon_h.view(-1, weapon_h.size(-1))
        weapon_msgs_flat = weapon_msgs.view(-1, weapon_msgs.size(-1))
        weapon_h_new = self.weapon_update(weapon_msgs_flat, weapon_h_flat)
        weapon_h_new = weapon_h_new.view(batch_size, para_size, num_weapons, -1)
        
        target_h_flat = target_h.view(-1, target_h.size(-1))
        target_msgs_flat = target_msgs.view(-1, target_msgs.size(-1))
        target_h_new = self.target_update(target_msgs_flat, target_h_flat)
        target_h_new = target_h_new.view(batch_size, para_size, num_targets, -1)
        
        return weapon_h_new, target_h_new, edge_h_new


class EdgeAwareGNN_ACTOR(nn.Module):
    """
    Edge-aware GNN Actor following BHGT framework structure.
    """
    def __init__(self, num_layers=3):
        super().__init__()
        
        self.num_layers = num_layers
        self.gnn_layers = nn.ModuleList([
            EdgeAwareGNNLayer() for _ in range(num_layers)
        ])
        
        # Residual edge scorer: residual block on edge embedding then score
        self.edge_residual = ResidualBlock(dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, dropout=0.1)
        self.edge_score = nn.Linear(EMBEDDING_DIM, 1)
        
        # No-action scorer: project global state to EMB -> residual -> score
        self.global_proj = nn.Linear(EMBEDDING_DIM * 2, EMBEDDING_DIM)
        self.global_residual = ResidualBlock(dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, dropout=0.1)
        self.no_action_score = nn.Linear(EMBEDDING_DIM, 1)
        
        # For compatibility
        self.assignment_embedding = None
        self.current_state = None
        
        # Add replay_memory for compatibility with BHGT sampling
        self.replay_memory = ReplayMemory(capacity=BUFFER_SIZE)
        
        # Add optimizer for compatibility with BHGT training
        import torch.optim as optim
        self.optimizer = optim.Adam(
            self.parameters(), 
            lr=ACTOR_LEARNING_RATE, 
            weight_decay=ACTOR_WEIGHT_DECAY
        )
        
        # Add learning rate scheduler
        self.lr_stepper = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=50,  # Decrease LR every 50 epochs
            gamma=0.9      # Multiply LR by 0.9 at each step
        )
        
    def forward(self, assignment_embedding, prob, mask):
        """
        Args:
            assignment_embedding: [batch, para, NUM_WEAPONS*NUM_TARGETS+1, 9]
            prob: [batch, para, num_weapons, num_targets] or None
            mask: [batch, para, num_assignments+1]
        Returns:
            policy: [batch, para, num_assignments+1]
            assignment_embedding: for compatibility
        """
        batch_size, para_size = assignment_embedding.shape[:2]
        
        # Handle prob=None case
        if prob is None:
            num_weapons, num_targets = NUM_WEAPONS, NUM_TARGETS
        else:
            num_weapons, num_targets = prob.shape[2], prob.shape[3]
        
        # Extract heterogeneous features from assignment encoding
        # Remove no-action and reshape
        assignments = assignment_embedding[:, :, :-1, :]  # [batch, para, W*T, 9]
        features_reshaped = assignments.view(batch_size, para_size, num_weapons, num_targets, 9)
        
        # Extract node and edge features
        weapon_features = features_reshaped[:, :, :, 0, :4]  # [batch, para, W, 4]
        target_features = features_reshaped[:, :, 0, :, 4:8]  # [batch, para, T, 4]
        edge_features = features_reshaped[:, :, :, :, 8:9]    # [batch, para, W, T, 1]
        
        # Apply GNN layers
        weapon_h, target_h, edge_h = weapon_features, target_features, edge_features
        for layer in self.gnn_layers:
            weapon_h, target_h, edge_h = layer(weapon_h, target_h, edge_h)
        
        # Residual process for edges then score
        edge_h_res = self.edge_residual(edge_h)
        edge_scores = self.edge_score(edge_h_res).squeeze(-1)  # [batch, para, W, T]
        edge_scores_flat = edge_scores.view(batch_size, para_size, num_weapons * num_targets)
        
        # No-action using global state with residual projection
        global_weapon = weapon_h.mean(dim=2)  # [batch, para, hidden]
        global_target = target_h.mean(dim=2)   # [batch, para, hidden]
        global_state = torch.cat([global_weapon, global_target], dim=-1)
        global_state_proj = self.global_proj(global_state)
        global_state_res = self.global_residual(global_state_proj)
        no_action_score = self.no_action_score(global_state_res).squeeze(-1)  # [batch, para]
        
        # Combine scores
        all_scores = torch.cat([edge_scores_flat, no_action_score.unsqueeze(-1)], dim=-1)
        
        # Apply mask and softmax
        all_scores = all_scores.masked_fill(~mask.bool(), float('-inf'))
        policy = F.softmax(all_scores, dim=-1)
        
        # Store for compatibility
        self.assignment_embedding = edge_h_res.view(batch_size, para_size, num_weapons * num_targets, -1)
        no_action_emb = torch.zeros(batch_size, para_size, 1, EMBEDDING_DIM, device=edge_h_res.device)
        self.assignment_embedding = torch.cat([self.assignment_embedding, no_action_emb], dim=2)
        self.current_state = self.assignment_embedding
        
        return policy, self.assignment_embedding


class EdgeAwareGNN_CRITIC(nn.Module):
    """
    Edge-aware GNN Critic following BHGT framework structure.
    """
    def __init__(self, num_layers=3):
        super().__init__()
        
        self.num_layers = num_layers
        self.gnn_layers = nn.ModuleList([
            EdgeAwareGNNLayer() for _ in range(num_layers)
        ])
        
        # Residual value head: project -> residual -> value
        self.value_proj = nn.Linear(EMBEDDING_DIM * 3, EMBEDDING_DIM)
        self.value_residual = ResidualBlock(dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, dropout=0.1)
        self.value_out = nn.Linear(EMBEDDING_DIM, 1)
        
        # Add optimizer for compatibility with BHGT training
        import torch.optim as optim
        self.optimizer = optim.Adam(
            self.parameters(), 
            lr=CRITIC_LEARNING_RATE, 
            weight_decay=CRITIC_WEIGHT_DECAY
        )
        
        # Add learning rate scheduler
        self.lr_stepper = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=50,  # Decrease LR every 50 epochs
            gamma=0.9      # Multiply LR by 0.9 at each step
        )
        
    def forward(self, state, mask):
        """
        Args:
            state: [batch, para, num_assignments+1, 9] or [batch, num_assignments+1, 9]
            mask: [batch, para, num_assignments+1] or [batch, num_assignments+1]
        Returns:
            value: [batch, para, 1] or [batch, 1]
        """
        # Handle 3D input by adding para dimension
        if state.dim() == 3:
            state = state.unsqueeze(1)
            
        batch_size, para_size = state.shape[:2]
        
        # Detect problem size dynamically from input tensor
        num_assignments = state.size(2) - 1  # Remove no-action
        # For now, assume square problems for simplicity in critic
        # In practice, you might want to pass this information explicitly
        estimated_size = int(num_assignments ** 0.5)
        if estimated_size * estimated_size == num_assignments:
            num_weapons = num_targets = estimated_size
        else:
            # Non-square problem - need to infer from tensor structure
            # Use the weapon_to_target_prob tensor if available, or make an educated guess
            # For now, let's try common factorizations
            factors = []
            for w in range(1, int(num_assignments**0.5) + 1):
                if num_assignments % w == 0:
                    factors.append((w, num_assignments // w))
            # Pick the most balanced factorization
            num_weapons, num_targets = min(factors, key=lambda x: abs(x[0] - x[1]))
        
        # Extract features same as actor
        assignments = state[:, :, :-1, :]
        features_reshaped = assignments.view(batch_size, para_size, num_weapons, num_targets, 9)
        
        weapon_features = features_reshaped[:, :, :, 0, :4]
        target_features = features_reshaped[:, :, 0, :, 4:8]
        edge_features = features_reshaped[:, :, :, :, 8:9]
        
        # Apply GNN layers (same as actor)
        weapon_h, target_h, edge_h = weapon_features, target_features, edge_features
        for layer in self.gnn_layers:
            weapon_h, target_h, edge_h = layer(weapon_h, target_h, edge_h)
        
        # Global pooling for value estimation
        weapon_global = weapon_h.mean(dim=2)  # [batch, para, 4]
        target_global = target_h.mean(dim=2)   # [batch, para, 4]
        edge_global = edge_h.mean(dim=(2, 3))  # [batch, para, 1]
        
        # Combine global features
        global_state = torch.cat([weapon_global, target_global, edge_global], dim=-1)  # [batch, para, 9]
        
        # Value head with residual connection
        value_hidden = self.value_proj(global_state)
        value_hidden = self.value_residual(value_hidden)
        state_value = self.value_out(value_hidden)
        
        return state_value


# Factory functions
def create_gnn_actor():
    """Create edge-aware GNN actor model."""
    return EdgeAwareGNN_ACTOR()


def create_gnn_critic():
    """Create edge-aware GNN critic model."""
    return EdgeAwareGNN_CRITIC()


if __name__ == "__main__":
    print("🔧 Testing Edge-Aware GNN model...")
    
    # Test model creation
    actor = create_gnn_actor()
    critic = create_gnn_critic()
    
    print(f"✅ GNN Actor created - Parameters: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"✅ GNN Critic created - Parameters: {sum(p.numel() for p in critic.parameters()):,}")
    
    # Test forward pass
    batch_size, para_size = 2, 1
    test_assignment = torch.randn(batch_size, para_size, NUM_WEAPONS * NUM_TARGETS + 1, 9)
    test_prob = torch.randn(batch_size, para_size, NUM_WEAPONS, NUM_TARGETS)
    test_mask = torch.ones(batch_size, para_size, NUM_WEAPONS * NUM_TARGETS + 1)
    
    with torch.no_grad():
        policy, embeddings = actor(test_assignment, test_prob, test_mask)
        value = critic(test_assignment, test_mask)
    
    print(f"✅ Policy output shape: {policy.shape}")
    print(f"✅ Value output shape: {value.shape}")
    print("🎉 Edge-Aware GNN model test completed successfully!") 