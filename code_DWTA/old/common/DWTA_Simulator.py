from .Dynamic_HYPER_PARAMETER import *
from .TORCH_OBJECTS import *

class Environment:
    """
    Dynamic Weapon-Target Assignment (DWTA) Environment
    
    Simulates a weapon-target assignment problem where weapons must be assigned
    to targets over time with constraints on ammunition, preparation times,
    and target availability windows.
    """

    def __init__(self, assignment_encoding, weapon_to_target_prob, max_time):
        """
        Initialize the DWTA environment.
        
        Args:
            assignment_encoding: Initial state encoding for all weapon-target pairs
            weapon_to_target_prob: Probability matrix for weapon-target engagement success
            max_time: Maximum simulation time steps
        """
        # Store original encoding with clone to avoid gradient issues
        self.assignment_encoding = assignment_encoding.clone()
        self.weapon_to_target_prob = weapon_to_target_prob.clone()
        
        # Pre-compute common tensor shapes for reuse
        self.batch_size = assignment_encoding.size(0)
        self.is_3d = len(assignment_encoding.shape) == 3
        
        if self.is_3d:
            # [batch, num_assignments, features] - no para dimension
            self.para_size = 1  # Default para size for compatibility
            # Expand to 4D for consistency: [batch, 1, num_assignments, features]
            self.assignment_encoding = self.assignment_encoding.unsqueeze(1).clone()
            if len(weapon_to_target_prob.shape) == 3:
                # [batch, weapons, targets] -> [batch, 1, weapons, targets]
                self.weapon_to_target_prob = weapon_to_target_prob.unsqueeze(1).clone()
        else:
            # [batch, para, num_assignments, features] - with para dimension
            self.para_size = assignment_encoding.size(1)
        
        # Detect actual problem size from input tensors
        self._detect_problem_size()

        # Build per-weapon preparation time (pad if global list is shorter)
        self.prep_time_per_weapon = self._build_prep_time_vector()
        
        # Initialize weapon-to-target assignment tracking
        self._initialize_weapon_target_assignments()
        
        # Initialize target values (optimized)
        self._initialize_target_values()
        
        # Initialize weapon status tracking
        self._initialize_weapon_status()
        
        # Initialize target availability tracking
        self._initialize_target_availability()
        
        # Initialize time tracking
        self._initialize_time_tracking()
        # Ensure initial action mask is consistent with current availability
        self._batch_update_action_mask()
    
    def _detect_problem_size(self):
        """Detect actual problem size from input tensors (size-agnostic)."""
        # assignment_encoding shape: [batch, para, num_assignments, features]
        # num_assignments = num_weapons * num_targets + 1 (for no-action)
        num_assignments = self.assignment_encoding.size(2) - 1  # Remove no-action
        
        # Get actual problem size from weapon_to_target_prob
        if len(self.weapon_to_target_prob.shape) == 4:
            # [batch, para, weapons, targets]
            self.actual_num_weapons = self.weapon_to_target_prob.size(2)
            self.actual_num_targets = self.weapon_to_target_prob.size(3)
        elif len(self.weapon_to_target_prob.shape) == 3:
            # [batch, weapons, targets] - already handled above with unsqueeze
            self.actual_num_weapons = self.weapon_to_target_prob.size(2)
            self.actual_num_targets = self.weapon_to_target_prob.size(3)
        else:
            raise ValueError(f"Unexpected weapon_to_target_prob shape: {self.weapon_to_target_prob.shape}")

    def _initialize_weapon_target_assignments(self):
        """Initialize weapon to target assignment tracking."""
        self.weapon_to_target_assign = torch.full(
            size=(self.batch_size, self.para_size, self.actual_num_weapons), 
            fill_value=-1
        ).to(DEVICE)

    def _initialize_target_values(self):
        """Initialize current and original target values (memory optimized)."""
        # Now assignment_encoding is always 4D: [batch, para, num_assignments, features]
        target_values = self.assignment_encoding[:, :, :-1, TARGET_VALUE_INDEX] * MAX_TARGET_VALUE
        # Fundamental fix: decouple from assignment_encoding view to avoid overlapping in-place ops
        self.current_target_value = target_values.clone()
        self.original_target_value = target_values.clone()  # Only clone once for original

    def _initialize_weapon_status(self):
        """Initialize weapon status tracking variables."""
        # Weapon availability tracking
        self.all_weapon_NOT_done = torch.tensor(True).to(DEVICE)
        
        # Pre-compute weapon indices for reuse
        self.weapon_indices = torch.arange(0, self.actual_num_weapons, device=DEVICE)
        self.possible_weapons = self.weapon_indices[None, None, :].expand(
            self.batch_size, self.para_size, self.actual_num_weapons
        ).clone()

        # Pre-compute action indices for reuse
        self.action_indices = torch.arange(0, self.actual_num_weapons*self.actual_num_targets, device=DEVICE)
        self.available_actions = self.action_indices[None, None, :].expand(
            self.batch_size, self.para_size, self.actual_num_weapons*self.actual_num_targets
        ).clone()
        
        # Initialize available actions mask
        initial_available_target = (self.assignment_encoding[:, :, :-1, TARGET_AVAILABILITY_INDEX])
        self.available_actions = initial_available_target.bool().clone()

        # Mask initialization
        self.mask = torch.full(
            size=(self.batch_size, self.para_size, self.actual_num_weapons*self.actual_num_targets+1,), 
            fill_value=1.0
        ).to(DEVICE)
        self.mask[:, :, -1] = 1.0  # No-action is always last index

        # Mask inactive targets (avoid overlapping in-place op)
        target_inactive = (self.assignment_encoding[:,:,:, TARGET_AVAILABILITY_INDEX] == 0)
        self.mask = self.mask * target_inactive.float()

        # Weapon availability
        self.weapon_availability = torch.full(
            size=(self.batch_size, self.para_size, self.actual_num_weapons), 
            fill_value=1.0
        ).to(DEVICE)
        
        # Ammunition availability (pad if AMM shorter than actual weapons)
        ammo_list = AMM[:self.actual_num_weapons]
        if len(ammo_list) < self.actual_num_weapons:
            pad_value = AMM[-1] if len(AMM) > 0 else 2
            ammo_list = ammo_list + [pad_value] * (self.actual_num_weapons - len(ammo_list))
        ammunition_availability = torch.tensor(ammo_list, device=DEVICE)
        self.ammunition_availability = ammunition_availability[None, None, :].expand(
            self.batch_size, self.para_size, ammunition_availability.size(0)
        ).clone()

        # Weapon wait time
        self.weapon_wait_time = torch.full(
            size=(self.batch_size, self.para_size, self.actual_num_weapons), 
            fill_value=0.0
        ).to(DEVICE)

    def _initialize_target_availability(self):
        """Initialize target availability tracking."""
        self.target_availability = torch.full(
            size=(self.batch_size, self.para_size, self.actual_num_targets), 
            fill_value=1.0
        ).to(DEVICE)
        
        # Pre-compute target indices for reuse
        self.target_indices = torch.arange(0, self.actual_num_targets, device=DEVICE)
        
        initial_available_target = (
            self.assignment_encoding[:, :, :, TARGET_AVAILABILITY_INDEX][:, :, :self.actual_num_targets] == 0
        )
        self.target_availability = initial_available_target.float()

        # Target time windows (avoid unnecessary clone)
        self.target_start_time = (
            self.assignment_encoding[:, :, :, TARGET_START_TIME_INDEX][:, :, :self.actual_num_targets] * MAX_TIME
        )
        self.target_end_time = (
            self.assignment_encoding[:, :, :, TARGET_END_TIME_INDEX][:, :, :self.actual_num_targets] * MAX_TIME
        )

        # Target hit count
        self.n_target_hit = torch.full(
            size=(self.batch_size, self.para_size, self.actual_num_targets), 
            fill_value=0.0
        ).to(DEVICE)

    def _initialize_time_tracking(self):
        """Initialize time tracking variables."""
        self.time_left = torch.tensor(MAX_TIME, device=DEVICE)
        self.clock = torch.tensor(0, device=DEVICE)
        self.n_fires = torch.tensor(0, device=DEVICE)
        self.mask[:, :, NO_ACTION_INDEX] = 1.0

    def update_internal_variables(self, selected_action):
        """Update environment state based on selected action."""
        # Convert action indices to weapon-target pairs
        batch_size = selected_action.size(0)
        para_size = selected_action.size(1)
        
        # Process each batch and parallel instance
        for batch_id in range(batch_size):
            for par_id in range(para_size):
                action_idx = selected_action[batch_id, par_id].item()
                
                # Skip no-action
                if action_idx == self.actual_num_weapons * self.actual_num_targets:
                    continue
                    
                # Convert flat index to weapon-target pair
                weapon_index = action_idx // self.actual_num_targets
                target_index = action_idx % self.actual_num_targets
                
                # Update state components (with cloned tensors)
                self._batch_update_target_values(batch_id, par_id, weapon_index, target_index)
                self._batch_update_weapon_assignments(batch_id, par_id, weapon_index, target_index)
                self._batch_update_weapon_status(batch_id, par_id, weapon_index)
                self._batch_update_ammunition(batch_id, par_id, weapon_index)
                self._batch_update_target_hit_count(batch_id, par_id, target_index)
                
        # Update availability and actions
        self._batch_update_availability_and_actions()
        
        # Update state encoding
        self._batch_update_state_encoding_for_time_step()
        
        return 0

    def _batch_update_target_values(self, batch_id, par_id, weapon_index, target_index):
        """Update target values based on weapon assignment."""
        # Clone tensors before modification
        self.current_target_value = self.current_target_value.clone()
        self.weapon_to_target_prob = self.weapon_to_target_prob.clone()
        
        # Update target value
        prob = self.weapon_to_target_prob[batch_id, par_id, weapon_index, target_index]
        self.current_target_value[batch_id, par_id, target_index] *= (1 - prob)

    def _batch_update_weapon_assignments(self, batch_id, par_id, weapon_index, target_index):
        """Update weapon-to-target assignments."""
        # Clone tensor before modification
        self.weapon_to_target_assign = self.weapon_to_target_assign.clone()
        self.weapon_to_target_assign[batch_id, par_id, weapon_index] = target_index

    def _batch_update_weapon_status(self, batch_id, par_id, weapon_index):
        """Update weapon status after assignment."""
        # Clone tensors before modification
        self.weapon_availability = self.weapon_availability.clone()
        self.weapon_wait_time = self.weapon_wait_time.clone()
        
        # Update weapon status
        self.weapon_availability[batch_id, par_id, weapon_index] = 0
        self.weapon_wait_time[batch_id, par_id, weapon_index] = self.prep_time_per_weapon[weapon_index]

    def _batch_update_ammunition(self, batch_id, par_id, weapon_index):
        """Update ammunition count for selected weapon."""
        # Clone tensor before modification
        self.ammunition_availability = self.ammunition_availability.clone()
        self.ammunition_availability[batch_id, par_id, weapon_index] -= 1

    def _batch_update_target_hit_count(self, batch_id, par_id, target_index):
        """Update hit count for target."""
        # Clone tensor before modification
        self.n_target_hit = self.n_target_hit.clone()
        self.n_target_hit[batch_id, par_id, target_index] += 1

    def _batch_update_availability_and_actions(self):
        """Update weapon and target availability."""
        # Clone tensors before modification
        self.available_actions = self.available_actions.clone()
        self.mask = self.mask.clone()
        
        # Update weapon availability
        weapon_available_mask = (self.weapon_wait_time == 0) & (self.ammunition_availability > 0)
        self.weapon_availability[weapon_available_mask] = 1.0
        
        # Update possible weapons
        possible_weapons_mask = (self.weapon_wait_time <= 0) & (self.ammunition_availability > 0)
        self.possible_weapons = possible_weapons_mask.long()
        
        # Update target availability
        target_available_mask = (self.target_start_time <= self.clock) & (self.target_end_time >= self.clock)
        self.target_availability = target_available_mask.long()
        
        # Update action mask
        self._batch_update_action_mask()

    def _batch_update_state_encoding_for_time_step(self):
        """Update state encoding with current time step information (speed optimized)."""
        # Use view for better performance
        encoding_view = self.assignment_encoding[:, :, :-1, :].view(
            self.batch_size, self.para_size, self.actual_num_weapons, self.actual_num_targets, -1
        )

        # Vectorized weapon availability update - simplified approach
        available_weapon_mask = self.weapon_availability == 1
        for b in range(self.batch_size):
            for p in range(self.para_size):
                for w in range(self.actual_num_weapons):
                    if available_weapon_mask[b, p, w]:
                        encoding_view[b, p, w, :, WEAPON_AVAILABILITY_FEATURE_INDEX] = 1.0

        # Vectorized weapon wait time update (use default prep time)
        max_prep = float(self.prep_time_per_weapon.max().item()) if self.prep_time_per_weapon.numel() > 0 else 1.0
        wait_time_normalized = (
            self.weapon_wait_time.unsqueeze(-1).expand(
                self.batch_size, self.para_size, self.actual_num_weapons, self.actual_num_targets
            ) / max(1.0, max_prep)
        )
        encoding_view[:, :, :, :, WEAPON_WAIT_TIME_FEATURE_INDEX] = wait_time_normalized

        # Vectorized time left update
        time_left_normalized = self.time_left / MAX_TIME
        encoding_view[:, :, :, :, TIME_LEFT_FEATURE_INDEX] = time_left_normalized
        
        # Safe write-back: avoid overlapping in-place ops
        new_assignment_encoding = self.assignment_encoding.clone()
        new_assignment_encoding[:, :, :-1, :] = encoding_view.reshape(
            self.batch_size, self.para_size, self.actual_num_weapons * self.actual_num_targets, -1
        )
        self.assignment_encoding = new_assignment_encoding

    def _build_prep_time_vector(self) -> torch.Tensor:
        """Return per-weapon preparation time tensor of length actual_num_weapons with padding if needed."""
        prep_list = PREPARATION_TIME[:self.actual_num_weapons]
        if len(prep_list) < self.actual_num_weapons:
            pad_value = PREPARATION_TIME[-1] if len(PREPARATION_TIME) > 0 else 1
            prep_list = prep_list + [pad_value] * (self.actual_num_weapons - len(prep_list))
        return torch.tensor(prep_list, device=DEVICE, dtype=torch.float32)


    def mask_probs(self, action_probs):
        """
        Apply action mask to action probabilities and normalize (speed optimized).
        
        Args:
            action_probs: Raw action probabilities
            
        Returns:
            Normalized probabilities with mask applied
        """
        # Vectorized masking and normalization
        mask_probs = action_probs * self.mask.to(DEVICE)
        sums = mask_probs.sum(dim=-1, keepdim=True)  # Sum along action dimension
        normalized_probability = mask_probs / (sums + 1e-8)  # Add epsilon for stability

        return normalized_probability

    def time_update(self):
        """
        Update the environment state for the next time step (speed optimized).
        
        This method advances the simulation clock and updates all time-dependent
        variables including weapon wait times, target availability, and state encoding.
        """
        # Advance clock
        self.clock += 1
        
        # Update weapon wait times (in-place operation)
        self.weapon_wait_time[self.weapon_wait_time > 0] -= 1
        
        # Update remaining time
        self.time_left = MAX_TIME - self.clock
        
        # Batch update all time-dependent components
        self._batch_update_time_dependent_components()
        
        # Update state encoding
        self._batch_update_state_encoding_for_time_step()

        return 0

    def _batch_update_time_dependent_components(self):
        """Batch update all time-dependent components using vectorized operations."""
        # Vectorized weapon availability update
        weapon_available_mask = (self.weapon_wait_time == 0) & (self.ammunition_availability > 0)
        self.weapon_availability[weapon_available_mask] = 1.0
        
        # Vectorized possible weapons update
        possible_weapons_mask = (self.weapon_wait_time <= 0) & (self.ammunition_availability > 0)
        self.possible_weapons = possible_weapons_mask.long()

        # Vectorized target availability update
        target_available_mask = (self.target_start_time <= self.clock) & (self.target_end_time >= self.clock)
        self.target_availability = target_available_mask.long()

        # Vectorized action mask creation and update
        self._batch_update_action_mask()

    def _batch_update_action_mask(self):
        """Batch update action mask using vectorized operations."""
        # Create action mask tensor
        available_actions = torch.ones(
            self.batch_size, self.para_size, self.actual_num_weapons * self.actual_num_targets, 
            device=DEVICE, dtype=torch.float32
        ).view(self.batch_size, self.para_size, self.actual_num_weapons, self.actual_num_targets)

        # Vectorized weapon masking
        weapon_unavailable = self.possible_weapons <= 0
        if weapon_unavailable.any():
            weapon_mask = weapon_unavailable.unsqueeze(-1).expand(-1, -1, -1, self.actual_num_targets)
            available_actions[weapon_mask] = 0.0

        # Vectorized target masking
        target_unavailable = self.target_availability <= 0
        if target_unavailable.any():
            target_mask = target_unavailable.unsqueeze(-1).expand(-1, -1, -1, self.actual_num_weapons)
            available_actions[..., target_mask.transpose(-1, -2)] = 0.0

        # Update in-place
        self.available_actions = available_actions.view(self.batch_size, self.para_size, -1)
        self.mask[:, :, :-1] = self.available_actions
        self.mask[:, :, NO_ACTION_INDEX] = 1.0

    def reward_calculation(self):
        """
        Calculate reward based on target value destruction (speed optimized).
        
        Returns:
            reward: Reward value (1 - remaining_value/original_value)
            current_target_value: Current target values for debugging
        """
        # Vectorized reward calculation
        current_sum = self.current_target_value[:, :, 0:self.actual_num_targets].sum()
        original_sum = self.original_target_value[:, :, 0:self.actual_num_targets].sum()
        reward = 1 - current_sum / original_sum
        return reward, self.current_target_value
