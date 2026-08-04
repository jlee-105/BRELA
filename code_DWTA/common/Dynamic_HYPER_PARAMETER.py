# This file includes parameter settings


################################
# Parameters for problem setting
################################
DEBUG = 'yes'
ENCODER_NAME = 'HGT'
NUM_TARGETS = 5
NUM_WEAPONS = 5
MIN_TARGET_VALUE = 1
MAX_TARGET_VALUE = 10
MAX_TIME = 5
PREPARATION_TIME = [1, 2, 1, 2, 1]*10  # length >= NUM_WEAPONS
# MAX_TIME_WINDOW = [10, 10, 10, 10, 10, 50, 50, 50, 50, 50]
# EMERGING_TIME = [0, 2, 1, 3, 0]*10
#AMM = [4] * 30
AMM = [2] * 50
#AMM = [4, 3, 4, 3, 4,   4, 6, 8, 6, 8]#,   8, 6, 8, 6, 8,   8, 6, 8, 6, 8]
LOW_PROB = 0.2
HIGH_PROB = 0.9

################################
# Cost parameters (multi-objective)
################################
WEAPON_COST = [100, 50, 100, 50, 20] * 10  # per-weapon firing cost, length >= NUM_WEAPONS
COST_WEIGHT = 0.5  # alpha: objective = COST_WEIGHT * remaining_value + (1 - COST_WEIGHT) * normalized_cost
MAX_BUDGET = None  # optional hard budget constraint (None = unconstrained)

################################
# ExIt (Expert Iteration) parameters
################################
N_EXIT_ITERS = 5
EXIT_IMITATION_WEIGHT = 0.8  # beta for hybrid loss, anneals toward 0.3
EXIT_IMITATION_WEIGHT_MIN = 0.3
EXIT_EXPERT_BUFFER_SIZE = 5000

################################
# Multi-policy rollout parameters
################################
NUM_ROLLOUT_POLICIES = 3  # greedy, stochastic, urgency-aware
ROLLOUT_SAMPLES_PER_POLICY = 1  # how many stochastic samples per policy

################################
# Assignment encoding indices for readability
################################
TARGET_START_TIME_INDEX = -4
TARGET_END_TIME_INDEX = -3
TARGET_VALUE_INDEX = -2
TARGET_AVAILABILITY_INDEX = -4
NO_ACTION_INDEX = -1

# Feature indices in assignment encoding (9 features total)
AMMUNITION_FEATURE_INDEX = 0
WEAPON_AVAILABILITY_FEATURE_INDEX = 1
TIME_LEFT_FEATURE_INDEX = 2
WEAPON_WAIT_TIME_FEATURE_INDEX = 3
TARGET_HIT_COUNT_FEATURE_INDEX = 4

NUM_FEATURES = 9           # total features per (weapon, target) pair
NUM_WEAPON_FEATURES = 4    # [ammo, availability, time_left, wait_time]
NUM_TARGET_FEATURES = 4    # [hit_count, start_time, end_time, value]
NUM_EDGE_FEATURES = 1      # [probability]

###############################
# Train: Hyper-Parameters
###############################`
TOTAL_EPISODE = 5                            # flexible
BUFFER_SIZE = 1000
UPDATE_PERIOD = 1
TOTAL_EPOCH = 200
#TOTAL_EPOCH = TOTAL_EPISODE*UPDATE_PERIOD
EVALUATION_PERIOD = UPDATE_PERIOD
NUM_EVALUATION = 1
TRAIN_BATCH = 15


VAL_BATCH =1
VAL_PARA = 1
NUM_PAR = 10
# SYNC_TARGET = 60
ACTOR_LEARNING_RATE = 5e-4
ACTOR_WEIGHT_DECAY = 1e-2
CRITIC_LEARNING_RATE = 1e-4
CRITIC_WEIGHT_DECAY = 1e-6
# START_PUCT = 4.0
# EPS_MIN = 1.414
# EPS_TARGET = 1000
#EPS_DECAY = (START_PUCT -EPS_MIN)/EPS_TARGET

##############################
# TRANSFORMER_MODEL-Parameters
##############################
# INPUT_DIM, OUTPUT_DIM will be computed dynamically based on actual problem size
# INPUT_DIM = num_weapons * num_targets + 1 (computed at runtime)
# OUTPUT_DIM = num_weapons * num_targets + 1 (computed at runtime)
EMBEDDING_DIM = 128
HIDDEN_DIM = 128  # Hidden dimension for policy network
HEAD_NUM = 8
KEY_DIM = 16
FF_DIM = 256
FF_HIDDEN_DIM = 256  # Feed-forward hidden dimension

# Model architecture parameters
ENCODER_LAYER_NUM = 4

# Chunking parameters for memory efficiency
CHUNK = 'False'  # Whether to use chunked attention
CHUNK_SIZE_Q = 128
CHUNK_SIZE_K = 128

# Logging and output folder configuration
# FOLDER_NAME = "TRAIN"

# Training parameters
# LEARNING_RATE = 0.001
# TEST_BATCH = 1

################################
# Hyperparameter Validation System
################################

def _validate_basic_constraints():
    """Validate basic parameter constraints."""
    errors = []
    
    # Basic positive constraints
    if NUM_WEAPONS <= 0:
        errors.append(f"NUM_WEAPONS must be positive, got {NUM_WEAPONS}")
    if NUM_TARGETS <= 0:
        errors.append(f"NUM_TARGETS must be positive, got {NUM_TARGETS}")
    if MAX_TIME <= 0:
        errors.append(f"MAX_TIME must be positive, got {MAX_TIME}")
    
    # Value range constraints
    if MIN_TARGET_VALUE > MAX_TARGET_VALUE:
        errors.append(f"MIN_TARGET_VALUE ({MIN_TARGET_VALUE}) must be <= MAX_TARGET_VALUE ({MAX_TARGET_VALUE})")
    
    # Probability constraints
    if not (0 < LOW_PROB < HIGH_PROB <= 1):
        errors.append(f"Invalid probability range: LOW_PROB={LOW_PROB}, HIGH_PROB={HIGH_PROB}")
    
    return errors


def _validate_array_lengths():
    """Validate array length consistency."""
    errors = []
    
    # AMM array length
    if len(AMM) < NUM_WEAPONS:
        errors.append(f"AMM array too short: need at least {NUM_WEAPONS} elements, got {len(AMM)}")
    
    # PREPARATION_TIME array length
    if len(PREPARATION_TIME) < NUM_WEAPONS:
        errors.append(f"PREPARATION_TIME array too short: need at least {NUM_WEAPONS} elements, got {len(PREPARATION_TIME)}")
    
    return errors


def _validate_training_parameters():
    """Validate training-related parameters."""
    errors = []
    
    # Learning rates
    if ACTOR_LEARNING_RATE <= 0:
        errors.append(f"ACTOR_LEARNING_RATE must be positive, got {ACTOR_LEARNING_RATE}")
    if CRITIC_LEARNING_RATE <= 0:
        errors.append(f"CRITIC_LEARNING_RATE must be positive, got {CRITIC_LEARNING_RATE}")
    
    # Batch sizes
    if TRAIN_BATCH <= 0:
        errors.append(f"TRAIN_BATCH must be positive, got {TRAIN_BATCH}")
    # MINI_BATCH is computed dynamically, skip validation
    
    # Epochs and episodes
    if TOTAL_EPOCH <= 0:
        errors.append(f"TOTAL_EPOCH must be positive, got {TOTAL_EPOCH}")
    if TOTAL_EPISODE <= 0:
        errors.append(f"TOTAL_EPISODE must be positive, got {TOTAL_EPISODE}")
    
    return errors


def _validate_model_parameters():
    """Validate neural network model parameters."""
    errors = []
    
    # Embedding dimensions
    if EMBEDDING_DIM <= 0:
        errors.append(f"EMBEDDING_DIM must be positive, got {EMBEDDING_DIM}")
    if HEAD_NUM <= 0:
        errors.append(f"HEAD_NUM must be positive, got {HEAD_NUM}")
    if KEY_DIM <= 0:
        errors.append(f"KEY_DIM must be positive, got {KEY_DIM}")
    
    # Dimension compatibility
    if EMBEDDING_DIM % HEAD_NUM != 0:
        errors.append(f"EMBEDDING_DIM ({EMBEDDING_DIM}) must be divisible by HEAD_NUM ({HEAD_NUM})")
    
    # Note: INPUT_DIM and OUTPUT_DIM are now computed dynamically at runtime
    # No validation needed for these parameters
    
    return errors


def validate_all_hyperparameters():
    """
    Comprehensive hyperparameter validation.
    
    Returns:
        tuple: (is_valid, error_messages)
    """
    all_errors = []
    
    # Run all validation checks
    all_errors.extend(_validate_basic_constraints())
    all_errors.extend(_validate_array_lengths())
    all_errors.extend(_validate_training_parameters())
    all_errors.extend(_validate_model_parameters())
    
    is_valid = len(all_errors) == 0
    
    if is_valid:
        print("[OK] All hyperparameters validated successfully!")
        print(f"   Problem size: {NUM_WEAPONS}W x {NUM_TARGETS}T, MAX_TIME={MAX_TIME}")
        print(f"   Model: EMBEDDING_DIM={EMBEDDING_DIM}, HEAD_NUM={HEAD_NUM}")
        print(f"   Training: {TOTAL_EPOCH} epochs, {TOTAL_EPISODE} episodes/epoch")
    else:
        print("[FAIL] Hyperparameter validation failed:")
        for error in all_errors:
            print(f"   - {error}")
    
    return is_valid, all_errors


def get_hyperparameter_summary():
    """Get a formatted summary of key hyperparameters."""
    summary = f"""
DWTA Hyperparameter Summary
==============================
Problem Configuration:
  - Weapons: {NUM_WEAPONS}
  - Targets: {NUM_TARGETS} 
  - Max Time: {MAX_TIME}
  - Target Values: [{MIN_TARGET_VALUE}, {MAX_TARGET_VALUE}]
  - Success Probability: [{LOW_PROB}, {HIGH_PROB}]

Training Configuration:
  - Total Epochs: {TOTAL_EPOCH}
  - Episodes/Epoch: {TOTAL_EPISODE}
  - Train Batch: {TRAIN_BATCH}
  - Mini Batch: (unused)
  - Actor LR: {ACTOR_LEARNING_RATE}
  - Critic LR: {CRITIC_LEARNING_RATE}

Model Architecture:
  - Embedding Dim: {EMBEDDING_DIM}
  - Attention Heads: {HEAD_NUM}
  - Key Dim: {KEY_DIM}
  - FF Dim: {FF_DIM}
==============================
"""
    return summary


# Automatic validation on import
if __name__ != "__main__":
    # Only validate when imported, not when run directly
    is_valid, errors = validate_all_hyperparameters()
    if not is_valid:
        print("\n WARNING: Hyperparameter validation failed!")
        print("Please fix the above errors before running the training.")
        print("Run 'python Dynamic_HYPER_PARAMETER.py' for detailed validation.")


# When run directly, provide detailed validation report
if __name__ == "__main__":
    print("DWTA Hyperparameter Validation Report")
    print("=" * 50)
    
    is_valid, errors = validate_all_hyperparameters()
    
    if not is_valid:
        print(f"\n Found {len(errors)} validation errors that must be fixed!")
        exit(1)
    else:
        print(get_hyperparameter_summary())
        print("[OK] All validations passed! Ready for training.")


