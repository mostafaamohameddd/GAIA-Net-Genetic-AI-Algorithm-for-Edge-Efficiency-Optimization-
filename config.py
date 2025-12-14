# Dataset Configuration
DATASET_NAME = "iris"  # Options: "iris", "breast_cancer", "wine"
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Training Configuration
MAX_EPOCHS = 10  # Keep LOW for fast GA iteration
BATCH_SIZE = 32
VERBOSE = 0

# GA Configuration
POPULATION_SIZE = 20  # Small for fast computation
GENERATIONS = 5
MUTATION_RATE = 0.03  # Per-gene mutation rate (3%)
CROSSOVER_RATE = 0.8
ELITE_SIZE = 2  # Keep top 2 individuals per generation

# Selection Method: Tournament Selection (k=2)
TOURNAMENT_SIZE = 2  # Select best from random pairs

# Neural Network Constraints (Chromosome)
MIN_LAYERS = 1
MAX_LAYERS = 4
MIN_NEURONS = 8
MAX_NEURONS = 128
NEURON_STEP = 8  # Quantize to 8, 16, 24, ..., 128

ACTIVATIONS = ['relu', 'sigmoid', 'tanh']

# Learning Rate
MIN_LR = 0.0001
MAX_LR = 0.1

# Multi-Objective Weights (Fitness Function)
W_ACCURACY = 1.0          # Maximize accuracy
W_SIZE = 0.5              # Penalize model size
W_INFERENCE_TIME = 0.2    # Penalize slow inference
W_TRAINING_TIME = 0.1     # Penalize slow training

# Hard Constraints
MIN_ACCURACY_THRESHOLD = 0.75  # 75% minimum accuracy
MAX_PARAMETERS = 10000  # Max parameters allowed (simulating IoT device)
MAX_TRAINING_TIME = 300  # 5 minutes in seconds
MAX_INFERENCE_TIME = 1.0  # 1 second per sample (normalized)

# Neural Network Hyperparameters (Previously hard-coded)
DROPOUT_RATE = 0.2  # Dropout regularization
L2_REGULARIZER = 0.001  # L2 regularization weight
EARLY_STOPPING_PATIENCE = 3  # Early stopping patience

# Output
RESULTS_DIR = "results"
SAVE_BEST_GENOME = True
PLOT_EVOLUTION = True

# Random Seed for Reproducibility
RANDOM_SEED = 42


# ============= CONFIGURATION VALIDATION =============
def validate_config():
    """Validate all configuration parameters on startup.
    
    Raises:
        ValueError: If any configuration parameter is invalid
    """
    errors = []
    
    # Dataset validation
    valid_datasets = ["iris", "breast_cancer", "wine"]
    if DATASET_NAME not in valid_datasets:
        errors.append(f"DATASET_NAME must be one of {valid_datasets}, got '{DATASET_NAME}'")
    
    # Split sizes validation
    if not (0 < TEST_SIZE < 1):
        errors.append(f"TEST_SIZE must be between 0 and 1, got {TEST_SIZE}")
    if not (0 < VALIDATION_SIZE < 1):
        errors.append(f"VALIDATION_SIZE must be between 0 and 1, got {VALIDATION_SIZE}")
    
    # Training validation
    if MAX_EPOCHS <= 0:
        errors.append(f"MAX_EPOCHS must be > 0, got {MAX_EPOCHS}")
    if BATCH_SIZE <= 0:
        errors.append(f"BATCH_SIZE must be > 0, got {BATCH_SIZE}")
    if VERBOSE not in [0, 1, 2]:
        errors.append(f"VERBOSE must be 0, 1, or 2, got {VERBOSE}")
    
    # GA validation
    if POPULATION_SIZE <= 0:
        errors.append(f"POPULATION_SIZE must be > 0, got {POPULATION_SIZE}")
    if GENERATIONS <= 0:
        errors.append(f"GENERATIONS must be > 0, got {GENERATIONS}")
    if not (0 <= MUTATION_RATE <= 1):
        errors.append(f"MUTATION_RATE must be between 0 and 1, got {MUTATION_RATE}")
    if not (0 <= CROSSOVER_RATE <= 1):
        errors.append(f"CROSSOVER_RATE must be between 0 and 1, got {CROSSOVER_RATE}")
    if ELITE_SIZE < 1 or ELITE_SIZE > POPULATION_SIZE:
        errors.append(f"ELITE_SIZE must be between 1 and POPULATION_SIZE ({POPULATION_SIZE}), got {ELITE_SIZE}")
    if TOURNAMENT_SIZE < 1 or TOURNAMENT_SIZE > POPULATION_SIZE:
        errors.append(f"TOURNAMENT_SIZE must be between 1 and POPULATION_SIZE ({POPULATION_SIZE}), got {TOURNAMENT_SIZE}")
    
    # Network architecture validation
    if MIN_LAYERS < 1 or MIN_LAYERS > MAX_LAYERS:
        errors.append(f"MIN_LAYERS must be >= 1 and <= MAX_LAYERS ({MAX_LAYERS}), got {MIN_LAYERS}")
    if MAX_LAYERS <= 0:
        errors.append(f"MAX_LAYERS must be > 0, got {MAX_LAYERS}")
    if MIN_NEURONS < 1 or MIN_NEURONS > MAX_NEURONS:
        errors.append(f"MIN_NEURONS must be >= 1 and <= MAX_NEURONS ({MAX_NEURONS}), got {MIN_NEURONS}")
    if MAX_NEURONS <= 0:
        errors.append(f"MAX_NEURONS must be > 0, got {MAX_NEURONS}")
    if NEURON_STEP <= 0:
        errors.append(f"NEURON_STEP must be > 0, got {NEURON_STEP}")
    if MAX_NEURONS % NEURON_STEP != 0:
        errors.append(f"MAX_NEURONS ({MAX_NEURONS}) must be divisible by NEURON_STEP ({NEURON_STEP})")
    if MIN_NEURONS % NEURON_STEP != 0:
        errors.append(f"MIN_NEURONS ({MIN_NEURONS}) must be divisible by NEURON_STEP ({NEURON_STEP})")
    if len(ACTIVATIONS) == 0:
        errors.append("ACTIVATIONS list cannot be empty")
    
    # Learning rate validation
    if MIN_LR <= 0 or MIN_LR > MAX_LR:
        errors.append(f"MIN_LR must be > 0 and <= MAX_LR ({MAX_LR}), got {MIN_LR}")
    if MAX_LR <= 0:
        errors.append(f"MAX_LR must be > 0, got {MAX_LR}")
    
    # Fitness weights validation
    if W_ACCURACY <= 0:
        errors.append(f"W_ACCURACY must be > 0, got {W_ACCURACY}")
    if W_SIZE < 0:
        errors.append(f"W_SIZE must be >= 0, got {W_SIZE}")
    if W_INFERENCE_TIME < 0:
        errors.append(f"W_INFERENCE_TIME must be >= 0, got {W_INFERENCE_TIME}")
    if W_TRAINING_TIME < 0:
        errors.append(f"W_TRAINING_TIME must be >= 0, got {W_TRAINING_TIME}")
    
    # Hard constraints validation
    if not (0 < MIN_ACCURACY_THRESHOLD < 1):
        errors.append(f"MIN_ACCURACY_THRESHOLD must be between 0 and 1, got {MIN_ACCURACY_THRESHOLD}")
    if MAX_PARAMETERS <= 0:
        errors.append(f"MAX_PARAMETERS must be > 0, got {MAX_PARAMETERS}")
    if MAX_TRAINING_TIME <= 0:
        errors.append(f"MAX_TRAINING_TIME must be > 0, got {MAX_TRAINING_TIME}")
    if MAX_INFERENCE_TIME <= 0:
        errors.append(f"MAX_INFERENCE_TIME must be > 0, got {MAX_INFERENCE_TIME}")
    
    # Neural network hyperparameters validation
    if not (0 < DROPOUT_RATE < 1):
        errors.append(f"DROPOUT_RATE must be between 0 and 1, got {DROPOUT_RATE}")
    if L2_REGULARIZER < 0:
        errors.append(f"L2_REGULARIZER must be >= 0, got {L2_REGULARIZER}")
    if EARLY_STOPPING_PATIENCE <= 0:
        errors.append(f"EARLY_STOPPING_PATIENCE must be > 0, got {EARLY_STOPPING_PATIENCE}")
    
    # If there are errors, raise them all
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)


# Run validation on import
try:
    validate_config()
except ValueError as e:
    print(f"ERROR: {e}")
    raise