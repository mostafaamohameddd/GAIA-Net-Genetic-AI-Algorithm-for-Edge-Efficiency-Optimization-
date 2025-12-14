import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config

def load_dataset():
    """Load the selected dataset.
    
    Returns:
        tuple: (X, y, n_features, n_classes) - Feature matrix, labels, and dataset info
    """
    
    if config.DATASET_NAME == "iris":
        data = datasets.load_iris()
    elif config.DATASET_NAME == "breast_cancer":
        data = datasets.load_breast_cancer()
    elif config.DATASET_NAME == "wine":
        data = datasets.load_wine()
    else:
        raise ValueError(f"Unknown dataset: {config.DATASET_NAME}")
    
    X = data.data
    y = data.target
    
    print(f"Loaded {config.DATASET_NAME} dataset")
    print(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}, Classes: {len(np.unique(y))}")
    
    return X, y, X.shape[1], len(np.unique(y))

def prepare_data():
    """Load, split, and normalize data.
    
    Returns:
        tuple: ((X_train, y_train), (X_val, y_val), (X_test, y_test), n_features, n_classes)
        
    Raises:
        ValueError: If TEST_SIZE or VALIDATION_SIZE are invalid
    """
    
    # Validate configuration to prevent division by zero
    if not (0 < config.TEST_SIZE < 1):
        raise ValueError(f"TEST_SIZE must be between 0 and 1, got {config.TEST_SIZE}")
    if not (0 < config.VALIDATION_SIZE < 1):
        raise ValueError(f"VALIDATION_SIZE must be between 0 and 1, got {config.VALIDATION_SIZE}")
    
    X, y, n_features, n_classes = load_dataset()
    
    # First split: train+val / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
    )
    
    # Second split: train / val
    val_size_adjusted = config.VALIDATION_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=config.RANDOM_SEED, stratify=y_temp
    )
    
    # Normalize - NO DATA LEAKAGE: fit only on training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"Data split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), n_features, n_classes