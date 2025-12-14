import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
import time
import config

class NeuralNetworkBuilder:
    """Build and train neural networks from GA chromosomes."""
    
    @staticmethod
    def chromosome_to_architecture(chromosome):
        """Convert chromosome to architecture dictionary.
        
        Chromosome format: [num_layers, neurons_1, activation_1, neurons_2, activation_2, ..., learning_rate]
        
        Args:
            chromosome: List of floats representing network architecture
            
        Returns:
            dict: Architecture with keys 'layers' (list of layer configs) and 'learning_rate' (float)
            
        Raises:
            ValueError: If chromosome is invalid or empty or insufficient length
        """
        if not chromosome or len(chromosome) < 2:
            raise ValueError(f"Invalid chromosome: must have at least 2 genes, got {len(chromosome)}")
        
        # Constrain number of layers
        num_layers = max(config.MIN_LAYERS, min(config.MAX_LAYERS, int(round(chromosome[0]))))
        
        # Validate chromosome length for declared num_layers (ISSUE-1 FIX)
        # Chromosome must have: 1 (num_layers) + num_layers*2 (neurons + activation) + 1 (learning_rate)
        expected_length = 1 + (num_layers * 2) + 1
        if len(chromosome) < expected_length:
            raise ValueError(
                f"Chromosome too short for {num_layers} layers: "
                f"expected minimum {expected_length} genes, got {len(chromosome)}"
            )
        
        # Constrain learning rate to valid range
        learning_rate = max(config.MIN_LR, min(config.MAX_LR, chromosome[-1]))
        
        # Validate ACTIVATIONS configuration (ISSUE-2 FIX)
        if not config.ACTIVATIONS or len(config.ACTIVATIONS) == 0:
            raise ValueError("ACTIVATIONS configuration is empty - cannot build network")
        
        layers_config = []
        for i in range(num_layers):
            # Get raw neuron value and constrain to valid range
            neurons_raw = chromosome[1 + i*2]
            neurons_raw = max(config.MIN_NEURONS, min(config.MAX_NEURONS, neurons_raw))
            
            # Quantize to nearest valid neuron count
            # FIX #3: Use consistent quantization logic
            neurons = NeuralNetworkBuilder._quantize_neurons(neurons_raw)
            
            # Get activation function index
            # FIX #4: Validate bounds explicitly, use clamping instead of modulo (ISSUE-4 FIX)
            activation_idx = int(round(chromosome[2 + i*2]))
            if not (0 <= activation_idx < len(config.ACTIVATIONS)):
                # Use explicit clamping instead of modulo for better error visibility
                activation_idx = max(0, min(activation_idx, len(config.ACTIVATIONS) - 1))
            activation = config.ACTIVATIONS[activation_idx]
            
            layers_config.append({
                'neurons': neurons,
                'activation': activation
            })
        
        return {
            'layers': layers_config,
            'learning_rate': learning_rate
        }
    
    @staticmethod
    def _quantize_neurons(neurons_value):
        """Quantize neuron value to valid step increments.
        
        FIX #3: Centralized quantization logic for consistency
        
        Args:
            neurons_value: Raw neuron value (float)
            
        Returns:
            int: Quantized neuron value, multiple of NEURON_STEP
        """
        quantized = int(round(neurons_value / config.NEURON_STEP) * config.NEURON_STEP)
        # Ensure within bounds
        quantized = max(config.MIN_NEURONS, min(config.MAX_NEURONS, quantized))
        return quantized
    
    @staticmethod
    def build_model(n_features, n_classes, architecture):
        """Build Keras model from architecture.
        
        Args:
            n_features: Number of input features
            n_classes: Number of output classes
            architecture: Dict with 'layers' (list of dicts) and 'learning_rate' (float)
            
        Returns:
            Compiled Keras Sequential model
        """
        
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(n_features,)))
        
        # Hidden layers with dropout and L2 regularization
        for layer_config in architecture['layers']:
            model.add(layers.Dense(
                units=layer_config['neurons'],
                activation=layer_config['activation'],
                kernel_regularizer=keras.regularizers.l2(config.L2_REGULARIZER)  # FIX #8: Use config
            ))
            model.add(layers.Dropout(config.DROPOUT_RATE))  # FIX #7: Use config
        
        # Output layer
        if n_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(n_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
        
        optimizer = keras.optimizers.Adam(learning_rate=architecture['learning_rate'])
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        
        return model
    
    @staticmethod
    def count_parameters(model):
        """Count total parameters in model.
        
        Args:
            model: Keras model
            
        Returns:
            int: Total number of trainable parameters
        """
        return model.count_params()
    
    @staticmethod
    def train_and_evaluate(model, train_data, val_data, test_data):
        """Train model and return metrics.
        
        Args:
            model: Keras model to train
            train_data: Tuple of (X_train, y_train)
            val_data: Tuple of (X_val, y_val)
            test_data: Tuple of (X_test, y_test)
            
        Returns:
            dict: Metrics including 'accuracy', 'inference_time', 'training_time', 'loss'
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        # Training with early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,  # FIX #7: Use config
            restore_best_weights=True
        )
        
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            epochs=config.MAX_EPOCHS,
            batch_size=config.BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=config.VERBOSE
        )
        training_time = time.time() - start_time
        
        # Evaluation
        start_time = time.time()
        predictions = model.predict(X_test, verbose=0)
        
        # ISSUE-3 FIX: Check for empty test set before division
        if len(X_test) == 0:
            raise ValueError("Test dataset is empty - cannot calculate inference time")
        
        inference_time = (time.time() - start_time) / len(X_test)  # Per-sample time
        
        if model.output_shape[-1] == 1:
            y_pred = (predictions > 0.5).astype(int).flatten()
        else:
            y_pred = np.argmax(predictions, axis=1)
        
        accuracy = np.mean(y_pred == y_test)
        
        return {
            'accuracy': accuracy,
            'inference_time': inference_time,
            'training_time': training_time,
            'loss': history.history['loss'][-1]
        }