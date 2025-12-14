"""
Multi-objective fitness calculation with HARD constraints.

Fitness Function:
  Fitness = w_acc * accuracy_norm - w_size * size_penalty - w_inf * inf_penalty - w_train * train_penalty

Hard Constraints (MUST satisfy all or return -1000):
  - Minimum accuracy >= 75%
  - Maximum parameters <= 10,000
  - Training time <= 300 seconds
  - Inference time <= 1.0 seconds

Soft Objectives (all normalized to [0,1]):
  - Maximize accuracy (normalized above threshold)
  - Minimize model size (normalized)
  - Minimize inference time (normalized)
  - Minimize training time (normalized)
"""

import config
from neural_network import NeuralNetworkBuilder


class FitnessCalculator:
    """Calculate multi-objective fitness with HARD constraints."""

    @staticmethod
    def calculate_fitness(metrics, n_parameters):
        """
        Calculate fitness score for a neural network.

        Args:
            metrics: Dict with keys:
                - 'accuracy': float [0, 1]
                - 'inference_time': float (seconds per sample)
                - 'training_time': float (seconds total)
            n_parameters: int, total trainable parameters

        Returns:
            float: Fitness score. -1000 if constraints violated, else in [-0.8, 1.0]

        Constraints:
            - accuracy >= MIN_ACCURACY_THRESHOLD (0.75)
            - n_parameters <= MAX_PARAMETERS (10,000)
            - training_time <= MAX_TRAINING_TIME (300s)
            - inference_time <= MAX_INFERENCE_TIME (1.0s)
        """
        accuracy = metrics['accuracy']
        inference_time = metrics['inference_time']
        training_time = metrics['training_time']

        # ===== HARD CONSTRAINT CHECKS =====
        # If ANY constraint violated, return -1000 (REJECT individual)

        if accuracy < config.MIN_ACCURACY_THRESHOLD:
            return -1000.0  # REJECT: insufficient accuracy

        if n_parameters > config.MAX_PARAMETERS:
            return -1000.0  # REJECT: model too large for edge device

        if training_time > config.MAX_TRAINING_TIME:
            return -1000.0  # REJECT: training too slow

        if inference_time > config.MAX_INFERENCE_TIME:
            return -1000.0  # REJECT: inference too slow

        # ===== SOFT OBJECTIVES (all scores normalized to [0, 1]) =====

        # Normalize accuracy above threshold to [0, 1] range
        # This ensures better solutions get higher scores
        accuracy_normalized = (accuracy - config.MIN_ACCURACY_THRESHOLD) / (1.0 - config.MIN_ACCURACY_THRESHOLD)
        accuracy_normalized = max(0.0, min(1.0, accuracy_normalized))

        # Size penalty: normalized to [0, 1], where 0 = smallest, 1 = largest allowed
        size_penalty = n_parameters / config.MAX_PARAMETERS

        # Inference time penalty: normalized to [0, 1]
        inference_penalty = inference_time / config.MAX_INFERENCE_TIME

        # Training time penalty: normalized to [0, 1]
        training_penalty = training_time / config.MAX_TRAINING_TIME

        # ===== MULTI-OBJECTIVE FITNESS CALCULATION =====
        # Formula ensures fitness range for valid solutions
        # Best case: 1.0 * 1.0 - 0.5 * 0.0 - 0.2 * 0.0 - 0.1 * 0.0 = 1.0
        # Worst valid: 1.0 * 0.0 - 0.5 * 1.0 - 0.2 * 1.0 - 0.1 * 1.0 = -0.8
        fitness = (
            config.W_ACCURACY * accuracy_normalized -
            config.W_SIZE * size_penalty -
            config.W_INFERENCE_TIME * inference_penalty -
            config.W_TRAINING_TIME * training_penalty
        )

        return fitness

    @staticmethod
    def evaluate_model(model, train_data, val_data, test_data):
        """
        Evaluate model and calculate fitness.

        Eliminates code duplication by reusing NeuralNetworkBuilder.train_and_evaluate().

        Args:
            model: Compiled Keras model
            train_data: (X_train, y_train) tuple
            val_data: (X_val, y_val) tuple
            test_data: (X_test, y_test) tuple

        Returns:
            float: Fitness score
        """
        # Use standard training pipeline from neural_network.py
        metrics = NeuralNetworkBuilder.train_and_evaluate(
            model, train_data, val_data, test_data
        )

        n_parameters = model.count_params()
        fitness = FitnessCalculator.calculate_fitness(metrics, n_parameters)

        return fitness