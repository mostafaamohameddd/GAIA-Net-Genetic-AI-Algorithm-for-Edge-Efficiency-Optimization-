"""
GA Service - Business logic layer for Genetic Algorithm operations.

Provides configuration management, architecture decoding, and result formatting.
"""

import logging
import pickle
from pathlib import Path
import config
from neural_network import NeuralNetworkBuilder

logger = logging.getLogger(__name__)
RESULTS_DIR = Path(config.RESULTS_DIR)


class GAService:
    """Service layer for GA operations."""
    
    @staticmethod
    def get_configuration():
        """
        Get current GA configuration.
        
        Returns:
            dict: Configuration parameters
        """
        return {
            'dataset': config.DATASET_NAME,
            'population_size': config.POPULATION_SIZE,
            'generations': config.GENERATIONS,
            'mutation_rate': config.MUTATION_RATE,
            'crossover_rate': config.CROSSOVER_RATE,
            'elite_size': config.ELITE_SIZE,
            'tournament_size': config.TOURNAMENT_SIZE,
            'min_accuracy': config.MIN_ACCURACY_THRESHOLD,
            'max_parameters': config.MAX_PARAMETERS,
            'w_accuracy': config.W_ACCURACY,
            'w_size': config.W_SIZE,
            'w_inference_time': config.W_INFERENCE_TIME,
            'w_training_time': config.W_TRAINING_TIME
        }
    
    @staticmethod
    def update_configuration(new_config):
        """
        Update GA configuration (runtime update).
        
        Args:
            new_config: dict with configuration parameters
            
        Returns:
            tuple: (is_valid, message)
        """
        try:
            # Validate and update config
            if 'population_size' in new_config:
                if new_config['population_size'] < 5 or new_config['population_size'] > 100:
                    return False, "Population size must be between 5 and 100"
                config.POPULATION_SIZE = new_config['population_size']
            
            if 'generations' in new_config:
                if new_config['generations'] < 1 or new_config['generations'] > 200:
                    return False, "Generations must be between 1 and 200"
                config.GENERATIONS = new_config['generations']
            
            if 'mutation_rate' in new_config:
                if new_config['mutation_rate'] < 0 or new_config['mutation_rate'] > 1:
                    return False, "Mutation rate must be between 0 and 1"
                config.MUTATION_RATE = new_config['mutation_rate']
            
            if 'crossover_rate' in new_config:
                if new_config['crossover_rate'] < 0 or new_config['crossover_rate'] > 1:
                    return False, "Crossover rate must be between 0 and 1"
                config.CROSSOVER_RATE = new_config['crossover_rate']
            
            if 'elite_size' in new_config:
                if new_config['elite_size'] < 1 or new_config['elite_size'] > config.POPULATION_SIZE:
                    return False, f"Elite size must be between 1 and {config.POPULATION_SIZE}"
                config.ELITE_SIZE = new_config['elite_size']
            
            if 'dataset_name' in new_config:
                valid_datasets = ['iris', 'breast_cancer', 'wine']
                if new_config['dataset_name'] not in valid_datasets:
                    return False, f"Dataset must be one of {valid_datasets}"
                config.DATASET_NAME = new_config['dataset_name']
            
            return True, "Configuration updated successfully"
            
        except Exception as e:
            logger.error(f"Error updating configuration: {str(e)}")
            return False, f"Error: {str(e)}"
    
    @staticmethod
    def validate_configuration():
        """
        Validate current configuration.
        
        Returns:
            tuple: (is_valid, message)
        """
        try:
            config.validate_config()
            return True, "Configuration is valid"
        except ValueError as e:
            return False, str(e)
    
    @staticmethod
    def decode_best_architecture():
        """
        Decode and format best neural network architecture.
        
        Returns:
            tuple: (formatted_architecture, error_message)
        """
        try:
            genome_file = RESULTS_DIR / "best_genome.pkl"
            
            if not genome_file.exists():
                return None, "No best genome found. Run optimization first."
            
            # Load the best genome
            with open(genome_file, 'rb') as f:
                best_genome = pickle.load(f)
            
            # Decode chromosome to architecture
            architecture = NeuralNetworkBuilder.chromosome_to_architecture(best_genome)
            
            # Format for display
            formatted = {
                'num_layers': len(architecture['layers']),
                'learning_rate': architecture['learning_rate'],
                'layers': [],
                'n_features': 4,  # Will be dynamic in production
                'n_classes': 3,   # Will be dynamic in production
            }
            
            for i, layer in enumerate(architecture['layers'], 1):
                formatted['layers'].append({
                    'layer_id': i,
                    'neurons': layer['neurons'],
                    'activation': layer['activation']
                })
            
            return formatted, None
            
        except FileNotFoundError:
            return None, "Best genome file not found."
        except Exception as e:
            logger.error(f"Error decoding architecture: {str(e)}")
            return None, f"Error decoding architecture: {str(e)}"
    
    @staticmethod
    def get_architecture_html():
        """
        Get formatted HTML for best architecture visualization.
        
        Returns:
            str: HTML string for architecture display
        """
        arch, error = GAService.decode_best_architecture()
        
        if error:
            return f"""
            <div class='alert alert-warning'>
                <i class='fas fa-exclamation-circle'></i>
                {error}
            </div>
            """
        
        if not arch:
            return """
            <div class='alert alert-info'>
                <i class='fas fa-info-circle'></i>
                No architecture available yet. Run optimization to generate best model.
            </div>
            """
        
        # Build HTML
        html = """
        <div class='architecture-display'>
            <style>
                .architecture-display {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 15px;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 8px;
                }
                .io-layer {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    padding: 15px 25px;
                    background: #e3f2fd;
                    border: 2px solid #2196F3;
                    border-radius: 8px;
                    min-width: 150px;
                }
                .io-label {
                    font-weight: bold;
                    color: #1976D2;
                    font-size: 14px;
                }
                .io-shape {
                    color: #0d47a1;
                    font-size: 18px;
                    font-weight: bold;
                    margin-top: 5px;
                }
                .arrow {
                    font-size: 24px;
                    color: #666;
                    margin: 5px 0;
                }
                .layer {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    padding: 15px 20px;
                    background: #f3e5f5;
                    border: 2px solid #7c4dff;
                    border-radius: 8px;
                    min-width: 140px;
                }
                .layer-title {
                    font-weight: bold;
                    color: #512da8;
                    font-size: 13px;
                }
                .layer-neurons {
                    color: #7c4dff;
                    font-size: 16px;
                    font-weight: bold;
                    margin-top: 5px;
                }
                .layer-activation {
                    color: #512da8;
                    font-size: 12px;
                    margin-top: 5px;
                    background: rgba(124, 77, 255, 0.1);
                    padding: 3px 8px;
                    border-radius: 4px;
                }
                .architecture-info {
                    margin-top: 20px;
                    padding: 15px;
                    background: white;
                    border: 1px solid #ddd;
                    border-radius: 6px;
                    width: 100%;
                }
                .architecture-info p {
                    margin: 8px 0;
                    font-size: 14px;
                }
                .architecture-info code {
                    background: #f5f5f5;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-family: monospace;
                }
            </style>

            <!-- INPUT LAYER -->
            <div class='io-layer'>
                <span class='io-label'>INPUT</span>
                <span class='io-shape'>{n_features}</span>
                <span style='font-size: 12px; color: #1976D2;'>features</span>
            </div>
        """.format(n_features=arch.get('n_features', '?'))
        
        # Add each hidden layer
        for layer in arch.get('layers', []):
            html += """
            <div class='arrow'>↓</div>
            <div class='layer'>
                <div class='layer-title'>Layer {layer_id}</div>
                <div class='layer-neurons'>{neurons} neurons</div>
                <div class='layer-activation'>{activation}</div>
            </div>
            """.format(
                layer_id=layer['layer_id'],
                neurons=layer['neurons'],
                activation=layer['activation'].upper()
            )
        
        # Output layer
        html += """
            <div class='arrow'>↓</div>
            <div class='io-layer'>
                <span class='io-label'>OUTPUT</span>
                <span class='io-shape'>{n_classes}</span>
                <span style='font-size: 12px; color: #1976D2;'>classes</span>
            </div>
        </div>

        <!-- ARCHITECTURE INFO -->
        <div class='architecture-info'>
            <p>
                <strong>Total Layers:</strong> {num_layers}
            </p>
            <p>
                <strong>Learning Rate:</strong> <code>{learning_rate:.6f}</code>
            </p>
            <p>
                <strong>Activation Functions Used:</strong>
                {activations_used}
            </p>
        </div>
        """.format(
            n_classes=arch.get('n_classes', '?'),
            num_layers=arch['num_layers'],
            learning_rate=arch['learning_rate'],
            activations_used=', '.join(
                sorted(set(
                    layer['activation'].upper()
                    for layer in arch.get('layers', [])
                ))
            )
        )
        
        return html