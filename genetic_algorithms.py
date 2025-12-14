"""
Genetic Algorithm Optimizer for Neural Network Architecture Search.

This module implements a complete genetic algorithm using DEAP framework
to optimize neural network architectures for edge deployment.
"""

import random
import numpy as np
import logging
from typing import Tuple, List
from deap import base, creator, tools

import config
from neural_network import NeuralNetworkBuilder
from fitness_calculator import FitnessCalculator

logger = logging.getLogger(__name__)


def create_chromosome() -> list:
    """
    Create a random chromosome representing a neural network architecture.
    
    Chromosome structure:
        [num_layers, neuron_1, activation_1, ..., neuron_n, activation_n, learning_rate]
    
    Returns:
        list: Chromosome with genes for architecture
    """
    chromosome = []
    
    # Gene 0: Number of layers
    num_layers = random.randint(config.MIN_LAYERS, config.MAX_LAYERS)
    chromosome.append(float(num_layers))
    
    # Genes for each layer (neurons + activation)
    for _ in range(num_layers):
        # Neurons: quantized to NEURON_STEP
        neurons = random.randrange(
            config.MIN_NEURONS,
            config.MAX_NEURONS + 1,
            config.NEURON_STEP
        )
        chromosome.append(float(neurons))
        
        # Activation function index
        activation = random.randint(0, len(config.ACTIVATIONS) - 1)
        chromosome.append(float(activation))
    
    # Last gene: Learning rate
    lr = random.uniform(config.MIN_LR, config.MAX_LR)
    chromosome.append(lr)
    
    return chromosome


class GeneticAlgorithmOptimizer:
    """Genetic Algorithm optimizer for neural network architecture search."""
    
    def __init__(self, n_features, n_classes, train_data, val_data, test_data):
        """
        Initialize GA optimizer.
        
        Args:
            n_features: Number of input features
            n_classes: Number of output classes
            train_data: (X_train, y_train) tuple
            val_data: (X_val, y_val) tuple
            test_data: (X_test, y_test) tuple
        """
        self.n_features = n_features
        self.n_classes = n_classes
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        self.toolbox = None
        self.stats = None
        self.hall_of_fame = None
        
        # Setup DEAP framework
        self.setup_deap()
        
        logger.info(f"GA Optimizer initialized: {n_features} features, {n_classes} classes")
    
    def setup_deap(self):
        """Setup DEAP genetic algorithm framework."""
        
        # Create fitness and individual classes
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        
        # Register individual and population creation
        self.toolbox.register("individual", tools.initIterate,
                            creator.Individual, create_chromosome)
        self.toolbox.register("population", tools.initRepeat,
                            list, self.toolbox.individual)
        
        # Register genetic operators
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", self.random_resetting_mutation)
        self.toolbox.register("select", self.tournament_selection_wrapper)
        
        # Setup statistics
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        
        # Hall of Fame to keep best individuals
        self.hall_of_fame = tools.HallOfFame(config.ELITE_SIZE)
        
        logger.info("DEAP framework setup complete")
    
    def evaluate_individual(self, individual) -> Tuple[float]:
        """
        Evaluate fitness of an individual.
        
        Args:
            individual: Chromosome to evaluate
            
        Returns:
            Tuple with single fitness value
        """
        try:
            # Decode chromosome to architecture
            architecture = NeuralNetworkBuilder.chromosome_to_architecture(individual)
            
            # Build model
            model = NeuralNetworkBuilder.build_model(
                self.n_features,
                self.n_classes,
                architecture
            )
            
            # Evaluate fitness
            fitness = FitnessCalculator.evaluate_model(
                model,
                self.train_data,
                self.val_data,
                self.test_data
            )
            
            return (fitness,)
            
        except Exception as e:
            logger.warning(f"Evaluation failed: {str(e)}")
            return (-1000.0,)  # Return penalty for invalid individuals
    
    def random_resetting_mutation(self, individual) -> Tuple[list]:
        """
        Random resetting mutation: mutate each gene independently.
        
        Args:
            individual: Chromosome to mutate
            
        Returns:
            Tuple containing mutated individual
        """
        # Get current number of layers
        num_layers = max(
            config.MIN_LAYERS,
            min(config.MAX_LAYERS, int(round(individual[0])))
        )
        
        # Mutate num_layers gene
        if random.random() < config.MUTATION_RATE:
            individual[0] = float(random.randint(config.MIN_LAYERS, config.MAX_LAYERS))
            num_layers = int(individual[0])
        
        # Mutate layer genes (only for active layers)
        for layer_idx in range(num_layers):
            # Mutate neurons
            neuron_idx = 1 + layer_idx * 2
            if neuron_idx < len(individual) - 1:  # Ensure within bounds
                if random.random() < config.MUTATION_RATE:
                    neurons = random.randrange(
                        config.MIN_NEURONS,
                        config.MAX_NEURONS + 1,
                        config.NEURON_STEP
                    )
                    individual[neuron_idx] = float(neurons)
            
            # Mutate activation
            activation_idx = 2 + layer_idx * 2
            if activation_idx < len(individual) - 1:  # Ensure within bounds
                if random.random() < config.MUTATION_RATE:
                    activation = random.randint(0, len(config.ACTIVATIONS) - 1)
                    individual[activation_idx] = float(activation)
        
        # Mutate learning rate (last gene)
        if random.random() < config.MUTATION_RATE:
            lr = random.uniform(config.MIN_LR, config.MAX_LR)
            individual[-1] = lr
        
        return (individual,)
    
    def tournament_selection(self, population) -> list:
        """
        Tournament selection: select best from random tournament.
        
        Args:
            population: Current population
            
        Returns:
            Selected individual
        """
        tournament = random.sample(population, config.TOURNAMENT_SIZE)
        return max(tournament, key=lambda ind: ind.fitness.values[0])
    
    def tournament_selection_wrapper(self, population, k) -> List:
        """
        Wrapper for tournament selection to select k individuals.
        
        Args:
            population: Current population
            k: Number of individuals to select
            
        Returns:
            List of selected individuals
        """
        return [self.tournament_selection(population) for _ in range(k)]
    
    def evolve(self, progress_callback=None) -> Tuple:
        """
        Run the genetic algorithm evolution.
        
        Args:
            progress_callback: Optional callback function(gen, total_gen) to report progress
        
        Returns:
            Tuple of (hall_of_fame, final_population, logbook)
        """
        logger.info(f"Starting evolution: {config.GENERATIONS} generations, {config.POPULATION_SIZE} population")
        
        # Create initial population
        population = self.toolbox.population(n=config.POPULATION_SIZE)
        
        # Logbook for statistics
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + self.stats.fields
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        self.hall_of_fame.update(population)
        record = self.stats.compile(population)
        logbook.record(gen=0, nevals=len(population), **record)
        logger.info(f"Gen 0: {record}")
        
        # Report initial progress
        if progress_callback:
            progress_callback(0, config.GENERATIONS)
        
        # Evolution loop
        for gen in range(1, config.GENERATIONS + 1):
            # Select next generation
            offspring = self.toolbox.select(population, len(population) - config.ELITE_SIZE)
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < config.CROSSOVER_RATE:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Apply mutation
            for mutant in offspring:
                if random.random() < config.MUTATION_RATE:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate individuals with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Elitism: add best individuals from previous generation
            offspring.extend(self.hall_of_fame.items)
            
            # Update population
            population[:] = offspring
            
            # Update hall of fame and statistics
            self.hall_of_fame.update(population)
            record = self.stats.compile(population)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            
            logger.info(f"Gen {gen}: Max={record['max']:.4f}, Avg={record['avg']:.4f}, Min={record['min']:.4f}")
            
            # Report progress after each generation
            if progress_callback:
                progress_callback(gen, config.GENERATIONS)
        
        logger.info("Evolution complete!")
        logger.info(f"Best fitness: {self.hall_of_fame[0].fitness.values[0]:.4f}")
        
        return self.hall_of_fame, population, logbook