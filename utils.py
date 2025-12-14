    @staticmethod
    def create_results_directory():
        """Create results directory if it doesn't exist.
        
        Creates the results directory and plots subdirectory for output files.
        """
        if not os.path.exists(config.RESULTS_DIR):
            os.makedirs(config.RESULTS_DIR)
            os.makedirs(os.path.join(config.RESULTS_DIR, "plots"))
        print(f"Results directory: {config.RESULTS_DIR}")
    
    @staticmethod
    def save_best_genome(best_individual, filename="best_genome.pkl"):
        """Save best genome to file.
        
        Args:
            best_individual: The best chromosome found
            filename: Output filename (default: best_genome.pkl)
        """
        filepath = os.path.join(config.RESULTS_DIR, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(best_individual, f)
        print(f"Best genome saved to {filepath}")
    
    @staticmethod
    def load_best_genome(filename="best_genome.pkl"):
        """Load best genome from file.
        
        Args:
            filename: Input filename (default: best_genome.pkl)
            
        Returns:
            list: The loaded chromosome
        """
        filepath = os.path.join(config.RESULTS_DIR, filename)
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def save_evolution_history(logbook):
        """Save evolution history to CSV.
        
        Args:
            logbook: DEAP logbook with generation statistics
        """
        df = pd.DataFrame(logbook)
        filepath = os.path.join(config.RESULTS_DIR, "evolution_history.csv")
        df.to_csv(filepath, index=False)
        print(f"Evolution history saved to {filepath}")
    
    @staticmethod
    def plot_evolution(logbook):
        """Generate comprehensive fitness evolution plots.
        
        Creates a 2x3 grid of plots showing:
        1. Fitness over generations (max/avg/min)
        2. Generation-to-generation improvements
        3. Fitness convergence rate
        4. Population diversity (max-min gap)
        5. Generational evaluation count
        6. Cumulative fitness distribution
        
        Args:
            logbook: DEAP logbook with generation statistics
        """
        df = pd.DataFrame(logbook)
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # ===== PLOT 1: Fitness Over Generations (Main) =====
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df['gen'], df['max'], 'g-', label='Best Fitness (Max)', 
                linewidth=3, marker='o', markersize=5, markerfacecolor='lightgreen', markeredgewidth=2)
        ax1.plot(df['gen'], df['avg'], 'b--', label='Average Fitness', 
                linewidth=2.5, marker='s', markersize=4, markerfacecolor='lightblue', markeredgewidth=1.5)
        ax1.plot(df['gen'], df['min'], 'r:', label='Worst Fitness (Min)', 
                linewidth=2, marker='^', markersize=4, markerfacecolor='lightcoral', markeredgewidth=1.5)
        
        # Set reasonable Y-axis limits
        ax1.set_ylim(-10, 1.0)
        ax1.fill_between(df['gen'], df['avg'], df['max'], alpha=0.15, color='green', label='Best-Avg Gap')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Constraint Threshold')
        ax1.axhline(y=config.MIN_ACCURACY_THRESHOLD, color='orange', linestyle='--', linewidth=1, 
                   alpha=0.5, label=f'Min Accuracy ({config.MIN_ACCURACY_THRESHOLD})')
        
        ax1.set_xlabel('Generation', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Fitness Score', fontsize=12, fontweight='bold')
        ax1.set_title('GA Evolution: Complete Fitness Progression', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10, framealpha=0.95)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(df['gen'].min() - 1, df['gen'].max() + 1)
        
        # ===== PLOT 2: Generation-to-Generation Fitness Change =====
        ax2 = fig.add_subplot(gs[1, 0])
        changes = df['max'].diff().fillna(0)
        
        colors = []
        for x in changes:
            if x > 0.001:
                colors.append('green')
            elif x < -0.001:
                colors.append('red')
            else:
                colors.append('gray')
        
        ax2.bar(df['gen'], changes, color=colors, alpha=0.7, edgecolor='black', linewidth=0.8)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
        ax2.set_xlabel('Generation', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Fitness Improvement', fontsize=11, fontweight='bold')
        ax2.set_title('Gen-to-Gen Improvement', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Improvement'),
            Patch(facecolor='red', alpha=0.7, label='Degradation'),
            Patch(facecolor='gray', alpha=0.7, label='Stagnation')
        ]
        ax2.legend(handles=legend_elements, loc='best', fontsize=9)
        
        # ===== PLOT 3: Convergence Rate =====
        ax3 = fig.add_subplot(gs[1, 1])
        max_fitness_array = df['max'].values
        convergence_rate = np.gradient(max_fitness_array)
        
        ax3.plot(df['gen'], convergence_rate, 'purple', linewidth=2.5, marker='D', markersize=4)
        ax3.fill_between(df['gen'], convergence_rate, alpha=0.3, color='purple')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax3.set_xlabel('Generation', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Convergence Rate', fontsize=11, fontweight='bold')
        ax3.set_title('Convergence Speed (Derivative)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # ===== PLOT 4: Population Diversity (Max-Min Gap) =====
        ax4 = fig.add_subplot(gs[2, 0])
        diversity = df['max'] - df['min']
        
        ax4.fill_between(df['gen'], diversity, alpha=0.5, color='cyan', label='Max-Min Gap')
        ax4.plot(df['gen'], diversity, 'c-', linewidth=2.5, marker='o', markersize=4)
        ax4.set_xlabel('Generation', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Fitness Gap', fontsize=11, fontweight='bold')
        ax4.set_title('Population Diversity', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(['Diversity Gap'], loc='best', fontsize=9)
        
        # ===== PLOT 5: Evaluation Count per Generation =====
        ax5 = fig.add_subplot(gs[2, 1])
        colors_eval = ['red' if x == config.POPULATION_SIZE else 'blue' for x in df['nevals']]
        
        ax5.bar(df['gen'], df['nevals'], color=colors_eval, alpha=0.7, edgecolor='black', linewidth=0.8)
        ax5.axhline(y=config.POPULATION_SIZE, color='red', linestyle='--', linewidth=1.5, 
                   label=f'Population Size ({config.POPULATION_SIZE})', alpha=0.7)
        ax5.set_xlabel('Generation', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Number of Evaluations', fontsize=11, fontweight='bold')
        ax5.set_title('Evaluations per Generation (Elitism Impact)', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.legend(fontsize=9)
        ax5.set_ylim(0, config.POPULATION_SIZE + 5)
        
        # Save figure
        plt.suptitle('Comprehensive GA Evolution Analysis', fontsize=16, fontweight='bold', y=0.995)
        filepath = os.path.join(config.RESULTS_DIR, "plots", "evolution_fitness.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Comprehensive evolution plot saved to {filepath}")
        plt.close()
    
    @staticmethod
    def plot_statistics_summary(logbook):
        """Generate statistical summary plots.
        
        Creates plots showing:
        1. Fitness distribution boxplot
        2. Statistical measures over time
        3. Improvement histogram
        
        Args:
            logbook: DEAP logbook with generation statistics
        """
        df = pd.DataFrame(logbook)
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # ===== PLOT 1: Fitness Statistics Box Plot =====
        ax1 = axes[0]
        stats_data = [df['min'], df['avg'], df['max']]
        bp = ax1.boxplot(stats_data, labels=['Min', 'Avg', 'Max'], patch_artist=True)
        
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax1.set_ylabel('Fitness Score', fontsize=11, fontweight='bold')
        ax1.set_title('Fitness Statistics Distribution', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # ===== PLOT 2: Key Statistics Over Time =====
        ax2 = axes[1]
        ax2.plot(df['gen'], df['max'], 'g-', linewidth=2.5, marker='o', label='Max', markersize=4)
        ax2.plot(df['gen'], df['avg'], 'b--', linewidth=2, marker='s', label='Average', markersize=3)
        ax2.plot(df['gen'], df['min'], 'r:', linewidth=2, marker='^', label='Min', markersize=3)
        
        ax2.set_xlabel('Generation', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Fitness Score', fontsize=11, fontweight='bold')
        ax2.set_title('Fitness Metrics Over Generations', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-10, 1.0)
        
        # ===== PLOT 3: Improvement Distribution =====
        ax3 = axes[2]
        improvements = df['max'].diff().fillna(0)
        
        ax3.hist(improvements, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        ax3.axvline(improvements.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {improvements.mean():.4f}')
        ax3.axvline(improvements.median(), color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {improvements.median():.4f}')
        
        ax3.set_xlabel('Fitness Improvement', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax3.set_title('Distribution of Gen-to-Gen Improvements', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = os.path.join(config.RESULTS_DIR, "plots", "statistics_summary.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Statistics summary plot saved to {filepath}")
        plt.close()
    
    @staticmethod
    def plot_performance_analysis(logbook):
        """Generate performance analysis plots.
        
        Creates plots showing:
        1. Cumulative best fitness
        2. Generation efficiency (improvement per evaluation)
        3. Elite preservation effectiveness
        
        Args:
            logbook: DEAP logbook with generation statistics
        """
        df = pd.DataFrame(logbook)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # ===== PLOT 1: Cumulative Best Fitness =====
        ax1 = axes[0, 0]
        cumulative_best = df['max'].cummax()
        
        ax1.fill_between(df['gen'], cumulative_best, alpha=0.3, color='green')
        ax1.plot(df['gen'], cumulative_best, 'g-', linewidth=2.5, marker='o', markersize=4, label='Cumulative Best')
        ax1.plot(df['gen'], df['max'], 'b--', linewidth=1.5, alpha=0.6, label='Gen Best', marker='s', markersize=3)
        
        ax1.set_xlabel('Generation', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Best Fitness Found', fontsize=11, fontweight='bold')
        ax1.set_title('Cumulative Best Fitness Progress', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # ===== PLOT 2: Generation Efficiency =====
        ax2 = axes[0, 1]
        # Improvements weighted by number of evaluations
        improvements = df['max'].diff().fillna(0)
        efficiency = improvements / df['nevals']
        
        colors_eff = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in efficiency]
        ax2.bar(df['gen'], efficiency, color=colors_eff, alpha=0.7, edgecolor='black', linewidth=0.8)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        ax2.set_xlabel('Generation', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Efficiency (Improvement/Evaluation)', fontsize=11, fontweight='bold')
        ax2.set_title('Generation Efficiency', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # ===== PLOT 3: Elite Preservation Effectiveness =====
        ax3 = axes[1, 0]
        elite_effectiveness = 1 - (df['min'] / df['max']).replace([np.inf, -np.inf], 0)
        elite_effectiveness = elite_effectiveness.clip(0, 1)
        
        ax3.fill_between(df['gen'], elite_effectiveness, alpha=0.4, color='orange')
        ax3.plot(df['gen'], elite_effectiveness, 'orange', linewidth=2.5, marker='D', markersize=4)
        ax3.set_xlabel('Generation', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Elite Effectiveness', fontsize=11, fontweight='bold')
        ax3.set_title('Elite Preservation Effectiveness', fontsize=12, fontweight='bold')
        ax3.set_ylim(-0.1, 1.1)
        ax3.grid(True, alpha=0.3)
        
        # ===== PLOT 4: Convergence Indicator =====
        ax4 = axes[1, 1]
        # Standard deviation as measure of convergence
        population_std = (df['max'] - df['min']).rolling(window=3, min_periods=1).std()
        
        ax4.fill_between(df['gen'], population_std, alpha=0.3, color='purple')
        ax4.plot(df['gen'], population_std, 'purple', linewidth=2.5, marker='o', markersize=4)
        ax4.set_xlabel('Generation', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Population Spread (std)', fontsize=11, fontweight='bold')
        ax4.set_title('Convergence Indicator (Lower = More Converged)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(config.RESULTS_DIR, "plots", "performance_analysis.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Performance analysis plot saved to {filepath}")
        plt.close()
    
    @staticmethod
    def plot_convergence_diagnostics(logbook):
        """Generate convergence diagnostic plots.
        
        Creates plots for:
        1. Fitness improvement rate (log scale)
        2. Normalized improvement trend
        3. Early vs late generation analysis
        
        Args:
            logbook: DEAP logbook with generation statistics
        """
        df = pd.DataFrame(logbook)
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # ===== PLOT 1: Fitness Improvement Rate (Log Scale) =====
        ax1 = axes[0]
        improvements = df['max'].diff().fillna(df['max'].iloc[0]).abs()
        improvements = improvements.replace(0, 1e-6)  # Avoid log(0)
        
        ax1.semilogy(df['gen'], improvements, 'b-', linewidth=2.5, marker='o', markersize=4)
        ax1.set_xlabel('Generation', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Abs(Improvement) [log scale]', fontsize=11, fontweight='bold')
        ax1.set_title('Improvement Rate Decay', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, which='both')
        
        # ===== PLOT 2: Normalized Improvement Trend =====
        ax2 = axes[1]
        max_improvement = improvements.max()
        normalized_improvements = improvements / max_improvement if max_improvement > 0 else improvements
        
        ax2.plot(df['gen'], normalized_improvements, 'g-', linewidth=2.5, marker='s', markersize=4)
        ax2.fill_between(df['gen'], normalized_improvements, alpha=0.3, color='green')
        
        # Exponential decay reference
        decay_ref = np.exp(-df['gen'] / (len(df) * 0.5))
        ax2.plot(df['gen'], decay_ref, 'r--', linewidth=1.5, alpha=0.7, label='Exponential decay reference')
        
        ax2.set_xlabel('Generation', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Normalized Improvement', fontsize=11, fontweight='bold')
        ax2.set_title('Improvement Trend (Normalized)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # ===== PLOT 3: Early vs Late Analysis =====
        ax3 = axes[2]
        midpoint = len(df) // 2
        
        early_improvements = df['max'].diff().fillna(0)[:midpoint].sum()
        late_improvements = df['max'].diff().fillna(0)[midpoint:].sum()
        
        categories = ['Early\nGenerations\n(0-50%)', 'Late\nGenerations\n(50-100%)']
        values = [early_improvements, late_improvements]
        colors_early_late = ['green', 'orange']
        
        bars = ax3.bar(categories, values, color=colors_early_late, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax3.set_ylabel('Cumulative Improvement', fontsize=11, fontweight='bold')
        ax3.set_title('Early vs Late Generation Improvement', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = os.path.join(config.RESULTS_DIR, "plots", "convergence_diagnostics.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Convergence diagnostics plot saved to {filepath}")
        plt.close()
    
    @staticmethod
    def decode_chromosome(chromosome):
        """Decode chromosome into readable architecture parameters.
        
        Args:
            chromosome: The chromosome (list of floats)
        
        Returns:
            dict: Decoded architecture with num_layers, layers details, and learning_rate
        """
        from neural_network import NeuralNetworkBuilder
        
        architecture = NeuralNetworkBuilder.chromosome_to_architecture(chromosome)
        
        decoded = {
            'num_layers': len(architecture['layers']),
            'learning_rate': architecture['learning_rate'],
            'layers': []
        }
        
        for i, layer in enumerate(architecture['layers'], 1):
            decoded['layers'].append({
                'layer_id': i,
                'neurons': layer['neurons'],
                'activation': layer['activation']
            })
        
        return decoded
    
    @staticmethod
    def format_architecture_report(decoded_chromosome):
        """Format decoded chromosome into readable report string.
        
        Args:
            decoded_chromosome: Decoded architecture dict from decode_chromosome()
            
        Returns:
            str: Formatted report string
        """
        
        report = "\nBEST ARCHITECTURE DETAILS\n"
        report += "-" * 70 + "\n"
        report += f"Number of Layers: {decoded_chromosome['num_layers']}\n"
        report += f"Learning Rate: {decoded_chromosome['learning_rate']:.6f}\n\n"
        
        report += "Layer Configuration:\n"
        for layer in decoded_chromosome['layers']:
            report += f"  Layer {layer['layer_id']}: {layer['neurons']} neurons, {layer['activation']} activation\n"
        
        return report
    
    @staticmethod
    def generate_report(best_genome, best_fitness, logbook, n_parameters):
        """Generate final comprehensive report and save to file.
        
        Args:
            best_genome: The best chromosome found
            best_fitness: Its fitness score
            logbook: DEAP logbook with evolution history
            n_parameters: Number of parameters in best model
        """
        
        df = pd.DataFrame(logbook)
        
        # Decode chromosome for detailed architecture info
        decoded = EvolutionUtils.decode_chromosome(best_genome)
        architecture_details = EvolutionUtils.format_architecture_report(decoded)
        
        # Calculate statistics
        improvements = df['max'].diff().fillna(0)
        total_evaluations = sum(df['nevals'])
        
        report = f"""
{'='*70}
GREEN AI: NEURAL NETWORK OPTIMIZATION WITH GENETIC ALGORITHMS
Final Comprehensive Report
{'='*70}

PROJECT CONFIGURATION
{'-'*70}
Dataset: {config.DATASET_NAME}
Population Size: {config.POPULATION_SIZE}
Generations: {config.GENERATIONS}
Mutation Rate: {config.MUTATION_RATE}
Crossover Rate: {config.CROSSOVER_RATE}
Tournament Size: {config.TOURNAMENT_SIZE}
Random Seed: {config.RANDOM_SEED}

MULTI-OBJECTIVE WEIGHTS
{'-'*70}
Accuracy Weight (W_ACCURACY): {config.W_ACCURACY}
Model Size Weight (W_SIZE): {config.W_SIZE}
Inference Time Weight (W_INFERENCE_TIME): {config.W_INFERENCE_TIME}
Training Time Weight (W_TRAINING_TIME): {config.W_TRAINING_TIME}

NEURAL NETWORK HYPERPARAMETERS
{'-'*70}
Dropout Rate: {config.DROPOUT_RATE}
L2 Regularization: {config.L2_REGULARIZER}
Early Stopping Patience: {config.EARLY_STOPPING_PATIENCE}
Max Epochs per Evaluation: {config.MAX_EPOCHS}
Batch Size: {config.BATCH_SIZE}

OPTIMIZATION RESULTS
{'-'*70}
Best Fitness Score: {best_fitness:.6f}
Model Parameters: {n_parameters:,}
Max Parameters Allowed: {config.MAX_PARAMETERS:,}
Parameter Efficiency: {(n_parameters / config.MAX_PARAMETERS * 100):.2f}%

{architecture_details}

Best Genome (Raw Chromosome):
{best_genome}

EVOLUTION STATISTICS
{'-'*70}
Starting Average Fitness: {df['avg'].iloc[0]:.6f}
Final Average Fitness: {df['avg'].iloc[-1]:.6f}
Average Improvement: {df['avg'].iloc[-1] - df['avg'].iloc[0]:.6f}

Starting Best Fitness: {df['max'].iloc[0]:.6f}
Final Best Fitness: {df['max'].iloc[-1]:.6f}
Best Overall Found: {df['max'].max():.6f}

Total Evaluations: {total_evaluations}
Avg Evaluations per Generation: {total_evaluations / len(df):.1f}
Total Improvements: {improvements[improvements > 0.001].shape[0]}
Major Improvements (>0.01): {improvements[improvements > 0.01].shape[0]}

CONVERGENCE ANALYSIS
{'-'*70}
Early Phase (First 25%): {improvements[:max(1, len(df)//4)].sum():.6f}
Late Phase (Last 25%): {improvements[-max(1, len(df)//4):].sum():.6f}
Peak Improvement Generation: {df['gen'].iloc[improvements.idxmax()]}
Peak Improvement Value: {improvements.max():.6f}

CONSTRAINT SATISFACTION
{'-'*70}
Minimum Accuracy Threshold: {config.MIN_ACCURACY_THRESHOLD} (75%)
Maximum Parameters Allowed: {config.MAX_PARAMETERS:,}
Maximum Training Time: {config.MAX_TRAINING_TIME}s
Maximum Inference Time: {config.MAX_INFERENCE_TIME}s

GENERATED PLOTS
{'-'*70}
1. evolution_fitness.png - Comprehensive fitness analysis (6 subplots)
2. statistics_summary.png - Statistical distributions and metrics
3. performance_analysis.png - Efficiency and effectiveness metrics
4. convergence_diagnostics.png - Convergence behavior analysis

NEXT STEPS
{'-'*70}
1. Review generated plots for optimization insights
2. Deploy best genome to edge device
3. Perform cross-validation on unseen test sets
4. Benchmark against manual baselines
5. Test on actual IoT hardware
6. Monitor performance in production

{'='*70}
        """
        
        filepath = os.path.join(config.RESULTS_DIR, "final_report.txt")
        with open(filepath, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"Report saved to {filepath}")
        