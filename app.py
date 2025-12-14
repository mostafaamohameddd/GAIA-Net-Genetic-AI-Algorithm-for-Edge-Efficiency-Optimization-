"""
Flask Application for IA Project - Neural Network Optimization with Genetic Algorithms
"""

import os
import sys
import json
import logging
import threading
import time
import pickle
from datetime import datetime
from pathlib import Path
from functools import wraps

import numpy as np
import pandas as pd

from flask import Flask, render_template, request, jsonify, session, send_file
from flask_session import Session
import secrets

# Import GA components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_loader import prepare_data
from genetic_algorithm import GeneticAlgorithmOptimizer
from neural_network import NeuralNetworkBuilder
from ga_service import GAService

# ===== FLASK SETUP =====
app = Flask(__name__, static_folder='static', static_url_path='/static')

app.config['SECRET_KEY'] = secrets.token_hex(32)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['PERMANENT_SESSION_LIFETIME'] = 3600
Session(app)

# ===== GLOBAL STATE =====
ga_state = {
    'running': False,
    'progress': 0,
    'current_generation': 0,
    'total_generations': 0,
    'start_time': None,
    'error': None
}

ga_state_lock = threading.Lock()

# ===== LOGGING SETUP =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('green_ai.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ===== PATHS =====
RESULTS_DIR = Path(config.RESULTS_DIR)
PLOTS_DIR = RESULTS_DIR / "plots"

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# ===== CSRF PROTECTION =====
def generate_csrf_token():
    """Generate a secure CSRF token for the session"""
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(32)
    return session['csrf_token']


def verify_csrf_token(token):
    """Verify CSRF token from request"""
    return token == session.get('csrf_token', None)


# ===== API RESPONSE HELPER =====
def api_response(status, message='', data=None, code=200):
    """
    Format API response
    
    Args:
        status: 'success' or 'error'
        message: Optional message string
        data: Optional data object
        code: HTTP status code
        
    Returns:
        Tuple of (JSON response, HTTP code)
    """
    response = {'status': status}
    if message:
        response['message'] = message
    if data is not None:
        response['data'] = data
    
    # Add CSRF token to all responses
    response['csrf_token'] = generate_csrf_token()
    
    return jsonify(response), code


# ===== DECORATORS =====
def handle_errors(f):
    """Decorator for error handling in routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {str(e)}", exc_info=True)
            return api_response('error', f'Server Error: {str(e)}', None, 500)
    return decorated_function


# ===== ROUTES - Pages =====

@app.route('/')
def index():
    """Home page"""
    logger.info("Homepage accessed")
    return render_template('index.html')


@app.route('/optimizer')
def optimizer():
    """Optimizer configuration page"""
    logger.info("Optimizer page accessed")
    return render_template('optimizer.html')


@app.route('/dashboard')
def dashboard():
    """Dashboard view"""
    logger.info("Dashboard accessed")
    return render_template('dashboard.html')


@app.route('/explanation')
def explanation():
    """Problem explanation page"""
    logger.info("Explanation page accessed")
    return render_template('explanation.html')


@app.route('/architecture')
def architecture_page():
    """Architecture visualization page"""
    logger.info("Architecture page accessed")
    return render_template('analysis.html')


# ===== API ROUTES - Configuration =====

@app.route('/api/config', methods=['GET'])
@handle_errors
def api_config():
    """Get current configuration"""
    config_data = GAService.get_configuration()
    return api_response('success', 'Configuration retrieved', config_data, 200)


@app.route('/api/config', methods=['POST'])
@handle_errors
def api_config_update():
    """Update configuration"""
    data = request.get_json()
    if not data:
        return api_response('error', 'Invalid JSON', None, 400)
    
    # Verify CSRF token for POST requests
    token = request.headers.get('X-CSRF-Token') or request.form.get('csrf_token')
    if not verify_csrf_token(token):
        return api_response('error', 'CSRF validation failed', None, 403)
    
    is_valid, message = GAService.update_configuration(data)
    if not is_valid:
        return api_response('error', message, None, 400)
    
    return api_response('success', message, None, 200)


@app.route('/api/config/validate', methods=['GET'])
@handle_errors
def api_config_validate():
    """Validate configuration"""
    is_valid, message = GAService.validate_configuration()
    return api_response('success', message, {'valid': is_valid}, 200)


# ===== API ROUTES - GA Status & Control =====

@app.route('/api/ga/status', methods=['GET'])
@handle_errors
def api_ga_status():
    """Get GA optimization status"""
    with ga_state_lock:
        state_copy = ga_state.copy()
    return api_response('success', 'Status retrieved', state_copy, 200)


@app.route('/api/ga/run', methods=['POST'])
@handle_errors
def api_ga_run():
    """Start GA optimization in background thread"""
    with ga_state_lock:
        if ga_state['running']:
            return api_response('error', 'Optimization already running', None, 400)
        
        ga_state['running'] = True
        ga_state['progress'] = 0
        ga_state['error'] = None
        ga_state['start_time'] = time.time()
        ga_state['current_generation'] = 0
        ga_state['total_generations'] = config.GENERATIONS
    
    # Run optimization in background thread
    thread = threading.Thread(target=run_optimization_thread)
    thread.daemon = True
    thread.start()
    
    logger.info("Optimization started")
    return api_response('success', 'Optimization started', None, 200)


@app.route('/api/ga/stop', methods=['POST'])
@handle_errors
def api_ga_stop():
    """Stop GA optimization"""
    with ga_state_lock:
        ga_state['running'] = False
    
    logger.info("Optimization stopped by user")
    return api_response('success', 'Optimization stopped', None, 200)


def run_optimization_thread():
    """Background thread for GA optimization"""
    try:
        with ga_state_lock:
            ga_state['total_generations'] = config.GENERATIONS
        
        # Load data
        (X_train, y_train), (X_val, y_val), (X_test, y_test), n_features, n_classes = prepare_data()
        train_data = (X_train, y_train)
        val_data = (X_val, y_val)
        test_data = (X_test, y_test)
        
        # Initialize GA
        optimizer = GeneticAlgorithmOptimizer(
            n_features=n_features,
            n_classes=n_classes,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data
        )
        
        # Progress callback to update state
        def update_progress(current_gen, total_gen):
            with ga_state_lock:
                ga_state['current_generation'] = current_gen
                ga_state['progress'] = int((current_gen / total_gen) * 100)
        
        # Run evolution with progress tracking
        best_hof, population, logbook = optimizer.evolve(progress_callback=update_progress)
        
        # Save results
        from utils import EvolutionUtils
        EvolutionUtils.create_results_directory()
        
        if config.SAVE_BEST_GENOME:
            EvolutionUtils.save_best_genome(best_hof[0])
        
        EvolutionUtils.save_evolution_history(logbook)
        
        if config.PLOT_EVOLUTION:
            EvolutionUtils.plot_evolution(logbook)
            EvolutionUtils.plot_statistics_summary(logbook)
            EvolutionUtils.plot_performance_analysis(logbook)
            EvolutionUtils.plot_convergence_diagnostics(logbook)
        
        # Update state on success
        with ga_state_lock:
            ga_state['running'] = False
            ga_state['progress'] = 100
            ga_state['current_generation'] = config.GENERATIONS
        
        logger.info("Optimization completed successfully")
        
    except Exception as e:
        with ga_state_lock:
            ga_state['running'] = False
            ga_state['error'] = str(e)
        logger.error(f"Optimization error: {str(e)}", exc_info=True)


# ===== API ROUTES - Results =====

@app.route('/api/results', methods=['GET'])
@handle_errors
def api_results():
    """Get optimization results"""
    genome_file = RESULTS_DIR / "best_genome.pkl"
    history_file = RESULTS_DIR / "evolution_history.csv"
    
    if not genome_file.exists():
        return api_response('error', 'No results available. Run optimization first.', None, 404)
    
    # Load best genome
    with open(genome_file, 'rb') as f:
        best_genome = pickle.load(f)
    
    # Load history if available
    history_count = 0
    if history_file.exists():
        df = pd.read_csv(history_file)
        history_count = len(df)
    
    results = {
        'genome': str(best_genome),
        'history_count': history_count
    }
    
    return api_response('success', 'Results retrieved', results, 200)


@app.route('/api/architecture', methods=['GET'])
@handle_errors
def api_architecture():
    """Get best architecture"""
    arch, error = GAService.decode_best_architecture()
    
    if error:
        return api_response('error', error, None, 404)
    
    return api_response('success', 'Architecture retrieved', arch, 200)


@app.route('/api/dashboard/summary', methods=['GET'])
@handle_errors
def api_dashboard_summary():
    """Get dashboard summary statistics"""
    history_file = RESULTS_DIR / "evolution_history.csv"
    
    if not history_file.exists():
        return api_response('error', 'No data available', None, 404)
    
    df = pd.read_csv(history_file)
    
    summary = {
        'total_generations': int(df['gen'].max()),
        'total_evaluations': int(df['nevals'].sum()),
        'best_fitness': float(df['max'].max()),
        'final_avg_fitness': float(df['avg'].iloc[-1])
    }
    
    return api_response('success', 'Summary retrieved', summary, 200)


@app.route('/api/plots', methods=['GET'])
@handle_errors
def api_plots_list():
    """List available plots"""
    plots = {}
    
    plot_files = [
        'evolution_fitness.png',
        'statistics_summary.png',
        'performance_analysis.png',
        'convergence_diagnostics.png'
    ]
    
    for plot_file in plot_files:
        plot_path = PLOTS_DIR / plot_file
        if plot_path.exists():
            plots[plot_file.replace('.png', '')] = True
    
    return api_response('success', 'Plots listed', {'plots': plots}, 200)


@app.route('/api/plots/<plot_name>', methods=['GET'])
@handle_errors
def api_plot_image(plot_name):
    """Get specific plot image"""
    plot_path = PLOTS_DIR / f"{plot_name}.png"
    
    if not plot_path.exists():
        return api_response('error', 'Plot not found', None, 404)
    
    return send_file(plot_path, mimetype='image/png')


# ===== ERROR HANDLERS =====

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return api_response('error', 'Resource not found', None, 404)


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return api_response('error', 'Internal server error', None, 500)


# ===== MAIN =====

if __name__ == '__main__':
    logger.info("Starting Green AI Flask Application")
    logger.info(f"Results directory: {config.RESULTS_DIR}")
    logger.info(f"Debug mode: {os.environ.get('FLASK_DEBUG', 'False')}")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Disable reloader for GA threads
    )