## GAIA-Net: Hardware-Aware NAS for Green IoT Edge
(Genetic AI Algorithm for Edge Efficiency Optimization)

A production-grade Neural Architecture Search (NAS) system that utilizes a Genetic Algorithm (GA) to discover and optimize Deep Learning model architectures for deployment on resource-constrained IoT Edge devices. The primary focus is achieving Green AI by balancing high accuracy with minimized model size and inference time.

The final optimized architecture is packaged and served through a Flask Web Interface, demonstrating deployment readiness and adherence to strict hardware constraints.

Table of Contents
Overview

Key Features

Problem & Solution

Methodology: The Genetic Algorithm

System Architecture

Technologies Used

Key Results & Achievements

Deployment & Usage

Project Structure

License

Author


## 1. Overview
This project implements a complete end-to-end NAS pipeline focused on efficiency. It addresses the critical industry challenge of deploying sophisticated AI on hardware with strict limitations (low memory, low power, slow CPU).

The pipeline covers:

Automated Architecture Discovery using the DEAP framework.

Hardware-Aware Fitness Evaluation with hard constraints on model size and latency.

Model Training and Evaluation using TensorFlow/Keras.

Deployment of the final optimized genome via a Flask dashboard.

The system is engineered to reflect industry-level NAS practices for power-efficient computing.

## 2. Key Features
Hardware-Aware NAS: Automatically searches for architectures that strictly comply with predefined physical constraints (Max Parameters, Max Inference Time).

Green AI Focus: Prioritizes efficiency and size reduction to minimize energy consumption on the Edge (e.g., battery-powered IoT devices).

Multi-Objective Optimization: Fitness function balances four conflicting objectives: Accuracy (Maximize), Size (Minimize), Inference Time (Minimize), and Training Time (Minimize).

Robust GA Implementation: Features variable-length chromosome representation (to encode variable number of layers) and adapted crossover/mutation operators.

Web Deployment: A Flask server hosts the system, allowing users to monitor the evolution progress and view the winning model's architecture and performance metrics.

## 3. Problem & Solution
The ProblemStandard Deep Learning models are often too large and computationally expensive (High FLOPs) to run efficiently on small Edge devices (MCUs, low-power SoCs) due to limited RAM, ROM, and stringent power budgets.The GAIA-Net SolutionWe use the Genetic Algorithm to intelligently explore the immense space of possible neural network architectures, guided by a fitness score that penalizes models exceeding the hardware limits first, before rewarding high accuracy.Hard Constraints Imposed:| Constraint | Value | Purpose || :--- | :--- | :--- || Minimum Accuracy | e.g., $75\%$ | Ensures basic functional performance. || Maximum Parameters | e.g., $10,000$ | Memory footprint constraint. || Maximum Inference Time | e.g., $1.0$ second/sample | Real-time latency constraint. |

## 4. Methodology: The Genetic Algorithm
The system is built around the DEAP framework for evolutionary computation.Chromosome RepresentationThe chromosome (Individual) is a variable-length list representing the entire network:$$[\text{Num\_Layers}, (\text{Neurons}_1, \text{Activation}_1), \dots, (\text{Neurons}_n, \text{Activation}_n), \text{Learning\_Rate}]$$Core OperationsInitialization (create_chromosome): Generates random initial architectures within layer/neuron ranges.Crossover (tools.cxOnePoint): Used due to the variable length of the chromosomes (variable number of layers), ensuring the integrity of the architecture design during mating.Mutation (random_resetting_mutation): Each gene (Layer count, Neurons, Activation, LR) is mutated independently with a specified rate.Fitness Evaluation (FitnessCalculator): The core logic that builds, trains, evaluates, and applies the hard constraints before calculating the soft objective score.

## 5. System Architecture
The project employs a clear separation of concerns, facilitating unit testing and scalability:

Data Layer (data_loader.py): Handles data loading, preprocessing, and train/validation/test splitting.

GA Core (genetic_algorithms.py): Manages the DEAP evolution loop (Selection, Crossover, Mutation).

NN Builder (neural_network.py): Responsible for compiling and configuring Keras models based on the GA's proposed genome.

Fitness Layer (fitness_calculator.py): Crucial Layer. Performs model training, metric measurement (Time, Accuracy, Size), and returns the final constrained fitness score.

Presentation Layer (app.py / Flask): Exposes the evolve function to the web interface and displays real-time progress and final results.

## 6. Technologies Used
Category,Component,Description
Core Languages,Python 3.x,Main development language.
AI / ML,TensorFlow / Keras,Deep Learning framework for model building and training.
Evolutionary Comp.,DEAP Library,Framework used to implement the Genetic Algorithm pipeline.
Data Science,"NumPy, Pandas, Scikit-learn",Data handling and preprocessing utilities.
Web Deployment,Flask,Lightweight Python framework for serving the GA execution and results dashboard.
Testing,"test_endpoints.py, test_ga_functional.py",Ensures robustness of the web application and core GA logic.

## 7. Key Results & Achievements
Successful Edge Model Discovery: Identified a minimal architecture (e.g., two layers, 1,611 parameters) that maintained high classification accuracy while staying well within the $10,000$ parameter limit.
Robustness of GA: Successfully implemented variable-length chromosome support through tailored crossover and mutation operations, solving common instability issues in NAS.
Deployment Readiness: The system provides a final, trained, and optimized model configuration (Best Genome) that can be directly exported (e.g., to TensorFlow Lite) for Edge deployment.
Performance: Achieved high scores (e.g., Fitness $0.5$ to $0.8$) on viable individuals, demonstrating the superior performance of the GA-optimized models compared to randomly generated onesز

## 8. Deployment & Usage
The Deployment Artifact (Best Genome)
The result of the evolutionary process is the Best Genome—the architecture configuration that yields the highest fitness score. This configuration is ready for integration into a target IoT firmware.

Web Interface Deployment
The entire system, including the GA search and result visualization, is managed by a local Flask application:

1.Clone the Repository:
git clone [https://github.com/mostafaamohameddd/GAIA-Net-Genetic-AI-Algorithm-for-Edge-Efficiency-Optimization-]
cd GAIA-Net-Project


2.Setup Environment:

Bash

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

3.Run the Flask Application:
python app.py

4.Access Dashboard: 
Open your browser and navigate to http://127.0.0.1:5000 to start the evolution process and monitor statistics.

## 9. Project Structure

GAIA-Net-Project/
├── templates/                 # Frontend HTML pages (Dashboard, Analysis)
├── static/                    # CSS, JavaScript, and asset files
├── tests/
│   ├── test_endpoints.py      # Flask API unit tests
│   └── test_ga_functional.py  # Core GA logic functional tests
├── app.py                     # Flask entry point and web routing
├── config.py                  # Global variables and constants (Crucial for defining Constraints)
├── data_loader.py             # Data preparation
├── fitness_calculator.py      # Fitness function logic
├── ga_service.py              # Service layer for starting and managing GA execution
├── genetic_algorithms.py      # DEAP implementation
├── neural_network.py          # Keras model builder
├── utils.py                   # Helper functions
├── requirements.txt           # Dependencies list
└── .gitignore                 # Files excluded from GitHub (e.g., __pycache__, venv, results)

## 10. License
This project is licensed under the MIT License

## 11. Author
[Mostafa Mohamed]
