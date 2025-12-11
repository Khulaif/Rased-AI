"""
Rased (راصد) - AI Fraud Detection System
Main package initialization.
"""

__version__ = "1.0.0"
__author__ = "Rased Security Team"
__description__ = "AI-based behavioral fraud detection for government services"

# Convenient imports
from config.settings import config, RiskLevel
from engine.inference_engine import InferenceEngine
from engine.decision_matrix import DecisionMatrix
from models.lstm_model import RasedLSTMModel
from models.digital_twin import DigitalTwinManager

__all__ = [
    'config',
    'RiskLevel',
    'InferenceEngine',
    'DecisionMatrix',
    'RasedLSTMModel',
    'DigitalTwinManager',
]
