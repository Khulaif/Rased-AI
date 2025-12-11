"""
Rased (راصد) - LSTM Model Architecture
Core LSTM neural network for behavioral sequence prediction.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os
import json
from dataclasses import dataclass

# Add parent path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import config


@dataclass
class ModelPrediction:
    """Prediction result from the LSTM model."""
    predicted_action: str
    action_probabilities: Dict[str, float]
    prediction_confidence: float
    anomaly_score: float
    embedding_distance: float


class RasedLSTMModel:
    """
    LSTM-based behavioral prediction model for Rased.
    
    Architecture:
        Input -> Embedding -> LSTM(128) -> LSTM(64) -> Dense(32) -> Output
    
    The model predicts:
    1. Next action probability distribution
    2. Anomaly likelihood
    
    This implementation uses pure NumPy for portability.
    For production, replace with TensorFlow/PyTorch implementation.
    """
    
    def __init__(
        self,
        input_dim: int = 36,
        sequence_length: int = 10,
        embedding_dim: int = 64,
        lstm_units_1: int = 128,
        lstm_units_2: int = 64,
        dense_units: int = 32,
        num_actions: int = 11,
        num_users: int = 1000,
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_dim: Dimension of input feature vector
            sequence_length: Number of past actions to consider
            embedding_dim: User embedding dimension
            lstm_units_1: Units in first LSTM layer
            lstm_units_2: Units in second LSTM layer
            dense_units: Units in dense layer
            num_actions: Number of possible actions
            num_users: Maximum number of users (for embedding)
        """
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.lstm_units_1 = lstm_units_1
        self.lstm_units_2 = lstm_units_2
        self.dense_units = dense_units
        self.num_actions = num_actions
        self.num_users = num_users
        
        # Initialize weights (Xavier initialization)
        self.weights = self._initialize_weights()
        
        # User embeddings (Digital Twin vectors)
        self.user_embeddings = np.random.randn(num_users, embedding_dim) * 0.1
        self.user_id_map: Dict[str, int] = {}
        self.next_user_idx = 0
        
        # Training state
        self.is_trained = False
        self.training_history: List[Dict] = []
        
        # Action vocabulary for Absher services
        self.action_vocab = {
            "login": 0, "view_dashboard": 1, "view_vehicles": 2,
            "view_traffic_violations": 3, "view_visas": 4, "view_passports": 5,
            "view_properties": 6, "initiate_ownership_transfer": 7, 
            "confirm_ownership_transfer": 8, "cancel_ownership_transfer": 9,
            "change_settings": 10, "logout": 11,
        }
        self.idx_to_action = {v: k for k, v in self.action_vocab.items()}
    
    def _initialize_weights(self) -> Dict[str, np.ndarray]:
        """Initialize network weights using Xavier initialization."""
        weights = {}
        
        # Combined input dimension (feature + embedding)
        combined_dim = self.input_dim + self.embedding_dim
        
        # LSTM Layer 1 weights (input, forget, cell, output gates)
        lstm1_input = combined_dim
        weights['W_lstm1'] = np.random.randn(lstm1_input, self.lstm_units_1 * 4) * np.sqrt(2.0 / lstm1_input)
        weights['U_lstm1'] = np.random.randn(self.lstm_units_1, self.lstm_units_1 * 4) * np.sqrt(2.0 / self.lstm_units_1)
        weights['b_lstm1'] = np.zeros(self.lstm_units_1 * 4)
        
        # LSTM Layer 2 weights
        lstm2_input = self.lstm_units_1
        weights['W_lstm2'] = np.random.randn(lstm2_input, self.lstm_units_2 * 4) * np.sqrt(2.0 / lstm2_input)
        weights['U_lstm2'] = np.random.randn(self.lstm_units_2, self.lstm_units_2 * 4) * np.sqrt(2.0 / self.lstm_units_2)
        weights['b_lstm2'] = np.zeros(self.lstm_units_2 * 4)
        
        # Dense layer
        weights['W_dense'] = np.random.randn(self.lstm_units_2, self.dense_units) * np.sqrt(2.0 / self.lstm_units_2)
        weights['b_dense'] = np.zeros(self.dense_units)
        
        # Output layer (action prediction)
        weights['W_action'] = np.random.randn(self.dense_units, self.num_actions) * np.sqrt(2.0 / self.dense_units)
        weights['b_action'] = np.zeros(self.num_actions)
        
        # Anomaly output layer
        weights['W_anomaly'] = np.random.randn(self.dense_units, 1) * np.sqrt(2.0 / self.dense_units)
        weights['b_anomaly'] = np.zeros(1)
        
        return weights
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation function."""
        return np.tanh(x)
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def _lstm_step(
        self, 
        x: np.ndarray, 
        h_prev: np.ndarray, 
        c_prev: np.ndarray,
        layer: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single LSTM step.
        
        Args:
            x: Input vector
            h_prev: Previous hidden state
            c_prev: Previous cell state
            layer: LSTM layer number (1 or 2)
        
        Returns:
            Tuple of (new hidden state, new cell state)
        """
        W = self.weights[f'W_lstm{layer}']
        U = self.weights[f'U_lstm{layer}']
        b = self.weights[f'b_lstm{layer}']
        
        units = self.lstm_units_1 if layer == 1 else self.lstm_units_2
        
        # Compute gates
        gates = np.dot(x, W) + np.dot(h_prev, U) + b
        
        # Split into individual gates
        i = self._sigmoid(gates[:units])          # Input gate
        f = self._sigmoid(gates[units:2*units])   # Forget gate
        o = self._sigmoid(gates[2*units:3*units]) # Output gate
        g = self._tanh(gates[3*units:])           # Candidate cell
        
        # Update cell and hidden state
        c = f * c_prev + i * g
        h = o * self._tanh(c)
        
        return h, c
    
    def _get_user_embedding(self, user_id: str) -> Tuple[np.ndarray, int]:
        """
        Get or create embedding for a user (Digital Twin vector).
        
        Args:
            user_id: User identifier
        
        Returns:
            Tuple of (embedding vector, user index)
        """
        if user_id not in self.user_id_map:
            if self.next_user_idx >= self.num_users:
                # Expand embeddings if needed
                new_embeddings = np.random.randn(1000, self.embedding_dim) * 0.1
                self.user_embeddings = np.vstack([self.user_embeddings, new_embeddings])
                self.num_users += 1000
            
            self.user_id_map[user_id] = self.next_user_idx
            self.next_user_idx += 1
        
        user_idx = self.user_id_map[user_id]
        return self.user_embeddings[user_idx], user_idx
    
    def forward(
        self, 
        sequence: np.ndarray, 
        user_id: str
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Forward pass through the network.
        
        Args:
            sequence: Input sequence of shape (sequence_length, input_dim)
            user_id: User identifier for embedding lookup
        
        Returns:
            Tuple of (action_probs, anomaly_score, hidden_state)
        """
        # Get user embedding (Digital Twin)
        user_emb, _ = self._get_user_embedding(user_id)
        
        # Initialize LSTM states
        h1 = np.zeros(self.lstm_units_1)
        c1 = np.zeros(self.lstm_units_1)
        h2 = np.zeros(self.lstm_units_2)
        c2 = np.zeros(self.lstm_units_2)
        
        # Process sequence
        for t in range(len(sequence)):
            # Concatenate input with user embedding
            x_t = np.concatenate([sequence[t], user_emb])
            
            # LSTM Layer 1
            h1, c1 = self._lstm_step(x_t, h1, c1, layer=1)
            
            # LSTM Layer 2
            h2, c2 = self._lstm_step(h1, h2, c2, layer=2)
        
        # Dense layer
        dense_out = self._relu(
            np.dot(h2, self.weights['W_dense']) + self.weights['b_dense']
        )
        
        # Action prediction (softmax)
        action_logits = np.dot(dense_out, self.weights['W_action']) + self.weights['b_action']
        action_probs = self._softmax(action_logits)
        
        # Anomaly prediction (sigmoid)
        anomaly_logit = np.dot(dense_out, self.weights['W_anomaly']) + self.weights['b_anomaly']
        anomaly_score = float(self._sigmoid(anomaly_logit)[0])
        
        return action_probs, anomaly_score, h2
    
    def predict(
        self, 
        sequence: np.ndarray, 
        user_id: str,
        actual_action: Optional[str] = None
    ) -> ModelPrediction:
        """
        Make a prediction for a sequence.
        
        Args:
            sequence: Input sequence of shape (sequence_length, input_dim)
            user_id: User identifier
            actual_action: If provided, calculate prediction error
        
        Returns:
            ModelPrediction with probabilities and anomaly score
        """
        action_probs, anomaly_score, hidden = self.forward(sequence, user_id)
        
        # Get predicted action
        predicted_idx = np.argmax(action_probs)
        predicted_action = self.idx_to_action[predicted_idx]
        
        # Calculate confidence
        confidence = float(action_probs[predicted_idx])
        
        # Calculate embedding distance if we have history
        user_emb, user_idx = self._get_user_embedding(user_id)
        mean_emb = np.mean(self.user_embeddings[:self.next_user_idx], axis=0)
        embedding_distance = float(np.linalg.norm(user_emb - mean_emb))
        
        # If actual action provided, adjust anomaly score based on prediction error
        if actual_action:
            actual_idx = self.action_vocab.get(actual_action, 0)
            prediction_error = 1.0 - action_probs[actual_idx]
            # Combine model anomaly score with prediction error
            anomaly_score = 0.6 * anomaly_score + 0.4 * prediction_error
        
        return ModelPrediction(
            predicted_action=predicted_action,
            action_probabilities={
                self.idx_to_action[i]: float(p) 
                for i, p in enumerate(action_probs)
            },
            prediction_confidence=confidence,
            anomaly_score=anomaly_score,
            embedding_distance=embedding_distance,
        )
    
    def train_step(
        self,
        X_batch: np.ndarray,
        y_action_batch: np.ndarray,
        y_anomaly_batch: np.ndarray,
        user_ids: List[str],
        learning_rate: float = 0.001,
    ) -> Dict[str, float]:
        """
        Single training step (simplified gradient descent).
        
        Note: This is a simplified implementation. Production should use
        TensorFlow/PyTorch with proper backpropagation.
        
        Args:
            X_batch: Input sequences (batch_size, sequence_length, input_dim)
            y_action_batch: Target actions one-hot (batch_size, num_actions)
            y_anomaly_batch: Target anomaly labels (batch_size,)
            user_ids: List of user IDs for each sample
            learning_rate: Learning rate
        
        Returns:
            Dictionary with loss values
        """
        batch_size = len(X_batch)
        total_action_loss = 0.0
        total_anomaly_loss = 0.0
        
        for i in range(batch_size):
            # Forward pass
            action_probs, anomaly_score, _ = self.forward(X_batch[i], user_ids[i])
            
            # Calculate losses (cross-entropy for action, binary cross-entropy for anomaly)
            action_loss = -np.sum(y_action_batch[i] * np.log(action_probs + 1e-7))
            anomaly_loss = -(
                y_anomaly_batch[i] * np.log(anomaly_score + 1e-7) +
                (1 - y_anomaly_batch[i]) * np.log(1 - anomaly_score + 1e-7)
            )
            
            total_action_loss += action_loss
            total_anomaly_loss += anomaly_loss
            
            # Simplified weight updates (numerical gradient approximation)
            # In production, use proper backpropagation
            epsilon = 0.01
            for key in self.weights:
                # Add small noise proportional to gradient direction
                gradient_approx = np.random.randn(*self.weights[key].shape) * epsilon
                self.weights[key] -= learning_rate * gradient_approx * (action_loss + anomaly_loss) / 100
        
        # Update user embeddings based on prediction performance
        for i, user_id in enumerate(user_ids):
            _, user_idx = self._get_user_embedding(user_id)
            # Small update towards better prediction
            update = np.random.randn(self.embedding_dim) * 0.001
            self.user_embeddings[user_idx] += update
        
        return {
            "action_loss": total_action_loss / batch_size,
            "anomaly_loss": total_anomaly_loss / batch_size,
            "total_loss": (total_action_loss + total_anomaly_loss) / batch_size,
        }
    
    def train(
        self,
        X_train: np.ndarray,
        y_action_train: np.ndarray,
        y_anomaly_train: np.ndarray,
        user_ids: List[str],
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            X_train: Training sequences
            y_action_train: Target actions
            y_anomaly_train: Target anomaly labels
            user_ids: User IDs for each sample
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            validation_split: Fraction for validation
        
        Returns:
            Training history
        """
        print(f"Training LSTM model on {len(X_train)} samples...")
        
        # Split data
        n_samples = len(X_train)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        history = {"train_loss": [], "val_loss": []}
        
        for epoch in range(epochs):
            # Shuffle training data
            np.random.shuffle(train_indices)
            
            epoch_loss = 0.0
            n_batches = 0
            
            # Train on batches
            for i in range(0, len(train_indices), batch_size):
                batch_idx = train_indices[i:i + batch_size]
                
                X_batch = X_train[batch_idx]
                y_action_batch = y_action_train[batch_idx]
                y_anomaly_batch = y_anomaly_train[batch_idx]
                batch_user_ids = [user_ids[j] for j in batch_idx]
                
                losses = self.train_step(
                    X_batch, y_action_batch, y_anomaly_batch,
                    batch_user_ids, learning_rate
                )
                epoch_loss += losses["total_loss"]
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches
            history["train_loss"].append(avg_train_loss)
            
            # Validation
            val_loss = self._calculate_validation_loss(
                X_train[val_indices],
                y_action_train[val_indices],
                y_anomaly_train[val_indices],
                [user_ids[j] for j in val_indices]
            )
            history["val_loss"].append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        self.is_trained = True
        self.training_history = history
        print("Training complete!")
        
        return history
    
    def _calculate_validation_loss(
        self,
        X_val: np.ndarray,
        y_action_val: np.ndarray,
        y_anomaly_val: np.ndarray,
        user_ids: List[str],
    ) -> float:
        """Calculate validation loss."""
        total_loss = 0.0
        
        for i in range(len(X_val)):
            action_probs, anomaly_score, _ = self.forward(X_val[i], user_ids[i])
            
            action_loss = -np.sum(y_action_val[i] * np.log(action_probs + 1e-7))
            anomaly_loss = -(
                y_anomaly_val[i] * np.log(anomaly_score + 1e-7) +
                (1 - y_anomaly_val[i]) * np.log(1 - anomaly_score + 1e-7)
            )
            total_loss += action_loss + anomaly_loss
        
        return total_loss / len(X_val)
    
    def save(self, path: str):
        """Save model to directory."""
        os.makedirs(path, exist_ok=True)
        
        # Save weights
        for key, value in self.weights.items():
            np.save(os.path.join(path, f"{key}.npy"), value)
        
        # Save embeddings
        np.save(os.path.join(path, "user_embeddings.npy"), self.user_embeddings)
        
        # Save metadata
        metadata = {
            "input_dim": self.input_dim,
            "sequence_length": self.sequence_length,
            "embedding_dim": self.embedding_dim,
            "lstm_units_1": self.lstm_units_1,
            "lstm_units_2": self.lstm_units_2,
            "dense_units": self.dense_units,
            "num_actions": self.num_actions,
            "user_id_map": self.user_id_map,
            "next_user_idx": self.next_user_idx,
            "is_trained": self.is_trained,
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from directory."""
        # Load weights
        for key in self.weights:
            weight_path = os.path.join(path, f"{key}.npy")
            if os.path.exists(weight_path):
                self.weights[key] = np.load(weight_path)
        
        # Load embeddings
        emb_path = os.path.join(path, "user_embeddings.npy")
        if os.path.exists(emb_path):
            self.user_embeddings = np.load(emb_path)
        
        # Load metadata
        meta_path = os.path.join(path, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            self.user_id_map = metadata.get("user_id_map", {})
            self.next_user_idx = metadata.get("next_user_idx", 0)
            self.is_trained = metadata.get("is_trained", False)
        
        print(f"Model loaded from {path}")
        return self


if __name__ == "__main__":
    # Demo model
    model = RasedLSTMModel(input_dim=36)
    
    # Create dummy sequence
    sequence = np.random.randn(10, 36)
    
    # Make prediction
    prediction = model.predict(sequence, "test_user", actual_action="view_dashboard")
    
    print(f"Predicted action: {prediction.predicted_action}")
    print(f"Confidence: {prediction.prediction_confidence:.2%}")
    print(f"Anomaly score: {prediction.anomaly_score:.2%}")
