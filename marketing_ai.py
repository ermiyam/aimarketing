import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import wandb
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('marketing_ai.log'),
        logging.StreamHandler()
    ]
)

class MarketingAI:
    def __init__(self, config=None):
        self.config = config or {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'validation_split': 0.2,
            'model_path': 'models/marketing_model'
        }
        self.model = None
        self.scaler = StandardScaler()
        
        # Initialize wandb
        wandb.init(
            project="marketing-ai",
            config=self.config
        )

    def build_model(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        self.model = model
        return model

    def preprocess_data(self, data):
        """Preprocess the input data"""
        if isinstance(data, str):
            data = pd.read_csv(data)
        
        # Handle missing values
        data = data.fillna(data.mean())
        
        # Convert categorical variables
        categorical_columns = data.select_dtypes(include=['object']).columns
        data = pd.get_dummies(data, columns=categorical_columns)
        
        return data

    def train(self, X, y):
        """Train the marketing AI model"""
        logging.info("Starting model training...")
        
        # Preprocess features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y,
            test_size=self.config['validation_split'],
            random_state=42
        )
        
        # Build model if not exists
        if self.model is None:
            self.build_model((X_train.shape[1],))
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=(X_val, y_val),
            callbacks=[
                wandb.keras.WandbCallback(),
                tf.keras.callbacks.EarlyStopping(
                    patience=10,
                    restore_best_weights=True
                )
            ]
        )
        
        # Save the model
        self.save_model()
        
        logging.info("Model training completed")
        return history

    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions

    def save_model(self):
        """Save the model and scaler"""
        if not os.path.exists('models'):
            os.makedirs('models')
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{self.config['model_path']}_{timestamp}"
        
        self.model.save(model_path)
        np.save(f"{model_path}_scaler.npy", self.scaler.get_params())
        logging.info(f"Model saved to {model_path}")

    def load_model(self, model_path):
        """Load a saved model and scaler"""
        self.model = tf.keras.models.load_model(model_path)
        scaler_params = np.load(f"{model_path}_scaler.npy", allow_pickle=True).item()
        self.scaler.set_params(**scaler_params)
        logging.info(f"Model loaded from {model_path}")

if __name__ == "__main__":
    # Example usage
    logging.info("Initializing Marketing AI system")
    ai = MarketingAI()
    
    # Add your training data loading and model training code here
    # Example:
    # data = pd.read_csv('marketing_data.csv')
    # X = data.drop('target', axis=1)
    # y = data['target']
    # ai.train(X, y)
