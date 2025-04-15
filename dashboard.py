import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from datetime import datetime
import wandb
import os
from typing import Dict, List, Optional
import logging

class TrainingDashboard:
    def __init__(self, wandb_project: str = "mak-research"):
        """Initialize the training dashboard.
        
        Args:
            wandb_project: Weights & Biases project name
        """
        self.wandb_project = wandb_project
        self.setup_logging()
        
        # Initialize session state
        if "metrics" not in st.session_state:
            st.session_state.metrics = []
        if "search_stats" not in st.session_state:
            st.session_state.search_stats = []
        if "model_info" not in st.session_state:
            st.session_state.model_info = {}

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def _load_wandb_data(self) -> pd.DataFrame:
        """Load data from Weights & Biases."""
        try:
            api = wandb.Api()
            runs = api.runs(self.wandb_project)
            
            data = []
            for run in runs:
                history = run.history()
                if not history.empty:
                    history["run_id"] = run.id
                    history["run_name"] = run.name
                    data.append(history)
            
            if data:
                return pd.concat(data, ignore_index=True)
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error loading W&B data: {e}")
            return pd.DataFrame()

    def _load_search_stats(self) -> pd.DataFrame:
        """Load search statistics."""
        try:
            stats_path = Path("./logs/search_stats.json")
            if stats_path.exists():
                with open(stats_path, "r") as f:
                    stats = json.load(f)
                return pd.DataFrame(stats)
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error loading search stats: {e}")
            return pd.DataFrame()

    def _load_model_info(self) -> Dict:
        """Load model information."""
        try:
            info_path = Path("./models/model_info.json")
            if info_path.exists():
                with open(info_path, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"Error loading model info: {e}")
            return {}

    def update_data(self):
        """Update dashboard data."""
        # Load W&B data
        wandb_data = self._load_wandb_data()
        if not wandb_data.empty:
            st.session_state.metrics = wandb_data
        
        # Load search stats
        search_stats = self._load_search_stats()
        if not search_stats.empty:
            st.session_state.search_stats = search_stats
        
        # Load model info
        model_info = self._load_model_info()
        if model_info:
            st.session_state.model_info = model_info

    def render_metrics_plot(self):
        """Render metrics plot."""
        if not st.session_state.metrics:
            st.warning("No metrics data available")
            return
        
        # Select metrics to plot
        metrics = st.multiselect(
            "Select metrics to plot",
            options=st.session_state.metrics.columns,
            default=["loss", "f1", "accuracy"]
        )
        
        # Create plot
        fig = go.Figure()
        for metric in metrics:
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.metrics["_step"],
                    y=st.session_state.metrics[metric],
                    name=metric,
                    mode="lines+markers"
                )
            )
        
        fig.update_layout(
            title="Training Metrics",
            xaxis_title="Step",
            yaxis_title="Value",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig)

    def render_search_stats(self):
        """Render search statistics."""
        if not st.session_state.search_stats:
            st.warning("No search statistics available")
            return
        
        # Create search stats plot
        fig = px.bar(
            st.session_state.search_stats,
            x="timestamp",
            y="num_searches",
            title="Number of Searches Over Time"
        )
        
        st.plotly_chart(fig)
        
        # Display search success rate
        success_rate = (
            st.session_state.search_stats["successful_searches"].sum() /
            st.session_state.search_stats["total_searches"].sum()
        ) * 100
        
        st.metric(
            "Search Success Rate",
            f"{success_rate:.2f}%"
        )

    def render_model_info(self):
        """Render model information."""
        if not st.session_state.model_info:
            st.warning("No model information available")
            return
        
        # Display model details
        st.subheader("Model Information")
        for key, value in st.session_state.model_info.items():
            st.text(f"{key}: {value}")
        
        # Display model size
        model_size = st.session_state.model_info.get("model_size", 0)
        st.metric(
            "Model Size",
            f"{model_size / 1024 / 1024 / 1024:.2f} GB"
        )

    def render_training_progress(self):
        """Render training progress."""
        if not st.session_state.metrics:
            st.warning("No training progress data available")
            return
        
        # Get latest metrics
        latest = st.session_state.metrics.iloc[-1]
        
        # Display progress
        st.subheader("Training Progress")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Epoch",
                f"{latest.get('epoch', 0):.1f}"
            )
        
        with col2:
            st.metric(
                "Current Loss",
                f"{latest.get('loss', 0):.4f}"
            )
        
        with col3:
            st.metric(
                "Best F1 Score",
                f"{latest.get('best_f1', 0):.4f}"
            )
        
        # Display progress bar
        progress = latest.get("epoch", 0) / st.session_state.model_info.get("total_epochs", 1)
        st.progress(progress)

    def render(self):
        """Render the dashboard."""
        st.title("Mak ReSearch Training Dashboard")
        
        # Update data
        self.update_data()
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Training Progress",
            "Metrics",
            "Search Statistics",
            "Model Information"
        ])
        
        with tab1:
            self.render_training_progress()
        
        with tab2:
            self.render_metrics_plot()
        
        with tab3:
            self.render_search_stats()
        
        with tab4:
            self.render_model_info()
        
        # Add refresh button
        if st.button("Refresh Data"):
            self.update_data()
            st.experimental_rerun()

if __name__ == "__main__":
    # Initialize dashboard
    dashboard = TrainingDashboard()
    
    # Render dashboard
    dashboard.render() 