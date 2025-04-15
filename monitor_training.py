import wandb
import time
from pathlib import Path

def monitor_training():
    """Monitor the training progress using wandb."""
    print("Monitoring training progress...")
    
    # Initialize wandb
    wandb.init(project="marketing-failure-predictor", resume=True)
    
    # Create monitoring directory
    monitoring_dir = Path("monitoring")
    monitoring_dir.mkdir(exist_ok=True)
    
    try:
        while True:
            # Get the latest run
            runs = wandb.Api().runs("marketing-failure-predictor")
            if not runs:
                print("No active training runs found.")
                time.sleep(60)
                continue
                
            latest_run = runs[0]
            
            # Get metrics
            metrics = latest_run.history()
            if not metrics.empty:
                latest_metrics = metrics.iloc[-1]
                
                # Print current status
                print("\nCurrent Training Status:")
                print(f"Step: {latest_metrics.get('step', 'N/A')}")
                print(f"Epoch: {latest_metrics.get('epoch', 'N/A')}")
                print(f"Training Loss: {latest_metrics.get('loss', 'N/A')}")
                print(f"Validation Loss: {latest_metrics.get('eval_loss', 'N/A')}")
                print(f"Learning Rate: {latest_metrics.get('learning_rate', 'N/A')}")
                
                # Save metrics to file
                with open(monitoring_dir / "latest_metrics.txt", "w") as f:
                    f.write(f"Step: {latest_metrics.get('step', 'N/A')}\n")
                    f.write(f"Epoch: {latest_metrics.get('epoch', 'N/A')}\n")
                    f.write(f"Training Loss: {latest_metrics.get('loss', 'N/A')}\n")
                    f.write(f"Validation Loss: {latest_metrics.get('eval_loss', 'N/A')}\n")
                    f.write(f"Learning Rate: {latest_metrics.get('learning_rate', 'N/A')}\n")
            
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    except Exception as e:
        print(f"Error monitoring training: {e}")

if __name__ == "__main__":
    monitor_training() 