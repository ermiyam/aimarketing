import subprocess
from pathlib import Path
import os

def transfer_data():
    """Transfer data to RunPod instance."""
    print("Transferring data to RunPod instance...")
    
    # Transfer learning data
    subprocess.run([
        'scp', '-i', str(Path.home() / '.ssh/id_ed25519'),
        'combined_learning_data/combined_marketing_learning_database.jsonl',
        'xdw41dclw40c27-64410f1a@ssh.runpod.io:/workspace/combined_learning_data/'
    ], check=True)
    
    # Transfer training script
    subprocess.run([
        'scp', '-i', str(Path.home() / '.ssh/id_ed25519'),
        'train_model.py',
        'xdw41dclw40c27-64410f1a@ssh.runpod.io:/workspace/'
    ], check=True)

def run_training():
    """Run training on RunPod instance."""
    print("Starting training on RunPod GPU...")
    
    # SSH into RunPod and run training
    ssh_command = [
        'ssh', '-i', str(Path.home() / '.ssh/id_ed25519'),
        'xdw41dclw40c27-64410f1a@ssh.runpod.io',
        'cd /workspace && python3 train_model.py'
    ]
    
    subprocess.run(ssh_command, check=True)

def download_results():
    """Download trained model and results."""
    print("Downloading trained model and results...")
    
    # Create local models directory if it doesn't exist
    Path("./models").mkdir(exist_ok=True)
    
    # Download trained model
    subprocess.run([
        'scp', '-i', str(Path.home() / '.ssh/id_ed25519'),
        '-r',
        'xdw41dclw40c27-64410f1a@ssh.runpod.io:/workspace/models/marketing_failure_predictor',
        './models/'
    ], check=True)

def main():
    try:
        # Transfer data and scripts
        transfer_data()
        
        # Run training
        run_training()
        
        # Download results
        download_results()
        
        print("Training complete! Model downloaded to ./models/marketing_failure_predictor")
        
    except subprocess.CalledProcessError as e:
        print(f"Error during execution: {e}")
        return

if __name__ == "__main__":
    main() 