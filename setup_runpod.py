import subprocess
import os
from pathlib import Path
import time
import paramiko
import requests
import json
from typing import Dict, Any

def get_runpod_instance() -> Dict[str, Any]:
    """Get the RunPod instance details using the API."""
    if not os.getenv('RUNPOD_API_KEY'):
        raise ValueError("RUNPOD_API_KEY environment variable is not set. Please set it using: export RUNPOD_API_KEY='your-api-key'")

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {os.getenv("RUNPOD_API_KEY")}',
    }

    response = requests.get('https://api.runpod.io/graphql', headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to get RunPod instance: {response.text}")

    return response.json()

def connect_to_runpod():
    """Connect to the RunPod instance via SSH and set up the environment."""
    try:
        instance = get_runpod_instance()
        
        # SSH connection details
        hostname = instance['data']['myself']['pods'][0]['runtime']['ipAddress']
        port = instance['data']['myself']['pods'][0]['runtime']['ports'][0]['publicPort']
        username = 'root'
        password = '0hick7ufhfjbjf1ioqn9'  # Consider using environment variable
        
        print(f"Connecting to RunPod instance {instance['data']['myself']['pods'][0]['name']} ({instance['data']['myself']['pods'][0]['id']})...")
        
        # Create SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            ssh.connect(hostname, port=port, username=username, password=password)
            print("Successfully connected to RunPod instance")
            
            # Setup commands
            commands = [
                "apt-get update",
                "apt-get install -y cuda-toolkit-12-0",
                "mkdir -p /workspace",
                "cd /workspace"
            ]
            
            # Execute setup commands
            for cmd in commands:
                print(f"Executing: {cmd}")
                stdin, stdout, stderr = ssh.exec_command(cmd)
                exit_status = stdout.channel.recv_exit_status()
                
                if exit_status != 0:
                    error = stderr.read().decode().strip()
                    print(f"Warning: Command '{cmd}' failed with status {exit_status}")
                    print(f"Error output: {error}")
                else:
                    print("Command completed successfully")
            
            print("RunPod GPU environment setup complete")
            print(f"You can now connect to your instance at {hostname}:{port}")
            
            return True
            
        except paramiko.AuthenticationException:
            raise Exception("Failed to authenticate with RunPod instance")
        except paramiko.SSHException as e:
            raise Exception(f"SSH error: {str(e)}")
        finally:
            ssh.close()
            
    except Exception as e:
        print(f"Failed to connect to RunPod instance: {str(e)}")
        print("Please verify:")
        print("1. Your RunPod API key is set correctly")
        print("2. The instance is running")
        print("3. The instance is accessible")
        return False

def setup_runpod():
    """Set up the RunPod instance for training."""
    try:
        instance = get_runpod_instance()
        print("Successfully connected to RunPod instance")
        return instance
    except Exception as e:
        print(f"Error setting up RunPod: {str(e)}")
        return None

def main():
    print("Setting up RunPod GPU environment...")
    
    # Set the API key from the provided value
    os.environ['RUNPOD_API_KEY'] = 'rpa_LTZJSAH7J3P12M8IUKYOXO3XKTB9PMS9P2PRU9PX15tpdf'
    
    if not connect_to_runpod():
        print("Failed to connect to RunPod instance")
        return
    
    print("RunPod GPU environment setup complete!")
    print("You can now run your training scripts on the GPU instance.")

if __name__ == "__main__":
    setup_runpod() 