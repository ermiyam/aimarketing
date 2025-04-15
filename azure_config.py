from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.monitor import MonitorClient
from azure.monitor.query import LogsQueryClient
import os
import logging

class AzureConfig:
    def __init__(self):
        self.credential = DefaultAzureCredential()
        self.key_vault_url = os.getenv('AZURE_KEY_VAULT_URL')
        self.workspace_id = os.getenv('AZURE_WORKSPACE_ID')
        self.app_name = os.getenv('APP_NAME', 'gen')
        self.setup_logging()
        
    def setup_logging(self):
        """Configure Azure Monitor logging"""
        if self.workspace_id:
            self.logs_client = LogsQueryClient(self.credential)
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger('azure_monitor')
            
    def get_secret(self, secret_name: str) -> str:
        """Retrieve secret from Azure Key Vault"""
        try:
            if self.key_vault_url:
                secret_client = SecretClient(
                    vault_url=self.key_vault_url,
                    credential=self.credential
                )
                return secret_client.get_secret(secret_name).value
            return os.getenv(secret_name)
        except Exception as e:
            self.logger.error(f"Error retrieving secret {secret_name}: {str(e)}")
            return None
            
    def log_metric(self, metric_name: str, value: float, properties: dict = None):
        """Log metric to Azure Monitor"""
        try:
            if self.workspace_id:
                monitor_client = MonitorClient(
                    credential=self.credential,
                    subscription_id=os.getenv('AZURE_SUBSCRIPTION_ID')
                )
                monitor_client.metrics.create(
                    metric_name=f"{self.app_name}_{metric_name}",
                    value=value,
                    properties=properties or {}
                )
        except Exception as e:
            self.logger.error(f"Error logging metric {metric_name}: {str(e)}")
            
    def log_event(self, event_name: str, properties: dict = None):
        """Log custom event to Azure Monitor"""
        try:
            if self.workspace_id:
                self.logs_client.query(
                    workspace_id=self.workspace_id,
                    query=f"customEvents | where name == '{self.app_name}_{event_name}'",
                    properties=properties or {}
                )
        except Exception as e:
            self.logger.error(f"Error logging event {event_name}: {str(e)}")

# Global Azure configuration instance
azure_config = AzureConfig() 