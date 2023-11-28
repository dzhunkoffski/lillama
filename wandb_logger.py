import wandb

class WandbLogger:
    def __init__(self, project_name: str, config: dict):
        wandb.init(
            project=project_name,
            config=config
        )
        self.wandb = wandb
    
    def log_metrics(self, metrics: dict):
        self.wandb.log(metrics)