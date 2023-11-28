import wandb

class WandbLogger:
    def __init__(self, project_name: str, config: dict):
        wandb.init(
            project=project_name,
            config=config
        )
        self.wandb = wandb
        self.step = 0
    
    def set_step(self, step: int):
        self.step = step

    def log_state(self, name: str, value: float):
        self.wandb.log({name: value}, step=self.step)
    