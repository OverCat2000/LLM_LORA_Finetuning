class NullLogger:
    run_id = ""

    def log_params(self, params):
        pass

    def log(self, metrics, step):
        pass

    def set_tag(self, key, value):
        pass

    def close(self):
        pass


class TensorBoardLogger:
    def __init__(self, cfg, resume_id=""):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(f"{cfg.log_dir}/{cfg.run_name}")
        self.run_id = cfg.run_name
        print(f"tensorboard: {cfg.log_dir}/{cfg.run_name}")

    def log_params(self, params):
        text = "\n".join(f"{k}: {v}" for k, v in sorted(params.items()))
        self.writer.add_text("config", text)

    def log(self, metrics, step):
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)

    def set_tag(self, key, value):
        self.writer.add_text(key, str(value))

    def close(self):
        self.writer.close()


class WandbLogger:
    def __init__(self, cfg, resume_id=""):
        import wandb

        self.wandb = wandb
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.run_name,
            id=resume_id or cfg.run_name,
            resume="allow",
            config=cfg.to_dict(),
        )
        self.run_id = wandb.run.id
        print(f"wandb: {wandb.run.url}")

    def log_params(self, params):
        self.wandb.config.update(params, allow_val_change=True)

    def log(self, metrics, step):
        self.wandb.log(metrics, step=step)

    def set_tag(self, key, value):
        self.wandb.run.summary[key] = value

    def close(self):
        self.wandb.finish()


class MLflowLogger:
    def __init__(self, cfg, resume_id=""):
        import mlflow

        self.mlflow = mlflow
        mlflow.set_tracking_uri(cfg.mlflow_uri)
        mlflow.set_experiment(cfg.experiment)
        mlflow.start_run(run_id=resume_id or None, run_name=cfg.run_name)
        self.run_id = mlflow.active_run().info.run_id
        self.fresh = not resume_id
        print(f"mlflow run {self.run_id} at {cfg.mlflow_uri}")

    def log_params(self, params):
        if self.fresh:
            self.mlflow.log_params(params)

    def log(self, metrics, step):
        self.mlflow.log_metrics(metrics, step=step)

    def set_tag(self, key, value):
        self.mlflow.set_tag(key, str(value))

    def close(self):
        self.mlflow.end_run()


BACKENDS = {
    "none": NullLogger,
    "tensorboard": TensorBoardLogger,
    "wandb": WandbLogger,
    "mlflow": MLflowLogger,
}


def make_logger(cfg, resume_id=""):
    if cfg.tracker not in BACKENDS:
        raise ValueError(f"unknown tracker: {cfg.tracker}")
    if cfg.tracker == "none":
        return NullLogger()
    return BACKENDS[cfg.tracker](cfg, resume_id)
