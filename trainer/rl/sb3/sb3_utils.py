from stable_baselines3.common.logger import Logger

class SB3InternalLogger(Logger):
    """
    An internal logger for SB3 models.
    SB3's learn() method logs metrics to this logger
    The trainer.log_epoch() method will extract data from this logger
    """

    def __init__(self, trainer=None):
        super().__init__(folder=None, output_formats=['json'])
        if trainer is not None:
            self.set_trainer(trainer)

    def set_trainer(self, trainer):
        self.trainer = trainer

    def dump(self, step: int = 0) -> None:
        # TODO: when dumping add to train.log or eval.log?
        super().dump(step)
