from loss import BaseLoss
from data import BaseDataSampler
from model import BaseModel
from experiment import BaseExperiment


class Classification(BaseExperiment):
    """Experiment which does things."""

    def __init__(self, sampler: BaseDataSampler, model: BaseModel, lossfn: BaseLoss):
        super().__init__(sampler, model, lossfn)
