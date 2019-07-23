from model import BaseModel
from model.rcnn.network import network


class RCNN(BaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def network(self, tensors):
        return network(*tensors, classes=self.classes)
