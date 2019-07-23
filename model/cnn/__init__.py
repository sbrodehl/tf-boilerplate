from model import BaseModel
from model.cnn.network import network


class CNN(BaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def network(self, tensors):
        return network(*tensors, classes=self.classes)
