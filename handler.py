from ts.torch_handler.base_handler import BaseHandler
import torch
import torchvision.transforms as T
from models import build_model
import numpy as np
import argparse
from PIL import Image

class MyHandler(BaseHandler):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.transform = T.Compose([
            T.Resize(800),
            T.toTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def preprocess(self, req):
        
        img = req.get("data")
        if img is None:
            img = req.get("body")

        rgb = img['rgb']
        origin = img['origin']
        rgb = Image.frombytes(mode="RGB", size=(300,400), data=rgb)
        img = self.transform(rgb)
        img = img.unsqueeze(0)

        return img

    def inference(self, x):

        outs = self.model.forward(x)
        
