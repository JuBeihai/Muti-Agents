import torch
import torch.nn.functional as F

class CLIP:
    def __init__(self, image_encoder, text_encoder, temperature = 0.07)
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.temperature = temperature
    
    def encode_image(self, images):
        image_features = self.image_encoder(images)
        