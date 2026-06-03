"""
CLIP simplified pseudocode.

Core idea:
1. Use an image encoder to get image embeddings.
2. Use a text encoder to get text embeddings.
3. Normalize both embeddings.
4. Maximize matched image-text similarity and minimize mismatched similarity.
"""

import torch
import torch.nn.functional as F


class CLIP:
    def __init__(self, image_encoder, text_encoder, temperature=0.07):
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.temperature = temperature

    def encode_image(self, images):
        image_features = self.image_encoder(images)
        image_features = F.normalize(image_features, dim=-1)
        return image_features

    def encode_text(self, texts):
        text_features = self.text_encoder(texts)
        text_features = F.normalize(text_features, dim=-1)
        return text_features

    def forward(self, images, texts):
        image_features = self.encode_image(images)  # [batch_size, dim]
        text_features = self.encode_text(texts)     # [batch_size, dim]

        # similarity[i][j] means how well image i matches text j
        logits = image_features @ text_features.T
        logits = logits / self.temperature
        return logits

    def loss(self, images, texts):
        logits_per_image = self.forward(images, texts)
        logits_per_text = logits_per_image.T

        # The correct pair is on the diagonal:
        # image 0 matches text 0, image 1 matches text 1, ...
        labels = torch.arange(len(images))

        image_to_text_loss = F.cross_entropy(logits_per_image, labels)
        text_to_image_loss = F.cross_entropy(logits_per_text, labels)

        return (image_to_text_loss + text_to_image_loss) / 2


"""
Training loop:

for images, texts in dataloader:
    loss = clip.loss(images, texts)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

After training:
- encode_image(image) and encode_text(text) are in the same embedding space.
- image retrieval: compare one text embedding with many image embeddings.
- zero-shot classification: compare one image embedding with class prompt embeddings.
"""
