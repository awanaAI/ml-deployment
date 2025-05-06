import bentoml
from bentoml.io import Image
import numpy as np
from PIL import Image as PILImage

@bentoml.service(resources={"gpu": 1})
class MnistClassifier:

    def __init__(self):
        import tensorflow as tf
        self.tf = tf
        self.model = tf.keras.models.load_model("./cnn_model.h5")

    @bentoml.api()
    def classify(self, img: PILImage.Image):
        img = img.resize((28, 28)).convert("L")
        img_array = np.expand_dims(np.array(img) / 255.0, axis=(0, -1))
        prediction = self.model.predict(img_array)
        return {"prediction": int(prediction.argmax())}
