from tensorflow import keras

from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np


import tensorflow as tf
# import tf2onnx
# import onnx


_IMAGE_NET_TARGET_SIZE = (224, 224)

class EmbGenerator(object):

    def __init__(self):
        
        model = resnet50.ResNet50(weights='imagenet')
        layer_name = 'avg_pool'
        self.intermediate_layer_model = Model(inputs=model.input, 
                                              outputs=model.get_layer(layer_name).output)
    
    def convert_to_onnx(self):
        input_signature = [tf.TensorSpec([None, 224, 224, 3], tf.float32, name='input_1')]
        onnx_model, _ = tf2onnx.convert.from_keras(self.intermediate_layer_model, input_signature, opset=13)
        onnx.save(onnx_model, "Embedding_Model.onnx")
        

    def get_emb(self, image_path):

        img = image.load_img(image_path, target_size=_IMAGE_NET_TARGET_SIZE)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = resnet50.preprocess_input(x)
        intermediate_output = self.intermediate_layer_model.predict(x)
        return intermediate_output[0]


if __name__ == "__main__":
    from glob import glob

    emb_generator = EmbGenerator()

    emb_generator.get_emb("/mnt/829A20D99A20CB8B/projects/img2vec/3d6fbe0a-661d-11ec-8953-f319442a69b7.jpg")

    # for file in glob("/mnt/829A20D99A20CB8B/projects/Datasets/cifar-100-python/dataset/*/*.png"):
    #     print(file)