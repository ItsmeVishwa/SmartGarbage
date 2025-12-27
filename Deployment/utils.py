from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

def gen_labels():
    base_path = os.path.abspath(__file__)
    # Correctly identify project root to find Labels.txt
    project_root = os.path.dirname(os.path.dirname(base_path))
    labels_path = os.path.join(project_root, 'Labels.txt')
    
    with open(labels_path, 'r') as f:
        labels_list = [line.strip() for line in f.readlines()]
    
    labels = {i: label for i, label in enumerate(labels_list)}
    return labels

def preprocess(image):
    image = image.resize((224, 224), Image.Resampling.LANCZOS)
    image = np.array(image) / 255.0
    return image

def model_arc():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(6, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Enable OneDNN optimizations
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
