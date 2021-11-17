import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub
from utility_functions import getClassNames, loadModel, processImage

import argparse
import numpy as np
import json

def predict(image_path, model_path, top_k, class_names):

    im = Image.open(image_path)
    im = np.asarray(im)
    im = processImage(im)
    expanded_im = np.expand_dims(im, axis=0)

    model = loadModel(model_path)
    
    probs = model.predict(expanded_im)
    
    top_k_probs, top_k_indices = tf.nn.top_k (probs, k = top_k)

    top_k_probs = top_k_probs.numpy()
    top_k_indices = top_k_indices.numpy()
    
    fl_classes = []   
    for idx in top_k_indices[0]:
        fl_classes.append(class_names[str(idx+1)])        

    print(f'\nTop {top_k} predictions for image: {image_path}')
    for i in range(top_k):
        print('Prediction {}: \"{}\" (index: {}) ---> {:,.4%}'.format(i+1,fl_classes[i],top_k_indices[0][i],top_k_probs[0][i]))
        
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "Parser description")
    parser.add_argument("image_path", help = "Image Path", default = "")
    parser.add_argument("saved_model", help = "Model Path",  default = "")
    parser.add_argument("--top_k", help = "Fetch top k predictions", required = False, default = 5)
    parser.add_argument("--category_names", help = "Class map json file", required = False, default = "label_map.json")
    args = parser.parse_args()
    

    #print (tf.cast(args.top_k), tf.float32)
    class_names = getClassNames(args.category_names)
    
    predict(args.image_path, args.saved_model, int(args.top_k), class_names)
    