#BUKASA MUYOMBO
#BACKEND SERVER
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow import keras
import numpy as np
import cv2
import os
import torch
import io
# HEQLTH = 0, OTHER = 1, RUST = 2
class_names = ["Health", "Other", "Rust"]

# initialize the Flask app
app = Flask(__name__)
CORS(app)

# then load the trained Keras model
keras_model = keras.models.load_model('C:/Users/User/Desktop/BIG DATA/Project/final_model.h5')
print("Keras model loaded.")

class MyPyTorchModel(torch.nn.Module):
    def __init__(self):
        #constructor for the mytorch model
        super(MyPyTorchModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(32 * 256 * 256, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def loadWeights(i):

    try:
        # load the content from the file object
        #with some error handling
        content = torch.load(i, map_location=torch.device('cpu'))
        
        #then check if the content is a state dictIionary or a single multi dimensional tensor tensor
        if isinstance(content, dict):
            numpyWeights = {key: tensor.cpu().numpy() for key, tensor in content.items()}
        elif isinstance(content, torch.Tensor):
            numpyWeights = {'model_weights': content.cpu().numpy()}
        else:
            raise ValueError(f"Error-----: {type(content)}")
        
        print("Loaded weights structure:", {k: v.shape for k, v in numpyWeights.items()})
        return numpyWeights
    except Exception as e:
        print(f"---Error loading the .pt file: {str(e)}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    global class_names
    predictions = []

    # take the Keras image predictions
    for i in request.files.getlist('is'):
        if i:
            if not os.path.exists('static'):
                os.makedirs('static')

            imagEPath = os.path.join("static", i.filename)
            i.save(imagEPath)
            print(f"-----Saved Keras image: {imagEPath}")

            if os.path.exists(imagEPath) and os.path.getsize(imagEPath) > 0:
                img = cv2.imread(imagEPath)
                if img is None:
                    print(f"Failed to load image: {imagEPath}")
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (256, 256))
                imageArray = np.expand_dims(img, axis=-1)
                #EXPAND DIMENSIONS
                imageArray = np.expand_dims(imageArray, axis=0)

                prediction = keras_model.predict(imageArray)
                predicted_class = np.argmax(prediction, axis=-1)
                #give them
                predictions.append(f"Keras Prediction: {class_names[predicted_class[0]]}")
    """
    # handle multiple .PT files
    for i in request.files.getlist('i'):
        if i:
            try:
                print(f"Processing the file: {i.filename}")
                # convert PT file to numpy arrays
                numpyWeights = loadWeights(i)
                
                
                model = MyPyTorchModel()
                #instantiate it
                
                if len(numpyWeights) == 1 and 'model_weights' in numpyWeights:
                    tensorWeight = torch.from_numpy(numpyWeights['model_weights'])
                    print(" Tensor weights shape:", tensorWeight.shape)
                    
                else:
                    state_dict = {
                        key: torch.from_numpy(arr)
                        for key, arr in numpyWeights.items()
                    }
        #            model.load_state_dict(state_dict)
                
        #        model.eval()

    
       # except Exception as e:
                print(f"Error processing PT file {i.filename}: {str(e)}")
                predictions.append(f"Error with PyTorch model {i.filename}: {str(e)}")
"""
    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)