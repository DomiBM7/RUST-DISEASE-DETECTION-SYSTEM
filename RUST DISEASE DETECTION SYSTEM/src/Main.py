# BUKASA MUYOMBO 2
# 
import os
import numpy as np
from PIL import Image
from matplotlib import cm
import tifffile as tiffS
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image
import cv2
from skimage.feature import hog, local_binary_pattern
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# set random seed for reproducibility, 
#which is easier for comparision
seed = 50
train_dir = 'C:/Users/User/Downloads/beyond-visible-spectrum-ai-for-agriculture-2024/archive/train'
val_dir = 'C:/Users/User/Downloads/beyond-visible-spectrum-ai-for-agriculture-2024/archive/val'

# a function to convert a .tif to greyscale .png
def convertTifToPNG(directory):
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".tif") or filename.endswith(".tiff"):
                image_path = os.path.join(root, filename)
                try:
                    tifImage = tiff.imread(image_path)
                    grayImage = tifImage.mean(axis=-1).astype(np.uint8)
                    img = Image.fromarray(grayImage)
                    pngFileName = os.path.splitext(filename)[0] + ".png"
                    img.save(os.path.join(root, pngFileName))
                    print(f"Converted {filename} to {pngFileName}")
                except Exception as e:
                    print(f"Failed to convert {filename}: {e}")


# convertTifToPNG(train_dir)
# convertTifToPNG(val_dir)

# function handles pectral normalization function
def spectralNormalization(image):
    
    # calculate of the mean and variance per channel
    mean, variance = tf.nn.moments(image, axes=[0, 1])  
    stddev = tf.sqrt(variance)
    return (image - mean) / (stddev + 1e-6)

# load the dataset of training
trainDataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(256, 256),  
    batch_size=32,
    color_mode='grayscale',
    seed=seed
)
 #same with validation
valDataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(256, 256),  
    batch_size=32,
    color_mode='grayscale',
    seed=seed
)

#get the class names , health = 0, other =1, rust = 2
classNames = trainDataset.class_names
print("Class names:", classNames)

# data augmentation.
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    # Normalize pixel values to [0, 1] to avoid teh exploding gradient problem and for easier
    #training
    layers.Rescaling(1./255)  
])

# then we apply spectral normalization and then data augmentation
trainDataset = trainDataset.map(lambda x, y: (spectralNormalization(x), y))
valDataset = valDataset.map(lambda x, y: (spectralNormalization(x), y))
trainDataset = trainDataset.map(lambda x, y: (data_augmentation(x, training=True), y))

# Feature extraction methods!!
#hog features
def getHogFeatures(images):
    hogFeatures = []
    for img in images:
        # get the HOG features
        features = hog(img.squeeze(), orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=False)
        hogFeatures.append(features)
        #return them as a numpy array
    return np.array(hogFeatures)

def getLBPFeatures(images):
    LBPFeatures = []
    for img in images:
        # same with LBP
        lbp = local_binary_pattern(img.squeeze(), P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp, bins=np.arange(0, 11), density=True)  
        LBPFeatures.append(hist)
        # and it comes back as 10 bins
    return np.array(LBPFeatures)


# get the features from training and validation datasets
trainImages, trainLabels = [], []
for images, labels in trainDataset:
    trainImages.append(images)
    trainLabels.append(labels)

#get their combinations
trainImages = np.concatenate(trainImages)
trainLabels = np.concatenate(trainLabels)
#same with labels and pass them all for training
valImages, valLabels = [], []
for images, labels in valDataset:
    valImages.append(images)
    valLabels.append(labels)

valImages = np.concatenate(valImages)
valLabels = np.concatenate(valLabels)

# get the HOG and LBP features
trainHogFeatures = getHogFeatures(trainImages)
trainLBPFeatures = getLBPFeatures(trainImages)
valHogFeatures = getHogFeatures(valImages)
valLBPFeatures = getLBPFeatures(valImages)

# Model definition
model = tf.keras.Sequential([
    #input layer
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
    # 1 because we are working with grey images
    tf.keras.layers.BatchNormalization(),
    #to avoid covariate shift
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #pool size of (2,2)
    tf.keras.layers.Dropout(0.25),
    # to prevent overfitting
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), #leakyrelu might lead to better training since it 
    #has non zero gradients
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    #flatten layer

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(len(classNames), activation='softmax')  # output layer
])

#save the model weights
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
 #compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# train it
epochs = 200  

#early stopping prevents us from wasting time
earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lrSCheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

history = model.fit(trainDataset, validation_data=valDataset, epochs=epochs, callbacks=[earlyStopping, lrSCheduler, checkpoint])

#thn we load the best saved model
model.load_weights('best_model.h5')

# evaluate on validation data and collect predictions, by using the ground truth valesu
yTrue = []
yPrediction = []

for images, labels in valDataset:
    preds = model.predict(images)
    predictedClasses = np.argmax(preds, axis=1)
    
    yTrue.extend(labels.numpy())
    yPrediction.extend(predictedClasses)

# Compute precision, recall, and F1 score
precision = precision_score(yTrue, yPrediction, average='weighted')
recall = recall_score(yTrue, yPrediction, average='weighted')
f1 = f1_score(yTrue, yPrediction, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

#Confusion matrix
confusionMatrix = confusion_matrix(yTrue, yPrediction)

# llot the confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusionMatrix, annot=True, fmt="d", cmap="cubehelix")
plt.title('Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()

# Check label distributions
print("True label distribution:", np.bincount(yTrue))
print("Predicted label distribution:", np.bincount(yPrediction))

# then sample for true and predicted values for verification
for true, pred in zip(yTrue[:10], yPrediction[:10]):
    print(f"True: {true}, Predicted: {pred}")

# then we evaluate the model
#
val_loss, val_accuracy = model.evaluate(valDataset)
print(f"Validation accuracy: {val_accuracy}")

    # Plot accuracy and other metrics, through the pochs
    # as rhe independent values
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy', color=cm.cubehelix(0.2))
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color=cm.cubehelix(0.6))
plt.title('Model Accuracy', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

#Plot loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss', color=cm.cubehelix(0.4))
plt.plot(history.history['val_loss'], label='Validation Loss', color=cm.cubehelix(0.8))
plt.title('Model Loss', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# Plot THe precision, recall, and F1 score
metrics = ['Precision', 'Recall', 'F1 Score']
values = [precision, recall, f1]

plt.figure(figsize=(8, 6))
plt.bar(metrics, values, color=cm.cubehelix(np.linspace(0.3, 0.7, len(values))))
plt.title('Classification Metrics', fontsize=14)
plt.ylabel('Score', fontsize=12)
plt.grid(True)
plt.show()



#Save the final model as a Keras model
model.save("C:/Users/User/Desktop/BIG DATA/Project/final_model.h5")

#Load and prepare an image for prediction
imagePath = "C:/Users/User/Desktop/BIG DATA/Project/a.png"
img = image.load_img(imagePath, target_size=(256, 256))  # resize image to (256, 256)
imageArray = image.img_to_array(img)

#convert to grayscale (because model was trained on grayscale images)
imageArray = cv2.cvtColor(imageArray, cv2.COLOR_RGB2GRAY)

#reshape for Conv2D and batch dimension
imageArray = np.expand_dims(imageArray, axis=-1)  # add a channel dimension
imageArray = np.expand_dims(imageArray, axis=0)   # add a batch dimension

#normalizing the the images is required to
imageArray = imageArray / 255.0

#make prediction with the Keras model
predictions = model.predict(imageArray)
predicted_class = np.argmax(predictions, axis=1)
print(f"Predicted class: {predicted_class}")

#convert the Keras model to A  tensorFlow lite model (tflite)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfliteModel = converter.convert()

#Save the TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tfliteModel)

#Initialize then the TFLite interpreter
interpreter = tf.lite.Interpreter(model_content=tfliteModel)
interpreter.allocate_tensors()

# get some input and output details for TFLite
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# put the input tensor to the interpreter
interpreter.set_tensor(input_details[0]['index'], imageArray)

# run inference on the model
interpreter.invoke()

# and get output tensor and class prediction
output_data = interpreter.get_tensor(output_details[0]['index'])
tflite_predicted_class = np.argmax(output_data, axis=1)
print(f"TFLite predicted class: {tflite_predicted_class}")
