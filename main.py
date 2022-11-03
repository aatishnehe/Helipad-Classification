import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
# from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_ResNet
# from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array, array_to_img
import os
import h5py
import copy
import keras
import cv2
import copy
import warnings
warnings.filterwarnings('ignore')

import os
import cv2
import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Input
from keras.layers import Conv2D, Activation, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array, array_to_img
#from keras.applications.xception import Xception,preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


import matplotlib
import matplotlib.pylab as plt
import matplotlib.cm as cm
import numpy as np
# import seaborn as sns
# import shap
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from annotated_text import annotated_text




def superimpose(img, cam):
    """superimpose original image and cam heatmap"""

    heatmap = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * .5 + img * .5
    superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)  # scale 0 to 255
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    return img, heatmap, superimposed_img


def _plot(model, cam_func, img):
    """plot original image, heatmap from cam and superimpose image"""

    # for cam
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(copy.deepcopy(x))

    # for superimpose
    img = np.uint8(img)

    # cam / superimpose
    cls_pred, cam = cam_func(x, model=model, last_conv_layer_name = model.layers[-2].name)
    img, heatmap, superimposed_img = superimpose(img, cam)

    return superimposed_img


def grad_cam(img_array, model, last_conv_layer_name, classifier_layer_names = ['sequential']):
    """Grad-CAM function"""
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)


    return top_pred_index, heatmap

#Set page layout
st.set_page_config(page_title = 'Helipad Identification App',
                    page_icon = ':smile:',
                    layout = 'wide',
                    initial_sidebar_state = 'expanded')


st.title = 'Image Classification'


st.subheader('Helipad Identification')
col1, col2 = st.columns([2,4])
models_list = ['ResNet50', 'VGG19', 'Inception', 'Xception', 'ResNet50_Trainable', 'MobileNet']
network = st.sidebar.selectbox('Select the model', models_list)
accuracy = {
            'ResNet50' : 96.56,
            'VGG19' : 93.44,
            'Inception' : 95.63,
            'Xception' : 97.5,
            'ResNet50_Trainable' : 98.12,
            'MobileNet' : 87.81
}
uploaded_file = st.sidebar.file_uploader('Choose an Image: ', type = ['jpg', 'jpeg', 'png'])


with col1:

    st.write(network, ' has an Accuracy of ', str(accuracy[network]), '%')

    if uploaded_file:
        image = load_img(uploaded_file)
        image = image.resize((350,350))
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        st.image(image)

if uploaded_file:

    test_image = image.resize((224,224))
    img_array = img_to_array(test_image)
    Images = np.array(img_array, dtype=np.float32)
    #test_image = np.expand_dims(test_image, axis = 0)
    test_image = preprocess_input(img_array)
    ##model = MODELS[network]

    classifier_model = load_model(
    'C:/Users/AatishNehe/Downloads/Helipad/saved_model_' + network + '/my_model')

    prediction = classifier_model.predict(test_image.reshape(1,224,224,3))
    prob = (np.max(prediction)*100)

    if np.argmax(prediction) == 1:
        annotated_text(('There is a Helipad for safe landing!!!', '',"#afa"))
        st.write('(Probabilty ' , str(int(prob)), '%)')

        with col2:

            heatmap = _plot(model=classifier_model, cam_func=grad_cam, img=Images)
            if st.button('Click here to get a Heatmap'):
                st.image(heatmap, width = 350)

    else:
        annotated_text(('No Helipad...Keep looking...', '', "#faa"))
        st.write('(Probabilty ' , str(int(prob)), '%)')
