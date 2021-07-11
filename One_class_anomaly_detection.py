###################### ONE CLASS ######################
import PIL
import h5py
import six
import keras
import scipy
import numpy
import yaml
!pip install git+https://github.com/rcmalli/keras-vggface.git
!pip show keras-vggface
!pip install Keras-Applications
import keras_vggface
from keras_vggface.vggface import VGGFace
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from keras.applications import VGG19
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm
from keras_vggface import utils
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.datasets import make_blobs
from numpy import quantile, where, random
import glob
from PIL import Image

####################################################### Data
x_re = x_ry_re.astype('float32')
y_re = t_ry_re.astype('float32')
w_re = z_ry_re.astype('float32')
x_re = utils.preprocess_input(x_re, version=1)
y_re = utils.preprocess_input(y_re, version=1)
w_re = utils.preprocess_input(w_re, version=1)

face_model = VGGFace(model='vgg16', include_top=False, input_shape=(64,64,3))

#new_layer = face_model.output
new_layer = face_model.get_layer('conv5_3').output
flatten_layer = Flatten()(new_layer)
final_model = Model(face_model.input,flatten_layer)

x_train_re = final_model.predict(x_re)
y_val_re = final_model.predict(y_re)
w_val_re = final_model.predict(w_re)


'''

# Apply standard scaler to output from resnet50
ss = StandardScaler(with_mean=True, with_std=True)
ss.fit(feature_array_x)
x_train = ss.transform(feature_array_x)
y_val = ss.transform(feature_array_y)
w_val = ss.transform(feature_array_w)
'''

'''
# Take PCA to reduce feature space dimensionality
pca = PCA(n_components=0.99, svd_solver='full',whiten=False)

pca.fit(x_train)

x_train = pca.transform(x_train)
y_val = pca.transform(y_val)
w_val = pca.transform(w_val)

'''
final_model.summary()
'''
tf.keras.utils.plot_model(
    final_model,
    to_file="/content/model.png",
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96)
'''


oc_svm_clf_re = svm.OneClassSVM(gamma='auto', kernel='linear', nu=.419)  
oc_svm_clf_re.fit(x_train_re)

###FRR###(normal)
oc_svm_preds_yre = oc_svm_clf_re.predict(y_val_re)
f = 0
t = 0
for k in range(oc_svm_preds_yre.size):
  if (oc_svm_preds_yre[k]==-1):
    f+=1
FRR = (f/oc_svm_preds_yre.size)*100
print('False Rejection Rate for normal data : {:.2f} %'.format(FRR))

###FAR###(abnormal)
oc_svm_preds_wre = oc_svm_clf_re.predict(w_val_re)
f = 0
t = 0
k = 0
for k in range(oc_svm_preds_wre.size):
  if (oc_svm_preds_wre[k]==+1):
    f+=1
FAR = (f/oc_svm_preds_wre.size)*100
print('False Acceptance Rate for abnormal data : {:.2f} %'.format(FAR))
####################################
############### HTER ###############
HTER_re = (FAR+FRR)/2
#re_res = 100-HTER_re
print('Half Total Error Rate : {:.2f} %'.format(HTER_re))

