# Aman Kumar
# 2020CSB1153


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
from numpy.linalg import norm
from sklearn.metrics import classification_report
import math

# import json
# from sklearn.cluster import KMeans

# wcss2 = []
# for i in range(1,200,1):
#     print("Starting new K ")
#     kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
#     kmeans.fit(global_descript)
#     wcss2.append(kmeans.inertia_)
#     print(f"Clusters : {i}  -  WCSS : {kmeans.inertia_}")
#     with open('wcss2.txt', 'w') as f:
#       f.write(json.dumps(wcss2))

def ComputeHistogram(visual_words_arr):     # Computing Histogram

  frequency_vectors = []
  for img_visual_words in visual_words_arr:
      img_frequency_vector = np.zeros(128)
      for word in img_visual_words:
          img_frequency_vector[word] += 1
      frequency_vectors.append(img_frequency_vector)
  frequency_vectors = np.stack(frequency_vectors)
  return frequency_vectors

def CreateVisualDictionary(img,CodeBook):             # Creating Visual Dictionart and Returning Frequency Vectors and Labels
  extractor = cv2.ORB_create(fastThreshold = 0 , edgeThreshold = 0)

  img_2d = []

  for im in img:
    if len(im.shape) == 3:
      img_2d.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    else:
      img_2d.append(im)
  
  data_keypts = {}
  data_descript = {}
  i = 0
  for img in img_2d:
    x , y = extractor.detectAndCompute(img,None)
    data_keypts[i] = x
    data_descript[i] = y
    i+=1

  visual_words_arr = []
  vis_label = []
  i = 0
  for x,img_descriptors in data_descript.items():
    if img_descriptors is None:
      i+=1
      continue
    img_visual_words, distance = vq(img_descriptors, CodeBook)
    visual_words_arr.append(img_visual_words)
    vis_label.append(train_labels[i])
    i+=1

  frequency_vectors = ComputeHistogram(visual_words_arr)

  return frequency_vectors,vis_label

def KMean(global_descript,k):             # Kmean Function
  ind = np.random.choice(len(global_descript), k, replace=False)
  codebook = global_descript[ind, :]
    
  dist = cdist(global_descript, codebook ,'euclidean')
    
  points = np.array([np.argmin(i) for i in dist])
    
  for iter in range(100): 
      codebook = []
      for ind in range(k):
        temp_cent = global_descript[points==ind].mean(axis=0) 
        codebook.append(temp_cent)
      codebook = np.vstack(codebook)
      dist = cdist(global_descript, codebook ,'euclidean')
      points = np.array([np.argmin(i) for i in dist])
  return codebook

def CreateDictionary(global_descript):          # Returning Centroids (CodeBook)
  Kmean_arr = KMean(global_descript,128)
  np.save("Dictionary.npy",Kmean_arr)
  return Kmean_arr

def MatchHistogram(tf,label,tfidf,vis_label):                         # Calculating Cosine Similarity and Returning Labels
  extractor = cv2.ORB_create(fastThreshold = 0 , edgeThreshold = 0)
  cos_sim = []
  for i in range(len(tf)):
    arr = []
    a = tf[i]
    b = tfidf
    cosine_simi = np.dot(a,b.T)/(norm(a)*norm(b,axis = 1))

    idx = np.argsort(-cosine_simi)[:1]

    # print(i,vis_label[idx[0]])
    cos_sim.append(vis_label[idx[0]])

  arr = []
  i = 0
  for x in test_images:
    if len(x.shape)==3:
      x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    a , b = extractor.detectAndCompute(x,None)
    if b is not None:
      arr.append(test_labels[i])
    i+=1

  return arr,cos_sim

# Main Function
if __name__ == '__main__':
  fashion_mnist = keras.datasets.fashion_mnist   # Importing Data

  (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

  extractor = cv2.ORB_create(fastThreshold = 0 , edgeThreshold = 0)

  img_2d = []

  for img in train_images:
    if len(img.shape) == 3:
      img_2d.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    else:
      img_2d.append(img)
  data_keypts = {}
  data_descript = {}
  i = 0
  for img in img_2d:
    x , y = extractor.detectAndCompute(img,None)
    data_keypts[i] = x
    data_descript[i] = y
    i+=1
  global_descript = []
  sz = len(data_descript)

  for i in range(0,sz):
    if (data_descript[i] is not None):
      for j in data_descript[i]:
        global_descript.append(j)

  global_descript = np.stack(global_descript)
  CodeBook = CreateDictionary(global_descript)      # Calculating CodeBook

  visual_words_arr = []
  vis_label = []
  i = 0
  for x,img_descriptors in data_descript.items():
      if img_descriptors is None:
        i+=1
        continue
      img_visual_words, distance = vq(img_descriptors, CodeBook)
      visual_words_arr.append(img_visual_words)
      vis_label.append(train_labels[i])
      i+=1

  frequency_vectors = ComputeHistogram(visual_words_arr)

  length,y = frequency_vectors.shape
  df = np.sum(frequency_vectors > 0, axis=0)

  idf = np.log(length/ df)

  tfidf = frequency_vectors * idf

  tf,label = CreateVisualDictionary(test_images,CodeBook)

  x , y = MatchHistogram(tf,label,tfidf,vis_label)

  print(classification_report(x,y))       # Printing Classification Report
  