### Author: Agastya Kapur

### May need to
# !pip install pyyaml h5py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from google. colab import files
import glob
import os.path
from PIL import Image
import re
from pathlib import Path    
import pandas as pd
import glob

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from skimage.io import imread, imshow
from sklearn.cluster import KMeans

import cv2 
from google.colab.patches import cv2_imshow
import tensorflow as tf
import shutil
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

from collections import Counter
from skimage.metrics import structural_similarity as ssim


### Some basic variables

homedir  = '/content/drive/MyDrive/ASTRPACE/PACE_F/PLOTS/'
types = ['CLEAN', 'CORR_DROP', 'RFI', 'ANT_ERROR']
average_color_std = [158.4, 230.66666667, 232.06666667] #Hard coded base clean minibox colour
#class_names  = ['ANT_ERROR', 'CLEAN', 'CORR_DROP', 'RFI']

### Load stuff make sure correct training set chosen (Not necessary if uncomment above class_names)

batch_size = 40
img_height = 200
img_width = 200

train_ds = tf.keras.utils.image_dataset_from_directory(
  '/content/drive/MyDrive/ASTRPACE/PACE_F/TRAINING/',
  validation_split=0.3,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  '/content/drive/MyDrive/ASTRPACE/PACE_F/TRAINING/',
  validation_split=0.3,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

### Code to crop labels

def firstCrop(SBIDC):
  os.chdir(homedir+SBIDC)
  print(SBIDC)
  pngList = glob.glob('*.png')
  for raw in pngList:
    dirtomk = raw+'_DIR'
    dir_exists = os.path.exists(dirtomk)
    if(dir_exists):
      print('Have dir for: ' + raw)
    else:
      os.mkdir(dirtomk)
    
    file_exists = os.path.exists(dirtomk+'/'+raw)
    if(file_exists):
      print('Have cropped file for: ' + raw)
    else:
      imtocrop = Image.open(raw)
      im_crop = imtocrop.crop((72,112,3330,3349))
      print(imtocrop.filename)  
      im_crop.save(dirtomk+'/'+imtocrop.filename)
      
### Make miniboxes

def makeMini(SBIDC):
  os.chdir(homedir+SBIDC)
  print(SBIDC)
  dirList = glob.glob('*.png_DIR')
  for dirs in dirList:
    os.chdir(homedir+SBIDC)
    print(dirs)
    #croppedpngs = glob.glob(dirs+'/*.png')
    file_exists = os.path.exists(dirs+'/'+'miniboxes')
    if(file_exists):
      print('Have miniboxes dir for SBID: ' + SBIDC)
    else:
      print('Making miniboxes and dir')
      os.chdir(homedir+SBIDC)
      os.mkdir(dirs+'/'+'miniboxes')
      os.chdir(homedir+SBIDC+'/'+dirs)
      croppedList = glob.glob('*.png')
      for x in croppedList:
        imtocrop = Image.open(x)
        print(imtocrop.filename)  
        for i in np.arange(0,36):
          for j in np.arange(0,36):
            im_crop = imtocrop.crop((90.5*i,89.9166666*j,90.5*(i+1),89.9166666*(j+1)))
            im_crop.save(homedir+SBIDC+'/'+dirs+'/'+'miniboxes'+'/'+str(i)+"_"+str(j)+"_"+imtocrop.filename) 

### Classify and get necessary information

def getClass(imgname):
  img_path = imgname
  img = tf.keras.utils.load_img(
    img_path, target_size=(img_height, img_width)
  )
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions = model.predict(img_array,verbose = 0)
  score = tf.nn.softmax(predictions[0])

  # print(
  #     "This image most likely belongs to {} with a {:.2f} percent confidence."
  #     .format(class_names[np.argmax(score)], 100 * np.max(score))
  # )
  conf = 100*np.max(score)
  newsco = class_names[np.argmax(score)]
  new_string = imgname
  new_result = re.findall('[0-9]+', new_string)
  #print(img_path)
  #print(new_result)
  beam = int(new_result[6])+1
  ant = int(new_result[7])+1
  beamstr = str(beam)
  antstr = str(ant)
  #print("This image is from antenna: "+ antstr + " and beam: "+ beamstr)
  #print("Filename is: "+ imgname)
  #src_img = cv2.imread(img_path)
  #cv2_imshow(src_img)
  return conf, newsco
  
  
### Run the classifications and save information in a text file

def makeImg(SBIDC,txtfile):
  
  os.chdir(homedir)

  if(os.path.exists('/content/drive/MyDrive/ASTRPACE/PACE_F/'+txtfile) == False):
    f = open('/content/drive/MyDrive/ASTRPACE/PACE_F/'+txtfile,'a')
    print("MAKING NOW")
  # else:
  #   #print("HAAAAS F1 TEAM IS BAD")

  os.chdir(homedir+SBIDC)
  croppedList = glob.glob('*.png_DIR')

  #print(croppedList)
  for x in croppedList:
    plt.close('all')
    #print(x)
    fig, ax = plt.subplots(1,3,figsize =(23, 7),dpi=100)

    bigImg = glob.glob(homedir+SBIDC+'/'+x+'/*.png')
    print(bigImg)
    image = mpimg.imread(bigImg[0])
    ax[0].imshow(image)
    plt.clf()
    imname = str(x)[:-8]

    with open('/content/drive/MyDrive/ASTRPACE/PACE_F/'+txtfile) as myfile:
      if imname in myfile.read():
        print('HAVE PROCESSED: '+imname)
      else:
        freqstr = str(x)
        freq = freqstr.split("_autoscale_")[-1].split("MHz")[0]

        polstr = str(x)
        pol = polstr.split("MHz_")[-1].split(".png")[0]

        imgList = glob.glob(homedir+SBIDC+'/'+x+'/miniboxes/*.png')
        #print(imgList)
        if(len(imgList) == 1296):
            print('Have all miniboxes')
        else:
            print(str(len(imgList))+' is NOT ENOUGH MINIBOXES, MAKING NEW ONES')
            shutil.rmtree(homedir+SBIDC+'/'+x)
            firstCrop(SBID)
            makeMini(SBIDC)
            imgList = glob.glob(homedir+SBIDC+'/'+x+'/miniboxes/*.png')
            print('AFTER MAKING NEW MINIBOXES, NOW HAVE: '+str(len(imgList)))
        arrofsc = []
        arrofmse = []
        arrofssim = []
        arrofconf = []
        for img in imgList:
          conf, sco = getClass(img)
          arrofconf.append(conf)
          arrofsc.append(sco)
          if(sco == 'RFI'):
            box1im = cv2.imread(img)
            average_color = np.average(average_color_std, axis=0)
            d_img = np.ones(box1im.shape, dtype=np.uint8)
            d_img[:,:] = average_color
            dim = d_img
            m = mse(box1im, dim)
            ssi = compare_images(box1im, dim)
            arrofmse.append(m)
            arrofssim.append(ssi)
        
        confarr = np.array(arrofconf)
        confmedian = np.median(confarr)

        msearr = np.array(arrofmse)
        ssimarr = np.array(arrofssim)

        msemean = np.mean(msearr)
        msemed = np.median(msearr)

        ssimmean = np.mean(ssimarr)
        ssimmed = np.median(ssimarr)

        
        cnt = Counter()
        for word in arrofsc:
          cnt[word] += 1
        cnt
          ### https://www.geeksforgeeks.org/plot-a-pie-chart-in-python-using-matplotlib/
          # Creating dataset
        cl = cnt['CLEAN']
        co = cnt['CORR_DROP']
        rf = cnt['RFI']
        an = cnt['ANT_ERROR']
        data = [cl,co,rf,an]
        
        test = "IMGNAME = {} SBID = {} CLEAN = {} CORR_DROP = {} OTHER_ERR = {} ANT = {} FREQ = {} POL = {} CONFMEDIAN = {} MSEMEAN = {} MSEMEDIAN = {} SSIMMEAN = {} SSIMMEDIAN = {}".format(imname, SBIDC, cl, co, rf, an, freq, pol,confmedian,msemean,msemed,ssimmean,ssimmed)
        print(test)

        with open(homedir+SBIDC+'/'+x+'/'+x+'.txt','w') as f:
          f.write(test)
        
        with open('/content/drive/MyDrive/ASTRPACE/PACE_F/'+txtfile,'a') as appf:
          appf.write(test + "\n")

          # Creating explode data
        explode = (0.1, 0.0, 0.2, 0.3)
        
        # Creating color parameters
        colors = ( "orange", "cyan", "brown",
                  "grey")
        
        # Wedge properties
        wp = { 'linewidth' : 1, 'edgecolor' : "green" }
        
        # Creating autocpt arguments
        def func(pct, allvalues):
            absolute = int(pct / 100.*np.sum(allvalues))
            return "{:.1f}%\n({:d} g)".format(pct, absolute)
        
        # Creating plot
        
        wedges, texts, autotexts = ax[1].pie(data,
                                          autopct = lambda pct: func(pct, data),
                                          explode = explode,
                                          labels = types,
                                          shadow = False,
                                          colors = colors,
                                          startangle = 90,
                                          wedgeprops = wp,
                                          textprops = dict(color ="magenta"))
        
        # Adding legend
        ax[1].legend(wedges, types,
                  title ="CATEGORIES",
                  loc ="center left",
                  bbox_to_anchor =(1, 0, 0.5, 1))
        
        ax[1].set_title(str(x))
        plt.setp(autotexts, size = 4, weight ="bold")
        
        #ax[2].figure(figsize=(10,20))
        #ax[2] = fig.add_axes([0,0,1,1])
        ax[2].bar(types,data)
        ax[2].legend()
        plt.savefig(homedir+SBIDC+'/'+x+'/'+x+'.png')
        #plt.show()
        #plt.clf()
        #plt.close('all')
        # show plot
