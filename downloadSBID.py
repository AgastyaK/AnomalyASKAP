### Author: Agastya Kapur#
### File to download images directory from the website 'https://www.atnf.csiro.au/projects/askap/diagnostics/'

import numpy as np
import matplotlib.pyplot as plt
import wget
import matplotlib.image as mpimg
from google. colab import files
import glob
import os.path
from PIL import Image
import os.path 
import re
from pathlib import Path    
import pandas as pd


def geturls(url,SBID):
  with open(SBID) as f:
    inwget = f.readlines()
    URL = []
    for line in np.arange(len(inwget)):
      string = str(inwget[line])
      if string.find("ampantbeam_waterfall_autoscale") != -1:
        URL.append(re.findall("<a href='(.*)'>", string))
  return URL

### Would need to have a file called SBID.txt (or anything really)
### with SBID names separated in lines

with open('SBID.txt') as f:
    SBID = f.readlines()
    for idnum in np.arange(len(SBID)):
      SBIDC = str(SBID[idnum])[0:5]
      weburl = 'https://www.atnf.csiro.au/projects/askap/diagnostics/'+SBIDC
      if os.path.exists(SBIDC):
          print("HAS SBID FILE")
          #print("REMOVING AND DOWNLOADING")
          #os.remove(SBIDC)
          #wget.download(weburl)
      else:
          print("DOWNLOADING")
          wget.download(weburl)
      URLS = geturls(weburl,SBIDC)
      #print(URLS)
      print(SBIDC)
      for url in np.arange(len(URLS)):
          precurl = str(URLS[url])
          fname = precurl[2:-2]
          print(fname)
          file_exists = os.path.exists(fname)
          if(file_exists):
              print('Have: ' + fname)
          else:
              print("Don't have: " +fname)
              print(weburl+"/"+fname)
              wget.download(weburl+"/"+fname)
      
