# Author: Agastya Kapur
# Sample file 'FOUR.txt'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def doPlots(txtfile):
  firstload = pd.read_csv(txtfile, delim_whitespace=True,header = None)
  for col in firstload:
    uniq = firstload[20].unique()
  uniq = np.sort(uniq)
  print(uniq)
  for frequ in uniq:
    freq = frequ
    #print('FREQ: '+str(freq))

    #data = firstload[20:40].loc[firstload[20] == freq]
    data = firstload.loc[firstload[20] == freq]
    #df.loc[df['col1'] == value]
    cols_as_np = data[data.columns[0:]].to_numpy()

    #print(cols_as_np)
    # #print(cols_as_np)
    SBIDarr = []
    Cleanarr = []
    Corrarr = []
    Otherarr = []
    Antarr = []
    Freqarr = []
    Polarr = []
    msearr = []
    ssimarr = []

    for each in np.arange(0,cols_as_np.shape[0]):
      SBID = cols_as_np[int(each)][5]
      SBIDarr.append(SBID)

      Clean = cols_as_np[int(each)][8]
      Cleanarr.append(Clean)

      Corr = cols_as_np[int(each)][11]
      Corrarr.append(Corr)

      Other = cols_as_np[int(each)][14]
      Otherarr.append(Other)

      Ant = cols_as_np[int(each)][17]
      Antarr.append(Ant)

      Freq = cols_as_np[int(each)][17]
      Freqarr.append(Freq)

      Pol = cols_as_np[int(each)][23]
      Polarr.append(Pol)

      mse = cols_as_np[int(each)][29]
      msearr.append(mse)

      ssim = cols_as_np[int(each)][35]
      ssimarr.append(ssim)

    # print(SBIDarr)
    # print(Cleanarr) 
    # print(Corrarr)
    # print(Otherarr)
    # print(Antarr)
    # print(Polarr)
    # print(msearr)
    # print(ssimarr)

    markarr = []
    for pol in np.arange(0,np.size(Polarr)):
      if(Polarr[pol] == "XX"):
        markarr.append("x")
      elif(Polarr[pol] == 'YY'):
        markarr.append("1")
    #print(markarr)
    #np.size(Polarr)
    plt.figure(figsize=(16,8))
    #print('FREQ: '+str(freq))
    for _m, _s, _cl, _co, _ot, _an in zip(markarr, SBIDarr, Cleanarr, Corrarr, Otherarr, Antarr):
        plt.title('Classifications of Anomalies at Frequency: '+ str(freq)+' MHz')
        plt.ylabel('Number of miniboxes')
        plt.xlabel('SBID number')
        plt.scatter(_s, _cl, marker=_m, c='g',label = 'Clean')
        plt.scatter(_s, _co, marker=_m, c='r',label = 'Correlator Dropout')
        plt.scatter(_s, _ot, marker=_m, c='b',label = 'Other Error')
        plt.scatter(_s, _an, marker=_m, c='y',label = 'Antenna/Beam Error')
        plt.ticklabel_format(useOffset=False)
        #plt.legend()
    plt.show()
    plt.figure(figsize=(16,8))
    for _m, _s, _mse in zip(markarr, SBIDarr,msearr):
        plt.title('MSE Values of RFI classified anomalies at Frequency: '+ str(freq)+' MHz')
        plt.ylabel('MSE Value')
        plt.xlabel('SBID number')
        plt.scatter(_s, _mse, marker=_m, c='m',label = 'MSE Value')
        plt.ticklabel_format(useOffset=False)
        #plt.legend()
    plt.show()
    plt.figure(figsize=(16,8))
    for _m, _s, _ssim in zip(markarr, SBIDarr,ssimarr):
        plt.title('(1 - SSIM) Values of RFI classified anomalies at Frequency: '+ str(freq)+' MHz')
        plt.ylabel('SSIM Value')
        plt.xlabel('SBID number')
        plt.scatter(_s, 1-_ssim, marker=_m, c='c',label='SSIM Value')
        #plt.legend()
        plt.ticklabel_format(useOffset=False)
    plt.show()
    
    
doPlots("FOUR.txt")
    
