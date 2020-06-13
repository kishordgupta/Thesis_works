import cv2
import numpy as np
import os, os.path
from matplotlib import pyplot as plt
import scipy
import scipy.stats
from PIL import Image, ImageChops
import sys
import statistics
sys.path.append('./drive/My Drive/Code/')
from LoadImage import loadim
from LoadImage import signaltonoise
from LoadImage import runclean
from imgaug import augmenters as iaa
from joblib import Parallel,delayed


def Filters(imgs,FT='FT1',test=1,phase2=False):
  notdetectedimgs= Parallel(n_jobs=20,verbose=1)(delayed(evalcall)(im,FT,test,phase2) for im in imgs)
  #notdetectedimgs = [evalcall(im,FT,test,phase2) for im in imgs]
  notdetectedimgs = [a for a in notdetectedimgs if len(np.array(a))>0]  
  return notdetectedimgs

def evalcall(im,FT,test=1,phase2=False):
  return eval(FT+'(im,test,phase2)')

def FT1(im,test=1,phase2=False,upperrange=0.7,lowerrange=0.15):#medianblur
  img = np.array(im)
  if test!= 0:
    img = img[:,:,:3]
  kernel = np.ones((3,3),np.float32)/25
  dst = cv2.medianBlur(img,25)
  snr2 = float(signaltonoise(dst, axis=None))
  index = img==dst
  dst2 = np.abs(img-dst)
  snr3 = float(signaltonoise(dst2, axis=None))
  if phase2 == True:
    if snr3<upperrange and snr3>lowerrange:
      return im
  return []
  
def FT2(im,test=1,phase2=False,upperrange=0.85,lowerrange=0.25):#GaussianBlur
  img = np.array(im)
  if test!= 0:
    img = img[:,:,:3]
  kernel = np.ones((3,3),np.float32)/25
  dst = cv2.GaussianBlur(img,(3,3),0)
  snr2 = float(signaltonoise(dst, axis=None))
  index = img==dst
  dst2 = np.abs(img-dst)
  snr3 = float(signaltonoise(dst2, axis=None))
  if phase2 == True:
    if snr3<upperrange and snr3>lowerrange:
      return im
  return []

def FT3(im,test=1,phase2=False,upperrange=1.1,lowerrange=0.45):#AverageBlur
  img = np.array(im)
  if test!= 0:
    img = img[:,:,:3]
  kernel = np.ones((3,3),np.float32)/25
  dst = cv2.blur(img,(5,5),0)
  snr2 = float(signaltonoise(dst, axis=None))
  index = img==dst
  dst2 = np.abs(img-dst)
  snr3 = float(signaltonoise(dst2, axis=None))
  if phase2 == True:
    if snr3<upperrange and snr3>lowerrange:
      return im
  return []
  
def FT4(im,test=1,phase2=False,upperrange=1.1,lowerrange=0.3):#Bilateral blur
  img = np.array(im)
  if test!= 0:
    img = img[:,:,:3]
  try:img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  except:img=img    
  dst = cv2.bilateralFilter(img,6,75,75)
  snr2 = float(signaltonoise(dst, axis=None))
  index = img==dst
  dst2 = np.abs(img-dst)
  snr3 = float(signaltonoise(dst2, axis=None))
  if phase2 == True:
    if snr3<upperrange and snr3>lowerrange:
      return im
  return []
  
def FT5(im,test=1,phase2=False,upperrange=1.3,lowerrange=0.7):#AdditivePoissonNoise
  img = np.array(im)
  if test!= 0:
    img = img[:,:,:3]
  kernel = np.ones((3,3),np.float32)/25
  aug = iaa.AdditivePoissonNoise(lam=10.0, per_channel=True)
  im_arr = aug.augment_image(img)    
  dst2 = np.abs(img-np.array(im_arr))
  snr3 = float(signaltonoise(dst2, axis=None))
  if phase2 == True:
    if snr3<upperrange and snr3>lowerrange:
      return im
  return []
  
def FT6(im,test=1,phase2=False,upperrange=1.3,lowerrange=0.7):#AdditivePoissonNoise
  img = np.array(im)
  if test!= 0:
    img = img[:,:,:3]
  aug = iaa.AdditiveGaussianNoise(loc=0, scale=0.1*255)
  im_arr = aug.augment_image(img)    
  dst2 = np.abs(img-np.array(im_arr))
  snr3 = float(signaltonoise(dst2, axis=None))
  if phase2 == True:
    if snr3<upperrange and snr3>lowerrange:
      return im
  return []
  
def FT7(im,test=1,phase2=False,upperrange=0.75,lowerrange=0.25):#Erode
  img = np.array(im)
  if test!= 0:
    img = img[:,:,:3]
  try:img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  except:img=img
  kernel = np.ones((5,5),np.uint8)
  dst = cv2.erode(img,kernel,iterations = 1)
  snr2 = float(signaltonoise(dst, axis=None))
  index = img==dst
  dst2 = np.abs(img-dst)
  snr3 = float(signaltonoise(dst2, axis=None))
  if phase2 == True:
    if snr3<upperrange and snr3>lowerrange:
      return im
  return []
  
def FT8(im,test=1,phase2=False,upperrange=0.7,lowerrange=0.25):#Dialte
  img = np.array(im)
  if test!= 0:
    img = img[:,:,:3]
  try:img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  except:img=img
  kernel = np.ones((5,5),np.uint8)
  dst = cv2.dilate(img,kernel,iterations = 1)
  snr2 = float(signaltonoise(dst, axis=None))
  index = img==dst
  dst2 = np.abs(img-dst)
  snr3 = float(signaltonoise(dst2, axis=None))
  if phase2 == True:
    if snr3<upperrange and snr3>lowerrange:
      return im
  return []
  
def FT9(im,test=1,phase2=False,upperrange=0.7,lowerrange=0.25):#opening
  img = np.array(im)
  if test!= 0:
    img = img[:,:,:3]
  try:img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  except:img=img
  kernel = np.ones((5,5),np.uint8)
  dst = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
  snr2 = float(signaltonoise(dst, axis=None))
  index = img==dst
  dst2 = np.abs(img-dst)
  snr3 = float(signaltonoise(dst2, axis=None))
  if phase2 == True:
    if snr3<upperrange and snr3>lowerrange:
      return im
  return []
  
def FT10(im,test=1,phase2=False,upperrange=0.55,lowerrange=0.7):#closing
  img = np.array(im)
  if test!= 0:
    img = img[:,:,:3]
  try:img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  except:img=img
  kernel = np.ones((5,5),np.uint8)
  dst = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
  snr2 = float(signaltonoise(dst, axis=None))
  index = img==dst
  dst2 = np.abs(img-dst)
  snr3 = float(signaltonoise(dst2, axis=None))
  if phase2 == True:
    if snr3<upperrange and snr3>lowerrange:
      return im
  return []
  
def FT11(im,test=1,phase2=False,upperrange=0.7,lowerrange=0.25):#Morphology_gradient
  img = np.array(im)
  if test!= 0:
    img = img[:,:,:3]
  try:img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  except:img=img
  kernel = np.ones((5,5),np.uint8)
  dst = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
  snr2 = float(signaltonoise(dst, axis=None))
  index = img==dst
  dst2 = np.abs(img-dst)
  snr3 = float(signaltonoise(dst2, axis=None))
  if phase2 == True:
    if snr3<upperrange and snr3>lowerrange:
      return im
  return []
  
def FT12(im,test=1,phase2=False,upperrange=0.6,lowerrange=0.15):#TopHat
  img = np.array(im)
  if test!= 0:
    img = img[:,:,:3]
  try:img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  except:img=img
  kernel = np.ones((3,3),np.uint8)
  dst = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
  snr2 = float(signaltonoise(dst, axis=None))
  index = img==dst
  dst2 = np.abs(img-dst)
  snr3 = float(signaltonoise(dst2, axis=None))
  if phase2 == True:
    if snr3<upperrange and snr3>lowerrange:
      return im
  return []
  
def FT13(im,test=1,phase2=False,upperrange=0.65,lowerrange=0.15):#Blackhat
  img = np.array(im)
  if test!= 0:
    img = img[:,:,:3]
  try:img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  except:ing=img
  kernel = np.ones((3,3),np.uint8)
  dst = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
  snr2 = float(signaltonoise(dst, axis=None))
  index = img==dst
  dst2 = np.abs(img-dst)
  snr3 = float(signaltonoise(dst2, axis=None))
  if phase2 == True:
    if snr3<upperrange and snr3>lowerrange:
      return im
  return []
  

