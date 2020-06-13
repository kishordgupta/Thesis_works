from PIL import Image
import os, os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def loadim(path):
  imgs = []
  path = path#"./drive/My Drive/BPDA/"
  valid_images = [".jpg",".gif",".png",".tga"]
  for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append(Image.open(os.path.join(path,f)))
  return imgs

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return "%.3f" % np.where(sd == 0, 0, m/sd)

def runclean():
  imgs =loadim("./drive/My Drive/MNIST/test")
  print("clean")
  return imgs

#runallaverageblur2(imgs4,0)
#b("./drive/My Drive/BPDA/")
#img=mpimg.imread('your_image.png')
#imgplot = plt.imshow(imgs[0])
#plt.show()
#print(len(imgs))
#print(imgs[0])