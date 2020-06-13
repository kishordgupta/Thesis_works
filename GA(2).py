import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy
import scipy.stats
from PIL import Image, ImageChops
import sys
import statistics
sys.path.append('./drive/My Drive/Code/')
from LoadImage import loadim
from joblib import Parallel,delayed
import random
import time

def totaltestcollection():
  imgs=[]
  imgs = loadim("./drive/My Drive/MNIST/jsma")
  imgs = imgs + loadim("./drive/My Drive/MNIST/cw2")
  imgs = imgs + loadim("./drive/My Drive/MNIST/df")
  imgs = imgs + loadim("./drive/My Drive/MNIST/fsgm")
  imgs = imgs + loadim("./drive/My Drive/MNIST/advgan")
  imgs = imgs + loadim("./drive/My Drive/MNIST/BPDA")
  imgs = imgs + loadim("./drive/My Drive/MNIST/zoo")
  imgs = imgs + loadim("./drive/My Drive/MNIST/lbs")
  imgs = imgs + loadim("./drive/My Drive/MNIST/lbfgs")
  imgs = imgs + loadim("./drive/My Drive/MNIST/localbinarysearch")
  return imgs

def totalcleancollection():
  imgs=[]
  imgs =loadim("./drive/My Drive/MNIST/test")
  return imgs

def evaluate1(filterlist,cd):
  from FilterList import MedianBlur as FT1
  from FilterList import AverageBlur as FT2
  from FilterList import bilateral as FT3
  from FilterList import GaussianBlur as FT4
  from FilterList import anp as FT5
  from FilterList import an as FT6
  from FilterList import errosion as FT7
  from FilterList import dilate as FT8
  from FilterList import opening as FT9
  from FilterList import closing as FT10
  from FilterList import morphology as FT11
  from FilterList import TopHat as FT12
  from FilterList import blackhat as FT13
  count =len(cd)
  #print(filterlist)
  #print(str(len(cd)))
  for filter in filterlist:
    cd=eval('FT'+str(filter)+'(cd,0,True)')
    #print(str(len(cd)))
    #if len(cd)==0:
      #print("100%detected")
  tot =count - len(cd)
  accuracy=tot/count;        
  return accuracy

def remove_duplicates(l):
    newl=[]
    for a in l:
      if a in newl:continue
      newl.append(a)
    return newl
def randomge(min=2,max=13):
  data = []
  len =random.randint(3,9)
  while len>0:
    len=len-1
    data.append(random.randint(min,max))
  return remove_duplicates(data)

def get_data(pop,ic):
    tic = time.clock()
    fitnessvalue = evaluate(pop,ic)
    toc = time.clock()
    timedata=toc - tic
    dictind ={'Individual':pop,'fitness':fitnessvalue,'time':timedata}
    return dictind

def create_starting_population(individuals,ic, chromosome_length=0,Varaiable_length=True):
    # Set up an initial array of all zeros
    indicator = []
    population=[]
    # Loop through each row (individual)
    for i in range(individuals):
        pop = randomge()
        if pop in indicator:
          continue
        indicator.append(pop)
        population.append(get_data(pop,ic))


    #population = Parallel(n_jobs=10,verbose=1)(delayed(get_data)(pop,ic) for pop in indicator)
    return sorted(population, key = lambda i: i['fitness'],reverse=True) 

def singlepontcrossover(population,ind1,ind2,ic):  
  point1=random.randint(2,len(ind1['Individual']))
  point2=random.randint(2,len(ind2['Individual']))
  newpop1=list(ind1['Individual'])[:point1]+list(ind2['Individual'])[point2:]
  newpop1=remove_duplicates(newpop1)
  tic = time.clock()
  fitnessvalue = evaluate(newpop1,ic)
  toc = time.clock()
  timedata=toc - tic
  dictind1 ={'Individual':newpop1,'fitness':fitnessvalue,'time':timedata}
  newpop2=list(ind2['Individual'])[:point2]+list(ind1['Individual'])[point1:]
  newpop2=remove_duplicates(newpop2)
  tic = time.clock()
  fitnessvalue = evaluate(newpop2,ic)
  toc = time.clock()
  timedata=toc - tic
  dictind2 ={'Individual':newpop2,'fitness':fitnessvalue,'time':timedata}
  population.append(dictind1)
  population.append(dictind2)
  return population

def clean_population(a):
  pop = []
  final_list = []
  for value in a:
    if value['Individual'] not in pop:
     final_list.append(value)
     pop.append(value['Individual'])
  return final_list[:10] 

def mutation(population,ic,probaility=1,min=1,max=13):
  pr=random.randint(1,100)
  if pr/100<=probaility:
    id=random.randint(1,len(population)-1)
    if len(population[id]['Individual'])>2:
      index =random.randint(1,len(population[id]['Individual']))
      newvalue=random.randint(min,max)
      population[id]['Individual'][index-1]=newvalue
      newpop1=remove_duplicates(population[id]['Individual'])
      population[id]['Individual']=newpop1
  return sorted(population, key = lambda i: i['fitness'],reverse=True)

def crossovers(population,ic,top=8):
  #top=int(len(population)/top)
  i=0
  for i in range(0,top):
    j=i+1
    for j in range(0,top):
      ind1=population[i]
      ind2=population[j]
      population=population+singlepontcrossover(population,ind1,ind2,ic)
  return sorted(population, key = lambda i: i['fitness'],reverse=True)

def ga(ic,terminationtime=10):
  i=0
  a = create_starting_population(10,ic)
  print(a)
  for i in range(0,terminationtime):
     a=crossovers(a[:10],ic)
     a=mutation(a[:10],ic)
     a=clean_population(a[:10])
     selection(a)
     print(a)
  return a
def evaluate(filterlist,cd):
  from FilterListPhase2 import Filters
  from FilterListPhase2 import FT1,FT2,FT3,FT4,FT5,FT6,FT7,FT8,FT9,FT10,FT11,FT12,FT13
  count =len(cd)
  #print(filterlist)
  #print(str(len(cd)))
  for filter in filterlist:
    #cd=eval('Filters(cd,FT'+str(filter)+',0,True)')
    cd=Filters(cd,'FT'+str(filter),0,True)
    #print(str(len(cd)))
    #if len(cd)==0:
      #print("100%detected")
  tot =count - len(cd)
  accuracy=tot/count;        
  return accuracy
def selection(population):
  return sorted(population, key = lambda i: i['fitness'],reverse=True)
