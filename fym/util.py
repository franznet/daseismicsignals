import glob, os
import numpy as np
from obspy import read
from PIL import Image
import io

# ========================================= GENERAL =========================================
# Intersection of 3 lists
def intersection(lLista1, lLista2, lLista3):
    return list(set(lLista1) & set(lLista2) & set(lLista3))

# List of simple file names
def lista_archivos_simple(sRutaDirectorio:str):
  # Fix directory string
  if sRutaDirectorio[len(sRutaDirectorio)-1]!='/':
    sRutaDirectorio+='/'
  # List of files
  return list(map(os.path.basename, glob.glob(sRutaDirectorio+"*.*")))

# List of common file names for the 3 channels
def lista_archivos_comunes(sRutaDirectorio:str):
  # Fix directory string
  if sRutaDirectorio[len(sRutaDirectorio)-1]!='/':
    sRutaDirectorio+='/'
  # List of files
  mz=list(map(os.path.basename, glob.glob(sRutaDirectorio+"Z/*.*")))
  me=list(map(os.path.basename, glob.glob(sRutaDirectorio+"EW/*.*")))
  mn=list(map(os.path.basename, glob.glob(sRutaDirectorio+"NS/*.*")))
  # File names without extension
  mz=[i.rsplit('.')[0] for i in mz]
  me=[i.rsplit('.')[0] for i in me]
  mn=[i.rsplit('.')[0] for i in mn]
  # Intersection of lists
  return intersection(mz, me, mn)

# List of files in one channel
def archivos_canal(sRutaCanal,sArchivo:str, bRutaCompleta:bool=True):
    # Channel file
    mz=list(map(os.path.basename, glob.glob(sRutaCanal+"/Z/"+sArchivo+".*")))[0]
    me=list(map(os.path.basename, glob.glob(sRutaCanal+"/EW/"+sArchivo+".*")))[0]
    mn=list(map(os.path.basename, glob.glob(sRutaCanal+"/NS/"+sArchivo+".*")))[0]
    # Adding full path
    if bRutaCompleta:
      mz=sRutaCanal+"/Z/"+mz
      me=sRutaCanal+"/EW/"+me
      mn=sRutaCanal+"/NS/"+mn
    # Return the 3 channels
    return mz, me, mn

# List of files in one channel
def archivos_canal_simple(sRutaDirectorio, sArchivo:str, bRutaCompleta:bool=True):
  # Fix directory string
  if sRutaDirectorio[len(sRutaDirectorio)-1]!='/':
    sRutaDirectorio+='/'
  # Channel file
  mz=list(map(os.path.basename, glob.glob(sRutaDirectorio+sArchivo)))[0]
  # Adding full path
  if bRutaCompleta:
    mz=sRutaDirectorio+mz
  # Return the channel file
  return mz

# Create directory path
def create_folders(sDirectory):
  if not os.path.exists(sDirectory):
    os.makedirs(sDirectory)

def now_string():
  from datetime import datetime
  dateTimeObj = datetime.now()
  return dateTimeObj.strftime("%Y%m%d_%H%M%S")

# ===================================== MATPLOTLIB ========================================
# Create a Function for Converting a figure to a PIL Image.
def fig2img(fig):
  buf = io.BytesIO()
  fig.savefig(buf, dpi=100)
  buf.seek(0)
  img = Image.open(buf)
  return img

# ========================================= SIGNAL =========================================
def signal_preprocess(sRutaArchivo:str):
  # Open file
  tr = read(sRutaArchivo)[0]
  # Resample to 100
  tr.resample(100.0)
  # Subtract the mean from the signal
  tr.data = tr.data-np.mean(tr.data)
  # Highpass and bandpass filter
  tr.filter('highpass', freq=1)
  tr.filter('bandpass', freqmin=1, freqmax=10, corners=10)

  return tr

def tr_duration(tr):
  #return tr.stats.endtime-tz.stats.starttime)
  return tr.stats.npts/tr.stats.sampling_rate
