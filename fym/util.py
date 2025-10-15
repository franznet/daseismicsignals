import glob, os
import numpy as np
from obspy import read
from PIL import Image
import io

# ========================================= GENERALES =========================================
# Interseccion de 3 listas
def intersection(lLista1, lLista2, lLista3):
    return list(set(lLista1) & set(lLista2) & set(lLista3))

# Lista de nombres de archivo simples
def lista_archivos_simple(sRutaDirectorio:str):
  # Corregir cadena directorio
  if sRutaDirectorio[len(sRutaDirectorio)-1]!='/':
    sRutaDirectorio+='/'
  # Lista de archivos
  return list(map(os.path.basename, glob.glob(sRutaDirectorio+"*.*")))

# Lista de nombres de archivo comunes en los 3 canales
def lista_archivos_comunes(sRutaDirectorio:str):
  # Corregir cadena directorio
  if sRutaDirectorio[len(sRutaDirectorio)-1]!='/':
    sRutaDirectorio+='/'
  # Lista de archivos
  mz=list(map(os.path.basename, glob.glob(sRutaDirectorio+"Z/*.*")))
  me=list(map(os.path.basename, glob.glob(sRutaDirectorio+"EW/*.*")))
  mn=list(map(os.path.basename, glob.glob(sRutaDirectorio+"NS/*.*")))
  #Lista de archivos son extension
  mz=[i.rsplit('.')[0] for i in mz]
  me=[i.rsplit('.')[0] for i in me]
  mn=[i.rsplit('.')[0] for i in mn]
  # Interseccion de listas
  return intersection(mz, me, mn)

#Lista de archivos en los 3 canales
def archivos_canal(sRutaCanal,sArchivo:str, bRutaCompleta:bool=True):
    # Archivo d canal
    mz=list(map(os.path.basename, glob.glob(sRutaCanal+"/Z/"+sArchivo+".*")))[0]
    me=list(map(os.path.basename, glob.glob(sRutaCanal+"/EW/"+sArchivo+".*")))[0]
    mn=list(map(os.path.basename, glob.glob(sRutaCanal+"/NS/"+sArchivo+".*")))[0]
    # Agregando ruta completa
    if bRutaCompleta:
      mz=sRutaCanal+"/Z/"+mz
      me=sRutaCanal+"/EW/"+me
      mn=sRutaCanal+"/NS/"+mn
    # Retorno de los 3 canales
    return mz, me, mn

#Lista de archivos en 1 canal
def archivos_canal_simple(sRutaDirectorio, sArchivo:str, bRutaCompleta:bool=True):
  # Corregir cadena directorio
  if sRutaDirectorio[len(sRutaDirectorio)-1]!='/':
    sRutaDirectorio+='/'
  # Archivo d canal
  mz=list(map(os.path.basename, glob.glob(sRutaDirectorio+sArchivo)))[0]
  # Agregando ruta completa
  if bRutaCompleta:
    mz=sRutaDirectorio+mz
  # Retorno de los 3 canales
  return mz

# Crear ruta de directorios
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

# ========================================= SEÑAL =========================================
def signal_preprocess(sRutaArchivo:str):
  #Apertura de archivo
  tr = read(sRutaArchivo)[0]
  # Resampling a 100
  tr.resample(100.0)
  # Restando la media de la señal
  tr.data = tr.data-np.mean(tr.data)
  # Filtro paso alto y paso banda
  tr.filter('highpass', freq=1)
  tr.filter('bandpass', freqmin=1, freqmax=10, corners=10)

  return tr

def tr_duration(tr):
  #return tr.stats.endtime-tz.stats.starttime)
  return tr.stats.npts/tr.stats.sampling_rate
