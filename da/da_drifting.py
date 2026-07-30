import numpy as np
import matplotlib.pyplot as plt
from obspy.core import read
import fym.util as fym
from fym.signal import TSignal, TListSignal
from random import randint, choice, sample
from subprocess import call
import pickle
import os, glob, shutil

# Constants
class FLAG_DA:
  sRutaEntrada = 'escenario2/data/'       # Folder of original MiniSEED signals
  sRutaSalida  = 'E:/spectrograms224_00/' # Output folder for spectrograms
  lEvento      = ['HY','LP','TR','VT']    # List of events to process

def guarda_lista(nombre_archivo, lista_a_guardar):
  archivo = open(nombre_archivo, "wb")
  pickle.dump(lista_a_guardar, archivo)
  archivo.close()

def lee_lista(nombre_archivo):
  archivo = open(nombre_archivo, "rb")
  lista_leida = pickle.load(archivo)
  archivo.close()
  return lista_leida

def generaMuestraArchivos(sRutaEntrada:str, sRutaSalida:str, dEvento:dict):
  """Generate a random sample of events without repetition into the destination folder.
  Args:
      sRutaEntrada (str): Folder where the original events are located organized by event subfolders
      sRutaSalida (str): Folder where the sampled events will be created
      dEvento (dict): Dictionary of events and quantities to generate per event, e.g. {'HY':2000, 'LP':1500, 'TC':0, 'TR':1800, 'VT':1350}
  """
  # Reading events
  for sEvento in dEvento:
    print("Generando muestra de eventos:", sEvento, fym.now_string())
    # Read list of event files from folder
    #m=fym.lista_archivos_simple(sRutaEntrada+sEvento)
    m=list(map(os.path.basename, glob.glob(sRutaEntrada+sEvento+"/*.*")))
    if len(m)>0:
      # Generate a list containing a sample of events without repetition
      lMuestra = sample(m, dEvento[sEvento])
      # Create output folder if it does not exist
      #fym.create_folders(sRutaSalida+sEvento)
      if not os.path.exists(sRutaSalida+sEvento):
        os.makedirs(sRutaSalida+sEvento)
      # Copy files to output path
      for fArchivo in lMuestra:
        shutil.copy(sRutaEntrada+sEvento+'/'+fArchivo, sRutaSalida+sEvento)
    # Message
    print("Fin de generación de eventos :", sEvento, fym.now_string())
  return

def generarEventoDrifting(sRutaEntrada:str, sRutaSalida:str, dEvento:dict, fDerivaInicio:int, fDerivaFin:int):
  """Generate spectrogram images by applying drift within a defined percentage range at AIF locations without repetition.
  Args:
      sRutaEntrada (str): Folder where events are stored organized by event subfolders
      sRutaSalida (str): Folder where the generated events will be saved
      dEvento (dict): Dictionary of events and quantities to generate per event, e.g. {'HY':2000, 'LP':1500, 'TC':0, 'TR':1800, 'VT':1350}
      fDerivaInicio (float): Drift start value (initial range value)
      fDerivaFin (float): Drift end value (final range value)
  """
  # Reading events
  for sEvento in dEvento:
    print("Generando eventos:", sEvento, fym.now_string())
    # Read list of event files from folder
    m=fym.lista_archivos_simple(sRutaEntrada+sEvento)
    if len(m)>0:
      # Generate list of tuples (fileIndex, percentage)
      lJittering=[]
      for i in range(len(m)):                                                   # File index
        for d in np.linspace(fDerivaInicio, fDerivaFin, num=dEvento[sEvento]):  # Drift range
          lJittering.append((i, round(d,10)))
      # Generate a list that contains sample items to generate spectrograms, without repetition
      lMuestra = sample(lJittering, dEvento[sEvento])
      # Create output folders if they do not exist
      fym.create_folders(sRutaSalida+sEvento)
      # Generate spectrograms for the requested sample size
      for iArchivo, fDeriva in lMuestra:
        # Get event file path
        sRuta = fym.archivos_canal_simple(sRutaEntrada+sEvento, m[iArchivo])
        # Open the event
        tr = TSignal(sRuta)
        # Preprocess (variant 2)
        tr.preproceso2()  #<==============================================================================
        # Normalize signals
        tr.normaliza()
        # Remove noise (optional)
        #tr.eliminaRuido(fRango=0.1, fTolerancia=1.0)
        # Generate new signal (apply drifting/jittering)
        tr.daJittering(fDeriva)
        # Save spectrogram to disk
        tr.espectrograma_guardar_canal(0, sRutaSalida+sEvento, 224,'_'+str(fDeriva))
      # Free variables from memory
      del lJittering
      del lMuestra
    # Message
    print("Fin de generación de eventos :", sEvento, fym.now_string())

#=============================================== SPECTROGRAM PROCESS SCENARIO 2 20HZ ============================================
print("Start:", fym.now_string())
#generarEventoDrifting(FLAG_DA.sRutaEntrada, 'escenario2/spectrograms224_00 210708 20Hz+da_drifting/', {'HY':2473, 'LP':2109, 'TC':488 ,'TR':2215}, 0.01, 0.1)
#generarEventoDrifting(FLAG_DA.sRutaEntrada, 'escenario2/spectrograms224_00 210708 20Hz+da_drifting/', {'HY':2473, 'LP':2109, 'TC':488 ,'TR':2215}, 0.001, 0.01)
#generarEventoDrifting(FLAG_DA.sRutaEntrada, 'escenario2/spectrograms224_00 210708 20Hz+da_drifting/', {'HY':2473, 'LP':2109, 'TC':488 ,'TR':2215}, 0.0001, 0.001)
#generarEventoDrifting(FLAG_DA.sRutaEntrada, 'escenario2/spectrograms224_00 210708 20Hz+da_drifting/', {'HY':13419, 'LP':13419, 'TC':13419 ,'TR':13419, 'VT':13419}, 0.001, 0.01)
#generarEventoDrifting(FLAG_DA.sRutaEntrada, 'escenario2/spectrograms224_00 210708 20Hz+da_drifting/', {'HY':13419, 'LP':13419, 'TC':13419 ,'TR':13419, 'VT':13419}, 0.01, 0.1)
#generarEventoDrifting(FLAG_DA.sRutaEntrada, 'escenario2/spectrograms224_00 210708 20Hz+da_drifting/', {'HY':13419, 'LP':13419, 'TC':13419 ,'TR':13419, 'VT':13419}, 0.0001, 0.001)
generaMuestraArchivos('escenario2/spectrograms224_00/', 'escenario2/spectrograms224_00 muestra/', {'HY':2686, 'LP':2686, 'TC':2686 ,'TR':2686, 'VT':2686})
print("End   :", fym.now_string())


