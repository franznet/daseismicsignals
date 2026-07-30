import numpy as np
import matplotlib.pyplot as plt
from obspy.core import read
import fym.util as fym
from fym.signal import TSignal, TListSignal
from random import randint, choice, sample
from subprocess import call

# Constants
class FLAG_DA:
  sRutaEntrada = 'escenario2/data/'                             # Folder of original MiniSEED signals
  sRutaSalida  = 'escenario2/01/data_augmentation_jittering/' # Output folder for spectrograms
  lEvento      = ['HY','LP','TR','VT']                          # List of events to process

def generarEventoJittering(sRutaEntrada:str, sRutaSalida:str, dEvento:dict, fSigmaInicio:float=0.2, fSigmaFin:float=None):
  """Generate spectrogram images by applying jittering (noise) within a sigma range.
  Args:
    sRutaEntrada (str): Folder where events are stored organized by event subfolders
    sRutaSalida (str): Folder where generated events will be saved
    dEvento (dict): Dictionary of events and quantities to generate per event, e.g. {'HY':2000, 'LP':1500, 'TC':0, 'TR':1800, 'VT':1350}
    fSigmaInicio (float): Noise standard deviation start value
    fSigmaFin (float): Noise standard deviation end value
  """
  if fSigmaFin is None:
    sCarpetaProceso=str(fSigmaInicio)
  else:
    sCarpetaProceso=str(fSigmaInicio)+'-'+str(fSigmaFin)
  # Reading events
  for sEvento in dEvento:
    print("Generando eventos:", sEvento, fym.now_string())
    # Read list of event files from folder
    m=fym.lista_archivos_simple(sRutaEntrada+sEvento)
    if len(m)>0:
      # Generate list of tuples (fileIndex, sigma)
      lJittering=[]
      for i in range(len(m)):                                                 # File index
        if fSigmaFin is None:
          for s in range(dEvento[sEvento]):  # Quantity
            lJittering.append((i, fSigmaInicio))
        else:
          # Generate points across sigma range
          for s in np.linspace(fSigmaInicio, fSigmaFin, num=dEvento[sEvento]):  # Sigma range
            lJittering.append((i, round(s, 10)))
        # Generate a sample list of items to create spectrograms, without repetition
        lMuestra = sample(lJittering, dEvento[sEvento])
      # Create output folders if they do not exist
      fym.create_folders(sRutaSalida+sCarpetaProceso+'/'+sEvento)
      # Generate spectrograms for the requested sample size
      for iCont, (iArchivo, fSigma) in enumerate(lMuestra):
        # Get event file path
        sRuta = fym.archivos_canal_simple(sRutaEntrada+sEvento, m[iArchivo])
        # Open the event
        tr = TSignal(sRuta)
        # Preprocess
        tr.preproceso()  #<==============================================================================
        # Normalize signals
        tr.normaliza()
        # Remove noise (optional)
        #tr.eliminaRuido(fRango=0.1, fTolerancia=1.0)
        # Generate new signal (apply jittering)
        tr.daJittering(fSigma)
        # Save spectrogram to disk
        tr.espectrograma_guardar_canal(0, sRutaSalida+sCarpetaProceso+'/'+sEvento, 224,'_'+str(fSigma)+'-'+str(iCont+1))

      # Free variables from memory
      del lJittering
      del lMuestra
    # Message
    print("Fin de generación de eventos :", sEvento, fym.now_string())

#=============================================== PROCESO ESPECTROGRAMA ESCENARIO 2 20HZ ============================================
print("Start:", fym.now_string())
#generarEventoJittering(FLAG_DA.sRutaEntrada, FLAG_DA.sRutaSalida, {'HY':2473, 'LP':2109, 'TC':488 ,'TR':2215}, 0.2)
generarEventoJittering(FLAG_DA.sRutaEntrada, FLAG_DA.sRutaSalida, {'HY':2686, 'LP':2686, 'TC':2686 ,'TR':2686, 'VT':2686}, 0.2)
#generarEventoJittering(FLAG_DA.sRutaEntrada, FLAG_DA.sRutaSalida, {'HY':13419, 'LP':13419, 'TC':13419 ,'TR':13419, 'VT':13419}, 0.2)
print("End   :", fym.now_string())


