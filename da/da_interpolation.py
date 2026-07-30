import numpy as np
import matplotlib.pyplot as plt
from obspy.core import read
import fym.util as fym
from fym.signal import TSignal, TListSignal
from random import randint, choice, sample
from subprocess import call
import random
import torch.nn as nn
from fym.autoencoder import AE, CAE3

# Constants
RUTA_ENTRADA    ='escenario2/data/'
RUTA_SALIDA     ='escenario2/01/data_augmentation_interpolation/'
EVENTO_ESTUDIO  =['HY','LP','TC','TR','VT']

# Returns a list of unique random integer indices within the specified range
def listaEnteroAleatorio(iMenor:int, iMaximo:int, iElementos:int):
  lLista=[]
  while len(lLista)<iElementos:
    iAleatorio=random.randint(iMenor, iMaximo)
    if iAleatorio not in lLista: lLista.append(iAleatorio)
    #else: print(iAleatorio, ' ya esta en lista ',lLista, ' rango[', iMenor,',',iMaximo,']')
  lLista.sort()
  return lLista

# Generate new events by interpolation (spectrogram saving)
def generarEventoRangoAleatorio(sRutaEntrada:str, sRutaSalida:str, lEvento:list,  iCantidad:int, ModeloAE:nn.Module, fPorcentajeMin:float=0.4, fPorcentajeMax:float=0.6):
  """Generate seismogram and spectrogram images via interpolation using an Autoencoder.
  Args:
      sRutaEntrada (str): Folder where events are stored in event subfolders
      sRutaSalida (str): Folder where generated events will be saved
      lEvento (list): Events to consider for generation
      iCantidad (int): Number of events to generate
      fPorcentaje(float, optional): Percentage used for interpolation between two signals. Range [0, 1]. 0=self, 1=other. Uses 3 decimal places like 45.123%. Defaults to 0.5.
  """
  # Reading events
  for sEvento in lEvento:
    # Read list of event files from folder
    m=fym.lista_archivos_simple(sRutaEntrada+sEvento)
    if len(m)>0:
      # Create output folders if they do not exist
      fym.create_folders(sRutaSalida+"{0:.3f}".format(fPorcentajeMin*100)+'-'+"{0:.3f}".format(fPorcentajeMax*100)+'/'+sEvento)
      # Generate the requested number of events
      lElementos=[]
      iContador=0
      while iContador<iCantidad:
        # Randomly choose 2 events
        lEventoIndice=listaEnteroAleatorio(0,len(m)-1,2) # e.g. [1, 2]
        # Extract paths of the randomly chosen events
        evento1, evento2=m[lEventoIndice[0]], m[lEventoIndice[1]]
        # Generate percentage for signal1 between [0, 1] with 3 decimals
        fPorcentaje = random.randint(fPorcentajeMin*10**5, fPorcentajeMax*10**5)/10**5
        # Verify it's not already generated (no duplicates)
        if (lEventoIndice[0], lEventoIndice[1], fPorcentaje) not in lElementos:
          lElementos.append((lEventoIndice[0], lEventoIndice[1], fPorcentaje))
        else:
          continue
        # Build file paths for the selected events
        sRuta1, sRuta2 = fym.archivos_canal_simple(sRutaEntrada+sEvento, evento1), fym.archivos_canal_simple(sRutaEntrada+sEvento, evento2)
        # Open the events
        tr1, tr2 = TSignal(sRuta1), TSignal(sRuta2)
        # Preprocess
        tr1.preproceso()
        tr2.preproceso()
        # Normalize signals
        tr1.normaliza()
        tr2.normaliza()
        # Remove noise (optional)
        #tr1.eliminaRuido(fRango=0.1, fTolerancia=1.0)
        #tr2.eliminaRuido(fRango=0.1, fTolerancia=1.0)
        # Generate new event via AE interpolation
        tr1.daInterpolationAE(tr2, ModeloAE, 0, fPorcentaje, sRutaSalida+"{0:.3f}".format(fPorcentajeMin*100)+'-'+"{0:.3f}".format(fPorcentajeMax*100)+'/'+sEvento)
        # Increment counter
        iContador+=1
    # Mensaje
    print("Generado eventos:", sEvento, fym.now_string())

#=============================================== SPECTROGRAM PROCESS SCENARIO 2 20HZ ============================================
# Create autoencoder
ae = AE(CAE3, 'aes/AE3_ModeloCAE.pt')
print("Start:", fym.now_string())
#generarEventoTiempoRangoAleatorio(FLAG_DA.sRutaEntrada, 'escenario2/spectrograms224_00 210708 20Hz+da_rotacion/', {'HY':2473, 'LP':2109, 'TC':488 ,'TR':2215}, 5, 25)
generarEventoRangoAleatorio(RUTA_ENTRADA, RUTA_SALIDA, EVENTO_ESTUDIO, 2686, ae, 0.4, 0.6)
print("End   :", fym.now_string())


