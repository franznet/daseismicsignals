import numpy as np
import matplotlib.pyplot as plt
from obspy.core import read
import fym.util as fym
from fym.signal import TSignal, TListSignal
from random import randint, choice

# Constants
RUTA_ENTRADA    ='escenario2/data/'
RUTA_SALIDA     ='escenario2/01/data_augmentation_specaugment1/'
EVENTO_ESTUDIO  =['HY','LP','TC','TR','VT']

def generarEvento(sRutaEntrada:str, sRutaSalida:str, lEvento:list, iCantidad:int, fFrecuenciaPorcentaje:float=0.1, iFrecuenciaCantidad:int=2, fTiempoPorcentaje:float=0.1, iTiempoCantidad:int=2):
  """Generate spectrogram images using SpecAugment.
  Args:
      sRutaEntrada (str): Folder where events are stored in event subfolders
      sRutaSalida (str): Folder where generated events will be saved
      lEvento (list): Events to consider during generation
      iPorcentajeTiempoMax (int): Maximum time percentage when creating zero padding
      iCantidad (int): Count
  """
  sCarpetaProceso=str(fFrecuenciaPorcentaje)+'-'+str(iFrecuenciaCantidad)+' '+str(fTiempoPorcentaje)+'-'+str(iTiempoCantidad)
  # Reading events
  for sEvento in lEvento:
    # Read event file list from folder
    m=fym.lista_archivos_simple(sRutaEntrada+sEvento)
    if len(m)>0:
      # Create output folders if they do not exist
      fym.create_folders(sRutaSalida+sCarpetaProceso+'/'+sEvento)
      # Generate the requested number of events
      for iCont in range(iCantidad):
        # Choose a random event and generate file paths
        sRuta = fym.archivos_canal_simple(sRutaEntrada+sEvento, m[randint(0, len(m)-1)])
        # Open the event
        tr = TSignal(sRuta)
        # Preprocess
        tr.preproceso()
        # Normalize signals
        tr.normaliza()
        # Remove noise()
        tr.eliminaRuido(fRango=0.1, fTolerancia=1.0)
        # Generate new spectrogram ==========================================================================
        tr.daEspectrogramaSpecAugment(0, sRutaSalida+sCarpetaProceso+'/'+sEvento, 224, '-'+str(iCont+1),
                                      fFrecuenciaPorcentaje=fFrecuenciaPorcentaje, iFrecuenciaCantidad=iFrecuenciaCantidad,
                                      fTiempoPorcentaje=fTiempoPorcentaje, iTiempoCantidad=iTiempoCantidad) #,'sColor='red')
    # Message
    print("Generated events:", sEvento, fym.now_string())

print("Start:", fym.now_string())
generarEvento(RUTA_ENTRADA, RUTA_SALIDA, EVENTO_ESTUDIO, iCantidad=2000, fFrecuenciaPorcentaje=0.1, iFrecuenciaCantidad=2, fTiempoPorcentaje=0.1, iTiempoCantidad=2)
print("End   :", fym.now_string())