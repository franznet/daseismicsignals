import numpy as np
import matplotlib.pyplot as plt
from obspy.core import read
import fym.util as fym
from fym.signal import TSignal, TListSignal
from random import randint, choice, sample
from subprocess import call
import pickle

# Constants
class FLAG_DA:
  sRutaEntrada = 'escenario2/data/'       # Folder of original MiniSEED signals
  sRutaSalida  = 'H:/spectrograms224_00/'  # Output folder for spectrograms
  lEvento      = ['HY','LP','TR','VT']    # List of events to process
  iPorcentajeTiempoInicio = 5             # Signal time percentage for start rotation. Range 0-100.
  iPorcentajeTiempoFinal  = 51            # Signal time percentage for end rotation. Range 0-100.
  iPorcentajeTiempoInc    = 5             # Increment
  iPorcentajeTiempoLong   = 25            # Length of percentage ranges.
  #sUbicacion   = 'A'                     # Rotation zone location. I=start, F=end, A=both sides.

def guarda_lista(nombre_archivo, lista_a_guardar):
  archivo = open(nombre_archivo, "wb")
  pickle.dump(lista_a_guardar, archivo)
  archivo.close()

def lee_lista(nombre_archivo):
  archivo = open(nombre_archivo, "rb")
  lista_leida = pickle.load(archivo)
  archivo.close()
  return lista_leida

def generarEvento(sRutaEntrada:str, sRutaSalida:str, lEvento:list, iCantidad:int, iPorcentajeTiempoMax:int, iPorcentajeTiempoMin:int=1):
  """Generate spectrogram images using rotation.
  Args:
      sRutaEntrada (str): Folder where events are stored in event subfolders
      sRutaSalida (str): Folder where generated events will be saved
      lEvento (list): Events to consider during generation
      iCantidad (int/dict): Count or dictionary of counts per event, e.g. 200, {'HY':2000, 'LP':1500, 'TC':0, 'TR':1800, 'VT':1350}
      iPorcentajeTiempoMax (int): Maximum time percentage when creating the zero padding.
      iPorcentajeTiempoMin (int): Minimum time percentage when creating the zero padding. Default 1
  """
  # Reading events
  for sEvento in lEvento:
    # Read event file list from folder
    m=fym.lista_archivos_simple(sRutaEntrada+sEvento)
    if len(m)>0:
      # Create output folders if they do not exist
      fym.create_folders(sRutaSalida+str(iPorcentajeTiempoMin)+'_'+str(iPorcentajeTiempoMax)+'/'+sEvento)
      # Generate the requested number of events
      for iCont in range(iCantidad if isinstance(iCantidad, int) else iCantidad[sEvento]):
        # Choose a random event and generate file paths
        sRuta = fym.archivos_canal_simple(sRutaEntrada+sEvento, m[randint(0, len(m)-1)])
        # Open the event
        tr = TSignal(sRuta)
        # Preprocess
        tr.preproceso()
        # Normalize signals
        tr.normaliza()
        # Remove noise()
        #tr.eliminaRuido(fRango=0.1, fTolerancia=1.0)
        # Generate new signal ==========================================================================
        tr.daAgregaRotacion(randint(iPorcentajeTiempoMin, iPorcentajeTiempoMax)/100, choice('IFA'))
        # Save seismogram to disk
        #tr.sismograma_guardar_canal(0, sRutaSalida + sEvento, 224)
        # Save spectrogram to disk
        tr.espectrograma_guardar_canal(0, sRutaSalida+str(iPorcentajeTiempoMin)+'_'+str(iPorcentajeTiempoMax)+'/'+sEvento, 224,'-'+str(iCont+1))
    # Message
    print("Generated events:", sEvento, fym.now_string())

def generarEventoTiempoFijo(sRutaEntrada:str, sRutaSalida:str, lEvento:list, iCantidad:int, iPorcentajeTiempo:int, sUbicacion:str):
  """Generate spectrogram images using rotation with a fixed signal percentage.
  Args:
      sRutaEntrada (str): Folder where events are stored in event subfolders
      sRutaSalida (str): Folder where generated events will be saved
      lEvento (list): Events to consider during generation
      iCantidad (int/dict): Count or dictionary of counts per event, e.g. 200, {'HY':2000, 'LP':1500, 'TC':0, 'TR':1800, 'VT':1350}
      iPorcentajeTiempo (int): Time percentage of the rotation zone (zero padding).
      sUbicacin (str): Location of the rotation zone. I=start, F=end, or A=both sides.
  """
  # Reading events
  for sEvento in lEvento:
    # Read event file list from folder
    m=fym.lista_archivos_simple(sRutaEntrada+sEvento)
    if len(m)>0:
      # Create output folders if they do not exist
      fym.create_folders(sRutaSalida+sEvento)
      # Generate the requested number of events
      for iCont in range(iCantidad if isinstance(iCantidad, int) else iCantidad[sEvento]):
        # Choose a random event and generate file paths
        sRuta = fym.archivos_canal_simple(sRutaEntrada+sEvento, m[randint(0, len(m)-1)])
        # Open the event
        tr = TSignal(sRuta)
        # Preprocess
        tr.preproceso()
        # Normalize signals
        tr.normaliza()
        # Remove noise()
        #tr.eliminaRuido(fRango=0.1, fTolerancia=1.0)
        # Generate new signal ==========================================================================
        tr.daAgregaRotacion(iPorcentajeTiempo/100, sUbicacion)
        # Save seismogram to disk
        #tr.sismograma_guardar_canal(0, sRutaSalida + sEvento, 224)
        # Save spectrogram to disk
        tr.espectrograma_guardar_canal(0, sRutaSalida+sEvento, 224,'-'+str(iCont+1))
    # Message
    print("Generated events:", sEvento, fym.now_string())

def generarEventoDiccionario(sRutaEntrada:str, lEvento:list, iCantidad:int):
  """Generate spectrogram images using rotation with a fixed signal percentage.
  Args:
      sRutaEntrada (str): Folder where events are stored in event subfolders
      lEvento (list): Events to consider during generation
      iCantidad (int/dict): Count or dictionary of counts per event, e.g. 200, {'HY':2000, 'LP':1500, 'TC':0, 'TR':1800, 'VT':1350}
  """
  dResultado={}
  # Reading events
  for sEvento in lEvento:
    # Read event file list from folder
    m=fym.lista_archivos_simple(sRutaEntrada+sEvento)
    if len(m)>0:
      # File list
      lArchivo=[]
      # Generate the requested number of events
      for iCont in range(iCantidad if isinstance(iCantidad, int) else iCantidad[sEvento]):
        # Choose a random event and generate file paths
        lArchivo.append(m[randint(0, len(m)-1)])
      # Add to result list
      dResultado[sEvento]=lArchivo
    # Message
    print("Generated events:", sEvento, fym.now_string())
  return dResultado
def generarEventoTiempoFijoLista(sRutaEntrada:str, sRutaSalida:str, iPorcentajeTiempo:int, sUbicacion:str, dEvento):
  """Generate spectrogram images using rotation with a fixed signal percentage.
  Args:
      sRutaEntrada (str): Folder where events are stored in event subfolders
      sRutaSalida (str): Folder where generated events will be saved
      lEvento (list): Events to consider during generation
      iPorcentajeTiempo (int): Time percentage of the rotation zone (zero padding).
      sUbicacin (str): Location of the rotation zone. I=start, F=end, or A=both sides.
      dEvento (dict): Dictionary with file names, e.g. {'HY':['a1','a2',...], 'LP':['b1','b2',...], 'TR':['c1','c2',...], 'VT':['d1','d2',...]}
  """
  # Reading events
  for sEvento in dEvento:
    # Read event file list from folder
    m=dEvento[sEvento]
    if len(m)>0:
      # Create output folders if they do not exist
      fym.create_folders(sRutaSalida+sEvento)
      # Generate the requested number of events
      for iCont in range(len(m)):
        # Choose an event and generate file paths
        sRuta = fym.archivos_canal_simple(sRutaEntrada+sEvento, m[iCont])
        # Open the event
        tr = TSignal(sRuta)
        # Preprocess
        tr.preproceso()
        # Normalize signals
        tr.normaliza()
        # Remove noise()
        #tr.eliminaRuido(fRango=0.1, fTolerancia=1.0)
        # Generate new signal ==========================================================================
        tr.daAgregaRotacion(iPorcentajeTiempo/100, sUbicacion)
        # Save seismogram to disk
        #tr.sismograma_guardar_canal(0, sRutaSalida + sEvento, 224)
        # Save spectrogram to disk
        tr.espectrograma_guardar_canal(0, sRutaSalida+sEvento, 224,'-'+str(iCont+1))
    # Message
    print("Generated events:", sEvento, fym.now_string())

def generarEventoTiempoRango(sRutaEntrada:str, sRutaSalida:str, lEvento:list, iPorcentajeInicio:int, iPorcentajeFin:int):
  """Generate spectrogram images using rotation within a percentage range in A/I/F locations.
  Args:
      sRutaEntrada (str): Folder where events are stored in event subfolders
      sRutaSalida (str): Folder where generated events will be saved
      lEvento (list): Events to consider during generation
      iPorcentajeInicio (int): Initial time percentage of the rotation zone.
      iPorcentajeFin  (int): Final time percentage of the rotation zone.
      dEvento (dict): Dictionary with file names, e.g. {'HY':['a1','a2',...], 'LP':['b1','b2',...], 'TR':['c1','c2',...], 'VT':['d1','d2',...]}
  """
  # Reading events
  for sEvento in lEvento:
    # Read event file list from folder
    m=fym.lista_archivos_simple(sRutaEntrada+sEvento)
    if len(m)>0:
      # Create output folders if they do not exist
      fym.create_folders(sRutaSalida+sEvento)
      # Generate the requested number of events
      for iCont in range(len(m)):
        # Open event
        sRuta = fym.archivos_canal_simple(sRutaEntrada+sEvento, m[iCont])
        # Open the event
        tr = TSignal(sRuta)
        # Preprocess
        tr.preproceso()
        # Normalize signals
        tr.normaliza()
        # Remove noise()
        #tr.eliminaRuido(fRango=0.1, fTolerancia=1.0)
        # Generate new signal ==========================================================================
        for i in range(iPorcentajeInicio, iPorcentajeFin+1):
          for u in ['I','A','F']:   # Rotation location
            trc=tr.copy()
            trc.daAgregaRotacion(i/100, u)
            # Save spectrogram to disk
            trc.espectrograma_guardar_canal(0, sRutaSalida+sEvento, 224,'.'+str(i).zfill(2)+u)
            del trc
    # Message
    print("Generated events:", sEvento, fym.now_string())

def generarEventoTiempoRangoAleatorio(sRutaEntrada:str, sRutaSalida:str, dEvento:dict, iPorcentajeInicio:int, iPorcentajeFin:int):
  """Generate spectrogram images using rotation within a percentage range in A/I/F locations without repetition.
  Args:
      sRutaEntrada (str): Folder where events are stored in event subfolders
      sRutaSalida (str): Folder where generated events will be saved
      dEvento (dict): Dictionary of events and counts to generate per event, e.g. {'HY':2000, 'LP':1500, 'TC':0, 'TR':1800, 'VT':1350}
      iPorcentajeInicio (int): Initial time percentage of the rotation zone.
      iPorcentajeFin  (int): Final time percentage of the rotation zone.
  """
  # Reading events
  for sEvento in dEvento:
    print("Generating events:", sEvento, fym.now_string())
    # Read event file list from folder
    m=fym.lista_archivos_simple(sRutaEntrada+sEvento)
    if len(m)>0:
      # Generate list of tuples (fileIndex, percentage, location).
      lRotacion=[]
      for i in range(len(m)):                                 # File index
        for p in range(iPorcentajeInicio,iPorcentajeFin+1):   # Rotation percentage range
          for u in ['I','A','F']:                             # Rotation location
            lRotacion.append((i,p,u))
      # Generate sample list of items to create spectrograms, without repetition
      lMuestra = sample(lRotacion, dEvento[sEvento])
      # Create output folders if they do not exist
      fym.create_folders(sRutaSalida+sEvento)
      # Generate the requested number of spectrograms
      for iArchivo, iPorcentaje, sUbicacion in lMuestra:
        # Open event
        sRuta = fym.archivos_canal_simple(sRutaEntrada+sEvento, m[iArchivo])
        # Open the event
        tr = TSignal(sRuta)
        # Preprocess
        tr.preproceso()
        # Normalize signals
        tr.normaliza()
        # Remove noise()
        #tr.eliminaRuido(fRango=0.1, fTolerancia=1.0)
        # Generate new signal ==========================================================================
        tr.daAgregaRotacion(iPorcentaje/100, sUbicacion)
        # Save spectrogram to disk
        tr.espectrograma_guardar_canal(0, sRutaSalida+sEvento, 224,'.'+str(iPorcentaje).zfill(2)+sUbicacion)
      # Remove variables from memory
      del lRotacion
      del lMuestra
    # Message
    print("Generated events :", sEvento, fym.now_string())

#print("Start:", fym.now_string())
#generarEvento('escenario2/data/', 'escenario2/data_augmentation_rotacion/', FLAG_DA.lEvento, iCantidad={'HY':2000, 'LP':1500, 'TR':1800, 'VT':1350}, iPorcentajeTiempoMax=50, iPorcentajeTiempoMin=45)
#print("End   :", fym.now_string())

print("Start:", fym.now_string())
dEventos=lee_lista('eventos.pkl')
for i in range(FLAG_DA.iPorcentajeTiempoInicio, FLAG_DA.iPorcentajeTiempoFinal+1, FLAG_DA.iPorcentajeTiempoInc):
  print("Generating for time: ", i, '-', i+FLAG_DA.iPorcentajeTiempoLong, fym.now_string())
  # Generate new spectrograms
  #generarEventoTiempoFijo(FLAG_DA.sRutaEntrada, FLAG_DA.sRutaSalida, FLAG_DA.lEvento, iCantidad={'HY':2000, 'LP':1500, 'TR':1800, 'VT':1350}, iPorcentajeTiempo=i, sUbicacion=FLAG_DA.sUbicacion)
  #generarEventoTiempoFijoLista(FLAG_DA.sRutaEntrada, FLAG_DA.sRutaSalida, iPorcentajeTiempo=i, sUbicacion=FLAG_DA.sUbicacion, dEvento=dEventos)
  generarEventoTiempoRangoAleatorio(FLAG_DA.sRutaEntrada, FLAG_DA.sRutaSalida, {'HY':2473, 'LP':2109, 'TC':488 ,'TR':2215}, i, i+FLAG_DA.iPorcentajeTiempoLong)
  # Add original spectrograms without noise cut
  print("Copy original spectrograms:")
  call(r'xcopy /s /q "D:\UCN\Python\escenario2\spectrograms224_00 210708" H:\spectrograms224_00')
  # Compress folder
  print("Compress to ZIP:")
  call(r'C:\Program Files\7-Zip\7z.exe a -tzip H:\spectrograms224_00'+'.'+str(i).zfill(2)+'-'+str(i+FLAG_DA.iPorcentajeTiempoLong).zfill(2)+'.zip H:\spectrograms224_00')
  # Move file to Google Drive
  print('Move to Google Drive:' +'spectrograms224_00'+'.'+str(i).zfill(2)+'-'+str(i+FLAG_DA.iPorcentajeTiempoLong).zfill(2)+'.zip')
  call(r'move "H:\spectrograms224_00'+'.'+str(i).zfill(2)+'-'+str(i+FLAG_DA.iPorcentajeTiempoLong).zfill(2)+r'.zip" "D:\Google Drive\Espectrogramas\Escenario2\00\spectrograms224_00 +daRotacion3"', shell=True)
  # Remove generated spectrogram folder
  print(r"Remove generated spectrogram folder: H:\spectrograms224_00")
  call(r'rmdir /q /s H:\spectrograms224_00', shell=True)
print("End   :", fym.now_string())

