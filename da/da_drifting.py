import numpy as np
import matplotlib.pyplot as plt
from obspy.core import read
import fym.util as fym
from fym.signal import TSignal, TListSignal
from random import randint, choice, sample
from subprocess import call
import pickle
import os, glob, shutil

# Constantes
class FLAG_DA:
  sRutaEntrada = 'escenario2/data/'       # Carpeta de señales originales MiniSEED
  sRutaSalida  = 'E:/spectrograms224_00/' # Carpeta de salida de espectrogramas
  lEvento      = ['HY','LP','TR','VT']    # Lista de eventos a procesar

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
  """Genera una muestra aleatoria de los eventos sin repeticion en carpeta destino.
  Args:
      sRutaEntrada (str): Carpeta donde se encuentan los eventos originales organizado por carpetas Evento
      sRutaSalida (str):  Carpeta donde se generaran la muestra de eventos
      dEvento (dict): Diccionario de eventos y cantidades a generer por evento, Ej. {'HY':2000, 'LP':1500, 'TC':0, 'TR':1800, 'VT':1350}
  """
  # Leyendo eventos
  for sEvento in dEvento:
    print("Generando muestra de eventos:", sEvento, fym.now_string())
    # Leyendo lista de archivo de eventos desde carpeta
    #m=fym.lista_archivos_simple(sRutaEntrada+sEvento)
    m=list(map(os.path.basename, glob.glob(sRutaEntrada+sEvento+"/*.*")))
    if len(m)>0:
      # Generando lista que tiene muestra de eventos sin repeticion
      lMuestra = sample(m, dEvento[sEvento])
      # Crear carpetas de salida si no existe
      #fym.create_folders(sRutaSalida+sEvento)
      if not os.path.exists(sRutaSalida+sEvento):
        os.makedirs(sRutaSalida+sEvento)
      # Copiar archivos a ruta de salida
      for fArchivo in lMuestra:
        shutil.copy(sRutaEntrada+sEvento+'/'+fArchivo, sRutaSalida+sEvento)
    # Mensaje
    print("Fin de generación de eventos :", sEvento, fym.now_string())
  return

def generarEventoDrifting(sRutaEntrada:str, sRutaSalida:str, dEvento:dict, fDerivaInicio:int, fDerivaFin:int):
  """Genera imagenes espectrogramas mediante la rotación en rango porcentual definido en ubicaciones AIF sin repeticion.
  Args:
      sRutaEntrada (str): Carpeta donde se encuentan los eventos ordenados por carpetas Evento
      sRutaSalida (str): Carpeta donde se generaran los nuevos eventos generados
      dEvento (dict): Diccionario de eventos y cantidades a generera por evento, Ej. {'HY':2000, 'LP':1500, 'TC':0, 'TR':1800, 'VT':1350}
      fDerivaInicio (float): Valor de deriva o valor inicial de rango).
      fDerivaFin  o (float): Valor de la deviPorcentaje de tiempo de espacio de rotación final.
  """
  # Leyendo eventos
  for sEvento in dEvento:
    print("Generando eventos:", sEvento, fym.now_string())
    # Leyendo lista de archivo de eventos desde carpeta
    m=fym.lista_archivos_simple(sRutaEntrada+sEvento)
    if len(m)>0:
      # Generando lista de tuplas (indiceArchivo, porcentaje, ubicacion).
      lJittering=[]
      for i in range(len(m)):                                                   # Indice archivo
        for d in np.linspace(fDerivaInicio, fDerivaFin, num=dEvento[sEvento]):  # Rango deviva
          lJittering.append((i, round(d,10)))
      # Generando lista que tiene muestra de items a generar espectrogramas, sin repeticion
      lMuestra = sample(lJittering, dEvento[sEvento])
      # Crear carpetas de salida si no existen
      fym.create_folders(sRutaSalida+sEvento)
      # Generando espectrograms la cantidad de la muestra solicitada
      for iArchivo, fDeriva in lMuestra:
        # Abrir evento
        sRuta = fym.archivos_canal_simple(sRutaEntrada+sEvento, m[iArchivo])
        # Abrir los eventos
        tr = TSignal(sRuta)
        # Preproceso
        tr.preproceso2()  #<==============================================================================
        # Normalizar señales
        tr.normaliza()
        # Elimina ruido()
        #tr.eliminaRuido(fRango=0.1, fTolerancia=1.0)
        # Generación de nueva señal ==========================================================================
        tr.daJittering(fDeriva)
        # Espectrograma guarda en disco
        tr.espectrograma_guardar_canal(0, sRutaSalida+sEvento, 224,'_'+str(fDeriva))
      # Eliminando de memoria variables
      del lJittering
      del lMuestra
    # Mensaje
    print("Fin de generación de eventos :", sEvento, fym.now_string())

#=============================================== PROCESO ESPECTROGRAMA ESCENARIO 2 20HZ ============================================
print("Inicio:", fym.now_string())
#generarEventoDrifting(FLAG_DA.sRutaEntrada, 'escenario2/spectrograms224_00 210708 20Hz+da_drifting/', {'HY':2473, 'LP':2109, 'TC':488 ,'TR':2215}, 0.01, 0.1)
#generarEventoDrifting(FLAG_DA.sRutaEntrada, 'escenario2/spectrograms224_00 210708 20Hz+da_drifting/', {'HY':2473, 'LP':2109, 'TC':488 ,'TR':2215}, 0.001, 0.01)
#generarEventoDrifting(FLAG_DA.sRutaEntrada, 'escenario2/spectrograms224_00 210708 20Hz+da_drifting/', {'HY':2473, 'LP':2109, 'TC':488 ,'TR':2215}, 0.0001, 0.001)
#generarEventoDrifting(FLAG_DA.sRutaEntrada, 'escenario2/spectrograms224_00 210708 20Hz+da_drifting/', {'HY':13419, 'LP':13419, 'TC':13419 ,'TR':13419, 'VT':13419}, 0.001, 0.01)
#generarEventoDrifting(FLAG_DA.sRutaEntrada, 'escenario2/spectrograms224_00 210708 20Hz+da_drifting/', {'HY':13419, 'LP':13419, 'TC':13419 ,'TR':13419, 'VT':13419}, 0.01, 0.1)
#generarEventoDrifting(FLAG_DA.sRutaEntrada, 'escenario2/spectrograms224_00 210708 20Hz+da_drifting/', {'HY':13419, 'LP':13419, 'TC':13419 ,'TR':13419, 'VT':13419}, 0.0001, 0.001)
generaMuestraArchivos('escenario2/spectrograms224_00/', 'escenario2/spectrograms224_00 muestra/', {'HY':2686, 'LP':2686, 'TC':2686 ,'TR':2686, 'VT':2686})
print("Fin   :", fym.now_string())


