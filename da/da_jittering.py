import numpy as np
import matplotlib.pyplot as plt
from obspy.core import read
import fym.util as fym
from fym.signal import TSignal, TListSignal
from random import randint, choice, sample
from subprocess import call

# Constantes
class FLAG_DA:
  sRutaEntrada = 'escenario2/data/'                             # Carpeta de señales originales MiniSEED
  sRutaSalida  = 'escenario2/01/data_augmentation_jittering/' # Carpeta de salida de espectrogramas
  lEvento      = ['HY','LP','TR','VT']                          # Lista de eventos a procesar

def generarEventoJittering(sRutaEntrada:str, sRutaSalida:str, dEvento:dict, fSigmaInicio:float=0.2, fSigmaFin:float=None):
  """Genera imagenes espectrogramas mediante la rotación en rango porcentual definido en ubicaciones AIF sin repeticion.
  Args:
      sRutaEntrada (str): Carpeta donde se encuentan los eventos ordenados por carpetas Evento
      sRutaSalida (str): Carpeta donde se generaran los nuevos eventos generados
      dEvento (dict): Diccionario de eventos y cantidades a generera por evento, Ej. {'HY':2000, 'LP':1500, 'TC':0, 'TR':1800, 'VT':1350}
      fSigmaInicio (float): Desviación estandar de ruido o valor inicial de rango.
      fSigmaFin  o (float): desviación estandar de ruido final de rango.
  """
  if fSigmaFin is None:
    sCarpetaProceso=str(fSigmaInicio)
  else:
    sCarpetaProceso=str(fSigmaInicio)+'-'+str(fSigmaFin)
  # Leyendo eventos
  for sEvento in dEvento:
    print("Generando eventos:", sEvento, fym.now_string())
    # Leyendo lista de archivo de eventos desde carpeta
    m=fym.lista_archivos_simple(sRutaEntrada+sEvento)
    if len(m)>0:
      # Generando lista de tuplas (indiceArchivo, sigma).
      lJittering=[]
      for i in range(len(m)):                                                 # Indice archivo
        if fSigmaFin is None:
          for s in range(dEvento[sEvento]):  # Cantidad
            lJittering.append((i, fSigmaInicio))
        else:
          # Generando puntos
          for s in np.linspace(fSigmaInicio, fSigmaFin, num=dEvento[sEvento]):  # Rango sigma
            lJittering.append((i, round(s, 10)))
        # Generando lista que tiene muestra de items a generar espectrogramas, sin repeticion
        lMuestra = sample(lJittering, dEvento[sEvento])
      # Crear carpetas de salida si no existen
      fym.create_folders(sRutaSalida+sCarpetaProceso+'/'+sEvento)
      # Generando espectrograms la cantidad de la muestra solicitada
      for iCont, (iArchivo, fSigma) in enumerate(lMuestra):
        # Abrir evento
        sRuta = fym.archivos_canal_simple(sRutaEntrada+sEvento, m[iArchivo])
        # Abrir los eventos
        tr = TSignal(sRuta)
        # Preproceso
        tr.preproceso()  #<==============================================================================
        # Normalizar señales
        tr.normaliza()
        # Elimina ruido()
        #tr.eliminaRuido(fRango=0.1, fTolerancia=1.0)
        # Generación de nueva señal ==========================================================================
        tr.daJittering(fSigma)
        # Espectrograma guarda en disco
        tr.espectrograma_guardar_canal(0, sRutaSalida+sCarpetaProceso+'/'+sEvento, 224,'_'+str(fSigma)+'-'+str(iCont+1))

      # Eliminando de memoria variables
      del lJittering
      del lMuestra
    # Mensaje
    print("Fin de generación de eventos :", sEvento, fym.now_string())

#=============================================== PROCESO ESPECTROGRAMA ESCENARIO 2 20HZ ============================================
print("Inicio:", fym.now_string())
#generarEventoJittering(FLAG_DA.sRutaEntrada, FLAG_DA.sRutaSalida, {'HY':2473, 'LP':2109, 'TC':488 ,'TR':2215}, 0.2)
generarEventoJittering(FLAG_DA.sRutaEntrada, FLAG_DA.sRutaSalida, {'HY':2686, 'LP':2686, 'TC':2686 ,'TR':2686, 'VT':2686}, 0.2)
#generarEventoJittering(FLAG_DA.sRutaEntrada, FLAG_DA.sRutaSalida, {'HY':13419, 'LP':13419, 'TC':13419 ,'TR':13419, 'VT':13419}, 0.2)
print("Fin   :", fym.now_string())


