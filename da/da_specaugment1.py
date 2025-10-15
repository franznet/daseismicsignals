import numpy as np
import matplotlib.pyplot as plt
from obspy.core import read
import fym.util as fym
from fym.signal import TSignal, TListSignal
from random import randint, choice

# Constantes
RUTA_ENTRADA    ='escenario2/data/'
RUTA_SALIDA     ='escenario2/01/data_augmentation_specaugment1/'
EVENTO_ESTUDIO  =['HY','LP','TC','TR','VT']

def generarEvento(sRutaEntrada:str, sRutaSalida:str, lEvento:list, iCantidad:int, fFrecuenciaPorcentaje:float=0.1, iFrecuenciaCantidad:int=2, fTiempoPorcentaje:float=0.1, iTiempoCantidad:int=2):
  """Genera imagenes de espectrograma mediante algoritmos geneticos.
  Args:
      sRutaEntrada (str): Carpeta donde se encuentan los eventos ordenados por carpetas Evento
      sRutaSalida (str): Carpeta donde se generaran los nuevos eventos generados
      lEvento (list): Eventos a considerarse en la generación
      iPorcentajeTiempoMax (int): Porcentaje de tiempo máximo al crear el arreglo de ceros
      iCantidad (int): Cantidad
  """
  sCarpetaProceso=str(fFrecuenciaPorcentaje)+'-'+str(iFrecuenciaCantidad)+' '+str(fTiempoPorcentaje)+'-'+str(iTiempoCantidad)
  # Leyendo eventos
  for sEvento in lEvento:
    # Leyendo lista de archivo de eventos desde carpeta
    m=fym.lista_archivos_simple(sRutaEntrada+sEvento)
    if len(m)>0:
      # Crear carpetas de salida si no existen
      fym.create_folders(sRutaSalida+sCarpetaProceso+'/'+sEvento)
      # Generando la cantidad de eventos solicitado
      for iCont in range(iCantidad):
        # Elegir aleatoriamente un evento y generar rutas de archivo
        sRuta = fym.archivos_canal_simple(sRutaEntrada+sEvento, m[randint(0, len(m)-1)])
        # Abrir los eventos
        tr = TSignal(sRuta)
        # Preproceso
        tr.preproceso()
        # Normalizar señales
        tr.normaliza()
        # Elimina ruido()
        tr.eliminaRuido(fRango=0.1, fTolerancia=1.0)
        # Generación de nuevo espectrograma ==========================================================================
        tr.daEspectrogramaSpecAugment(0, sRutaSalida+sCarpetaProceso+'/'+sEvento, 224, '-'+str(iCont+1),
                                      fFrecuenciaPorcentaje=fFrecuenciaPorcentaje, iFrecuenciaCantidad=iFrecuenciaCantidad,
                                      fTiempoPorcentaje=fTiempoPorcentaje, iTiempoCantidad=iTiempoCantidad) #,'sColor='red')
    # Mensaje
    print("Generado eventos:", sEvento, fym.now_string())

print("Inicio:", fym.now_string())
generarEvento(RUTA_ENTRADA, RUTA_SALIDA, EVENTO_ESTUDIO, iCantidad=2000, fFrecuenciaPorcentaje=0.1, iFrecuenciaCantidad=2, fTiempoPorcentaje=0.1, iTiempoCantidad=2)
print("Fin   :", fym.now_string())