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

# Constantes
RUTA_ENTRADA    ='escenario2/data/'
RUTA_SALIDA     ='escenario2/01/data_augmentation_interpolation/'
EVENTO_ESTUDIO  =['HY','LP','TC','TR','VT']

# Retorna lista con indices aleatorio no repetidos de rango preveido. Incluye rangos más
def listaEnteroAleatorio(iMenor:int, iMaximo:int, iElementos:int):
  lLista=[]
  while len(lLista)<iElementos:
    iAleatorio=random.randint(iMenor, iMaximo)
    if iAleatorio not in lLista: lLista.append(iAleatorio)
    #else: print(iAleatorio, ' ya esta en lista ',lLista, ' rango[', iMenor,',',iMaximo,']')
  lLista.sort()
  return lLista

# Genera eventos nuevos por AG, espectrograma
def generarEventoRangoAleatorio(sRutaEntrada:str, sRutaSalida:str, lEvento:list,  iCantidad:int, ModeloAE:nn.Module, fPorcentajeMin:float=0.4, fPorcentajeMax:float=0.6):
  """Genera imagenes de sismograma y espectrograma mediante algoritmos geneticos.
  Args:
      sRutaEntrada (str): Carpeta donde se encuentan los eventos ordenados por carpetas Evento
      sRutaSalida (str): Carpeta donde se generaran los nuevos eventos generados
      lEvento (list): Eventos a considerarse en la generación
      iCantidad (int): Cantidad de eventos a generarse
      fPorcentaje(float, optional): Porcentaje de la interpolación de las dos señales. Rango [0, 1]. 0=self, 1=str2. Maneja 3 deciamles porcentuales como 45.123%. Defaults to 0.5.
  """
  # Leyendo eventos
  for sEvento in lEvento:
    # Leyendo lista de archivo de evetos desde carpeta
    m=fym.lista_archivos_simple(sRutaEntrada+sEvento)
    if len(m)>0:
      # Crear carpetas de salida si no existen
      fym.create_folders(sRutaSalida+"{0:.3f}".format(fPorcentajeMin*100)+'-'+"{0:.3f}".format(fPorcentajeMax*100)+'/'+sEvento)
      # Generando la cantidad de eventos solicitado
      lElementos=[]
      iContador=0
      while iContador<iCantidad:
        # Elegir aleatoriamente 2 eventos
        lEventoIndice=listaEnteroAleatorio(0,len(m)-1,2) # Mismo tiempo(26,74) #[1, 2] #
        # Extraer ruta de eventos elegidos al azar
        evento1, evento2=m[lEventoIndice[0]], m[lEventoIndice[1]]
        # Generar porcentaje de señal 1 entre [0, 1] con 3 decimales. Ej 45.567%
        fPorcentaje = random.randint(fPorcentajeMin*10**5, fPorcentajeMax*10**5)/10**5
        # Verificar que no este ya generado (Repetido)
        if (lEventoIndice[0], lEventoIndice[1], fPorcentaje) not in lElementos:
          lElementos.append((lEventoIndice[0], lEventoIndice[1], fPorcentaje))
        else:
          continue
        # Generar rutas de los elegidos
        sRuta1, sRuta2 = fym.archivos_canal_simple(sRutaEntrada+sEvento, evento1), fym.archivos_canal_simple(sRutaEntrada+sEvento, evento2)
        # Abrir los eventos
        tr1, tr2 = TSignal(sRuta1), TSignal(sRuta2)
        # Preproceso
        tr1.preproceso()
        tr2.preproceso()
        # Normalizar señales
        tr1.normaliza()
        tr2.normaliza()
        # Elimina ruido()
        #tr1.eliminaRuido(fRango=0.1, fTolerancia=1.0)
        #tr2.eliminaRuido(fRango=0.1, fTolerancia=1.0)
        # Genera nuevo evento por Interpolación de AE
        tr1.daInterpolationAE(tr2, ModeloAE, 0, fPorcentaje, sRutaSalida+"{0:.3f}".format(fPorcentajeMin*100)+'-'+"{0:.3f}".format(fPorcentajeMax*100)+'/'+sEvento)
        # Incrementar contador
        iContador+=1
    # Mensaje
    print("Generado eventos:", sEvento, fym.now_string())

#=============================================== PROCESO ESPECTROGRAMA ESCENARIO 2 20HZ ============================================
# Crear autoencoder
ae = AE(CAE3, 'aes/AE3_ModeloCAE.pt')
print("Inicio:", fym.now_string())
#generarEventoTiempoRangoAleatorio(FLAG_DA.sRutaEntrada, 'escenario2/spectrograms224_00 210708 20Hz+da_rotacion/', {'HY':2473, 'LP':2109, 'TC':488 ,'TR':2215}, 5, 25)
generarEventoRangoAleatorio(RUTA_ENTRADA, RUTA_SALIDA, EVENTO_ESTUDIO, 2686, ae, 0.4, 0.6)
print("Fin   :", fym.now_string())


