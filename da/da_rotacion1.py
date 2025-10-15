import numpy as np
import matplotlib.pyplot as plt
from obspy.core import read
import fym.util as fym
from fym.signal import TSignal, TListSignal
from random import randint, choice, sample
from subprocess import call
import pickle

# Constantes
class FLAG_DA:
  sRutaEntrada = 'escenario2/data/'       # Carpeta de señales originales MiniSEED
  sRutaSalida  = 'H:/spectrograms224_00/'  # Carpeta de salida de espectrogramas
  lEvento      = ['HY','LP','TR','VT']    # Lista de eventos a procesar
  iPorcentajeTiempoInicio = 5             # Porcentaje de tiempo de señal para rotación inicio. Rango 0-100.
  iPorcentajeTiempoFinal  = 51            # Porcentaje de tiempo de señal para rotación fin. Rango 0-100.
  iPorcentajeTiempoInc    = 5             # Incremento
  iPorcentajeTiempoLong   = 25            # Longitud de rangos porcentuales.
  #sUbicacion   = 'A'                     # Ubicación de espacio de rotación. I=inicio, F=Final o A=Ambos lados.

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
  """Genera imagenes de espectrograma mediante rotación.
  Args:
      sRutaEntrada (str): Carpeta donde se encuentan los eventos ordenados por carpetas Evento
      sRutaSalida (str): Carpeta donde se generaran los nuevos eventos generados
      lEvento (list): Eventos a considerarse en la generación
      iCantidad (int/dict): Cantidad o diccionario de catnidades por evento, Ej. 200, {'HY':2000, 'LP':1500, 'TC':0, 'TR':1800, 'VT':1350}
      iPorcentajeTiempoMax (int): Porcentaje de tiempo máximo al crear el arreglo de ceros.
      iPorcentajeTiempoMin (int): Porcentaje de tiempo mínimo al crear el arreglo de ceros. Defecto 1
  """
  # Leyendo eventos
  for sEvento in lEvento:
    # Leyendo lista de archivo de eventos desde carpeta
    m=fym.lista_archivos_simple(sRutaEntrada+sEvento)
    if len(m)>0:
      # Crear carpetas de salida si no existen
      fym.create_folders(sRutaSalida+str(iPorcentajeTiempoMin)+'_'+str(iPorcentajeTiempoMax)+'/'+sEvento)
      # Generando la cantidad de eventos solicitado
      for iCont in range(iCantidad if isinstance(iCantidad, int) else iCantidad[sEvento]):
        # Elegir aleatoriamente un evento y generar rutas de archivo
        sRuta = fym.archivos_canal_simple(sRutaEntrada+sEvento, m[randint(0, len(m)-1)])
        # Abrir los eventos
        tr = TSignal(sRuta)
        # Preproceso
        tr.preproceso()
        # Normalizar señales
        tr.normaliza()
        # Elimina ruido()
        #tr.eliminaRuido(fRango=0.1, fTolerancia=1.0)
        # Generación de nueva señal ==========================================================================
        tr.daAgregaRotacion(randint(iPorcentajeTiempoMin, iPorcentajeTiempoMax)/100, choice('IFA'))
        # Sismograma guarda en disco
        #tr.sismograma_guardar_canal(0, sRutaSalida + sEvento, 224)
        # Espectrograma guarda en disco
        tr.espectrograma_guardar_canal(0, sRutaSalida+str(iPorcentajeTiempoMin)+'_'+str(iPorcentajeTiempoMax)+'/'+sEvento, 224,'-'+str(iCont+1))
    # Mensaje
    print("Generado eventos:", sEvento, fym.now_string())

def generarEventoTiempoFijo(sRutaEntrada:str, sRutaSalida:str, lEvento:list, iCantidad:int, iPorcentajeTiempo:int, sUbicacion:str):
  """Genera imagenes espectrograma mediante la rotación con procentaje fijo de señal.
  Args:
      sRutaEntrada (str): Carpeta donde se encuentan los eventos ordenados por carpetas Evento
      sRutaSalida (str): Carpeta donde se generaran los nuevos eventos generados
      lEvento (list): Eventos a considerarse en la generación
      iCantidad (int/dict): Cantidad o diccionario de catnidades por evento, Ej. 200, {'HY':2000, 'LP':1500, 'TC':0, 'TR':1800, 'VT':1350}
      iPorcentajeTiempo (int): Porcentaje de tiempo de espacio de rotación(arreglo de ceros).
      sUbicacin (str): Ubicación del espacio de rotación. I=inicio, F=Final o A=Ambos lados.
  """
  # Leyendo eventos
  for sEvento in lEvento:
    # Leyendo lista de archivo de eventos desde carpeta
    m=fym.lista_archivos_simple(sRutaEntrada+sEvento)
    if len(m)>0:
      # Crear carpetas de salida si no existen
      fym.create_folders(sRutaSalida+sEvento)
      # Generando la cantidad de eventos solicitado
      for iCont in range(iCantidad if isinstance(iCantidad, int) else iCantidad[sEvento]):
        # Elegir aleatoriamente un evento y generar rutas de archivo
        sRuta = fym.archivos_canal_simple(sRutaEntrada+sEvento, m[randint(0, len(m)-1)])
        # Abrir los eventos
        tr = TSignal(sRuta)
        # Preproceso
        tr.preproceso()
        # Normalizar señales
        tr.normaliza()
        # Elimina ruido()
        #tr.eliminaRuido(fRango=0.1, fTolerancia=1.0)
        # Generación de nueva señal ==========================================================================
        tr.daAgregaRotacion(iPorcentajeTiempo/100, sUbicacion)
        # Sismograma guarda en disco
        #tr.sismograma_guardar_canal(0, sRutaSalida + sEvento, 224)
        # Espectrograma guarda en disco
        tr.espectrograma_guardar_canal(0, sRutaSalida+sEvento, 224,'-'+str(iCont+1))
    # Mensaje
    print("Generado eventos:", sEvento, fym.now_string())

def generarEventoDiccionario(sRutaEntrada:str, lEvento:list, iCantidad:int):
  """Genera imagenes espectrograma mediante la rotación con procentaje fijo de señal.
  Args:
      sRutaEntrada (str): Carpeta donde se encuentan los eventos ordenados por carpetas Evento
      lEvento (list): Eventos a considerarse en la generación
      iCantidad (int/dict): Cantidad o diccionario de cantidades por evento, Ej. 200, {'HY':2000, 'LP':1500, 'TC':0, 'TR':1800, 'VT':1350}
  """
  dResultado={}
  # Leyendo eventos
  for sEvento in lEvento:
    # Leyendo lista de archivo de eventos desde carpeta
    m=fym.lista_archivos_simple(sRutaEntrada+sEvento)
    if len(m)>0:
      # Lista de archivos
      lArchivo=[]
      # Generando la cantidad de eventos solicitado
      for iCont in range(iCantidad if isinstance(iCantidad, int) else iCantidad[sEvento]):
        # Elegir aleatoriamente un evento y generar rutas de archivo
        lArchivo.append(m[randint(0, len(m)-1)])
      # Agregar a lista de resultados
      dResultado[sEvento]=lArchivo
    # Mensaje
    print("Generado eventos:", sEvento, fym.now_string())
  return dResultado
def generarEventoTiempoFijoLista(sRutaEntrada:str, sRutaSalida:str, iPorcentajeTiempo:int, sUbicacion:str, dEvento):
  """Genera imagenes espectrograma mediante la rotación con procentaje fijo de señal.
  Args:
      sRutaEntrada (str): Carpeta donde se encuentan los eventos ordenados por carpetas Evento
      sRutaSalida (str): Carpeta donde se generaran los nuevos eventos generados
      lEvento (list): Eventos a considerarse en la generación
      iPorcentajeTiempo (int): Porcentaje de tiempo de espacio de rotación(arreglo de ceros).
      sUbicacin (str): Ubicación del espacio de rotación. I=inicio, F=Final o A=Ambos lados.
      dEvento (dict): Diccionario de con nombres de archivos, Ej. 200, {'HY':['a1','a2',...], 'LP':['b1','b2',...], 'TR':['c1','c2',...], 'VT':['d1','d2',...]}
  """
  # Leyendo eventos
  for sEvento in dEvento:
    # Leyendo lista de archivo de eventos desde carpeta
    m=dEvento[sEvento]
    if len(m)>0:
      # Crear carpetas de salida si no existen
      fym.create_folders(sRutaSalida+sEvento)
      # Generando la cantidad de eventos solicitado
      for iCont in range(len(m)):
        # Elegir aleatoriamente un evento y generar rutas de archivo
        sRuta = fym.archivos_canal_simple(sRutaEntrada+sEvento, m[iCont])
        # Abrir los eventos
        tr = TSignal(sRuta)
        # Preproceso
        tr.preproceso()
        # Normalizar señales
        tr.normaliza()
        # Elimina ruido()
        #tr.eliminaRuido(fRango=0.1, fTolerancia=1.0)
        # Generación de nueva señal ==========================================================================
        tr.daAgregaRotacion(iPorcentajeTiempo/100, sUbicacion)
        # Sismograma guarda en disco
        #tr.sismograma_guardar_canal(0, sRutaSalida + sEvento, 224)
        # Espectrograma guarda en disco
        tr.espectrograma_guardar_canal(0, sRutaSalida+sEvento, 224,'-'+str(iCont+1))
    # Mensaje
    print("Generado eventos:", sEvento, fym.now_string())

def generarEventoTiempoRango(sRutaEntrada:str, sRutaSalida:str, lEvento:list, iPorcentajeInicio:int, iPorcentajeFin:int):
  """Genera imagenes espectrogramas mediante la rotación en rango porcentual definido en ubicaciones AIF.
  Args:
      sRutaEntrada (str): Carpeta donde se encuentan los eventos ordenados por carpetas Evento
      sRutaSalida (str): Carpeta donde se generaran los nuevos eventos generados
      lEvento (list): Eventos a considerarse en la generación
      iPorcentajeInicio (int): Porcentaje de tiempo de espacio de rotación inicial).
      iPorcentajeFin  o (int): Porcentaje de tiempo de espacio de rotación final.
      dEvento (dict): Diccionario de con nombres de archivos, Ej. 200, {'HY':['a1','a2',...], 'LP':['b1','b2',...], 'TR':['c1','c2',...], 'VT':['d1','d2',...]}
  """
  # Leyendo eventos
  for sEvento in lEvento:
    # Leyendo lista de archivo de eventos desde carpeta
    m=fym.lista_archivos_simple(sRutaEntrada+sEvento)
    if len(m)>0:
      # Crear carpetas de salida si no existen
      fym.create_folders(sRutaSalida+sEvento)
      # Generando la cantidad de eventos solicitado
      for iCont in range(len(m)):
        # Abrir evento
        sRuta = fym.archivos_canal_simple(sRutaEntrada+sEvento, m[iCont])
        # Abrir los eventos
        tr = TSignal(sRuta)
        # Preproceso
        tr.preproceso()
        # Normalizar señales
        tr.normaliza()
        # Elimina ruido()
        #tr.eliminaRuido(fRango=0.1, fTolerancia=1.0)
        # Generación de nueva señal ==========================================================================
        for i in range(iPorcentajeInicio, iPorcentajeFin+1):
          for u in ['I','A','F']:   # Ubicación de rotación
            trc=tr.copy()
            trc.daAgregaRotacion(i/100, u)
            # Espectrograma guarda en disco
            trc.espectrograma_guardar_canal(0, sRutaSalida+sEvento, 224,'.'+str(i).zfill(2)+u)
            del trc
    # Mensaje
    print("Generado eventos:", sEvento, fym.now_string())

def generarEventoTiempoRangoAleatorio(sRutaEntrada:str, sRutaSalida:str, dEvento:dict, iPorcentajeInicio:int, iPorcentajeFin:int):
  """Genera imagenes espectrogramas mediante la rotación en rango porcentual definido en ubicaciones AIF sin repeticion.
  Args:
      sRutaEntrada (str): Carpeta donde se encuentan los eventos ordenados por carpetas Evento
      sRutaSalida (str): Carpeta donde se generaran los nuevos eventos generados
      dEvento (dict): Diccionario de eventos y cantidades a generera por evento, Ej. {'HY':2000, 'LP':1500, 'TC':0, 'TR':1800, 'VT':1350}
      iPorcentajeInicio (int): Porcentaje de tiempo de espacio de rotación inicial).
      iPorcentajeFin  o (int): Porcentaje de tiempo de espacio de rotación final.
  """
  # Leyendo eventos
  for sEvento in dEvento:
    print("Generando eventos:", sEvento, fym.now_string())
    # Leyendo lista de archivo de eventos desde carpeta
    m=fym.lista_archivos_simple(sRutaEntrada+sEvento)
    if len(m)>0:
      # Generando lista de tuplas (indiceArchivo, porcentaje, ubicacion).
      lRotacion=[]
      for i in range(len(m)):                                 # Indice archivo
        for p in range(iPorcentajeInicio,iPorcentajeFin+1):   # Rango porcentaje rotaciones
          for u in ['I','A','F']:                             # Ubicacion de rotacion
            lRotacion.append((i,p,u))
      # Generando lista que tiene muestra de items a generar espectrogramas, sin repeticion
      lMuestra = sample(lRotacion, dEvento[sEvento])
      # Crear carpetas de salida si no existen
      fym.create_folders(sRutaSalida+sEvento)
      # Generando espectrograms la cantidad de la muestra solicitada
      for iArchivo, iPorcentaje, sUbicacion in lMuestra:
        # Abrir evento
        sRuta = fym.archivos_canal_simple(sRutaEntrada+sEvento, m[iArchivo])
        # Abrir los eventos
        tr = TSignal(sRuta)
        # Preproceso
        tr.preproceso()
        # Normalizar señales
        tr.normaliza()
        # Elimina ruido()
        #tr.eliminaRuido(fRango=0.1, fTolerancia=1.0)
        # Generación de nueva señal ==========================================================================
        tr.daAgregaRotacion(iPorcentaje/100, sUbicacion)
        # Espectrograma guarda en disco
        tr.espectrograma_guardar_canal(0, sRutaSalida+sEvento, 224,'.'+str(iPorcentaje).zfill(2)+sUbicacion)
      # Eliminando de memoria variables
      del lRotacion
      del lMuestra
    # Mensaje
    print("Generado eventos :", sEvento, fym.now_string())

#print("Inicio:", fym.now_string())
#generarEvento('escenario2/data/', 'escenario2/data_augmentation_rotacion/', FLAG_DA.lEvento, iCantidad={'HY':2000, 'LP':1500, 'TR':1800, 'VT':1350}, iPorcentajeTiempoMax=50, iPorcentajeTiempoMin=45)
#print("Fin   :", fym.now_string())

print("Inicio:", fym.now_string())
dEventos=lee_lista('eventos.pkl')
for i in range(FLAG_DA.iPorcentajeTiempoInicio, FLAG_DA.iPorcentajeTiempoFinal+1, FLAG_DA.iPorcentajeTiempoInc):
  print("Generando para tiempo: ", i, '-', i+FLAG_DA.iPorcentajeTiempoLong, fym.now_string())
  # Generando espectrogramas nuevos
  #generarEventoTiempoFijo(FLAG_DA.sRutaEntrada, FLAG_DA.sRutaSalida, FLAG_DA.lEvento, iCantidad={'HY':2000, 'LP':1500, 'TR':1800, 'VT':1350}, iPorcentajeTiempo=i, sUbicacion=FLAG_DA.sUbicacion)
  #generarEventoTiempoFijoLista(FLAG_DA.sRutaEntrada, FLAG_DA.sRutaSalida, iPorcentajeTiempo=i, sUbicacion=FLAG_DA.sUbicacion, dEvento=dEventos)
  generarEventoTiempoRangoAleatorio(FLAG_DA.sRutaEntrada, FLAG_DA.sRutaSalida, {'HY':2473, 'LP':2109, 'TC':488 ,'TR':2215}, i, i+FLAG_DA.iPorcentajeTiempoLong)
  # Agregando espectrogramas originales sin corte ruido
  print("Copiar espectrogramas originales:")
  call(r'xcopy /s /q "D:\UCN\Python\escenario2\spectrograms224_00 210708" H:\spectrograms224_00')
  # Comprime carpeta
  print("Comprime a ZIP:")
  call(r'C:\Program Files\7-Zip\7z.exe a -tzip H:\spectrograms224_00'+'.'+str(i).zfill(2)+'-'+str(i+FLAG_DA.iPorcentajeTiempoLong).zfill(2)+'.zip H:\spectrograms224_00')
  # Mover archivo a google drive
  print('Mueve a Google Drive:' +'spectrograms224_00'+'.'+str(i).zfill(2)+'-'+str(i+FLAG_DA.iPorcentajeTiempoLong).zfill(2)+'.zip')
  call(r'move "H:\spectrograms224_00'+'.'+str(i).zfill(2)+'-'+str(i+FLAG_DA.iPorcentajeTiempoLong).zfill(2)+r'.zip" "D:\Google Drive\Espectrogramas\Escenario2\00\spectrograms224_00 +daRotacion3"', shell=True)
  # Eliminar carpeta de espectrogramas
  print(r"Elimina carpeta de espectrogramas generado: H:\spectrograms224_00")
  call(r'rmdir /q /s H:\spectrograms224_00', shell=True)
print("Fin   :", fym.now_string())

