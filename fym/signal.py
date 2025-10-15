import obspy
from obspy.core.stream import Stream
from obspy import read
import numpy as np
import os
from statistics import mean, median, mode, stdev, variance
import matplotlib.pyplot as plt
import copy
from matplotlib.patches import Rectangle
from random import randint, random, sample
from fym.util import fig2img
from fym.autoencoder import AE, CAE3
import torch
import torch.nn as nn
from torchvision import transforms
from scipy.stats import skew, kurtosis

# Usando backend no interactivo, para escribir en un archivo
#matplotlib.use('agg')

# Clase señal
class TSignal(Stream):
  # Constructor
  def __init__(self, sRutaArchivoZ:str, sRutaArchivoE:str=None, sRutaArchivoN:str=None):
    # Lectura de canal principal
    super(TSignal, self).__init__(read(sRutaArchivoZ)[0])
    #self.sRuta  = sRutaArchivoZ
    _, sArchivo = os.path.split(sRutaArchivoZ)
    self.nombre = sArchivo.rsplit('.')[0]
    self.ext    = [sArchivo.rsplit('.')[1]]
    # lectura de los otros canales
    if sRutaArchivoE is not None:
      super(TSignal, self).append(read(sRutaArchivoE)[0])
      _, sArchivo = os.path.split(sRutaArchivoE)
      self.ext.append(sArchivo.rsplit('.')[1])
    if sRutaArchivoN is not None:
      super(TSignal, self).append(read(sRutaArchivoN)[0])
      _, sArchivo = os.path.split(sRutaArchivoN)
      self.ext.append(sArchivo.rsplit('.')[1])
  # Copia de objeto
  def copy(self):
    return copy.deepcopy(self)
  # Preproceso obsoleto
  def preprocesoOld(self):
    # Resampling a 100
    super(TSignal, self).resample(100.0)
    # Restando la media de la señal
    for tr in self.traces:
      tr.data = tr.data-np.mean(tr.data)
    # Filtro paso alto y paso banda
    super(TSignal, self).filter('highpass', freq=1)
    super(TSignal, self).filter('bandpass', freqmin=1, freqmax=10, corners=10)
  # Preproceso segun recomendación AGU
  def preproceso(self):
    # Restando la media de la señal
    for tr in self.traces:
      tr.data = tr.data-np.mean(tr.data)
    # Filtro paso alto y paso banda
    super(TSignal, self).filter('highpass', freq=1)
    super(TSignal, self).filter('bandpass', freqmin=1, freqmax=20, corners=10)
    # Resampling a 100
    super(TSignal, self).resample(100.0)
  # Duración señal
  def duracion(self):
    #return self.tr.stats.endtime-self.tr.stats.starttime)
    if len(self.traces)>0:
      return self.traces[0].stats.npts/self.traces[0].stats.sampling_rate
    return -1
  # Duración señal de canal en segundos
  def duracionCanal(self, sCanal:str):
    """ Retorna la duraciçon en segundos de la señal en el canla dado.
    Args:
      sCanal (str): Canal: Z,[EW,E,W],[NS,N,S] o equivalente 0,1,2
    """
    i=-1
    if sCanal=='Z' or sCanal==0:
      i=0
    elif sCanal=='EW' or sCanal=='E' or sCanal=='W':
      i=1
    elif sCanal=='NS' or sCanal=='N' or sCanal=='S':
      i=2
    else:
      return None
    return self.traces[i].stats.npts/self.traces[i].stats.sampling_rate
  # Duración señal de traza
  def duracion_traza(self, iIndice:int):
    if len(self.traces)>iIndice:
      return self.traces[iIndice].stats.npts/self.traces[iIndice].stats.sampling_rate
    return -1
  # Todos los canales tienes la misma duración de tiempo
  def es_misma_duracion_canales(self):
    if len(self.traces)>1:
      iDuracion=None
      for tr in self.traces:
        if iDuracion is None:
          iDuracion=tr.stats.npts/tr.stats.sampling_rate
        else:
          if iDuracion!=(tr.stats.npts/tr.stats.sampling_rate):
            return False
    return True

  # Ajustar longitud en tiempo, agregando ceros
  def ajuste_tiempo(self, iTiempo:int):
    # Ajustar todas las trazas
    for tr in self.traces:
      # Tiempo total de señal(segundos)
      iTotal=tr.stats.npts/tr.stats.sampling_rate
      # Duración solicitada es mayor duracion de señal?
      if iTiempo>iTotal:
        # Tiempo restante
        iT1=int( ((iTiempo - iTotal)*tr.stats.sampling_rate)//2 )
        iT2=int( (iTiempo - iTotal)*tr.stats.sampling_rate - iT1 )
        # Ajustando data
        tr.data=np.concatenate((np.zeros(iT1, dtype=int), tr.data, np.zeros(iT2, dtype=int)))
      elif iTiempo<iTotal:
        # Corta data. Queda la parte inicial
        tr.data=tr.data[0:int(iTiempo*tr.stats.sampling_rate)]
      else:
        pass
  # Guarda espectrograma sin bordes de un canal, en la ruta especificada
  def espectrograma_guardar_canal(self, sCanal:str, sRutaDirectorio:str, iTamanioPixel:int, sCorrelativo:str=''):
    i=-1
    #sCanal=sCanal.upper()
    if sRutaDirectorio!='' and sRutaDirectorio[-1]!='/':
      sRutaDirectorio+='/'
    if sCanal=='Z' or sCanal==0:
      i=0
    elif sCanal=='EW' or sCanal=='E' or sCanal=='W':
      i=1
    elif sCanal=='NS' or sCanal=='N' or sCanal=='S':
      i=2
    else:
      pass
    if i>-1:
      # Cambiando backend para velocidad
      #backend_orig = plt.get_backend()
      plt.switch_backend('Agg')
      # Plotting
      fig = self.traces[i].spectrogram(show=False, cmap='jet', samp_rate=100.0, per_lap=0.95, wlen=1)
      fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
      fig.set_size_inches(iTamanioPixel/100, iTamanioPixel/100)
      ax = fig.axes[0]
      ax.axis('tight')
      ax.set_axis_off()
      ax.set_ylim(0.01, 20.0)
      plt.title('')
      plt.savefig(sRutaDirectorio+self.nombre+'.'+self.ext[i]+sCorrelativo+'.png', dpi=100, bbox_inches='tight', pad_inches=0)
      # Liberar memoria
      plt.clf()
      # Evitar que muestre la imagen en modo interactivo
      plt.close('all')
      # Restaurando backend original
      #plt.switch_backend(backend_orig)

  # Guarda espectrograma sin bordes de un canal, en la ruta especificada
  def sismograma_guardar_canal(self, sCanal:str, sRutaDirectorio:str, iTamanioPixel:int, sCorrelativo:str=''):
    i=-1
    #sCanal=sCanal.upper()
    if sRutaDirectorio!='' and sRutaDirectorio[-1]!='/':
      sRutaDirectorio+='/'
    if sCanal=='Z' or sCanal==0:
      i=0
    elif sCanal=='EW' or sCanal=='E' or sCanal=='W':
      i=1
    elif sCanal=='NS' or sCanal=='N' or sCanal=='S':
      i=2
    else:
      pass
    if i>-1:
      '''fig = self.traces[i].plot(show=False, cmap='jet', samp_rate=100.0, per_lap=0.95, wlen=1)
      fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
      fig.set_size_inches(iTamanioPixel/100, iTamanioPixel/100)
      ax = fig.axes[0]
      ax.axis('tight')
      ax.set_axis_off()
      ax.set_ylim(0.01, 20.0)
      plt.title('')
      plt.savefig(sRutaDirectorio+self.nombre+'.'+self.ext[i]+'.png', dpi=100, bbox_inches='tight', pad_inches=0)'''
      fig = plt.figure(frameon=False)
      fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
      #fig.set_size_inches(iTamanioPixel/100, iTamanioPixel/100)
      ax = fig.add_subplot(1, 1, 1)
      ax.axis('off')
      ax.plot(self.traces[i].data, "k")
      plt.autoscale(tight=True)
      plt.title('')
      #plt.savefig('prueba.png', dpi=100, frameon=False, aspect='normal', bbox_inches='tight', pad_inches=0)
      #plt.savefig(sRutaDirectorio+self.nombre+'.'+self.ext[i]+'.png', dpi=100, frameon=False, aspect='normal', bbox_inches='tight', pad_inches=0)
      plt.savefig(sRutaDirectorio+self.nombre+'.'+self.ext[i]+sCorrelativo+'.png', dpi=100, bbox_inches='tight', pad_inches=0)
      # Liberar memoria
      plt.clf()
      # Evitar que muestre la imagen en modo interactivo
      plt.close('all')
  # Promedia señal de los 3 canales en uno solo
  def promedio_canales(self, sRutaDirectorio:str=None, sNombreArchivo:str=None):
    """Retorna traza que promedia los tres canales
    Args:
        sRutaDirectorio (str, optional): Ruta de directorio para guardar traza promedio, por defecto no guarda. Defaults to None.
        sNombreArchivo (str, optional): Nombre de archivo especifico, por defecto guarda con nombre de traza original. Defaults to None.
    Returns:
        [type]: Trace promedio
    """
    trPromedio=None
    for tr in self.traces:
      if trPromedio is None:
        trPromedio=tr.copy()
      else:
        trPromedio.data+=tr.data
    trPromedio.data=trPromedio.data/len(self.traces)
    if sRutaDirectorio is not None:
      # Validando directorio
      if sRutaDirectorio[len(sRutaDirectorio)-1]!='/':
        sRutaDirectorio+='/'
      if sNombreArchivo is None:
        trPromedio.write(sRutaDirectorio+self.nombre+'.mseed', format="MSEED")
      else:
        trPromedio.write(sRutaDirectorio+sNombreArchivo+'.mseed', format="MSEED")
    return trPromedio
  # Pila de señales de los 3 canales en uno solo
  def apilado_canales(self, sRutaDirectorio:str=None, sNombreArchivo:str=None):
    """Retorna traza que apila los tres canales
    Args:
        sRutaDirectorio (str, optional): Ruta de directorio para guardar traza apilada, por defecto no guarda. Defaults to None.
        sNombreArchivo (str, optional): Nombre de archivo especifico, por defecto guarda con nombre de traza original. Defaults to None.
    Returns:
        [type]: Trace promedio
    """
    trApilado=None
    for tr in self.traces:
      if trApilado is None:
        trApilado=tr.copy()
      else:
        trApilado.data+=tr.data
    if sRutaDirectorio is not None:
      # Validando directorio
      if sRutaDirectorio[len(sRutaDirectorio)-1]!='/':
        sRutaDirectorio+='/'
      if sNombreArchivo is None:
        trApilado.write(sRutaDirectorio+self.nombre+'.mseed', format="MSEED")
      else:
        trApilado.write(sRutaDirectorio+sNombreArchivo+'.mseed', format="MSEED")
    return trApilado

  # Normalizar señal
  def normaliza(self):
    # Ajustar todas las trazas
    #self.traces.normalize()
    for tr in  self.traces:
      tr.normalize()
    '''for tr in self.traces:
      # Hallamos valores [máximo, mínimo]; el valor absoluto y el valor maximo de array
      fMaximo=np.max(np.abs(np.array([np.max(tr.data), np.min(tr.data)])))
      # Normalizamos señal
      tr.data = tr.data/fMaximo'''

  # ELimina ruido de inicial en señal que fluctua en el rango [-fRango, fRango]. Aplicarla en señales normalizadas.
  # La tolerancia (en segundos) se resta al punto inicial y se suma al punto final de corte, para que el corte no sea muy al filo del evento.
  # Tras recorte tiempo de señal podria se diferente en cada canal, si este tiene más de un canal.
  def eliminaRuido(self, fRango:float=0.1, fTolerancia:float=1.0):
    # Procesar las trazas
    for tr in self.traces:
      # Valores de corte de incio
      xi = 0
      #xc = tr.times("matplotlib")[0]
      # Buscar punto hasta donde el ruido dura. si oscila entre el rango dado
      for i in range(len(tr.data)):
        if abs(tr.data[i])>fRango: break
        else:
          xi = i
          #xc = tr.times("matplotlib")[i]
      # Valores de corte de final
      xf = len(tr.data)-1
      for i in reversed(range(len(tr.data))):
        if abs(tr.data[i])>fRango: break
        else:
          xf = i
    # Dando tolerancia
    if xi-int(tr.stats.sampling_rate*fTolerancia)>0:
      xi=xi-int(tr.stats.sampling_rate*fTolerancia)
    if xf+int(tr.stats.sampling_rate*fTolerancia)<len(tr.data)-1:
      xf=xf+int(tr.stats.sampling_rate*fTolerancia)
    # Recortar señal
    if not(xi==0 and xf==len(tr.data)-1):
      tr.trim(tr.stats.starttime + tr.times()[xi], tr.stats.starttime + tr.times()[xf])
      #tr.trim(tr.stats.starttime + (xi+1)*(tr.stats.npts/tr.stats.sampling_rate)/len(tr.data)  )
      #tr.trim(tr.stats.starttime + tr.times()[xi+1], tr.stats.starttime + tr.times()[len(tr.data)-1])

  # ===================================== FEATURES  ===============================================
  def features(self, sCanal:str):
    """[summary]
      Generar vector de features.
    Args:
        sCanal (str): Canal de traza a procesar
    """
    i=-1
    if sCanal=='Z' or sCanal==0:
      i=0
    elif sCanal=='EW' or sCanal=='E' or sCanal=='W':
      i=1
    elif sCanal=='NS' or sCanal=='N' or sCanal=='S':
      i=2
    else:
      return None
    # Array numpy
    v = np.array(self.duracionCanal(i))                   # Duración en segundos
    v = np.append(v, [
      self.traces[i].stats.npts,            # Frames
      self.traces[i].stats.sampling_rate,   # Frecuencia de muestreo
      np.mean(self.traces[i].data),         # Mean
      np.std(self.traces[i].data),          # Standard deviation
      skew(self.traces[i].data),            # Skewness
      kurtosis(self.traces[i].data),        # Kurtosis
    ])

    return np.around(v, 5)


  # ================================= DATA AUGMENTATION ===========================================
  # Agrega rotación/ceros a la señal, ya sea en la parte Inicial, Final o Ambos lados.
  def daAgregaRotacion(self, fPorcentaje:float=0.1, sUbicacion:str='I'):
    """ Agrega ceros a la señal, ya sea en la parte Inicial, Final o Ambos lados.
    Args:
      fPorcentaje (float, optional): Porcentaje de tiempo en relación al tiempo del evento, que se tomara para generar señal cero. Defaults to 0.1=10%.
      sPosiciones (str, optional):   Posición donne se agregara la señal cero, I=inicio, F=Final o A=Ambos lados. Defaults to 'I'.
    """
    # Procesar las trazas
    if sUbicacion in ['I','F','A']:
      # Agregar array cero
      for tr in self.traces:
        if sUbicacion=='I':
          tr.data=np.concatenate([np.zeros(int(len(tr.data)*fPorcentaje)), tr.data])
        elif sUbicacion=='F':
          tr.data=np.concatenate([tr.data, np.zeros(int(len(tr.data)*fPorcentaje))])
        else:
          tr.data=np.concatenate([np.zeros(int(len(tr.data)*fPorcentaje)//2), tr.data, np.zeros(int(len(tr.data)*fPorcentaje))//2])
  # Guarda espectrograma sin bordes de un canal, en la ruta especificada
  def daEspectrogramaSpecAugment(self, sCanal:str, sRutaDirectorio:str, iTamanioPixel:int, sCorrelativo:str='', sColor:str='#00007F',
                                 fFrecuenciaPorcentaje:float=0.1, iFrecuenciaCantidad:int=2, fTiempoPorcentaje:float=0.1, iTiempoCantidad:int=0):
    """[summary]
      Generar espectrograsmas modificados con data augmentation SpecAugment, mascaras de frecuencia(horizontal) y tiempo(vertical).
    Args:
        sCanal (str): Canal de traza a procesar
        sRutaDirectorio (str): Ruta de señal en disco
        iTamanioPixel (int): Tamaño en pixeles de espectrograma, tanto de altura y ancho.
        sCorrelativo (str, optional): Default ''. Se genera la imagen con el nombre del archivo de señal. Y se concatena este parametro si es diferente de ''
        fFrecuenciaPorcentaje (float, optional): Defaults 10%. Altura de máscara de frecuencia con referencia a iTamanioPixel.
                                           Se generara de forma aletatoria del rango (0, 1]*20.
        iFrecuenciaCantidad (int, optional): Defaults 2. Cantidad de máscaras de frecuencia a generarse.
        fTiempoPorcentaje (float, optional): Defaults 10%. Anchura de máscara de frecuencia con referencia a iTamanioPixel.
                                           Se generara de forma aletatoria del rango (0, iTamanioPixel*fTiempoPorcentaje].
        iTiempoCantidad (int, optional): [description]. Defaults to 0. Cantidad de máscaras de tiempo a generarse.
    """
    def GeneraMascara(fMaxValor:float, fPorcentaje:float, iCantidad):
      """ Retorna lista de mascaras en forma de tupla (x0, ancho) de acuerdo a un límite
      Args:
          fMaxValor (float): Máximo valor de rango [0, fMaxValor] parab generar mascaras.
          fPorcentaje (float): Porcentaje de fMaxValor a considerarse para generera anchos aleatorios.
          iCantidad ([type]): Cantidad de mascaras a generarse.
      Returns:
          [list]: Lista de mascaras. [(x0, ancho0),(x1, ancho1),...]
      """
      lMascara=[]
      for _ in range(iCantidad):
        bIntersectado=True   # Punto se intersecta
        while bIntersectado:
          xAncho=(1.0-random())*fPorcentaje*fMaxValor   # Altura de máscara
          x0=(randint(0, 1000)/1000)*(fMaxValor-xAncho)  # Punto de [0,fMaxValor] de inicio de mascara.
          # Revisamos si existe intersección con los existentes
          bIntersectado=False # Asumimos que no hay intersección
          for a, b in lMascara:
            if a<x0<b or a<x0+xAncho<b or x0+xAncho>fMaxValor:
              bIntersectado=True
              break
        # Agregar a lista de mascaras
        lMascara.append((x0, x0+xAncho))
      return lMascara

    i=-1
    #sCanal=sCanal.upper()
    if sRutaDirectorio!='' and sRutaDirectorio[-1]!='/':
      sRutaDirectorio+='/'
    if sCanal=='Z' or sCanal==0:
      i=0
    elif sCanal=='EW' or sCanal=='E' or sCanal=='W':
      i=1
    elif sCanal=='NS' or sCanal=='N' or sCanal=='S':
      i=2
    else:
      return

    plt.switch_backend('Agg')
    fig = self.traces[i].spectrogram(show=False, cmap='jet', samp_rate=100.0, per_lap=0.95, wlen=1)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    fig.set_size_inches(iTamanioPixel/100, iTamanioPixel/100)
    ax = fig.axes[0]
    ax.axis('tight')
    ax.set_axis_off()
    ax.set_ylim(0.01, 20.0)
    plt.title('')
    # ===========================================================================================
    fDuracion=self.duracionCanal(i)
    lMascaraFrecuencia=GeneraMascara(20.0, fFrecuenciaPorcentaje, iFrecuenciaCantidad)
    lMascaraTiempo    =GeneraMascara(fDuracion, fTiempoPorcentaje, iTiempoCantidad)
    # Pintar
    #f=open("specaugment.csv", "a")
    for y1, y2 in lMascaraFrecuencia:
      ax.add_patch(Rectangle((0, y1), fDuracion, y2-y1, edgecolor=sColor, facecolor=sColor, fill=True))
      #f.write(self.nombre+'.'+self.ext[i]+sCorrelativo+';y;'+str(y1).replace('.',',')+';'+str(y2).replace('.',',')+';'+str(fDuracion).replace('.',',')+'\n')
    for x1, x2 in lMascaraTiempo:
      ax.add_patch(Rectangle((x1, 0), x2-x1, 20.0, edgecolor=sColor, facecolor=sColor, fill=True))
      #f.write(self.nombre+'.'+self.ext[i]+sCorrelativo+';x;'+str(x1).replace('.',',')+';'+str(x2).replace('.',',')+';'+str(fDuracion).replace('.',',')+'\n')
    #f.close()
    # ===========================================================================================
    plt.savefig(sRutaDirectorio+self.nombre+'.'+self.ext[i]+sCorrelativo+'.png', dpi=100, bbox_inches='tight', pad_inches=0)
    # Liberar memoria
    plt.clf()
    # Evitar que muestre la imagen en modo interactivo
    plt.close('all')
    #plt.cla()
    del fig
  # Generar nuevo sismograma a partir de esta y otro objeto
  def daAlgortimoGenetico(self, str2:"TSignal", iSegmentoTiempo:int=1, iSegmentoCruce:int=5, bMostrarMensaje:bool=False):
    """ Genera un nuevo sismograma a partir de este y otro, mediante algoritmos geneticos.
    Args:
        str2 (TSignal): Otro objeto del mismo tipo
        iSegmentoTiempo (int, optional): Tiempo de cada segmento en segundos. Defaults to 1.
        iSegmentoCruce (int, optional): Número de segmentos de cruce. Defaults to 5.
    """
    # Retorna lista con indices aleatorio no repetidos de rango preveido
    def listaEnteroAleatorio(iMenor:int, iMaximo:int, iElementos:int):
      lLista=sample(range(iMenor, iMaximo+1), iElementos)
      lLista.sort()
      return lLista

    # Tienen el mismo número de canales
    if len(self.traces)!=len(str2.traces):
      if bMostrarMensaje:
        print("Error diferente número de canales entre los dos eventos")
      return None
    # Verificar tiempos
    if (self.duracion()//iSegmentoTiempo)<iSegmentoCruce or (str2.duracion()//iSegmentoTiempo)<iSegmentoCruce:
      if bMostrarMensaje:
        print("Número de segmentos insuficintes para cruce, dado el tamaño de segmento")
      return None
    # Nuevo evento como copia de evento
    tr3=str2.copy()
    # Procesamos trazas
    for i, tr in enumerate(self.traces):
      # Generando listas de puntos de tiempo de cruzamiento y tuplas de rangos correspondientes (CROSS OVER)
      if self.duracionCanal(i)<str2.duracionCanal(i):
        lTiempo=listaEnteroAleatorio(0, int(self.duracionCanal(i)//iSegmentoTiempo)-1, iSegmentoCruce)
      else:
        lTiempo=listaEnteroAleatorio(0, int(str2.duracionCanal(i)//iSegmentoTiempo)-1, iSegmentoCruce)
      lPunto = [(int(t*iSegmentoTiempo*tr.stats.sampling_rate), int((t+1)*iSegmentoTiempo*tr.stats.sampling_rate)) for t in lTiempo]
      for (x,y) in lPunto:
        # Escala - MUTACION
        iMax1, iMax2 = np.absolute(self.traces[i].data[x:y]).max(), np.absolute(str2.traces[i].data[x:y]).max()
        # Reemplazando segmento de cruzamiento y MUTACION(Escala)
        tr3.traces[i].data[x:y]=self.traces[i].data[x:y]*(iMax2/iMax1)
        # Promediando segmento(MUTATION - XOR)
        #for j in range(x+1, y):
        #  tr3.traces[i].data[j]=(tr3.traces[i].data[j]+tr3.traces[i].data[j-1])/2.0
    # Resultado
    return tr3
  # Generar nuevo sismograma a partir de esta y otro objeto, modificando tiempos
  def daAlgortimoGenetico1(self, str2:"TSignal", fPorcentaje:float=0.3, iSegmentoCruce:int=5, bAjusteSegmentoContiguo:bool=False, bMostrarMensaje:bool=False):
    """ Genera un nuevo sismograma a partir de este y otro, mediante algoritmos geneticos.
    Args:
        str2 (TSignal): Otro objeto del mismo tipo
        fPorcentaje(float, optional): Porcentaje de señal en hijo, (1-fporcentaje) de otra señal en hijo. Defaults to 0.3.
        iSegmentoCruce (int, optional): Número de segmentos de cruce. Defaults to 5.
        bAjusteSegmentoContiguo (bool, optional): Por True, se unira segmentos continguos en uno solo. eoc, no. Defauls to False.
    """
    # Retorna lista con indices aleatorio no repetidos de rango preveido. Incluye rangos más
    def listaEnteroAleatorio(iMenor:int, iMaximo:int, iElementos:int):
      lLista=sample(range(iMenor, iMaximo+1), iElementos)
      lLista.sort()
      return lLista
    # Retorna si existen segmentos contiguos. [..., (0, 200), (200, 400), ...]
    def bExisteSegmentoContiguo(lstSegmento:list):
      if(len(lstSegmento)>1):
        for i in range(len(lstSegmento)-1):
          if(lstSegmento[i][1]==lstSegmento[i+1][0]):
            return True
      return False
    # Une todos los segmentos contiguos existentes. [..., (0, 200), (200, 400), ...] => [..., (0, 400), ...]
    def UneSegmentoContiguo(lstSegmento:list):
      while bExisteSegmentoContiguo(lstSegmento):
        for i in range(len(lstSegmento)-1):
          if(lstSegmento[i][1]==lstSegmento[i+1][0]):
            lstSegmento[i]=(lstSegmento[i][0], lstSegmento[i+1][1])
            lstSegmento.pop(i+1) #del lstSegmento[i+1]
            break

    # Tienen el mismo número de canales
    if len(self.traces)!=len(str2.traces):
      if bMostrarMensaje:
        print("Error diferente número de canales entre los dos eventos")
      return None
    # Nuevo evento como copia de evento
    tr3=str2.copy()
    # Procesamos trazas
    for i, _ in enumerate(self.traces):
      # Calculando tamaños de segmentos
      iSegmentoSize1=int(self.traces[i].data.size*fPorcentaje/iSegmentoCruce)
      iSegmentoSize2=int(str2.traces[i].data.size*fPorcentaje/iSegmentoCruce)
      # Verificar tiempo de segmentos de cruce
      if iSegmentoSize1==0 or iSegmentoSize2==0:
        if bMostrarMensaje:
          print('iSegmentoSize=0: Tamaño de segmento de cruce igual a cero')
        return None
      lTiempo=listaEnteroAleatorio(0, int(self.traces[i].data.size//iSegmentoSize1)-1, iSegmentoCruce) # Puntos de corte en 1
      lPunto1 = [(int(t*iSegmentoSize1), int((t+1)*iSegmentoSize1)) for t in lTiempo]     # Segmentos de corte en 1
      lPunto2 = [(int(t*iSegmentoSize2), int((t+1)*iSegmentoSize2)) for t in lTiempo]     # Segmentos de corte en 2
      # Ajuste segementos contiguos
      if(bAjusteSegmentoContiguo):
        UneSegmentoContiguo(lPunto1)
        UneSegmentoContiguo(lPunto2)
      # Generndo señal nueva
      for (x1,x2),(y1,y2) in zip(reversed(lPunto1), reversed(lPunto2)):
        # Escala
        iMax1, iMax2 = np.absolute(self.traces[i].data[x1:x2]).max(), np.absolute(tr3.traces[i].data[y1:y2]).max()
        # Reemplazando segmento de cruzamiento y MUTACION(Escala)
        #tr3.traces[i].data[y1:y2]=self.traces[i].data[x1:x2]*(iMax2/iMax1)  # No funciona
        tr3.traces[i].data = np.delete(tr3.traces[i].data, slice(y1,y2))
        tr3.traces[i].data = np.insert(tr3.traces[i].data, y1, self.traces[i].data[x1:x2]*(iMax2/iMax1) )
    # Resultado
    return tr3

  # Agrega deriva a la señal.
  def daDrifting(self, fDeriva:float=0.1):
    """ Agrega deriva a la señal a las trazas.
    Args:
      fDeriva (float, optional): Valor de la deriva de la cual se generaran los pasos aleatorios. Defaults to 0.1.
    """
    # Generando pasos aleatorios
    vPaso=[]
    vPaso.append(-fDeriva if random()<0.5 else fDeriva)
    for i in range(1, self.traces[0].data.size):
      value = vPaso[i-1] + (-fDeriva if random()<0.5 else fDeriva)
      vPaso.append(value)

    # Agregando a la señal a las trazas
    for tr in self.traces:
      tr.data=np.add(tr.data, np.array(vPaso))
  # Agrega ruido gausiano a la señal.
  def daJittering(self, fSigma:float=0.2):
    """ Agrega ruido a la señal a las trazas.
    Args:
      fSigma (float, optional): Desviación estandar de la distribución del ruido aleatorio. Defaults to 2.
    """
    # Agregando ruido a las trazas
    for tr in self.traces:
      # Generando ruido gausiano aleatorio
      noise = np.random.normal(0, fSigma, len(tr.data)) #  μ = 0, σ = 2, size = length of x or y. Choose μ and σ wisely.
      # Agregando ruido a la señal
      tr.data = tr.data + noise   # Since both y and noise are numpy arrays of same size, the addition is done element-wise.
      #tr.data=np.add(tr.data, np.array(vPaso))
  def daScaling(self, fScale:float=1.1):
    """ Escala la señal de las trazas.
    Args:
      fScale (float, optional): Valor de escalamiento a agregar a la señal. Defaults to 1.1.
    """
    # Escalando señal mediante el parametro de escalación
    for tr in self.traces:
      tr.data=np.multiply(tr.data, fScale)
  def daFlipping(self):
    """ Invierte señal en eje de la señal(x) multiplicando por -1.
    """
    # Escalando señal mediante el parametro de escalación
    self.daScaling(-1)

  def daInterpolationAE(self, str2:"TSignal", ModeloAE:nn.Module, sCanal:str, fPorcentaje:float=0.5, sRutaDirectorio:str='', bMostrarMensaje:bool=False):
    """ Genera un nuevo espectrograma a partir de este y otro, mediante interpolación de AutoEncoder.
    Args:
        str2 (TSignal): Otro objeto de la misma clase
        ModeloAE: Instancia de la clase autoencoder.AE() que administra un AE.
        fPorcentaje(float, optional): Porcentaje de la interpolación de las dos señales. Rango [0, 1]. 0=self, 1=str2. Maneja 3 deciamles porcentuales como 45.123%. Defaults to 0.5.
        sRutaDirectorio (str): Ruta de señal en disco
        bMostrarMensaje (bool): Mostrar mensajes de error. Defaults to False
    """

    i=-1
    #sCanal=sCanal.upper()
    if sRutaDirectorio!='' and sRutaDirectorio[-1]!='/':
      sRutaDirectorio+='/'
    if sCanal=='Z' or sCanal==0:
      i=0
    elif sCanal=='EW' or sCanal=='E' or sCanal=='W':
      i=1
    elif sCanal=='NS' or sCanal=='N' or sCanal=='S':
      i=2
    else:
      if bMostrarMensaje:
        print("Error en el ingreso de canal de proceso de señales")
      return

    plt.switch_backend('Agg')
    # Espectrogramas para proceso
    fig = self[i].spectrogram(show=False, cmap='jet', samp_rate=100.0, per_lap=0.95, wlen=1)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    fig.set_size_inches(224/100, 224/100)
    ax = fig.axes[0]
    ax.axis('tight')
    ax.set_axis_off()
    ax.set_ylim(0.01, 20.0)
    plt.title('')
    img1=fig2img(fig)

    fig = str2[i].spectrogram(show=False, cmap='jet', samp_rate=100.0, per_lap=0.95, wlen=1)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    fig.set_size_inches(224/100, 224/100)
    ax = fig.axes[0]
    ax.axis('tight')
    ax.set_axis_off()
    ax.set_ylim(0.01, 20.0)
    plt.title('')
    img2=fig2img(fig)

    # ===========================================================================================
    # image to a Torch tensor
    transform = transforms.Compose([
      transforms.ToTensor()
    ])

    # Uniendo las dos imagenes en un array
    x=torch.stack([transform(img1.convert("RGB")), transform(img2.convert("RGB"))]).to(ModeloAE.device)
    # Codificando
    embedding = ModeloAE.modelo.encoder(x)

    # Interpolar las dos incrustaciones y decodificarlas
    #e = e1*(1-i/10) + e2*(i/10)
    e = embedding[0]*(1-fPorcentaje) + embedding[1]*(fPorcentaje)
    d = ModeloAE.modelo.decoder(torch.stack([e]))
    # Guardar nuena señal
    imgRes = transforms.ToPILImage()(d[0]).convert("RGB")
    #imgRes.save(sRutaDirectorio+self.nombre+'.'+self.ext[i]+'-'+str(fPorcentaje*100)+'.png')
    imgRes.save(sRutaDirectorio+self.nombre+'.'+self.ext[i]+'-'+"{0:.3f}".format(fPorcentaje*100)+'.png')

    # Liberar memoria
    plt.clf()
    # Evitar que muestre la imagen en modo interactivo
    plt.close('all')
    #plt.cla()
    del fig

# Clase lista de senales
class TListSignal:
  # Constructor
  def __init__(self, streams=None):
    self.streams = []
    if isinstance(streams, Stream):
      streams = [streams]
    if streams:
      self.streams.extend(streams)
  # Longitud
  def __len__(self):
    return len(self.streams)
  # No vacio
  def __nonzero__(self):
    return bool(len(self.streams))
  # Iterador
  def __iter__(self):
    return list(self.streams).__iter__()
  # Item
  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.__class__(streams=self.streams.__getitem__(index))
    else:
      return self.streams.__getitem__(index)
  # Agregar
  def add(self, st:TSignal):
    if isinstance(st, TSignal):
      self.streams.append(st)
    else:
      msg = 'Append solamente soporta un simple objeto TSignal como argumento.'
      raise TypeError(msg)
    return self
  # Duración mínima de los streams
  def duration(self):
    iDuracion=[]
    for tr in self.streams:
      iDuracion.append(tr.duracion())
    return min(iDuracion), max(iDuracion)
  # Duración máxima de los streams
  def max_duration(self):
    iDuracion=[]
    for tr in self.streams:
      iDuracion.append(tr.duracion())
    return max(iDuracion)
  # Duración mínima de los streams
  def min_duration(self):
    iDuracion=[]
    for tr in self.streams:
      iDuracion.append(tr.duracion())
    return min(iDuracion)
  # Duración promedio de los streams
  def mean_duration(self):
    iDuracion=[]
    for tr in self.streams:
      iDuracion.append(tr.duracion())
    return mean(iDuracion)
  # Duración mediana de los streams
  def median_duration(self):
    iDuracion=[]
    for tr in self.streams:
      iDuracion.append(tr.duracion())
    return median(iDuracion)
  # Duración moda de los streams
  def mode_duration(self):
    iDuracion=[]
    for tr in self.streams:
      iDuracion.append(tr.duracion())
    return mode(iDuracion)
  # Duración desviación estandar de los streams
  def stdev_duration(self):
    iDuracion=[]
    for tr in self.streams:
      iDuracion.append(tr.duracion())
    if len(iDuracion)>1:
      return stdev(iDuracion)
    else:
      return None
  # Duración^2 varianza de los streams
  def variance_duration(self):
    iDuracion=[]
    for tr in self.streams:
      iDuracion.append(tr.duracion())
    if len(iDuracion)>1:
      return variance(iDuracion)
    else:
      return None
  # Ajustar longitud en tiempo, agregando ceros
  def ajuste_tiempo(self, iTiempo:int):
    for tr in self.streams:
      tr.ajuste_tiempo(iTiempo)
