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

# Using a non-interactive backend for writing files
#matplotlib.use('agg')

# Signal class
class TSignal(Stream):
  # Constructor
  def __init__(self, sRutaArchivoZ:str, sRutaArchivoE:str=None, sRutaArchivoN:str=None):
    # Read primary channel
    super(TSignal, self).__init__(read(sRutaArchivoZ)[0])
    #self.sRuta  = sRutaArchivoZ
    _, sArchivo = os.path.split(sRutaArchivoZ)
    self.nombre = sArchivo.rsplit('.')[0]
    self.ext    = [sArchivo.rsplit('.')[1]]
    # Read other channels
    if sRutaArchivoE is not None:
      super(TSignal, self).append(read(sRutaArchivoE)[0])
      _, sArchivo = os.path.split(sRutaArchivoE)
      self.ext.append(sArchivo.rsplit('.')[1])
    if sRutaArchivoN is not None:
      super(TSignal, self).append(read(sRutaArchivoN)[0])
      _, sArchivo = os.path.split(sRutaArchivoN)
      self.ext.append(sArchivo.rsplit('.')[1])
  # Copy object
  def copy(self):
    return copy.deepcopy(self)
  # Deprecated preprocessing
  def preprocesoOld(self):
    # Resample to 100
    super(TSignal, self).resample(100.0)
    # Subtract mean from signal
    for tr in self.traces:
      tr.data = tr.data-np.mean(tr.data)
    # Highpass and bandpass filter
    super(TSignal, self).filter('highpass', freq=1)
    super(TSignal, self).filter('bandpass', freqmin=1, freqmax=10, corners=10)
  # Preprocessing according to AGU recommendation
  def preproceso(self):
    # Subtract mean from signal
    for tr in self.traces:
      tr.data = tr.data-np.mean(tr.data)
    # Highpass and bandpass filter
    super(TSignal, self).filter('highpass', freq=1)
    super(TSignal, self).filter('bandpass', freqmin=1, freqmax=20, corners=10)
    # Resample to 100
    super(TSignal, self).resample(100.0)
  # Signal duration
  def duracion(self):
    #return self.tr.stats.endtime-self.tr.stats.starttime)
    if len(self.traces)>0:
      return self.traces[0].stats.npts/self.traces[0].stats.sampling_rate
    return -1
  # Channel signal duration in seconds
  def duracionCanal(self, sCanal:str):
    """Return the duration in seconds of the signal for the given channel.
    Args:
      sCanal (str): Channel: Z,[EW,E,W],[NS,N,S] or equivalent 0,1,2
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
  # Trace signal duration
  def duracion_traza(self, iIndice:int):
    if len(self.traces)>iIndice:
      return self.traces[iIndice].stats.npts/self.traces[iIndice].stats.sampling_rate
    return -1
  # Check that all channels have the same duration
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

  # Adjust time length by padding with zeros
  def ajuste_tiempo(self, iTiempo:int):
    # Adjust all traces
    for tr in self.traces:
      # Total signal time (seconds)
      iTotal=tr.stats.npts/tr.stats.sampling_rate
      # Is requested duration longer than signal duration?
      if iTiempo>iTotal:
        # Remaining time
        iT1=int( ((iTiempo - iTotal)*tr.stats.sampling_rate)//2 )
        iT2=int( (iTiempo - iTotal)*tr.stats.sampling_rate - iT1 )
        # Adjust data
        tr.data=np.concatenate((np.zeros(iT1, dtype=int), tr.data, np.zeros(iT2, dtype=int)))
      elif iTiempo<iTotal:
        # Truncate data. Keep initial part
        tr.data=tr.data[0:int(iTiempo*tr.stats.sampling_rate)]
      else:
        pass
  # Save a borderless spectrogram for a channel to the specified path
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
      # Switch backend for speed
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
      # Release memory
      plt.clf()
      # Prevent interactive display
      plt.close('all')
      # Restore original backend
      #plt.switch_backend(backend_orig)

  # Save a borderless trace plot for a channel to the specified path
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
      # Release memory
      plt.clf()
      # Prevent interactive display
      plt.close('all')
  # Average signal across the 3 channels into one
  def promedio_canales(self, sRutaDirectorio:str=None, sNombreArchivo:str=None):
    """Return a trace that averages the three channels.
    Args:
        sRutaDirectorio (str, optional): Directory path to save the averaged trace, defaults to None.
        sNombreArchivo (str, optional): Specific file name, defaults to original trace name. Defaults to None.
    Returns:
        [type]: Averaged trace
    """
    trPromedio=None
    for tr in self.traces:
      if trPromedio is None:
        trPromedio=tr.copy()
      else:
        trPromedio.data+=tr.data
    trPromedio.data=trPromedio.data/len(self.traces)
    if sRutaDirectorio is not None:
      # Validate directory
      if sRutaDirectorio[len(sRutaDirectorio)-1]!='/':
        sRutaDirectorio+='/'
      if sNombreArchivo is None:
        trPromedio.write(sRutaDirectorio+self.nombre+'.mseed', format="MSEED")
      else:
        trPromedio.write(sRutaDirectorio+sNombreArchivo+'.mseed', format="MSEED")
    return trPromedio
  # Stack signals from the 3 channels into one
  def apilado_canales(self, sRutaDirectorio:str=None, sNombreArchivo:str=None):
    """Return a trace that stacks the three channels.
    Args:
        sRutaDirectorio (str, optional): Directory path to save the stacked trace, defaults to None.
        sNombreArchivo (str, optional): Specific file name, defaults to original trace name. Defaults to None.
    Returns:
        [type]: Stacked trace
    """
    trApilado=None
    for tr in self.traces:
      if trApilado is None:
        trApilado=tr.copy()
      else:
        trApilado.data+=tr.data
    if sRutaDirectorio is not None:
      # Validate directory
      if sRutaDirectorio[len(sRutaDirectorio)-1]!='/':
        sRutaDirectorio+='/'
      if sNombreArchivo is None:
        trApilado.write(sRutaDirectorio+self.nombre+'.mseed', format="MSEED")
      else:
        trApilado.write(sRutaDirectorio+sNombreArchivo+'.mseed', format="MSEED")
    return trApilado

  # Normalize signal
  def normaliza(self):
    # Normalize all traces
    #self.traces.normalize()
    for tr in  self.traces:
      tr.normalize()
    '''for tr in self.traces:
      # Find values [maximum, minimum]; absolute value and max value of array
      fMaximo=np.max(np.abs(np.array([np.max(tr.data), np.min(tr.data)])))
      # Normalize signal
      tr.data = tr.data/fMaximo'''

  # Remove initial noise in signal that fluctuates within range [-fRango, fRango]. Apply on normalized signals.
  # The tolerance (in seconds) is subtracted from the start cut point and added to the end cut point so the cut is not too close to the event edge.
  # After trimming, signal length may differ on each channel if there are multiple channels.
  def eliminaRuido(self, fRango:float=0.1, fTolerancia:float=1.0):
    # Process traces
    for tr in self.traces:
      # Start cut values
      xi = 0
      #xc = tr.times("matplotlib")[0]
      # Find where the noise ends, if it oscillates within the given range
      for i in range(len(tr.data)):
        if abs(tr.data[i])>fRango: break
        else:
          xi = i
          #xc = tr.times("matplotlib")[i]
      # End cut values
      xf = len(tr.data)-1
      for i in reversed(range(len(tr.data))):
        if abs(tr.data[i])>fRango: break
        else:
          xf = i
    # Apply tolerance
    if xi-int(tr.stats.sampling_rate*fTolerancia)>0:
      xi=xi-int(tr.stats.sampling_rate*fTolerancia)
    if xf+int(tr.stats.sampling_rate*fTolerancia)<len(tr.data)-1:
      xf=xf+int(tr.stats.sampling_rate*fTolerancia)
    # Trim signal
    if not(xi==0 and xf==len(tr.data)-1):
      tr.trim(tr.stats.starttime + tr.times()[xi], tr.stats.starttime + tr.times()[xf])
      #tr.trim(tr.stats.starttime + (xi+1)*(tr.stats.npts/tr.stats.sampling_rate)/len(tr.data)  )
      #tr.trim(tr.stats.starttime + tr.times()[xi+1], tr.stats.starttime + tr.times()[len(tr.data)-1])

  # ===================================== FEATURES  ===============================================
  def features(self, sCanal:str):
    """Generate a feature vector.
    Args:
        sCanal (str): Trace channel to process
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
    # Numpy array
    v = np.array(self.duracionCanal(i))                   # Duration in seconds
    v = np.append(v, [
      self.traces[i].stats.npts,            # Frames
      self.traces[i].stats.sampling_rate,   # Sampling rate
      np.mean(self.traces[i].data),         # Mean
      np.std(self.traces[i].data),          # Standard deviation
      skew(self.traces[i].data),            # Skewness
      kurtosis(self.traces[i].data),        # Kurtosis
    ])

    return np.around(v, 5)


  # ================================= DATA AUGMENTATION ===========================================
  # Add rotation/zeros to the signal at the start, end, or both sides.
  def daAgregaRotacion(self, fPorcentaje:float=0.1, sUbicacion:str='I'):
    """Add zeros to the signal at the start, end, or both sides.
    Args:
      fPorcentaje (float, optional): Percentage of event duration used to create zero padding. Defaults to 0.1=10%.
      sPosiciones (str, optional): Location where zero padding is added: I=start, F=end, or A=both sides. Defaults to 'I'.
    """
    # Process traces
    if sUbicacion in ['I','F','A']:
      # Add zero array
      for tr in self.traces:
        if sUbicacion=='I':
          tr.data=np.concatenate([np.zeros(int(len(tr.data)*fPorcentaje)), tr.data])
        elif sUbicacion=='F':
          tr.data=np.concatenate([tr.data, np.zeros(int(len(tr.data)*fPorcentaje))])
        else:
          tr.data=np.concatenate([np.zeros(int(len(tr.data)*fPorcentaje)//2), tr.data, np.zeros(int(len(tr.data)*fPorcentaje))//2])
  # Save a borderless spectrogram for a channel to the specified path
  def daEspectrogramaSpecAugment(self, sCanal:str, sRutaDirectorio:str, iTamanioPixel:int, sCorrelativo:str='', sColor:str='#00007F',
                                 fFrecuenciaPorcentaje:float=0.1, iFrecuenciaCantidad:int=2, fTiempoPorcentaje:float=0.1, iTiempoCantidad:int=0):
    """Generate modified spectrograms using SpecAugment data augmentation, frequency (horizontal) and time (vertical) masks.
    Args:
        sCanal (str): Trace channel to process
        sRutaDirectorio (str): Path to save the generated image
        iTamanioPixel (int): Spectrogram size in pixels for both height and width.
        sCorrelativo (str, optional): Default ''. The image is generated using the signal file name and this string is appended when non-empty.
        fFrecuenciaPorcentaje (float, optional): Defaults to 10%. Frequency mask height relative to iTamanioPixel.
                                           It is generated randomly in the range (0, 1]*20.
        iFrecuenciaCantidad (int, optional): Defaults to 2. Number of frequency masks to generate.
        fTiempoPorcentaje (float, optional): Defaults to 10%. Time mask width relative to iTamanioPixel.
                                           It is generated randomly in the range (0, iTamanioPixel*fTiempoPorcentaje].
        iTiempoCantidad (int, optional): Defaults to 0. Number of time masks to generate.
    """
    def GeneraMascara(fMaxValor:float, fPorcentaje:float, iCantidad):
      """Return a list of masks as tuples (x0, width) within a limit.
      Args:
          fMaxValor (float): Maximum range value [0, fMaxValor] for generating masks.
          fPorcentaje (float): Percentage of fMaxValor to use for generating random widths.
          iCantidad ([type]): Number of masks to generate.
      Returns:
          [list]: List of masks. [(x0, width0),(x1, width1),...]
      """
      lMascara=[]
      for _ in range(iCantidad):
        bIntersectado=True   # Overlap check
        while bIntersectado:
          xAncho=(1.0-random())*fPorcentaje*fMaxValor   # Mask height
          x0=(randint(0, 1000)/1000)*(fMaxValor-xAncho)  # Start point in [0,fMaxValor] for the mask
          # Check for intersection with existing masks
          bIntersectado=False # Assume there is no overlap
          for a, b in lMascara:
            if a<x0<b or a<x0+xAncho<b or x0+xAncho>fMaxValor:
              bIntersectado=True
              break
        # Add mask to list
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
    # Draw
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
    # Release memory
    plt.clf()
    # Prevent interactive display
    plt.close('all')
    #plt.cla()
    del fig
  # Generate a new trace from this and another object
  def daAlgortimoGenetico(self, str2:"TSignal", iSegmentoTiempo:int=1, iSegmentoCruce:int=5, bMostrarMensaje:bool=False):
    """Generate a new trace from this one and another using genetic algorithms.
    Args:
        str2 (TSignal): Another object of the same type
        iSegmentoTiempo (int, optional): Duration of each segment in seconds. Defaults to 1.
        iSegmentoCruce (int, optional): Number of crossover segments. Defaults to 5.
    """
    # Return a list of unique random indices within the given range
    def listaEnteroAleatorio(iMenor:int, iMaximo:int, iElementos:int):
      lLista=sample(range(iMenor, iMaximo+1), iElementos)
      lLista.sort()
      return lLista

    # Check that both have the same number of channels
    if len(self.traces)!=len(str2.traces):
      if bMostrarMensaje:
        print("Error different number of channels between the two events")
      return None
    # Verify segment counts
    if (self.duracion()//iSegmentoTiempo)<iSegmentoCruce or (str2.duracion()//iSegmentoTiempo)<iSegmentoCruce:
      if bMostrarMensaje:
        print("Insufficient number of segments for crossover given the segment size")
      return None
    # New event as a copy of the second event
    tr3=str2.copy()
    # Process traces
    for i, tr in enumerate(self.traces):
      # Generate crossover time points and corresponding range tuples
      if self.duracionCanal(i)<str2.duracionCanal(i):
        lTiempo=listaEnteroAleatorio(0, int(self.duracionCanal(i)//iSegmentoTiempo)-1, iSegmentoCruce)
      else:
        lTiempo=listaEnteroAleatorio(0, int(str2.duracionCanal(i)//iSegmentoTiempo)-1, iSegmentoCruce)
      lPunto = [(int(t*iSegmentoTiempo*tr.stats.sampling_rate), int((t+1)*iSegmentoTiempo*tr.stats.sampling_rate)) for t in lTiempo]
      for (x,y) in lPunto:
        # Scale - MUTATION
        iMax1, iMax2 = np.absolute(self.traces[i].data[x:y]).max(), np.absolute(str2.traces[i].data[x:y]).max()
        # Replace crossover segment and MUTATE (scale)
        tr3.traces[i].data[x:y]=self.traces[i].data[x:y]*(iMax2/iMax1)
        # Average segment (MUTATION - XOR)
        #for j in range(x+1, y):
        #  tr3.traces[i].data[j]=(tr3.traces[i].data[j]+tr3.traces[i].data[j-1])/2.0
    # Result
    return tr3
  # Generate a new trace from this and another object by modifying time segments
  def daAlgortimoGenetico1(self, str2:"TSignal", fPorcentaje:float=0.3, iSegmentoCruce:int=5, bAjusteSegmentoContiguo:bool=False, bMostrarMensaje:bool=False):
    """Generate a new trace from this and another using genetic algorithms.
    Args:
        str2 (TSignal): Another object of the same type
        fPorcentaje(float, optional): Percentage of the signal from self in the child; (1-fPorcentaje) from the other signal. Defaults to 0.3.
        iSegmentoCruce (int, optional): Number of crossover segments. Defaults to 5.
        bAjusteSegmentoContiguo (bool, optional): If True, contiguous segments are merged into one. Defaults to False.
    """
    # Return a list of unique random indices within the given range. Includes wider ranges.
    def listaEnteroAleatorio(iMenor:int, iMaximo:int, iElementos:int):
      lLista=sample(range(iMenor, iMaximo+1), iElementos)
      lLista.sort()
      return lLista
    # Return whether contiguous segments exist. [..., (0, 200), (200, 400), ...]
    def bExisteSegmentoContiguo(lstSegmento:list):
      if(len(lstSegmento)>1):
        for i in range(len(lstSegmento)-1):
          if(lstSegmento[i][1]==lstSegmento[i+1][0]):
            return True
      return False
    # Merge all existing contiguous segments. [..., (0, 200), (200, 400), ...] => [..., (0, 400), ...]
    def UneSegmentoContiguo(lstSegmento:list):
      while bExisteSegmentoContiguo(lstSegmento):
        for i in range(len(lstSegmento)-1):
          if(lstSegmento[i][1]==lstSegmento[i+1][0]):
            lstSegmento[i]=(lstSegmento[i][0], lstSegmento[i+1][1])
            lstSegmento.pop(i+1) #del lstSegmento[i+1]
            break

    # Check that both have the same number of channels
    if len(self.traces)!=len(str2.traces):
      if bMostrarMensaje:
        print("Error different number of channels between the two events")
      return None
    # New event as a copy of the second event
    tr3=str2.copy()
    # Process traces
    for i, _ in enumerate(self.traces):
      # Calculate segment sizes
      iSegmentoSize1=int(self.traces[i].data.size*fPorcentaje/iSegmentoCruce)
      iSegmentoSize2=int(str2.traces[i].data.size*fPorcentaje/iSegmentoCruce)
      # Verify crossover segment size
      if iSegmentoSize1==0 or iSegmentoSize2==0:
        if bMostrarMensaje:
          print('iSegmentoSize=0: Crossover segment size is zero')
        return None
      lTiempo=listaEnteroAleatorio(0, int(self.traces[i].data.size//iSegmentoSize1)-1, iSegmentoCruce) # Cut points in signal 1
      lPunto1 = [(int(t*iSegmentoSize1), int((t+1)*iSegmentoSize1)) for t in lTiempo]     # Segments in signal 1
      lPunto2 = [(int(t*iSegmentoSize2), int((t+1)*iSegmentoSize2)) for t in lTiempo]     # Segments in signal 2
      # Adjust contiguous segments
      if(bAjusteSegmentoContiguo):
        UneSegmentoContiguo(lPunto1)
        UneSegmentoContiguo(lPunto2)
      # Generate new signal
      for (x1,x2),(y1,y2) in zip(reversed(lPunto1), reversed(lPunto2)):
        # Scale
        iMax1, iMax2 = np.absolute(self.traces[i].data[x1:x2]).max(), np.absolute(tr3.traces[i].data[y1:y2]).max()
        # Replace crossover segment and MUTATE (scale)
        #tr3.traces[i].data[y1:y2]=self.traces[i].data[x1:x2]*(iMax2/iMax1)  # Does not work
        tr3.traces[i].data = np.delete(tr3.traces[i].data, slice(y1,y2))
        tr3.traces[i].data = np.insert(tr3.traces[i].data, y1, self.traces[i].data[x1:x2]*(iMax2/iMax1) )
    # Result
    return tr3

  # Add drifting to the signal.
  def daDrifting(self, fDeriva:float=0.1):
    """Add drift to the signal traces.
    Args:
      fDeriva (float, optional): Drift magnitude used to generate random steps. Defaults to 0.1.
    """
    # Generate random steps
    vPaso=[]
    vPaso.append(-fDeriva if random()<0.5 else fDeriva)
    for i in range(1, self.traces[0].data.size):
      value = vPaso[i-1] + (-fDeriva if random()<0.5 else fDeriva)
      vPaso.append(value)

    # Add drift to the traces
    for tr in self.traces:
      tr.data=np.add(tr.data, np.array(vPaso))
  # Add Gaussian noise to the signal.
  def daJittering(self, fSigma:float=0.2):
    """Add noise to the signal traces.
    Args:
      fSigma (float, optional): Standard deviation of the random noise distribution. Defaults to 0.2.
    """
    # Add noise to the traces
    for tr in self.traces:
      # Generate random Gaussian noise
      noise = np.random.normal(0, fSigma, len(tr.data)) #  μ = 0, σ = 2, size = length of x or y. Choose μ and σ wisely.
      # Add noise to the signal
      tr.data = tr.data + noise   # Since both y and noise are numpy arrays of same size, the addition is done element-wise.
      #tr.data=np.add(tr.data, np.array(vPaso))
  def daScaling(self, fScale:float=1.1):
    """Scale the signal traces.
    Args:
      fScale (float, optional): Scaling factor to apply to the signal. Defaults to 1.1.
    """
    # Scale the signal by the scaling parameter
    for tr in self.traces:
      tr.data=np.multiply(tr.data, fScale)
  def daFlipping(self):
    """Flip the signal by multiplying by -1.
    """
    # Flip the signal by scaling with -1
    self.daScaling(-1)

  def daInterpolationAE(self, str2:"TSignal", ModeloAE:nn.Module, sCanal:str, fPorcentaje:float=0.5, sRutaDirectorio:str='', bMostrarMensaje:bool=False):
    """Generate a new spectrogram from this and another signal using AutoEncoder interpolation.
    Args:
        str2 (TSignal): Another object of the same class
        ModeloAE: Instance of autoencoder.AE() managing an AE.
        fPorcentaje(float, optional): Interpolation percentage between the two signals. Range [0, 1]. 0=self, 1=str2. Supports three decimal percentages like 45.123%. Defaults to 0.5.
        sRutaDirectorio (str): Output path on disk
        bMostrarMensaje (bool): Show error messages. Defaults to False
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
        print("Error in the channel input for signal processing")
      return

    plt.switch_backend('Agg')
    # Spectrograms for processing
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

    # Combine the two images into an array
    x=torch.stack([transform(img1.convert("RGB")), transform(img2.convert("RGB"))]).to(ModeloAE.device)
    # Encode
    embedding = ModeloAE.modelo.encoder(x)

    # Interpolate the two embeddings and decode them
    #e = e1*(1-i/10) + e2*(i/10)
    e = embedding[0]*(1-fPorcentaje) + embedding[1]*(fPorcentaje)
    d = ModeloAE.modelo.decoder(torch.stack([e]))
    # Save new signal
    imgRes = transforms.ToPILImage()(d[0]).convert("RGB")
    #imgRes.save(sRutaDirectorio+self.nombre+'.'+self.ext[i]+'-'+str(fPorcentaje*100)+'.png')
    imgRes.save(sRutaDirectorio+self.nombre+'.'+self.ext[i]+'-'+"{0:.3f}".format(fPorcentaje*100)+'.png')

    # Release memory
    plt.clf()
    # Prevent the image from displaying in interactive mode
    plt.close('all')
    #plt.cla()
    del fig

# Signal list class
class TListSignal:
  # Constructor
  def __init__(self, streams=None):
    self.streams = []
    if isinstance(streams, Stream):
      streams = [streams]
    if streams:
      self.streams.extend(streams)
  # Length
  def __len__(self):
    return len(self.streams)
  # Non-empty
  def __nonzero__(self):
    return bool(len(self.streams))
  # Iterator
  def __iter__(self):
    return list(self.streams).__iter__()
  # Item
  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.__class__(streams=self.streams.__getitem__(index))
    else:
      return self.streams.__getitem__(index)
  # Add
  def add(self, st:TSignal):
    if isinstance(st, TSignal):
      self.streams.append(st)
    else:
      msg = 'Append only supports a single TSignal object as argument.'
      raise TypeError(msg)
    return self
  # Minimum duration of the streams
  def duration(self):
    iDuracion=[]
    for tr in self.streams:
      iDuracion.append(tr.duracion())
    return min(iDuracion), max(iDuracion)
  # Maximum duration of the streams
  def max_duration(self):
    iDuracion=[]
    for tr in self.streams:
      iDuracion.append(tr.duracion())
    return max(iDuracion)
  # Minimum duration of the streams
  def min_duration(self):
    iDuracion=[]
    for tr in self.streams:
      iDuracion.append(tr.duracion())
    return min(iDuracion)
  # Mean duration of the streams
  def mean_duration(self):
    iDuracion=[]
    for tr in self.streams:
      iDuracion.append(tr.duracion())
    return mean(iDuracion)
  # Median duration of the streams
  def median_duration(self):
    iDuracion=[]
    for tr in self.streams:
      iDuracion.append(tr.duracion())
    return median(iDuracion)
  # Mode duration of the streams
  def mode_duration(self):
    iDuracion=[]
    for tr in self.streams:
      iDuracion.append(tr.duracion())
    return mode(iDuracion)
  # Standard deviation of stream durations
  def stdev_duration(self):
    iDuracion=[]
    for tr in self.streams:
      iDuracion.append(tr.duracion())
    if len(iDuracion)>1:
      return stdev(iDuracion)
    else:
      return None
  # Variance of stream durations
  def variance_duration(self):
    iDuracion=[]
    for tr in self.streams:
      iDuracion.append(tr.duracion())
    if len(iDuracion)>1:
      return variance(iDuracion)
    else:
      return None
  # Adjust time length by adding zeros
  def ajuste_tiempo(self, iTiempo:int):
    for tr in self.streams:
      tr.ajuste_tiempo(iTiempo)
