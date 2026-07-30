import numpy as np
import matplotlib.pyplot as plt
from obspy.core import read
import fym.util as fym
from fym.signal import TSignal, TListSignal
import random

# Constants
RUTA_ENTRADA    ='escenario2/data/'
RUTA_SALIDA     ='escenario2/01/data_augmentation_ag/'
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

# Generate new events using GA with detailed processing
def generarEventoDetalle(sRutaEntrada:str, sRutaSalida:str, lEvento:list, iSegmentoTamanio:int, iSegmentoCruce:int, iCantidad:int):
  """Generate seismogram and spectrogram images using genetic algorithms.
  Args:
      sRutaEntrada (str): Folder where events are stored in event subfolders
      sRutaSalida (str): Folder where generated events will be saved
      lEvento (list): Events to consider for generation
      iSegmentoTamanio (int): Segment size in seconds
      iSegmentoCruce (int): Number of segments for crossover
      iCantidad (int): Number of events to generate
  """
  # Reading events
  for sEvento in lEvento:
    # Read list of event files from folder
    m=fym.lista_archivos_simple(sRutaEntrada+sEvento)
    if len(m)>0:
      # Create output folders if they do not exist
      fym.create_folders(sRutaSalida+str(iSegmentoTamanio)+'_'+str(iSegmentoCruce)+'/'+sEvento)
      # Generating the requested number of events
      iContador=0
      while iContador<iCantidad:
        # Generate random indices
        lEventoIndice=listaEnteroAleatorio(0,len(m)-1,2) # Mismo tiempo(26,74) #[1, 2] #
        # Select events at random (SELECTION)
        evento1, evento2=m[lEventoIndice[0]], m[lEventoIndice[1]]
        # Build file paths for the selected events
        sRuta1, sRuta2 = fym.archivos_canal_simple(sRutaEntrada+sEvento, evento1), fym.archivos_canal_simple(sRutaEntrada+sEvento, evento2)
        # Open the events
        tr1, tr2 = TSignal(sRuta1), TSignal(sRuta2)
        # Check durations
        if (tr1.duracion()//iSegmentoTamanio)<iSegmentoCruce or (tr2.duracion()//iSegmentoTamanio)<iSegmentoCruce:
          continue  # Reject
        # Preprocess
        tr1.preproceso()
        tr2.preproceso()
        # Normalize signals
        tr1.normaliza()
        tr2.normaliza()
        # Generate new signal ==================================================================
        tr3=tr2.copy()
        # Generate lists of crossover time points and corresponding range tuples (CROSS OVER)
        if tr1.duracion()<tr2.duracion():
          lTiempo=listaEnteroAleatorio(0, (tr1.duracion()//iSegmentoTamanio)-1, iSegmentoCruce)
        else:
          lTiempo=listaEnteroAleatorio(0, (tr2.duracion()//iSegmentoTamanio)-1, iSegmentoCruce)
        lPunto = [(int(t*iSegmentoTamanio*tr1.traces[0].stats.sampling_rate), int((t+1)*iSegmentoTamanio*tr1.traces[0].stats.sampling_rate)) for t in lTiempo]
        for (x,y) in lPunto:
          # Replace crossover segment
          tr3.traces[0].data[x:y]=tr1.traces[0].data[x:y]
          # Average the segment (MUTATION - XOR)
          for i in range(x+1, y):
            tr3.traces[0].data[i]=(tr3.traces[0].data[i]+tr3.traces[0].data[i-1])/2.0
        # Display seismograms
        #tr1.plot(size=(1500, 200), color='red',   number_of_ticks=10, tick_format='%I:%M %p')
        #tr2.plot(size=(1500, 200), color='green', number_of_ticks=tr2.duracion(), tick_format='%I:%M %p')
        #tr3.plot(size=(1500, 200), number_of_ticks=tr3.duracion(), tick_format='%I:%M %p')

        # Display seismogram using matplotlib
        fig = plt.figure(figsize=(25,8))
        ax = fig.add_subplot(3, 1, 1) # Plot1
        ax.plot(tr1.traces[0].times("matplotlib"), tr1.traces[0].data, "b-")
        ax.set_title('Evento'+str(lEventoIndice[0])+' ['+sRuta1+'] '+str(tr1.duracion())+'s')
        for (x,y) in lPunto: ax.plot(tr1.traces[0].times("matplotlib")[x:y], tr1.traces[0].data[x:y], "r-")
        ax = fig.add_subplot(3, 1, 2) # Plot2
        ax.plot(tr2.traces[0].times("matplotlib"), tr2.traces[0].data, "b-")
        ax.set_title('Evento'+str(lEventoIndice[1])+' ['+sRuta2+'] '+str(tr2.duracion())+'s')
        ax = fig.add_subplot(3, 1, 3) # Plot3
        ax.plot(tr3.traces[0].times("matplotlib"), tr3.traces[0].data, "b-")
        ax.set_title('EventoResultado '+str(tr3.duracion())+'s '+str(lTiempo) )
        for (x,y) in lPunto: ax.plot(tr3.traces[0].times("matplotlib")[x:y], tr3.traces[0].data[x:y], "r-")
        ax.xaxis_date()
        fig.autofmt_xdate()
        #plt.show()
        plt.savefig(sRutaSalida+str(iSegmentoTamanio)+'_'+str(iSegmentoCruce)+'/'+sEvento+'/'+evento1+'_'+evento2+'_sismograma.png')

        # Display spectrograms
        #tr1.spectrogram(title='Evento'+str(lEventoIndice[0])+' '+str(tr1.duracion())+'s', cmap='jet', per_lap=0.95, wlen=1, samp_rate=100)
        #tr2.spectrogram(title='Evento'+str(lEventoIndice[1])+' '+str(tr2.duracion())+'s', cmap='jet', per_lap=0.95, wlen=1, samp_rate=100)
        #tr3.spectrogram(title='EventoRes1 '+str(tr3.duracion())+'s '+str(lTiempo), cmap='jet', per_lap=0.95, wlen=1, samp_rate=100)

        # Display spectrogram using matplotlib
        fig = plt.figure(figsize=(25,6))
        plt.subplot(131)  # Plot1
        ax=plt.gca()
        ax.set_title('Evento'+str(lEventoIndice[0])+' '+str(tr1.duracion())+'s')
        tr1.traces[0].spectrogram(show=False,axes=ax, cmap='jet', samp_rate=100.0, per_lap=0.95, wlen=1)
        plt.subplot(132)  # Plot2
        ax=plt.gca()
        ax.set_title('Evento'+str(lEventoIndice[1])+' '+str(tr2.duracion())+'s')
        tr2.traces[0].spectrogram(show=False,axes=ax, cmap='jet', samp_rate=100.0, per_lap=0.95, wlen=1)
        plt.subplot(133)  # Plot3
        ax=plt.gca()
        ax.set_title('EventoResultado '+str(tr3.duracion())+'s '+str(lTiempo) )
        tr3.traces[0].spectrogram(show=False,axes=ax, cmap='jet', samp_rate=100.0, per_lap=0.95, wlen=1)
        #plt.show()
        plt.savefig(sRutaSalida+str(iSegmentoTamanio)+'_'+str(iSegmentoCruce)+'/'+sEvento+'/'+evento1+'_'+evento2+'_espectrograma.png')

        # Free memory
        plt.clf()
        # Prevent interactive display
        plt.close('all')

        # Increment counter
        iContador+=1

# Generate new events using GA (spectrogram saving)
def generarEvento(sRutaEntrada:str, sRutaSalida:str, lEvento:list, iSegmentoTamanio:int, iSegmentoCruce:int, iCantidad:int):
  """Generate seismogram and spectrogram images using genetic algorithms.
  Args:
      sRutaEntrada (str): Folder where events are stored in event subfolders
      sRutaSalida (str): Folder where generated events will be saved
      lEvento (list): Events to consider for generation
      iSegmentoTamanio (int): Segment size in seconds
      iSegmentoCruce (int): Number of segments for crossover
      iCantidad (int): Number of events to generate
  """
  # Reading events
  for sEvento in lEvento:
    # Read list of event files from folder
    m=fym.lista_archivos_simple(sRutaEntrada+sEvento)
    if len(m)>0:
      # Create output folders if they do not exist
      fym.create_folders(sRutaSalida+str(iSegmentoTamanio)+'_'+str(iSegmentoCruce)+'/'+sEvento)
      # Generating the requested number of events
      iContador=0
      while iContador<iCantidad:
        # Generate random indices
        lEventoIndice=listaEnteroAleatorio(0,len(m)-1,2) # Mismo tiempo(26,74) #[1, 2] #
        # Select events at random (SELECTION)
        evento1, evento2=m[lEventoIndice[0]], m[lEventoIndice[1]]
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
        # Remove noise
        tr1.eliminaRuido(fRango=0.1, fTolerancia=1.0)
        tr2.eliminaRuido(fRango=0.1, fTolerancia=1.0)
        # Generate new event via GA
        tr3=tr1.daAlgortimoGenetico(tr2, iSegmentoTiempo=iSegmentoTamanio, iSegmentoCruce=iSegmentoCruce)
        if tr3 is not None:
          # Save spectrogram to disk
          tr3.espectrograma_guardar_canal(0, sRutaSalida+str(iSegmentoTamanio)+'_'+str(iSegmentoCruce)+'/'+sEvento, 224,'-'+str(iContador+1))
          # Increment counter
          iContador+=1
    # Message
    print("Generado eventos:", sEvento, fym.now_string())


print("Start:", fym.now_string())
generarEvento(RUTA_ENTRADA, RUTA_SALIDA, EVENTO_ESTUDIO, iSegmentoTamanio=5, iSegmentoCruce=5, iCantidad=2686)
print("End  :", fym.now_string())