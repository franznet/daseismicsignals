import numpy as np
import matplotlib.pyplot as plt
from obspy.core import read
import fym.util as fym
from fym.signal import TSignal, TListSignal
import random

# Constantes
RUTA_ENTRADA    ='escenario2/data/'
RUTA_SALIDA     ='escenario2/01/data_augmentation_ag/'
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

# Genera eventos nuevos por AG con detalle del proceso
def generarEventoDetalle(sRutaEntrada:str, sRutaSalida:str, lEvento:list, iSegmentoTamanio:int, iSegmentoCruce:int, iCantidad:int):
  """Genera imagenes de sismograma y espectrograma mediante algoritmos geneticos.
  Args:
      sRutaEntrada (str): Carpeta donde se encuentan los eventos ordenados por carpetas Evento
      sRutaSalida (str): Carpeta donde se generaran los nuevos eventos generados
      lEvento (list): Eventos a considerarse en la generación
      iSegmentoTamanio (int): Tamano del segmento en segundos
      iSegmentoCruce (int):   Cantidad de segmentos para cruzamiento
      iCantidad (int): [description]
  """
  # Leyendo eventos
  for sEvento in lEvento:
    # Leyendo lista de archivo de evetos desde carpeta
    m=fym.lista_archivos_simple(sRutaEntrada+sEvento)
    if len(m)>0:
      # Crear carpetas de salida si no existen
      fym.create_folders(sRutaSalida+str(iSegmentoTamanio)+'_'+str(iSegmentoCruce)+'/'+sEvento)
      # Generando la cantidad de eventos solicitado
      iContador=0
      while iContador<iCantidad:
        # Generar números aleatorios
        lEventoIndice=listaEnteroAleatorio(0,len(m)-1,2) # Mismo tiempo(26,74) #[1, 2] #
        # Escoger eventos al azar (SELECTION)
        evento1, evento2=m[lEventoIndice[0]], m[lEventoIndice[1]]
        # Generar rutas de los elegidos
        sRuta1, sRuta2 = fym.archivos_canal_simple(sRutaEntrada+sEvento, evento1), fym.archivos_canal_simple(sRutaEntrada+sEvento, evento2)
        # Abrir los eventos
        tr1, tr2 = TSignal(sRuta1), TSignal(sRuta2)
        # Verificar tiempos
        if (tr1.duracion()//iSegmentoTamanio)<iSegmentoCruce or (tr2.duracion()//iSegmentoTamanio)<iSegmentoCruce:
          continue  # Rechazar
        # Preproceso
        tr1.preproceso()
        tr2.preproceso()
        # Normalizar señales
        tr1.normaliza()
        tr2.normaliza()
        # Generación de nueva señal ==========================================================================
        tr3=tr2.copy()
        # Generando listas de puntos de tiempo de cruzamiento y tuplas de rangos correspondientes (CROSS OVER)
        if tr1.duracion()<tr2.duracion():
          lTiempo=listaEnteroAleatorio(0, (tr1.duracion()//iSegmentoTamanio)-1, iSegmentoCruce)
        else:
          lTiempo=listaEnteroAleatorio(0, (tr2.duracion()//iSegmentoTamanio)-1, iSegmentoCruce)
        lPunto = [(int(t*iSegmentoTamanio*tr1.traces[0].stats.sampling_rate), int((t+1)*iSegmentoTamanio*tr1.traces[0].stats.sampling_rate)) for t in lTiempo]
        for (x,y) in lPunto:
          # Reemplazando segmento de cruzamiento
          tr3.traces[0].data[x:y]=tr1.traces[0].data[x:y]
          # Promediando segmento(MUTATION - XOR)
          for i in range(x+1, y):
            tr3.traces[0].data[i]=(tr3.traces[0].data[i]+tr3.traces[0].data[i-1])/2.0
        # Desplegar sismogramas
        #tr1.plot(size=(1500, 200), color='red',   number_of_ticks=10, tick_format='%I:%M %p')
        #tr2.plot(size=(1500, 200), color='green', number_of_ticks=tr2.duracion(), tick_format='%I:%M %p')
        #tr3.plot(size=(1500, 200), number_of_ticks=tr3.duracion(), tick_format='%I:%M %p')

        # Desplegar sismograma matplotlib
        fig = plt.figure(figsize=(25,8))
        ax = fig.add_subplot(3, 1, 1) # Grafico1
        ax.plot(tr1.traces[0].times("matplotlib"), tr1.traces[0].data, "b-")
        ax.set_title('Evento'+str(lEventoIndice[0])+' ['+sRuta1+'] '+str(tr1.duracion())+'s')
        for (x,y) in lPunto: ax.plot(tr1.traces[0].times("matplotlib")[x:y], tr1.traces[0].data[x:y], "r-")
        ax = fig.add_subplot(3, 1, 2) # Grafico2
        ax.plot(tr2.traces[0].times("matplotlib"), tr2.traces[0].data, "b-")
        ax.set_title('Evento'+str(lEventoIndice[1])+' ['+sRuta2+'] '+str(tr2.duracion())+'s')
        ax = fig.add_subplot(3, 1, 3) # Grafico3
        ax.plot(tr3.traces[0].times("matplotlib"), tr3.traces[0].data, "b-")
        ax.set_title('EventoResultado '+str(tr3.duracion())+'s '+str(lTiempo) )
        for (x,y) in lPunto: ax.plot(tr3.traces[0].times("matplotlib")[x:y], tr3.traces[0].data[x:y], "r-")
        ax.xaxis_date()
        fig.autofmt_xdate()
        #plt.show()
        plt.savefig(sRutaSalida+str(iSegmentoTamanio)+'_'+str(iSegmentoCruce)+'/'+sEvento+'/'+evento1+'_'+evento2+'_sismograma.png')

        # Desplegar espectrogramas
        #tr1.spectrogram(title='Evento'+str(lEventoIndice[0])+' '+str(tr1.duracion())+'s', cmap='jet', per_lap=0.95, wlen=1, samp_rate=100)
        #tr2.spectrogram(title='Evento'+str(lEventoIndice[1])+' '+str(tr2.duracion())+'s', cmap='jet', per_lap=0.95, wlen=1, samp_rate=100)
        #tr3.spectrogram(title='EventoRes1 '+str(tr3.duracion())+'s '+str(lTiempo), cmap='jet', per_lap=0.95, wlen=1, samp_rate=100)

        # Desplegar espectrograma matplotlib
        fig = plt.figure(figsize=(25,6))
        plt.subplot(131)  # Grafico1
        ax=plt.gca()
        ax.set_title('Evento'+str(lEventoIndice[0])+' '+str(tr1.duracion())+'s')
        tr1.traces[0].spectrogram(show=False,axes=ax, cmap='jet', samp_rate=100.0, per_lap=0.95, wlen=1)
        plt.subplot(132)  # Grafico2
        ax=plt.gca()
        ax.set_title('Evento'+str(lEventoIndice[1])+' '+str(tr2.duracion())+'s')
        tr2.traces[0].spectrogram(show=False,axes=ax, cmap='jet', samp_rate=100.0, per_lap=0.95, wlen=1)
        plt.subplot(133)  # Grafico3
        ax=plt.gca()
        ax.set_title('EventoResultado '+str(tr3.duracion())+'s '+str(lTiempo) )
        tr3.traces[0].spectrogram(show=False,axes=ax, cmap='jet', samp_rate=100.0, per_lap=0.95, wlen=1)
        #plt.show()
        plt.savefig(sRutaSalida+str(iSegmentoTamanio)+'_'+str(iSegmentoCruce)+'/'+sEvento+'/'+evento1+'_'+evento2+'_espectrograma.png')

        # Liberar memoria
        plt.clf()
        # Evitar que muestre la imagen en modo interactivo
        plt.close('all')

        # Incrementar contador
        iContador+=1

# Genera eventos nuevos por AG, espectrograma
def generarEvento(sRutaEntrada:str, sRutaSalida:str, lEvento:list, fPorcentaje:float, iSegmentoCruce:int, bAjusteSegmentoContiguo:bool, iCantidad:int):
  """Genera imagenes de sismograma y espectrograma mediante algoritmos geneticos.
  Args:
      sRutaEntrada (str): Carpeta donde se encuentan los eventos ordenados por carpetas Evento
      sRutaSalida (str): Carpeta donde se generaran los nuevos eventos generados
      lEvento (list): Eventos a considerarse en la generación
      fPorcentaje (float): Porcentaje de señal en hijo, (1-fporcentaje) de otra señal en hijo
      iSegmentoCruce (int): Cantidad de segmentos para cruzamiento
      bAjusteSegmentoContiguo (bool): Por True, se unira segmentos continguos en uno solo. eoc, no.
      iCantidad (int): [description]
  """
  # Leyendo eventos
  for sEvento in lEvento:
    # Leyendo lista de archivo de evetos desde carpeta
    m=fym.lista_archivos_simple(sRutaEntrada+sEvento)
    if len(m)>0:
      # Crear carpetas de salida si no existen
      fym.create_folders(sRutaSalida+str(fPorcentaje)+'_'+str(iSegmentoCruce)+('T' if bAjusteSegmentoContiguo else 'F')+'/'+sEvento)
      # Generando la cantidad de eventos solicitado
      iContador=0
      while iContador<iCantidad:
        # Generar números aleatorios
        lEventoIndice=listaEnteroAleatorio(0,len(m)-1,2) # Mismo tiempo(26,74) #[1, 2] #
        # Escoger eventos al azar (SELECTION)
        evento1, evento2=m[lEventoIndice[0]], m[lEventoIndice[1]]
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
        tr1.eliminaRuido(fRango=0.1, fTolerancia=1.0)
        tr2.eliminaRuido(fRango=0.1, fTolerancia=1.0)
        # Genera nuevo evento por AG
        tr3=tr1.daAlgortimoGenetico1(tr2, fPorcentaje=fPorcentaje, iSegmentoCruce=iSegmentoCruce, bAjusteSegmentoContiguo=bAjusteSegmentoContiguo)
        if tr3 is not None:
          # Espectrograma guarda en disco
          tr3.espectrograma_guardar_canal(0, sRutaSalida+str(fPorcentaje)+'_'+str(iSegmentoCruce)+('T' if bAjusteSegmentoContiguo else 'F')+'/'+sEvento, 224,'-'+str(iContador+1))
          # Incrementar contador
          iContador+=1
    # Mensaje
    print("Generado eventos:", sEvento, fym.now_string())


print("Inicio:", fym.now_string())
generarEvento(RUTA_ENTRADA, RUTA_SALIDA, EVENTO_ESTUDIO, fPorcentaje=0.45, iSegmentoCruce=10, bAjusteSegmentoContiguo=True, iCantidad=2686)
print("Fin   :", fym.now_string())