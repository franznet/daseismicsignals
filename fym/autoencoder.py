# Impotar librerias
import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Define parámetros
class CONST_AE:
  carpeta       = None  # Carpeta de espectrogramas 'spectrograms224_00'
  classes       = None  # Lista de clases           ['HY','LP','TC','TR','VT']
  num_classes   = None  # Número de clases          5
  batch_size    = None  # Tamaño de lote de espectrogramas 4
  num_workers   = None  # Procesos                  2
  learning_rate = None  # Tasa de aprendizaje       0.001
  num_epochs    = None  # Número de épocas de entrenamiento 100

# Arquitecturas de autoencoder Convolutional
class CAE1(nn.Module):  # 2.054 parámetros.
  def __init__(self):
    super().__init__()
    self.encoder = nn.Sequential( # Composición de capas
      nn.Conv2d(3, 8, 3, stride=2, padding=1),  #B*3*224*224 -> B*8*112*112
      nn.ReLU(),
      nn.Conv2d(8, 8, 3, stride=2, padding=1),  #B*8*112*112 -> B*8*56*56
      nn.ReLU(),
      nn.Conv2d(8, 3, 3)                        #B*8*56*56   -> B*3*54*54 = B*8748
    )
    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(3, 8, 3),
      nn.ReLU(),
      nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1, output_padding=1),
      nn.ReLU(),
      nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1),
      nn.Sigmoid()
    )
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

class CAE2(nn.Module):  # 5.542 parámetros.
  def __init__(self):
    super().__init__()
    self.encoder = nn.Sequential( # Composición de capas
      nn.Conv2d(3, 8, 3,  stride=2, padding=1), #B*3*224*224-> B*8*112*112
      nn.ReLU(),
      nn.Conv2d(8, 16, 3, stride=2, padding=1), #B*8*112*112-> B*16*56*56
      nn.ReLU(),
      nn.Conv2d(16, 8, 3, stride=2, padding=1), #B*16*56*56 -> B*8*28*28
      nn.ReLU(),
      nn.Conv2d(8, 3, 3)                   #B*8*28*28  -> B*3*26*26=B*2028
    )
    self.decoder = nn.Sequential( # Composición de capas
      nn.ConvTranspose2d(3, 8, 3),
      nn.ReLU(),
      nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1),
      nn.ReLU(),
      nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
      nn.ReLU(),
      nn.ConvTranspose2d(8, 3, 3,  stride=2, padding=1, output_padding=1),
      nn.Sigmoid()
    )
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

class CAE3(nn.Module):  # 186.371 parámetros.
  def __init__(self):
    super().__init__()
    self.encoder = nn.Sequential( # Composición de capas
      nn.Conv2d(3, 32, 3, stride=1, padding=1),   #B*3*224*224 -> B*32*224*224
      nn.ReLU(),
      nn.MaxPool2d(2),                            #B*32*224*224 -> B*32*112*112
      nn.Conv2d(32, 64, 3, stride=1, padding=1),  #B*32*112*112 -> B*64*112*112
      nn.ReLU(),
      nn.MaxPool2d(2),                            #B*64*112*112 -> B*64*56*56
      nn.Conv2d(64, 128, 3, stride=1, padding=1), #B*64*56*56   -> B*128*56*56
      #nn.ReLU(),                                  #???
    )
    self.decoder = nn.Sequential( # Composición de capas
      nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),  #B*128*56*56   -> B*64*56*56
      nn.ReLU(),
      nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), #B*64*56*56  -> B*32*112*112
      nn.ReLU(),
      nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),  #B*32*112*112   -> B*3*224*224
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

# Administra autoencoders
class AE():
  # Define parámetros
  class FLAGS_AE:
    carpeta       = 'spectrograms224_00'        # Carpeta de espectrogramas
    classes       = ['HY','LP','TC','TR','VT']  # Lista de clases
    num_classes   = 5          # Número de clases
    batch_size    = 4         # Genera el dataset torch de los espectrogramas
    num_workers   = 2          # Procesos
    learning_rate = 0.001      # Tasa de aprendizaje
    num_epochs    = 100        # Número de épocas de entrenamiento

  def __init__(self, ModeloAE:nn.Module, sRutaModeloArchivo:str=None, bMostraDevice:bool=False):
    # Usar GPU si esta disponible
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if bMostraDevice:
      print("Device:", self.device)
    # Crear y cargar al dispositivo especifico, ya sea GPU o CPU
    self.modelo = ModeloAE().to(self.device)
    self.FLAGS_AE = CONST_AE()
    # Cargar modelo
    if sRutaModeloArchivo is not None:
      self.CargaModelo(sRutaModeloArchivo)
  def CargaModelo(self, sRutaModeloArchivo:str):
    self.modelo, self.FLAGS_AE, _, _, _, _, _ = self.modelo_carga(sRutaModeloArchivo, self.modelo, self.FLAGS_AE)
  def MostrarArquitectura(self):
    self.modelo.eval()
  # Carga modelo desde archivo
  def modelo_carga(self, sNombreArchivo:str, tModelo, tFlag):
    # Recuperando valores(modelos) de disco
    tArchivo  = torch.load(sNombreArchivo)

    # Recuperando parametros
    tFlag.carpeta = tArchivo['carpeta']
    tFlag.classes = tArchivo['clases']
    tFlag.num_classes = tArchivo['num_classes']
    tFlag.batch_size  = tArchivo['batch_size']
    tFlag.num_workers = tArchivo['num_workers']
    tFlag.learning_rate = tArchivo['learning_rate']
    tFlag.num_epochs  = tArchivo['num_epochs']
    tTimeTrain        = tArchivo['time_train'] # Tiempo de Entrenamiento
    #fTiempProceso ???
    vLossTrain        = tArchivo['train_loss']
    vLossTest         = tArchivo['test_loss']
    vAccuracyTrain    = tArchivo['train_acc']
    vAccuracyTest     = tArchivo['test_acc']

    # Recuperando modelo
    tModelo.load_state_dict(tArchivo['modelo'])

    # Retornando
    return tModelo, tFlag, vLossTrain, vLossTest, vAccuracyTrain, vAccuracyTest, tTimeTrain







