# Import libraries
import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Define parameters
class CONST_AE:
  carpeta       = None  # Spectrogram folder 'spectrograms224_00'
  classes       = None  # Class list           ['HY','LP','TC','TR','VT']
  num_classes   = None  # Number of classes    5
  batch_size    = None  # Spectrogram batch size 4
  num_workers   = None  # Worker processes     2
  learning_rate = None  # Learning rate        0.001
  num_epochs    = None  # Training epochs      100

# Convolutional autoencoder architectures
class CAE1(nn.Module):  # 2,054 parameters.
  def __init__(self):
    super().__init__()
    self.encoder = nn.Sequential( # Layer composition
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

class CAE2(nn.Module):  # 5,542 parameters.
  def __init__(self):
    super().__init__()
    self.encoder = nn.Sequential( # Layer composition
      nn.Conv2d(3, 8, 3,  stride=2, padding=1), #B*3*224*224-> B*8*112*112
      nn.ReLU(),
      nn.Conv2d(8, 16, 3, stride=2, padding=1), #B*8*112*112-> B*16*56*56
      nn.ReLU(),
      nn.Conv2d(16, 8, 3, stride=2, padding=1), #B*16*56*56 -> B*8*28*28
      nn.ReLU(),
      nn.Conv2d(8, 3, 3)                   #B*8*28*28  -> B*3*26*26=B*2028
    )
    self.decoder = nn.Sequential( # Layer composition
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

class CAE3(nn.Module):  # 186,371 parameters.
  def __init__(self):
    super().__init__()
    self.encoder = nn.Sequential( # Layer composition
      nn.Conv2d(3, 32, 3, stride=1, padding=1),   #B*3*224*224 -> B*32*224*224
      nn.ReLU(),
      nn.MaxPool2d(2),                            #B*32*224*224 -> B*32*112*112
      nn.Conv2d(32, 64, 3, stride=1, padding=1),  #B*32*112*112 -> B*64*112*112
      nn.ReLU(),
      nn.MaxPool2d(2),                            #B*64*112*112 -> B*64*56*56
      nn.Conv2d(64, 128, 3, stride=1, padding=1), #B*64*56*56   -> B*128*56*56
      #nn.ReLU(),                                  #???
    )
    self.decoder = nn.Sequential( # Layer composition
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

# Manages autoencoders
class AE():
  # Define parameters
  class FLAGS_AE:
    carpeta       = 'spectrograms224_00'        # Spectrogram folder
    classes       = ['HY','LP','TC','TR','VT']  # Class list
    num_classes   = 5          # Number of classes
    batch_size    = 4         # Creates the torch dataset from spectrograms
    num_workers   = 2          # Worker processes
    learning_rate = 0.001      # Learning rate
    num_epochs    = 100        # Number of training epochs

  def __init__(self, ModeloAE:nn.Module, sRutaModeloArchivo:str=None, bMostraDevice:bool=False):
    # Use GPU if available
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if bMostraDevice:
      print("Device:", self.device)
    # Create and load model on the selected device, either GPU or CPU
    self.modelo = ModeloAE().to(self.device)
    self.FLAGS_AE = CONST_AE()
    # Load model
    if sRutaModeloArchivo is not None:
      self.CargaModelo(sRutaModeloArchivo)
  def CargaModelo(self, sRutaModeloArchivo:str):
    self.modelo, self.FLAGS_AE, _, _, _, _, _ = self.modelo_carga(sRutaModeloArchivo, self.modelo, self.FLAGS_AE)
  def MostrarArquitectura(self):
    self.modelo.eval()
  # Load model from file
  def modelo_carga(self, sNombreArchivo:str, tModelo, tFlag):
    # Recover values (model data) from disk
    tArchivo  = torch.load(sNombreArchivo)

    # Recover parameters
    tFlag.carpeta = tArchivo['carpeta']
    tFlag.classes = tArchivo['clases']
    tFlag.num_classes = tArchivo['num_classes']
    tFlag.batch_size  = tArchivo['batch_size']
    tFlag.num_workers = tArchivo['num_workers']
    tFlag.learning_rate = tArchivo['learning_rate']
    tFlag.num_epochs  = tArchivo['num_epochs']
    tTimeTrain        = tArchivo['time_train'] # Training time
    #fTiempProceso ???
    vLossTrain        = tArchivo['train_loss']
    vLossTest         = tArchivo['test_loss']
    vAccuracyTrain    = tArchivo['train_acc']
    vAccuracyTest     = tArchivo['test_acc']

    # Recover model
    tModelo.load_state_dict(tArchivo['modelo'])

    # Return
    return tModelo, tFlag, vLossTrain, vLossTest, vAccuracyTrain, vAccuracyTest, tTimeTrain







