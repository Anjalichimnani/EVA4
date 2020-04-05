from imports.imports_eva import *

class ResBlock (nn.Module):
  def __init__(self, nf):
    super(ResBlock, self).__init__()

    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(num_features=nf),
        nn.ReLU()
    )

    self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(num_features=nf),
        nn.ReLU()
    )

  def forward (self, x):
    return x + self.conv2(self.conv1(x))


class CustomNet (nn.Module):
  
  def __init__(self):
    super(CustomNet, self).__init__()

    #Input Block 
    self.convinputblock = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
    
    #Layer 1, Convolution Block
    self.convblock1 = nn.Sequential(
        nn.Conv2d (in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(num_features=128),
        nn.ReLU()
    )

    #Layer 1, Define the ResBlock
    self.resblock128 = ResBlock (nf=128)

    #Layer2, Convolution Block
    self.convblock2 = nn.Sequential (
        nn.Conv2d (in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(num_features=256),
        nn.ReLU()
    )

    #Layer 3, Convolution Block
    self.convblock3 = nn.Sequential(
        nn.Conv2d (in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d (num_features=512),
        nn.ReLU()
    )

    #Layer 3, Define the ResBlock with features 512
    self.resblock512 = ResBlock (nf=512)

    self.pool4 = nn.MaxPool2d(kernel_size=4, ceil_mode=True)
    self.fc = nn.Linear (in_features=512, out_features=10, bias=False)
    
  def forward (self, x):
    x = self.convinputblock(x)

    #Layer1
    x = self.convblock1(x)
    x = x + self.resblock128 (x)

    #Layer 2
    x = self.convblock2(x)
    
    #Layer3
    x = self.convblock3(x)
    x = x + self.resblock512 (x)

    x = self.pool4(x)
    x = x.view (-1, 512)
    x = self.fc(x)
    return F.log_softmax(x, dim=-1)
