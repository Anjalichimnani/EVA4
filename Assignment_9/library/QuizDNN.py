from imports_eva import *
from datetime import datetime

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.05)
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.05)
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.05)
        )  #

        self.pool1 = nn.MaxPool2d((2, 2))

        self.convblock04 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.05)
        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.05)
        )

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.05)
        )

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.pool2 = nn.MaxPool2d((2, 2))  # out = 5

        self.convblock07 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=90, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(90),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=90, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(90),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=90, out_channels=90, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(90),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=90, out_channels=90, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(90),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        )

        self.linear1 = nn.Sequential(
            # nn.Conv2d(in_channels=90, out_channels=10, kernel_size=(1,1), padding=0, bias=False)
            nn.Linear(90, out_features=10)
        )

    def forward(self, x):
        #print("Current Date/Time: ", datetime.now())
        x1 = self.convblock1(x)  # 32*32*32
        x2 = self.convblock2(x1)  # 32*32*32
        x2 = x1 + x2  # 32*32*32
        x3 = self.convblock3(x2)  # 32*32*32
        x3 = x2 + x3  # 32*32*32
        x4 = self.pool1(x3)  # 16*16*32
        x41 = self.convblock04(x4)  # 16*16*64
        x5 = self.convblock4(x4)  # 16*16*64
        x5 = x41 + x5  # 16*16*64
        x6 = self.convblock5(x5)  # 16*16*64
        x6 = x5 + x6  # 16*16*64
        x7 = self.convblock6(x6)  # 16*16*64
        # xp = x5 + x6 + x7
        x8 = self.pool2(x7)  # 8*8*64
        x81 = self.convblock07(x8)  # 8*8*90
        x9 = self.convblock7(x8)  # 8*8*90
        x9 = x81 + x9  # 8*8*90
        x10 = self.convblock8(x9)  # 8*8*90
        x10 = x9 + x10  # 8*8*90
        x11 = self.convblock9(x10)  # 8*8*90
        x12 = self.gap(x11)  # 1*1*90
        x12 = x12.view(-1, 90)  # 90
        x13 = self.linear1(x12)  # 10
        return F.log_softmax(x13, dim=-1)