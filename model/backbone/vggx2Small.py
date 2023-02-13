from torch import nn
from model.backbone.vggx2Small_fpn import Vggx2SmallDeconvFPN
import torch

class Vggx2SmallNet(nn.Module):
    def __init__(self,  width_mult=1., out_stages=(2, ), last_channel=1280, activation='ReLU6'):
        super(Vggx2SmallNet, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=2,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )#x4

        self.layer2=nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128,out_channels=128, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )#x2

        self.layer3=nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=128,out_channels=128, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128,out_channels=128, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128,out_channels=128, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )#x2

        self.layer4=nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=128,out_channels=128, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128,out_channels=128, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128,out_channels=128, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )#x2

        self.layer5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=128,out_channels=128, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128,out_channels=128, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )#x2
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,x):
        output = []
        out1 = self.layer1(x)
        output.append(out1)
        out2 = self.layer2(out1)
        output.append(out2)
        out3 = self.layer3(out2)
        output.append(out3)
        out4 = self.layer4(out3)
        output.append(out4)
        out5 = self.layer5(out4)
        output.append(out5)
        
        return tuple(output)

    
class vgg_backfpn_ensemble(nn.Module):
    def __init__(self, ):
        super(vgg_backfpn_ensemble, self).__init__()
        self.backbone = Vggx2SmallNet()
        self.fpn = Vggx2SmallDeconvFPN()
        self.out_channels = 64
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        return x
    
if __name__ == '__main__':
    model= vgg_backfpn_ensemble().cuda()
    x = torch.rand(1, 3, 320, 640).cuda()
    y = model(x)
    print(y.shape)


