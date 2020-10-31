import torch
from torch import nn

# defining the architecture
class Unet(nn.Module):
    def __init__(self, mode='baseline'):
        super(Unet, self).__init__()
        self.mode = mode
        
        # convolution layers to be applied the input
        self.conv_down_1_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn_down_1_1 = nn.BatchNorm2d(64)
        self.conv_down_1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_down_1_2 = nn.BatchNorm2d(64)
        nn.init.kaiming_normal_(self.conv_down_1_1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_down_1_2.weight, mode='fan_in', nonlinearity='relu')
        
        # convolution layers to be applied for first downsampled feature maps
        self.conv_down_2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn_down_2_1 = nn.BatchNorm2d(128)
        self.conv_down_2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_down_2_2 = nn.BatchNorm2d(128)
        nn.init.kaiming_normal_(self.conv_down_2_1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_down_2_2.weight, mode='fan_in', nonlinearity='relu')
        
        # convolution layers to be applied for second downsampled feature maps
        self.conv_down_3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn_down_3_1 = nn.BatchNorm2d(256)
        self.conv_down_3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_down_3_2 = nn.BatchNorm2d(256)
        nn.init.kaiming_normal_(self.conv_down_3_1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_down_3_2.weight, mode='fan_in', nonlinearity='relu')
        
        # convolution layers at the bottommost region
        self.conv_down_4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn_down_4_1 = nn.BatchNorm2d(512)
        self.conv_down_4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn_down_4_2 = nn.BatchNorm2d(512)
        self.conv_transpose_4_3 = nn.ConvTranspose2d(512, 256, [2, 3], 2)
        nn.init.kaiming_normal_(self.conv_down_4_1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_down_4_2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_transpose_4_3.weight, mode='fan_in', nonlinearity='relu')
        
        # convolution layers to be applied to upsampled feature maps at 4 level
        self.conv_up_3_1 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn_up_3_1 = nn.BatchNorm2d(256)
        self.conv_up_3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_up_3_2 = nn.BatchNorm2d(256)
        self.conv_transpose_3_2 = nn.ConvTranspose2d(256, 128, 2, 2)
        nn.init.kaiming_normal_(self.conv_up_3_1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_up_3_2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_transpose_3_2.weight, mode='fan_in', nonlinearity='relu')
        
        # convolution layers to be applied to upsampled feature maps at 3 level
        self.conv_up_2_1 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn_up_2_1 = nn.BatchNorm2d(128)
        self.conv_up_2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_up_2_2 = nn.BatchNorm2d(128)
        self.conv_transpose_2_1 = nn.ConvTranspose2d(128, 64, 2, 2)
        nn.init.kaiming_normal_(self.conv_up_2_1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_up_2_2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_transpose_2_1.weight, mode='fan_in', nonlinearity='relu')
        
        # convolution to be applied at the final layer
        self.conv_up_1_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_up_1_1 = nn.BatchNorm2d(64)
        self.conv_up_1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_up_1_2 = nn.BatchNorm2d(64)
        self.conv_up_1_3 = nn.Conv2d(64, 2, 1)
        nn.init.kaiming_normal_(self.conv_up_1_1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_up_1_2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_up_1_3.weight, mode='fan_in', nonlinearity='relu')
        
    # for forward propagation
    def forward(self, batch):
        
        # some modules without parameters
        max_pool = nn.MaxPool2d(2, stride=2)
        log_softmax = nn.LogSoftmax(dim=1)
        
        # down_1 handling
        down_1_1_out = nn.functional.relu(self.bn_down_1_1(self.conv_down_1_1(batch)))
        down_1_2_out = nn.functional.relu(self.bn_down_1_2(self.conv_down_1_2(down_1_1_out)))
        down_2_inp = max_pool(down_1_2_out)
        
        # down 2 handling
        down_2_1_out = nn.functional.relu(self.bn_down_2_1(self.conv_down_2_1(down_2_inp)))
        down_2_2_out = nn.functional.relu(self.bn_down_2_2(self.conv_down_2_2(down_2_1_out)))
        down_3_inp = max_pool(down_2_2_out)
        
        # down 3 handling
        down_3_1_out = nn.functional.relu(self.bn_down_3_1(self.conv_down_3_1(down_3_inp)))
        down_3_2_out = nn.functional.relu(self.bn_down_3_2(self.conv_down_3_2(down_3_1_out)))
        down_4_inp = max_pool(down_3_2_out)
        
        # down 4 handling
        down_4_1_out = nn.functional.relu(self.bn_down_4_1(self.conv_down_4_1(down_4_inp)))
        down_4_2_out = nn.functional.relu(self.bn_down_4_2(self.conv_down_4_2(down_4_1_out)))
        up_3_inp = self.conv_transpose_4_3(down_4_2_out)
        
        # up 3 handling
        up_3_inp = torch.cat([down_3_2_out, up_3_inp], 1)
        up_3_1_out = nn.functional.relu(self.bn_up_3_1(self.conv_up_3_1(up_3_inp)))
        up_3_2_out = nn.functional.relu(self.bn_up_3_2(self.conv_up_3_2(up_3_1_out)))
        up_2_inp = self.conv_transpose_3_2(up_3_2_out)
        
        # up 2 handling
        up_2_inp = torch.cat([down_2_2_out, up_2_inp], 1)
        up_2_1_out = nn.functional.relu(self.bn_up_2_1(self.conv_up_2_1(up_2_inp)))
        up_2_2_out = nn.functional.relu(self.bn_up_2_2(self.conv_up_2_2(up_2_1_out)))
        up_1_inp = self.conv_transpose_2_1(up_2_2_out)
        
        # final layer handling
        up_1_inp = torch.cat([down_1_2_out, up_1_inp], 1)
        up_1_1_out = nn.functional.relu(self.bn_up_1_1(self.conv_up_1_1(up_1_inp)))
        up_1_2_out = nn.functional.relu(self.bn_up_1_2(self.conv_up_1_2(up_1_1_out)))
        out = log_softmax(self.conv_up_1_3(up_1_2_out))
        
        return out