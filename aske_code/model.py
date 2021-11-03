import torch 
from torch import nn

"Simple neural network with one convolution and activation"
class CNN(nn.Module):
    def __init__(self, in_channels, parameters = None, base_features = 32):
        super(CNN, self).__init__()
        self.in_channels =  in_channels
        self.base_features = base_features
        self.maxpool = nn.MaxPool3d
        self.askpool = nn.MaxPool3d((2,2,2))
        self.upconv = nn.ConvTranspose3d
        self.askconv = nn.ConvTranspose3d

        self.test_layer = self._conv_layer_set(self.in_channels,1)
        #Aske test network
        #Layer encode 
        self.aske1_layer1 = self._conv_layer_set(self.in_channels, 16)
        self.aske1_layer2 = self._conv_layer_set(16, self.base_features)
        
        #bottom layer
        self.aske2_layer1 = self._conv_layer_set(self.base_features, self.base_features)
        self.aske2_layer2 = self._conv_layer_set(self.base_features, self.base_features*2)
        self.aske_upconv1 = self.upconv(self.base_features*2, self.base_features*2, 2, 2)

        #decode layer
        self.aske3_layer1 = self._conv_layer_set(self.base_features*2 + self.base_features, self.base_features)
        self.aske3_layer2 = self._conv_layer_set(self.base_features, self.base_features)
        self.aske3_layer3 = self._conv_layer_set(self.base_features, 1)

        


        #Layer Encode 1
        self.conv1_layer1 = self._conv_layer_set(self.in_channels, self.base_features)
        self.conv1_layer2 = self._conv_layer_set(self.base_features, self.base_features)
        self.maxPool1 = self.maxpool((2,2,2))

        #Layer Encode 2
        self.conv2_layer1 = self._conv_layer_set(self.base_features, self.base_features*2)
        self.conv2_layer2 = self._conv_layer_set(self.base_features*2, self.base_features*4)
        self.maxPool2 = self.maxpool((2,2,2))

        #Layer Encode 3
        self.conv3_layer1 = self._conv_layer_set(self.base_features*4, self.base_features*4)
        self.conv3_layer2 = self._conv_layer_set(self.base_features*4, self.base_features*8)
        self.maxPool3 = self.maxpool((2,2,2))

        #Bottom Layer
        self.conv4_layer1 = self._conv_layer_set(self.base_features*8, self.base_features*8)
        self.conv4_layer2 = self._conv_layer_set(self.base_features*8, self.base_features*16)
        self.upconv1 = self.upconv(self.base_features*16, self.base_features*16, 2, 2)

        #Decode Layer 1
        self.conv5_layer1 = self._conv_layer_set((self.base_features*8 + self.base_features*16), self.base_features*8)
        self.conv5_layer2 = self._conv_layer_set(self.base_features*8, self.base_features*8)
        self.upconv2 = self.upconv(self.base_features*8, self.base_features*8, 2, 2)

        #Decode Layer 2
        self.conv6_layer1 = self._conv_layer_set((self.base_features*4 + self.base_features*8), self.base_features*4)
        self.conv6_layer2 = self._conv_layer_set(self.base_features*4, self.base_features*4)
        self.upconv3 = self.upconv(self.base_features*4, self.base_features*4, 2, 2)

        #Decode Layer 3
        self.conv7_layer1 = self._conv_layer_set((self.base_features + self.base_features*4), self.base_features)
        self.conv7_layer2 = self._conv_layer_set(self.base_features, self.base_features)

        #Output
        self.final_conv = self._conv_layer_set(self.base_features, 1)

    def _conv_layer_set(self, feat_in, feat_out):
            conv_layer = nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm3d(feat_out),
            nn.LeakyReLU(),
            )
            return conv_layer



#Original forward
    def forward(self, x):
        out = x.float()
        skips = []
        
        out = self.conv1_layer1(out)
        out = self.conv1_layer2(out)
        #print('Layer 1:' , out.shape)
        skips.append(out)
        out = self.maxPool1(out)
        #print(out.shape)

        #Encode Layer 2
        out = self.conv2_layer1(out)
        out = self.conv2_layer2(out)
        #print('Layer 2:' , out.shape)
        skips.append(out)
        out = self.maxPool2(out)
        

        #Encode Layer 3
        out = self.conv3_layer1(out)
        out = self.conv3_layer2(out)
        #print('Layer 3:' , out.shape)
        skips.append(out)
        out = self.maxPool3(out)

        #Bottom Layer
        out = self.conv4_layer1(out)
        out = self.conv4_layer2(out)
        #print('Layer 4:' , out.shape)
        out = self.upconv1(out)

        #Decode Layer 1
        out = torch.cat((skips.pop(), out), dim=1)
        out = self.conv5_layer1(out)
        out = self.conv5_layer2(out)
        #print('Layer 5:' , out.shape)
        out = self.upconv2(out)
        
        #Decode Layer 2
        out = torch.cat((skips.pop(), out), dim=1)
        out = self.conv6_layer1(out)
        out = self.conv6_layer2(out)
        #print('Layer 6:' , out.shape)
        out = self.upconv3(out)

        #Decode Layer 3
        out = torch.cat((skips.pop(), out), dim=1)
        out = self.conv7_layer1(out)
        out = self.conv7_layer2(out)
        #print('Layer 7:' , out.shape)

        #Final
        out = self.final_conv(out)
        return out
