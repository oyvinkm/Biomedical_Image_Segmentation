from torch import nn

"Simple neural network with one convolution and activation"
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = self._conv_layer_set(3, 16)
        self.conv2 = self._conv_layer_set(16,32)
        self.finconv = self._conv_layer_set(64,1)
        
    def _conv_layer_set(self, in_c, out_c):
            conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, 1),
            nn.LeakyReLU(),
            )
            return conv_layer

    def forward(self, x):
        x = x.float()
        print('1st convolution')
        out = self.conv1(x)
        print(out.shape)
        out = self.conv2(out)
        #out = self.finconv(out)
        return out