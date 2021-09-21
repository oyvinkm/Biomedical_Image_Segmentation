from torch import nn
from torch.nn import parameter

"Simple neural network with one convolution and activation"
class CNN(nn.Module):
    def __init__(self, in_channels, parameters):
        super(CNN, self).__init__()
        self.in_channels =  in_channels
        self.encoder_blocks = []
        self.decoder_blocks = []


    def _conv_layer_set(self, feat_in, feat_out, norm, active):
            conv_layer = nn.Sequential(
            nn.Conv3d(feat_in, feat_out, 1),
            norm(feat_out),
            active,
            )
            return conv_layer

    def encoder_block(self, feat_in, feat_out, norm, active, feat_mid = None):
        if feat_mid is None:
            feat_mid = feat_in
        conv1 = self._conv_layer_set(feat_in, feat_mid, norm, active)
        conv2 = self._conv_layer_set(feat_mid, feat_out, norm, active)
        block = nn.Sequential(conv1, conv2, nn.MaxPool3d((2,2,2)))
        return block

    def decoder_block(self, feat_in, feat_out, norm, active):
        conv1 = self._conv_layer_set(feat_in, feat_in, norm, active)
        conv2 = self._conv_layer_set(feat_in, feat_out, norm, active)
        block = nn.Sequential(conv1, conv2, nn.ConvTranspose3d((2,2,2)))
        return block

    def build_block(self, feat_in, feat_out):



        

        


    
    
    
    def forward(self, x):
        out = 0
        return out