import torch
from torch import nn
from torch.nn import parameter


"Simple neural network with one convolution and activation"
class CNN(nn.Module):
    def __init__(self, in_channels, parameters):
        super(CNN, self).__init__()
        self.in_channels =  in_channels
        #self.activation = nn.LeakyReLU()
        #self.norm = nn.InstanceNorm3d()
        self.encoder_blocks = nn.ModuleList(self.build_block('encode'))
        self.decoder_blocks = nn.ModuleList(self.build_block('decode'))
        self.depth = 4

        self.final_conv = self._conv_layer_set(64, 1)

    def _conv_layer_set(self, feat_in, feat_out):
            conv_layer = nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size = 3, stride=1, padding=1),
            nn.LeakyReLU(),
            )
            return conv_layer

    def encoder_block(self, feat_in, feat_out, feat_mid = None, maxpool = True):
        if feat_mid is None:
            feat_mid = feat_in
        conv1 = self._conv_layer_set(feat_in, feat_mid)
        conv2 = self._conv_layer_set(feat_mid, feat_out)
        if maxpool:
            block = nn.Sequential(nn.MaxPool3d(2,2), conv1, conv2)
        else:
            block = nn.Sequential(conv1,conv2, nn.ConvTranspose3d(feat_out, feat_out, 2, 2))
        return block

    def decoder_block(self, feat_in, feat_out, feat_uncat = None):
        if feat_uncat is None:
            conv1 = self._conv_layer_set(feat_in, feat_out)
        else:
            conv1 = self._conv_layer_set(feat_in, feat_uncat)
        conv2 = self._conv_layer_set(feat_out, feat_out)
        block = nn.Sequential(conv1, conv2, nn.ConvTranspose3d(feat_out, feat_out, 2, 2))
        return block

    def build_block(self, path_type):
        block_list = []
        if (path_type == 'encode'):
            print('encoding')
            block_list.append(self.encoder_block(3, 64, 32))
            block_list.append(nn.MaxPool3d(2,2))
            block_list.append(self.encoder_block(64, 128))
            block_list.append(nn.MaxPool3d(2,2))
            block_list.append(self.encoder_block(128,256))
            block_list.append(nn.MaxPool3d(2,2))
            block_list.append(self.encoder_block(256, 512))
            block_list.append(nn.ConvTranspose3d(512, 512, 2, 2))
        elif (path_type == 'decode'):
            print('decoding')
            block_list.append(self.decoder_block(768, 256))
            block_list.append(self.decoder_block(384,128))
            block_list.append(self.decoder_block(192, 64))
        else: 
            print('Given wrong type')
        return block_list

    def _get_length_(self):
        print('Encoder Length: ', len(self.encoder_blocks))
        print('Decoder Length: ', len(self.decoder_blocks))
            
    def forward(self, x):
        concat = []
        out = x.float()
        for d in range(len(self.encoder_blocks) - 1):
            out = self.encoder_blocks[d](out)
            if not isinstance(self.encoder_blocks[d], nn.MaxPool3d):
                concat.append(out)
        for i in range(len(concat)):
            print('Concat: ', concat[i].shape)
        out = self.encoder_blocks[-1](out)
        for u in range(len(self.decoder_blocks)):
            print(concat.pop().shape)
            print(out.shape)
            out = torch.cat([concat.pop(), out], dim=1)
            out = self.decoder_blocks[u](out)
            print('Decode: ', out.shape)
        out = self.final_conv(out)
        return out