import torch 
from torch import nn




"Simple neural network with one convolution and activation"
class CNN(nn.Module):
    def __init__(self, in_channels, base_features : int):
        super(CNN, self).__init__()
        self.in_channels =  in_channels
        self.base_features = base_features
        self.maxpool = nn.MaxPool3d
        self.upconv = nn.ConvTranspose3d
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
        self.final_conv = nn.Sequential(nn.Conv3d(self.base_features, 1, kernel_size=3, stride=1, padding=1),
                          nn.Sigmoid())


        """ self.encoder_blocks = nn.ModuleList(self.build_block('encode'))
        self.decoder_blocks = nn.ModuleList(self.build_block('decode'))
        self.depth = 4  """
        

    def _conv_layer_set(self, feat_in, feat_out):
            conv_layer = nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size = 3, stride=1, padding=1),
            nn.LeakyReLU(),
            )
            return conv_layer

    """def encoder_block(self, feat_in, feat_out, feat_mid = None, maxpool = True):
        if feat_mid is None:
            feat_mid = feat_in
        conv1 = self._conv_layer_set(feat_in, feat_mid)
        conv2 = self._conv_layer_set(feat_mid, feat_out)
        if maxpool:
            block = nn.Sequential(conv1, conv2)
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
        for d in range(len(self.encoder_blocks) - 2):
            out = self.encoder_blocks[d](out)
            if not isinstance(self.encoder_blocks[d], nn.MaxPool3d):
                concat.append(out)
        for i in range(len(concat)):
            print('Concat: ', concat[i].shape)
        out = self.encoder_blocks[-2](out)
        out = self.encoder_blocks[-1](out)
        for u in range(len(self.decoder_blocks)):
            print(concat[-1].shape)
            print(out.shape)
            out = torch.cat((concat.pop(), out), dim = 1)
            print('Concatted')
            out = self.decoder_blocks[u](out)
            print('Decode: ', out.shape)
        out = self.final_conv(out) """

    def forward(self, x):
        out = x.float()
        skips = []
        #Encode Layer 1
        out = self.conv1_layer1(out)
        out = self.conv1_layer2(out)
        print('Layer 1:' , out.shape)
        skips.append(out)
        out = self.maxPool1(out)

        #Encode Layer 2
        out = self.conv2_layer1(out)
        out = self.conv2_layer2(out)
        print('Layer 2:' , out.shape)
        skips.append(out)
        out = self.maxPool2(out)
        

        #Encode Layer 3
        out = self.conv3_layer1(out)
        out = self.conv3_layer2(out)
        print('Layer 3:' , out.shape)
        skips.append(out)
        out = self.maxPool3(out)

        #Bottom Layer
        out = self.conv4_layer1(out)
        out = self.conv4_layer2(out)
        print('Layer 4:' , out.shape)
        out = self.upconv1(out)

        #Decode Layer 1
        out = torch.cat((skips.pop(), out), dim=1)
        out = self.conv5_layer1(out)
        out = self.conv5_layer2(out)
        print('Layer 5:' , out.shape)
        out = self.upconv2(out)
        
        #Decode Layer 2
        out = torch.cat((skips.pop(), out), dim=1)
        out = self.conv6_layer1(out)
        out = self.conv6_layer2(out)
        print('Layer 6:' , out.shape)
        out = self.upconv3(out)

        #Decode Layer 3
        out = torch.cat((skips.pop(), out), dim=1)
        out = self.conv7_layer1(out)
        out = self.conv7_layer2(out)
        print('Layer 7:' , out.shape)

        #Final
        out = self.final_conv(out)
        return out