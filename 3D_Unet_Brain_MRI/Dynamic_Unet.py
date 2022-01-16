import torch 
from torch import nn
from torch.nn.modules.activation import LeakyReLU


class ConvDropoutNormNonlin(nn.Module):
    def __init__(   self, in_channels, out_channels, 
                    conv = nn.Conv3d, conv_kwargs = None,
                    dropout = nn.Dropout3d, dropout_kwargs = None,
                    norm = nn.BatchNorm3d, norm_kwargs = None,
                    nonlin = nn.LeakyReLU, nonlin_kwargs = None):
        super(ConvDropoutNormNonlin, self).__init__()
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size' : 3, 'stride' : 1, 'padding' : 1, 'bias' : True}
        if dropout_kwargs is None:
            dropout_kwargs = {'p': 0.5, 'inplace': True}
        if norm_kwargs is None:
            norm_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if nonlin_kwargs is None and isinstance(nonlin, nn.LeakyReLU):
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.conv_kwargs = conv_kwargs
        self.dropout_kwargs = dropout_kwargs
        self.dropout_op = dropout
        self.norm_kwargs = norm_kwargs
        self.nonlin_kwargs = nonlin_kwargs
        self.norm_op = norm
        self.conv_op = conv
        self.nonlin_op = nonlin

        self.conv = self.conv_op(in_channels, out_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_kwargs['p'] is not None and self.dropout_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_kwargs)
        else:
            self.dropout = None
        if self.norm_op is not None:    
            self.instnorm = self.norm_op(out_channels,**self.norm_kwargs)
        if self.nonlin_kwargs is not None and self.instnorm is not None:
            self.nonlin= self.nonlin_op(**self.nonlin_kwargs)
        else: 
            self.nonlin = self.nonlin_op()
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.instnorm is not None:
            x = self.instnorm(x)
        return self.nonlin(x)


class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.instnorm is not None:
            return self.instnorm(self.nonlin(x)) 
        else:
            return self.nonlin(x)

class StackedLayers(nn.Module):
    def __init__(self, in_feat_channels, out_feat_channels, num_cons,
                conv = nn.Conv3d, conv_kwargs = None,
                norm = nn.BatchNorm3d, norm_kwargs = None,
                dropout = nn.Dropout3d, dropout_kwargs = None,
                nonlin = nn.LeakyReLU, nonlin_kwargs = None, basic_block = ConvDropoutNormNonlin):
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size' : 3, 'stride' : 1, 'padding' : 1, 'bias' : True}
        if dropout_kwargs is None:
            dropout_kwargs = {'p': 0.5, 'inplace': True}
        if norm_kwargs is None:
            norm_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.in_channels = in_feat_channels
        self.out_channels = out_feat_channels
        self.conv_kwargs = conv_kwargs
        self.dropout_kwargs = dropout_kwargs
        self.dropout = dropout
        self.norm_kwargs = norm_kwargs
        self.nonlin_kwargs = nonlin_kwargs
        self.norm = norm
        self.nonlin = nonlin
        self.conv = conv

        super(StackedLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(in_feat_channels, out_feat_channels, self.conv, self.conv_kwargs,
            self.dropout, self.dropout_kwargs, self.norm, self.norm_kwargs, 
            self.nonlin, self.nonlin_kwargs)]
            + [basic_block(out_feat_channels, out_feat_channels, self.conv, 
            self.conv_kwargs, self.dropout, self.dropout_kwargs, self.norm, 
            self.norm_kwargs, self.nonlin, 
            self.nonlin_kwargs) for _ in range(num_cons - 1)]))
    def forward(self, x):
        return self.blocks(x)


class Dynamic_3DUnet(nn.Module):
    def __init__(self,   base_features, in_channels = 3, num_classes = 1,  num_conv_layer = 2, depth = 4,
                conv_op = nn.Conv3d, conv_kwargs = None, norm_op = nn.BatchNorm3d, norm_kwargs = None,
                dropout_op = nn.Dropout3d, dropout_kwargs = None, nonlin_op = nn.LeakyReLU, nonlin_kwargs = None,
                maxpool_op = nn.MaxPool3d, maxpool_kwargs = None, upconv_op = nn.ConvTranspose3d, basic_block = ConvDropoutNormNonlin, 
                final_nonlin = nn.Sigmoid):
        if maxpool_kwargs is None:
            maxpool_kwargs = (2,2,2)

        super(Dynamic_3DUnet, self).__init__()
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size' : 3, 'stride' : 1, 'padding' : 1, 'bias' : True}
        self.conv_kwargs = conv_kwargs
        self.dropout_kwargs = dropout_kwargs
        self.norm_kwargs = norm_kwargs
        self.nonlin_kwargs = nonlin_kwargs
        self.maxpool_kwargs = maxpool_kwargs

        self.final_nonlin_op = final_nonlin()
        self.basic_block_op = basic_block
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.conv_op = conv_op
        self.nonlin_op = nonlin_op
        self.num_conv_layer = num_conv_layer
        self.depth = depth
        self.base_features = base_features
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.decoder_block = []
        self.encoder_block = []
        self.num_maxpools = 0
        self.upconv = []
        self.maxpool_op = maxpool_op(self.maxpool_kwargs)
        self.upconv_op = upconv_op
        in_features = self.in_channels
        out_features = self.base_features
        
        self.first_layer = StackedLayers(in_features, out_features, 1, self.conv_op, 
                                                    self.conv_kwargs, self.norm_op, self.norm_kwargs, 
                                                    self.dropout_op, self.dropout_kwargs, self.nonlin_op, 
                                                    basic_block= basic_block)
        in_features = self.base_features
        out_features = in_features * 2
        for d in range(self.depth):
            self.decoder_block.append(StackedLayers(in_features, out_features, 2, self.conv_op, 
                                                    self.conv_kwargs, self.norm_op, self.norm_kwargs, 
                                                    self.dropout_op, self.dropout_kwargs, self.nonlin_op, 
                                                    basic_block= basic_block))                                             
            in_features = self.decoder_block[d].out_channels 
            if d != self.depth - 1:
                self.num_maxpools += 1
            out_features = in_features * 2


        for e in reversed(range(len(self.decoder_block)-1)):
            if not isinstance(self.decoder_block[e], nn.MaxPool3d):
                up_features = self.decoder_block[e + 1].out_channels
                out_features = self.decoder_block[e].out_channels
                in_channels = self.decoder_block[e].out_channels
                skip_feat = out_features + up_features
                self.upconv.append(self.upconv_op(up_features, up_features, (2,2,2), 2))
                if e != 0:
                    self.encoder_block.append(StackedLayers(skip_feat, in_channels, 2, self.conv_op, 
                                                        self.conv_kwargs, self.norm_op, self.norm_kwargs, 
                                                        self.dropout_op, self.dropout_kwargs, self.nonlin_op, 
                                                        basic_block= basic_block))
                if e == 0:
                    self.encoder_block.append(StackedLayers(skip_feat, out_features//2, 2, self.conv_op, 
                                                        self.conv_kwargs, self.norm_op, self.norm_kwargs, 
                                                        self.dropout_op, self.dropout_kwargs, self.nonlin_op, 
                                                        basic_block= basic_block))
                    self.conv_kwargs['kernel_size'] = 1
                    self.conv_kwargs['padding'] = 0
                    self.conv = self.conv_op(out_features//2, 1, **self.conv_kwargs)
                    self.encoder_block.append(nn.Sequential(self.conv, self.final_nonlin_op))
        self.decode_path = nn.ModuleList(self.decoder_block)
        self.encode_path = nn.ModuleList(self.encoder_block)
        self.upconv_path = nn.ModuleList(self.upconv)

    def forward(self, x):
        skips = []
        x = self.first_layer(x)
        for d in range(len(self.decode_path) - 1):
            print('Before path: ', x.shape)
            x = self.decode_path[d](x)
            print('After path: ',x.shape)
            skips.append(x)
            x = self.maxpool_op(x)
            print('After max_pool: ', x.shape)
        x = self.decode_path[-1](x)
        for e in range(len(self.encode_path)):
            if e != len(self.encode_path) - 1:
                print('Before upconv: ', x.shape)
                x = self.upconv[e](x)
                x = torch.cat((skips.pop(), x), dim=1)
                print('After concat: ', x.shape)
            x = self.encode_path[e](x)
            print('After encode path: ', x.shape)
        return x

        

