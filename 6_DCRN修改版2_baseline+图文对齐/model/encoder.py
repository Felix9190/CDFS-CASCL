import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralEncoder(nn.Module):
    def __init__(self, input_channels, patch_size, feature_dim):
        super(SpectralEncoder, self).__init__()
        self.input_channels = input_channels # 100
        self.patch_size = patch_size # 9
        self.feature_dim = feature_dim # 128
        self.inter_size = 24

        self.conv1 = nn.Conv3d(1, self.inter_size, kernel_size=(7, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), bias=True)  # stride = 2 -> 1 too slow ; add zero-padding
        self.bn1 = nn.BatchNorm3d(self.inter_size)
        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0), padding_mode='zeros', bias=True) # replicate useful???
        self.bn2 = nn.BatchNorm3d(self.inter_size)
        self.activation2 = nn.ReLU()

        self.conv3 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0), padding_mode='zeros', bias=True)
        self.bn3 = nn.BatchNorm3d(self.inter_size)
        self.activation3 = nn.ReLU()

        self.conv4 = nn.Conv3d(self.inter_size, self.feature_dim, kernel_size=(((self.input_channels - 7 + 2 * 1) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(self.feature_dim)
        self.activation4 = nn.ReLU()

        self.avgpool = nn.AvgPool3d((1, self.patch_size, self.patch_size))

        # self.bn_last = nn.BatchNorm1d(self.feature_dim)

    def forward(self, x): # (batchsize, 100, 9, 9)
        x = x.unsqueeze(1)  # (batchsize, 100, 9, 9) -> (batchsize, 1, 100, 9, 9)

        # Convolution layer 1
        x1 = self.conv1(x) # (batchsize, 16, 94, 9, 9)
        x1 = self.activation1(self.bn1(x1))

        # Residual layer 1
        residual = x1
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))
        x1 = self.conv3(x1)
        x1 = residual + x1
        x1 = self.activation3(self.bn3(x1))

        # Convolution layer to combine rest
        x1 = self.conv4(x1) # (batchsize, 128, 1, 9, 9)
        x1 = self.activation4(self.bn4(x1))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4)) # (batchsize, 128, 9, 9)

        x1 = self.avgpool(x1) # (batchsize, 128, 1, 1)
        x1 = x1.reshape((x1.size(0), -1)) # (batchsize, 128)

        # add bn_last
        # x1 = self.bn_last(x1)

        return x1


class SpatialEncoder(nn.Module):
    def __init__(self, input_channels, patch_size, feature_dim):
        super(SpatialEncoder, self).__init__()
        self.input_channels = input_channels  # 100
        self.patch_size = patch_size  # 9
        self.feature_dim = feature_dim # 128
        self.inter_size = 24

        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, self.inter_size, kernel_size=(self.input_channels, 1, 1))
        self.bn5 = nn.BatchNorm3d(self.inter_size)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv8 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 1, 1))

        self.conv6 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), padding_mode='zeros', bias=True)
        self.bn6 = nn.BatchNorm3d(self.inter_size)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), padding_mode='zeros', bias=True)
        self.bn7 = nn.BatchNorm3d(self.inter_size)
        self.activation7 = nn.ReLU()

        self.avgpool = nn.AvgPool3d((1, self.patch_size, self.patch_size))

        # Fully connected Layer   not add dropout
        # self.fc = nn.Linear(in_features=self.inter_size, out_features=self.feature_dim)

        self.fc = nn.Sequential(nn.Dropout(p=0.5),
                                nn.Linear(self.inter_size, out_features=self.feature_dim))

        # self.bn_last = nn.BatchNorm1d(self.feature_dim)


    def forward(self, x): # (batchsize, 100, 9, 9)
        x = x.unsqueeze(1)  # (batchsize, 100, 9, 9) -> (batchsize, 1, 100, 9, 9)

        x2 = self.conv5(x) # (batchsize, 16, 1, 9, 9)
        x2 = self.activation5(self.bn5(x2))

        # Residual layer 2
        residual = x2
        residual = self.conv8(residual) # (batchsize, 16, 1, 9, 9) why? this is not residual !
        x2 = self.conv6(x2)
        x2 = self.activation6(self.bn6(x2))
        x2 = self.conv7(x2)
        x2 = residual + x2 # (batchsize, 16, 1, 9, 9)

        x2 = self.activation7(self.bn7(x2))
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4)) # (batchsize, 16, 9, 9)

        x2 = self.avgpool(x2) # (batchsize, 16, 1, 1)
        x2 = x2.reshape((x2.size(0), -1)) # (batchsize, 16)

        x2 = self.fc(x2) # (batchsize, 128)

        # add bn_last
        # x2 = self.bn_last(x2)

        return x2


class WordEmbTransformers(nn.Module):
    def __init__(self, feature_dim, dropout):
        super(WordEmbTransformers, self).__init__()
        self.feature_dim = feature_dim
        self.dropout = dropout
        # not add BN
        self.fc = nn.Sequential(nn.Linear(in_features=768,
                                           out_features=128,
                                           bias=True),
                                # nn.BatchNorm1d(128),
                                nn.ReLU(),
                                nn.Dropout(p=self.dropout), # too important, without it, OA down to 65
                                nn.Linear(in_features=128,
                                          out_features=self.feature_dim,
                                          bias=True)
                                # nn.BatchNorm1d(self.feature_dim),
                                )

    def forward(self, x):
        # 0-1
        x = self.fc(x)
        return x


class AttentionWeight(nn.Module):
    def __init__(self, feature_dim, hidden_layer, dropout):
        super(AttentionWeight, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_layer = hidden_layer
        self.dropout = dropout

        self.getAttentionWeight = nn.Sequential(nn.Linear(in_features=self.feature_dim,
                                                          out_features=self.hidden_layer),
                                                nn.ReLU(),
                                                nn.Dropout(p=self.dropout), # 0.3
                                                nn.Linear(in_features=self.hidden_layer,
                                                          out_features=1),
                                                nn.Sigmoid()
                                                )

    def forward(self,x): # (batchsize, 128)
        x = self.getAttentionWeight(x) # (batchsize, 1)
        return x


class Encoder(nn.Module):
    def __init__(self, n_dimension, patch_size, emb_size, dropout = 0.5):
        super(Encoder, self).__init__()
        self.n_dimension = n_dimension
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.dropout = dropout

        self.spectral_encoder = SpectralEncoder(input_channels=self.n_dimension, patch_size=self.patch_size, feature_dim=self.emb_size)
        self.spatial_encoder = SpatialEncoder(input_channels=self.n_dimension, patch_size=self.patch_size, feature_dim=self.emb_size)
        self.word_emb_transformers = WordEmbTransformers(feature_dim=self.emb_size, dropout=self.dropout)
        # self.intra_model_attention_weight = AttentionWeight(feature_dim=self.emb_size, hidden_layer=64, dropout=0.5)
        # self.inter_model_attention_weight = AttentionWeight(feature_dim=self.emb_size, hidden_layer=64, dropout=0.5)

        # self.bn_last = nn.BatchNorm1d(self.emb_size)

    def forward(self, x, semantic_feature = "", s_or_q = "query"): # UP (9, 100, 9, 9)
        spatial_feature = self.spatial_encoder(x) # (9, 128)
        spectral_feature = self.spectral_encoder(x) # (9, 128)
        spatial_spectral_fusion_feature = 0.5 * spatial_feature + 0.5 * spectral_feature  # (9, 128)

        # support set extract fusion_feature
        if s_or_q == "support": # semantic_feature = (9, 768)
            semantic_feature = self.word_emb_transformers(semantic_feature) # (9, 128)
            return spatial_spectral_fusion_feature, semantic_feature
        # query set extract spatial_spectral_fusion_feature
        return spatial_spectral_fusion_feature





# def conv3x3x3(in_channel, out_channel):
#     layer = nn.Sequential(
#         nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3, stride=1,padding=1,bias=False),
#         nn.BatchNorm3d(out_channel),
#     )
#     return layer
#
# class residual_block(nn.Module):
#     def __init__(self, in_channel,out_channel):
#         super(residual_block, self).__init__()
#         self.conv1 = conv3x3x3(in_channel,out_channel)
#         self.conv2 = conv3x3x3(out_channel,out_channel)
#         self.conv3 = conv3x3x3(out_channel,out_channel)
#
#     def forward(self, x): # (1,1,100,9,9)
#         x1 = F.relu(self.conv1(x), inplace=True) # 对于block1 (1,8,100,9,9)
#         x2 = F.relu(self.conv2(x1), inplace=True) # (1,8,100,9,9)
#         x3 = self.conv3(x2) # (1,8,100,9,9)
#         out = F.relu(x1 + x3, inplace=True) # (1,8,100,9,9)
#         return out

# class D_Res_3d_CNN(nn.Module):
#     def __init__(self, in_channel, out_channel1, out_channel2, patch_size, emb_size):
#         super(D_Res_3d_CNN, self).__init__()
#         self.in_channel = in_channel
#         self.emb_size = emb_size # 没用上
#         self.patch_size = patch_size # 没用上
#         self.block1 = residual_block(in_channel, out_channel1)
#         self.maxpool1 = nn.MaxPool3d(kernel_size=(4, 2, 2), padding=(0, 1, 1), stride=(4, 2, 2))
#         self.block2 = residual_block(out_channel1, out_channel2)
#         self.maxpool2 = nn.MaxPool3d(kernel_size=(4, 2, 2), padding=(2, 1, 1), stride=(4, 2, 2))
#         self.conv = nn.Conv3d(in_channels=out_channel2, out_channels=32, kernel_size=3, bias=False)
#         # self.fc = nn.Sequential(nn.Dropout(p=0.5),
#         #                         nn.Linear(in_features=160,
#         #                                   out_features=128,
#         #                                   bias=True),
#         #                         nn.BatchNorm1d(128))
#
#         # self.fc = nn.Linear(in_features=160, out_features=128) # bias=True
#
#     def forward(self, x): # 对于UP作为目标域 (180,100,9,9)
#         x = x.unsqueeze(1) # (180,1,100,9,9)
#         x = self.block1(x) # (180,8,100,9,9)
#         x = self.maxpool1(x) # (180,8,25,5,5)
#         x = self.block2(x) # (180,16,25,5,5)
#         x = self.maxpool2(x) # (180,16,7,3,3)
#         x = self.conv(x) # (180,32,5,1,1)
#         x = x.view(x.shape[0], -1) # (180,160)
#         # x = self.fc(x) # (180, 128)
#         # 可以加上160->128的映射 或者 加上160->nums_class线性分类器
#         return x