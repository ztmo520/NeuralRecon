import torch
import torch.nn as nn

from .backbone import MnasMulti
from .neucon_network import NeuConNet
from .gru_fusion import GRUFusion
from utils import tocuda


class NeuralRecon(nn.Module):
    '''
    NeuralRecon main class.
    '''

    def __init__(self, cfg):
        super(NeuralRecon, self).__init__()
        self.cfg = cfg.MODEL
        # 根据训练配置文件，这里alpha = 1.0
        alpha = float(self.cfg.BACKBONE2D.ARC.split('-')[-1])
        # other hparams
        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1) # MODEL.PIXEL_MEAN = [103.53, 116.28, 123.675]
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1) # MODEL.PIXEL_STD = [1., 1., 1.]
        self.n_scales = len(self.cfg.THRESHOLDS) - 1

        # networks
        self.backbone2d = MnasMulti(alpha)
        self.neucon_net = NeuConNet(cfg.MODEL)
        # for fusing to global volume
        self.fuse_to_global = GRUFusion(cfg.MODEL, direct_substitute=True)

    def normalizer(self, x):
        """ Normalizes the RGB images to the input range"""
        return (x - self.pixel_mean.type_as(x)) / self.pixel_std.type_as(x)

    def forward(self, inputs, save_mesh=False):
        '''

        :param inputs: dict: {
            'imgs':                    (Tensor), images,
                                    (batch size, number of views, C, H, W)
            'vol_origin':              (Tensor), origin of the full voxel volume (xyz position of voxel (0, 0, 0)),
                                    (batch size, 3)
            'vol_origin_partial':      (Tensor), origin of the partial voxel volume (xyz position of voxel (0, 0, 0)),
                                    (batch size, 3)
            'world_to_aligned_camera': (Tensor), matrices: transform from world coords to aligned camera coords,
                                    (batch size, number of views, 4, 4)
            'proj_matrices':           (Tensor), projection matrix,
                                    (batch size, number of views, number of scales, 4, 4)
            when we have ground truth:
            'tsdf_list':               (List), tsdf ground truth for each level,
                                    [(batch size, DIM_X, DIM_Y, DIM_Z)]
            'occ_list':                (List), occupancy ground truth for each level,
                                    [(batch size, DIM_X, DIM_Y, DIM_Z)]
            others: unused in network
        }
        :param save_mesh: a bool to indicate whether or not to save the reconstructed mesh of current sample
        :return: outputs: dict: {
            'coords':                  (Tensor), coordinates of voxels,
                                    (number of voxels, 4) (4 : batch ind, x, y, z)
            'tsdf':                    (Tensor), TSDF of voxels,
                                    (number of voxels, 1)
            When it comes to save results:
            'origin':                  (List), origin of the predicted partial volume,
                                    [3]
            'scene_tsdf':              (List), predicted tsdf volume,
                                    [(nx, ny, nz)]
        }
                 loss_dict: dict: {
            'tsdf_occ_loss_X':         (Tensor), multi level loss
            'total_loss':              (Tensor), total loss
        }
        '''
        # 如果是tensor就放到cuda，否则原样返回
        inputs = tocuda(inputs)
        outputs = {}
        # 图像序列tuple,每个里面是9张图像
        # imgs是一个tuple, 每个图像的size: torch.Size([1, 3, 480, 640])
        imgs = torch.unbind(inputs['imgs'], 1)

        '''
        第一步，提取特征
            1. 对每张图像进行了归一化
            2. 送入特征提取网络
        '''
        # image feature extraction
        # in: images; out: feature maps
        # features也是一个tuple,9张图像的特征tuple,每个tuple是3个tensor，对应3个scale
        # 0: [1, 24, 120, 160]
        # 1: [1, 40, 60, 80]
        # 2: [1, 80, 30 ,40]
        features = [self.backbone2d(self.normalizer(img)) for img in imgs]

        '''
        第二步，稀疏卷积、GRU Fusion
        '''
        # coarse-to-fine decoder: SparseConv and GRU Fusion.
        # in: image feature; out: sparse coords and tsdf
        # 最终输出的是每个点的坐标以及每个点的tsdf值
        #           coords： torch.Size([点的数量, 4])
        #           tsdf:    torch.Size([点的数量, 1])
        outputs, loss_dict = self.neucon_net(features, inputs, outputs)

        # fuse to global volume.
        if not self.training and 'coords' in outputs.keys():
            outputs = self.fuse_to_global(outputs['coords'], outputs['tsdf'], inputs, self.n_scales, outputs, save_mesh)

        # gather loss.
        print_loss = 'Loss: '
        for k, v in loss_dict.items():
            print_loss += f'{k}: {v} '

        weighted_loss = 0

        for i, (k, v) in enumerate(loss_dict.items()):
            weighted_loss += v * self.cfg.LW[i]

        loss_dict.update({'total_loss': weighted_loss})
        return outputs, loss_dict
