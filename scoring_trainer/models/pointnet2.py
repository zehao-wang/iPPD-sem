import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.set_abstraction import PointNet_SA_Module, PointNet_SA_Module_MSG

class pointnet2_cls_ssg(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(pointnet2_cls_ssg, self).__init__()
        self.pt_sa1 = PointNet_SA_Module(M=512, radius=0.2, K=32, in_channels=in_channels, mlp=[64, 64, 128], group_all=False)
        self.pt_sa2 = PointNet_SA_Module(M=128, radius=0.4, K=64, in_channels=131, mlp=[128, 128, 256], group_all=False)
        self.pt_sa3 = PointNet_SA_Module(M=None, radius=None, K=None, in_channels=259, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, hidden_size, bias=False)

    def forward(self, xyz, points):
        batchsize = xyz.shape[0]
        new_xyz, new_points = self.pt_sa1(xyz, points)
        new_xyz, new_points = self.pt_sa2(new_xyz, new_points)
        new_xyz, new_points = self.pt_sa3(new_xyz, new_points)
        net = new_points.view(batchsize, -1)
        out = self.fc1(net)
        return out

class pointnet2_feat_msg(nn.Module):
    def __init__(self, feat_dim, hidden_size):
        super().__init__()
        self.pt_sa1 = PointNet_SA_Module_MSG(M=512,
                                             radiuses=[0.1, 0.2, 0.4],
                                            #  Ks=[16, 32, 128],
                                             Ks=[2, 4, 8], # TODO: change based on average number of surrounding objects
                                             in_channels=feat_dim,
                                             mlps=[[32, 32, 64],
                                                   [64, 64, 128],
                                                   [64, 96, 128]])
        self.pt_sa2 = PointNet_SA_Module_MSG(M=128,
                                             radiuses=[0.2, 0.4, 0.8],
                                            #  Ks=[32, 64, 128],
                                             Ks=[4, 8, 16],
                                             in_channels=323,
                                             mlps=[[64, 64, 128],
                                                   [128, 128, 256],
                                                   [128, 128, 256]])
        self.pt_sa3 = PointNet_SA_Module(M=None, radius=None, K=None, in_channels=643, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, hidden_size, bias=False)

    def forward(self, x):
        xyz, points = torch.split(x, [3, x.shape[2]-3], dim=2)
        batchsize = xyz.shape[0]
        new_xyz, new_points = self.pt_sa1(xyz, points)
        new_xyz, new_points = self.pt_sa2(new_xyz, new_points)
        new_xyz, new_points = self.pt_sa3(new_xyz, new_points)
        net = new_points.view(batchsize, -1)
        out = self.fc1(net)
        return out


class cls_loss(nn.Module):
    def __init__(self):
        super(cls_loss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
    def forward(self, pred, lable):
        '''
        :param pred: shape=(B, nclass)
        :param lable: shape=(B, )
        :return: loss
        '''
        loss = self.loss(pred, lable)
        return loss


if __name__ == '__main__':
    xyz = torch.randn(50, 100, 3) * 3
    points = torch.randn(50, 100, 50) * 3
    ssg_model = pointnet2_feat_msg(50+3, 768).cuda()
    ssg_model.eval()

    # print(ssg_model)
    # xyz:    point coord
    # points: point feature vector
    x = torch.cat([xyz, points], dim=2)

    perm = torch.randperm(x.size(1))
    idx = perm[:20]
    samples = x[:, idx, :]
    x_duplicated = torch.cat([x, samples], dim=1)
    out1 = ssg_model(x.cuda())
    print(out1)
    out2 = ssg_model(x_duplicated.cuda())
    print(out2)
    print(torch.equal(out1, out2))
    print(torch.nn.functional.mse_loss(out1, out2))

    x_another = torch.randn_like(x.cuda())
    out3 = ssg_model(x_another)
    print(torch.nn.functional.mse_loss(out1, out3))

    #loss = cls_loss()
    #loss = loss(net, label)
    #print(loss)