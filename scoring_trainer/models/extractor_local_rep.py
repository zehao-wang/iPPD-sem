from models import pointnet, pointnet2

def build_extractor(config, input_dim):
    if config.TYPE == 'pointnet':
        model = pointnet.PointNetfeat_lite(
            feat_dim=input_dim, hidden_size=config.hidden_size, global_feat=True
        )
    elif config.TYPE == 'pointnet2':
        model = pointnet2.pointnet2_feat_msg(
            feat_dim=input_dim, hidden_size=config.hidden_size
        )
    else:
        raise NotImplementedError(f"For model type {config.TYPE}")
    return model