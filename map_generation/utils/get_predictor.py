import numpy as np

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config
from .predictor import Predictor
from .classes_mapping import std_labels, stuff2std

def setup_cfg(config_file='./meta_data/mask2former_config/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml'):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = 'meta_data/mask2former_ckpt/model_final_f07440.pkl'
    cfg.freeze()
    return cfg

def map_by_dict(arr, mapping_dict):
    return np.vectorize(mapping_dict.get)(arr)

class Wrapped_Predictor(object):
    def __init__(self) -> None:
        cfg = setup_cfg()
        self.model = Predictor(cfg)
        meta = self.model.metadata
        self.raw_labels = meta.stuff_classes
 
    def predict(self, img):
        """ 
            input rgb image
            return semantic segmentation 
        """
        img = img[:,:,:3]
        img = img[:, :, ::-1]
        predictions = self.model.run_on_image(img)
        label_obs = self.post_process_segmentation(predictions["panoptic_seg"],  self.raw_labels)
        return label_obs

    def post_process_segmentation(self, pano_seg_raw, raw_labels):
        pano_seg, segments_info = pano_seg_raw
        pano_seg = pano_seg.cpu().numpy()

        # ins_id2ins = {0: -100}
        ins_id2label = {0: -100}

        for info in segments_info:
            # ins_id2ins[info['id']] = info['id']
            std_label = stuff2std[raw_labels[info['category_id']]]
            if std_label is None:
                ins_id2label[info['id']] = 0
                # ins_id2ins[info['id']] = 0
                # print('\n', raw_labels[info['category_id']], '\n')
            else:
                remapped_cat_id = std_labels.index(std_label)
                ins_id2label[info['id']] = remapped_cat_id
        
        label_obs = map_by_dict(pano_seg, ins_id2label)
        # ins_i_obs = map_by_dict(pano_seg, ins_id2ins) 
        return label_obs

