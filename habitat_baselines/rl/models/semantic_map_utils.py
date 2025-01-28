import torch
import torch.nn.functional as F
from habitat_baselines.rl.models.networks.resnetUnet import ResNetUNet
from habitat_baselines.common.utils import Model
from efficientunet import *
# import segmentation_models_pytorch as smp
import numpy as np
import matplotlib.pyplot as plt


def get_network_from_options(config, img_segmentor=None):
    if config.WITH_IMG_SEGM:
        with torch.no_grad():
            img_segmentor = ResNetUNet(n_channel_in=3, n_class_out=config.N_OBJECT_CLASSES)
            # img_segmentor = get_efficientunet_b5(out_channels=17, concat_input=True, pretrained=True)
        # img_segmentor = smp.create_model("FPN", encoder_name="resnet18", in_channels=3, classes=17) # "Unet", encoder_name="efficientnet-b7"
        # img_segmentor = smp.DeepLabV3(
        #     encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        #     encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        #     in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        #     classes=17
        #     ,  # model output channels (number of classes in your dataset)
        # )
            for p in img_segmentor.parameters():
                p.requires_grad = False

            model_utils = Model()
            latest_checkpoint = model_utils.get_latest_model(save_dir=config.IMG_SEGM_MODEL_DIR)
            print("Loading image segmentation checkpoint", latest_checkpoint, config.IMG_SEGM_MODEL_DIR)

            checkpoint = torch.load(latest_checkpoint)
            img_segmentor.load_state_dict(checkpoint['models']['img_segm_model'], strict=False)
            img_segmentor.eval()

            # for name, child in img_segmentor.named_children():
            #     for param in child.parameters():
            #         print(name, param.requires_grad)

    return img_segmentor


def run_img_segm(model, input_obs, img_labels=None):
    input = input_obs['rgb'].permute(0, 3, 1, 2) / 255.0
    for i in range(input_obs['rgb'].shape[0]):  # batch
        plt.imsave('visualization/rgb_input/rgb{0}.png'.format(i), np.array(input[i].permute(1, 2, 0).squeeze().cpu(), dtype=np.uint8))

    B, _, H, W = input.shape

    pred_segm_raw = model(input) # .detach()
    C = pred_segm_raw.shape[1]  # must be 27
    pred_segm_raw = pred_segm_raw.view(B, C, H, W)
    pred_segm = F.softmax(pred_segm_raw, dim=1)
    pred_img_segm = {'pred_segm': pred_segm}

    # get labels from prediction
    # img_labels = torch.argmax(pred_img_segm['pred_segm'], dim=1, keepdim=True)  # B x 1 x cH x cW .detach()

    return pred_img_segm['pred_segm']  # img_labels.permute(0, 2, 3, 1)  # [bs, 128, 128, 1]
