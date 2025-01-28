from efficientunet import *
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from habitat_baselines.rl.models.networks.resnetUnet import ResNetUNet
from habitat_baselines import test_hyeon_utils
import segmentation_models_pytorch as smp

def run_img_segm(model, model_name, input):
    if model_name.startswith("resnet_unet"):
        print("resnet:", input.shape)
        input = torch.Tensor(input).permute(0, 3, 1, 2) / 255.0

        B, _, H, W = input.shape
    elif model_name.startswith("efficient_unet"):
        print("efficient:", input.shape)
        input = torch.Tensor(input).permute(0, 3, 1, 2).unsqueeze(1) / 255.0
        B, T, _, H, W = input.shape
    else:
        input = torch.Tensor(input).permute(0, 3, 1, 2) / 255.0
        B, _, H, W = input.shape

    pred_segm_raw = model(input)
    C = pred_segm_raw.shape[1]  # must be 27
    pred_segm_raw = pred_segm_raw.view(B, C, H, W)
    pred_segm = F.softmax(pred_segm_raw, dim=2)
    pred_img_segm = {'pred_segm': pred_segm}

    # get labels from prediction
    img_labels = torch.argmax(pred_img_segm['pred_segm'], dim=1, keepdim=True)  # B x T x 1 x cH x cW .detach()
    if model_name.startswith("efficient"):
        img_labels = img_labels.unsqueeze(1)

    return img_labels


def main(model_type: str):
    if model_type.startswith("resnet_unet"):
        model = ResNetUNet(n_channel_in=3, n_class_out=17)
    elif model_type.startswith("efficient_unet"):
        model = get_efficientunet_b3(out_channels=17, pretrained=True)  # concat_input=True,
    else:
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b3",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=17
            ,  # model output channels (number of classes in your dataset)
        )
    model.eval()
    # for n, p in model.named_children():
    #     print(p.requires_grad_())
    for n, p in model.named_parameters():
        p.requires_grad = False
        # print(n, ":", p.requires_grad)

    print("number of parameters: {}".format(sum(param.numel() for param in model.parameters())))

    img_path = "/media/ailab8503/484d268c-c692-47d6-800f-b6c2d2f92790/jinhwan_workspace/images/1LXtFkjw3qL/rgb/test/"
    # image_w, image_h = 256, 256

    img_obs = []
    for top, dir, f in os.walk(img_path):
        for filename in f:
            img = cv2.imread(img_path+filename)
            img_obs.append(img)
    pred_ssegs = run_img_segm(model=model, model_name=model_type, input=np.array(img_obs))

    vis = test_hyeon_utils.colorize_grid(pred_ssegs, color_mapping=27).squeeze()
    for i in range(vis.shape[0]):  # batch
        plt.imsave('visualization/semantic/sem_color_{0}_{1}_4.png'.format(model_type, i),
                   np.array(vis[i].squeeze().cpu()))

    # for i in range(pred_ssegs.shape[0]):  # batch
    #     import matplotlib.pyplot as plt
    #     # plt.imshow(np.array(pred_ssegs[0].permute(1, 2, 0).squeeze().cpu(), dtype=np.uint8))
    #     # plt.show()
    #     plt.imsave('/home/ailab/LYON_OURS/visualization/semantic/sem_{0}_{1}.png'.format(model_type, i), np.array(pred_ssegs[i].permute(1, 2, 0).squeeze().cpu(), dtype=np.uint8))
    # print(pred_ssegs.shape)

    # for i in range(pred_ssegs.shape[0]):  # batch
    #     plt.imshow(np.array(pred_ssegs[0].permute(1, 2, 0).squeeze().cpu(), dtype=np.uint8))
    #     plt.show()
    #     np_sem = np.array(pred_ssegs[i][0].unsqueeze(dim=0).permute(1, 2, 0).cpu(), dtype=np.uint8)
    #     print(np_sem.shape)
    #     cv2.imwrite('/home/ailab/junghyeon_ws/image/semantic/sem{0}.png'.format(i), np_sem * 255)
    #
    # print("running model - finish")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-type",
        choices=["resnet_unet", "efficient_unet", "else"],
        required=True,
        help="model type of the experiment",
    )
    args = parser.parse_args()

    main(**vars(args))

