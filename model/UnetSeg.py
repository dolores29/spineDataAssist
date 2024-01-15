import numpy as np
import torch
import SimpleITK as sitk

from utils.util import norm_img, make_one_hot, MyDiceCoeff5, get_max_area_region


class SegPred:
    def __init__(self):
        self.model_path = r'./pth/weight_seg/UNet3D_spineSeg.pt'
        self.num_classes = 3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 方法一 直接用pth文件
        # self.seg_net = UNet(1, [32, 64, 96, 128, 192], 2, net_mode=self.config.mode,
        #                     conv_block=RecombinationBlock).to(self.config.device)
        # ckpt = torch.load(self.config.path_weight)
        # self.seg_net.load_state_dict(ckpt['net'])

        # 方法二 加载pt文件

    def load_model(self, img_sitk):
        model_load = torch.jit.load(self.model_path)
        model_load.eval()
        img_arr = sitk.GetArrayFromImage(img_sitk)
        img_arr = norm_img(img_arr)
        img_arr = np.expand_dims(np.expand_dims(img_arr, axis=0), axis=0)
        img_arr_input = torch.from_numpy(img_arr)

        data_input = img_arr_input.float().to(self.device)
        with torch.no_grad():
            img_arr_output = model_load(data_input)
        pred_one_hot = make_one_hot(img_arr_output.argmax(dim=1).unsqueeze(1).clone().detach(), self.num_classes)
        if torch.is_tensor(pred_one_hot):
            pred = pred_one_hot.data.cpu().numpy()
        else:
            pred = pred_one_hot
        # 提取标签为2的椎骨
        img_arr_output = get_max_area_region(pred[0, 2, :, :, :])
        return img_arr_output

