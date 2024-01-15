import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

from model.UnetSeg import SegPred
from model.YoloLoc import LocPred
from model.utils_bbox import get_boxes3d_pred, judge_bound_3d
from utils import util
from utils.util import draw_box_2d


class DataAssist:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.save_name = None

    def generate_data(self):
        for img_dir in os.listdir(self.dir_path):
            if '.mhd' in img_dir:
                sitkImg = util.get_sitkImage(os.path.join(self.dir_path, img_dir), isMhd=True)
            elif os.path.isdir(os.path.join(self.dir_path, img_dir)):
                sitkImg = util.get_sitkImage(os.path.join(self.dir_path, img_dir), isMhd=False)
            else:
                continue
            self.save_name = img_dir.split('.')[0]
            sitkImg_sam = util.isotropic_resampler(sitkImg, is_label=False)
            sitkImg_sam = util.padding_3D_data(sitkImg_sam)  # 不满足（128，128，64）进行扩充
            sitk.WriteImage(sitkImg_sam, os.path.join(save_seg_mhd, self.save_name + '.nrrd'))

            loc_boxes3d = self.loc_project(sitkImg_sam)
            crop_list = self.crop_img(sitkImg_sam, loc_boxes3d)
            seg_list = self.seg_img(crop_list)
            print(img_dir)

    def loc_project(self, sitkImg_sam):
        proj_f = np.squeeze(sitk.GetArrayFromImage(sitk.MaximumProjection(sitkImg_sam, projectionDimension=1)))
        proj_l = np.squeeze(sitk.GetArrayFromImage(sitk.MaximumProjection(sitkImg_sam, projectionDimension=0)))
        proj_f_flip = np.flip(proj_f, 0)  # 上下翻转
        proj_l_flip = np.flip(proj_l, 0)
        proj_f_img = util.arr2jpg(proj_f_flip)
        proj_l_img = util.arr2jpg(proj_l_flip)
        # proj_f_img.save('img_project_f.jpg')
        # proj_l_img.save('img_project_l.jpg')
        locPred = LocPred()
        box_f = locPred.load_model(proj_f_img)
        box_l = locPred.load_model(proj_l_img)
        loc_boxes3d = get_boxes3d_pred(box_f, box_l)
        loc_boxes3d = loc_boxes3d[np.lexsort(loc_boxes3d.T)]
        loc_boxes3d = judge_bound_3d(loc_boxes3d)

        # proj_f_img_draw = draw_box_2d(proj_f_img, box_f)
        # proj_l_img_draw = draw_box_2d(proj_l_img, box_l)
        # plt.subplot(211)
        # plt.imshow(proj_f_img_draw)
        # plt.subplot(212)
        # plt.imshow(proj_l_img_draw)
        # plt.show()
        # plt.pause(10)  # 10秒后自动关闭窗口

        return np.array(loc_boxes3d)

    def crop_img(self, sitkImg, boxes3d):
        sitkImg_crop_list = []
        for i,box3d in enumerate(boxes3d):
            sitkImg_crop = sitkImg[box3d[0]: box3d[3],
                                   box3d[1]:box3d[4],
                                   box3d[2]:box3d[5]]
            sitkImg_crop_list.append(sitkImg_crop)
            sitk.WriteImage(sitkImg_crop, os.path.join(save_crop_mhd, self.save_name+'_crop_' + str(i) + '.mhd'))

        return sitkImg_crop_list

    def seg_img(self, sitkImg_crop_list):
        sitkImg_seg_list = []
        segPred = SegPred()
        for i, sitkImg_crop in enumerate(sitkImg_crop_list):
            seg_arr = segPred.load_model(sitkImg_crop)
            seg_img = sitk.GetImageFromArray(seg_arr)
            seg_img.SetOrigin(sitkImg_crop.GetSpacing())
            seg_img.SetOrigin(sitkImg_crop.GetOrigin())
            sitkImg_seg_list.append(seg_img)
            sitk.WriteImage(seg_img, os.path.join(save_seg_mhd, self.save_name+'_seg_'+str(i)+'.nrrd'))

        return sitkImg_seg_list


if __name__ == '__main__':
    dir_path = r'E:\workdata\spine\temp\test_mhd\testOrigin'
    save_seg_mhd = r'E:\workdata\spine\temp\save_seg_mhd'
    save_crop_mhd = r'E:\workdata\spine\temp\save_crop_mhd'
    Dat = DataAssist(dir_path)
    Dat.generate_data()
















