import os
import SimpleITK as sitk
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
from medpy.metric.binary import assd, dc


def get_sitkImage(org_path, isMhd=True):
    if isMhd:
        sitkImg = sitk.ReadImage(org_path)
    else:
        sitkImg = read_dcm_series(org_path)
    return sitkImg


def read_dcm_series(path_series_dcm):
    reader = sitk.ImageSeriesReader()
    seriesIDs = reader.GetGDCMSeriesIDs(path_series_dcm)
    dcm_series = reader.GetGDCMSeriesFileNames(path_series_dcm, seriesIDs[0])
    reader.SetFileNames(dcm_series)
    itk_img = reader.Execute()
    return itk_img


def isotropic_resampler(img_mhd, new_spacing=None, is_label=False):
    if new_spacing is None:
        new_spacing = [1, 1, 1]
    resampler = sitk.ResampleImageFilter()
    if is_label:  # 如果是mask图像，就选择sitkNearestNeighbor这种插值
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:  # 如果是普通图像，就采用线性插值法
        resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(img_mhd.GetDirection())
    resampler.SetOutputOrigin(img_mhd.GetOrigin())
    resampler.SetOutputSpacing(new_spacing)
    orig_size = np.array(img_mhd.GetSize(), dtype=np.int)
    orig_spacing = img_mhd.GetSpacing()
    new_size = np.array([x * (y / z) for x, y, z in zip(orig_size, orig_spacing, new_spacing)])
    new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    resampler.SetSize(new_size)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    isotropic_img = resampler.Execute(img_mhd)
    return isotropic_img


def arr2jpg(img_proj):
    img_proj = np.array(img_proj, dtype=float)
    img_proj = (img_proj - img_proj.min()) / (img_proj.max() - img_proj.min())
    img_proj = np.array(img_proj * 255, dtype=np.uint8)
    img_proj = Image.fromarray(img_proj)
    return img_proj


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def draw_box_2d(img, box2ds):
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    for box2d in box2ds:
        red_colors = np.zeros([4, 3], np.uint8)
        red_colors[:, 0] = 255
        pt1 = int(box2d[1]), int(box2d[0])
        pt2 = int(box2d[3]), int(box2d[2])
        draw.rectangle([pt1, pt2], outline='red',width=2)


    return img_copy


def resize_image(image, size, letterbox_image):
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def preprocess_input(image):
    image /= 255.0
    return image


def norm_img(image):  # 归一化像素值到（0，1）之间，且将溢出值取边界值
    # image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


# 数据如果不够（128，128，64），需要添加边界
def padding_3D_data(sitk_image):
    img_size = sitk_image.GetSize()  # x,y,z(Width, Height, Depth)
    img_arr = sitk.GetArrayFromImage(sitk_image)
    img_shape = img_arr.shape # img_arr.shape :z,y,x
    if img_shape[1] < 128:  # y
        img_arr_new = np.zeros([img_shape[0], 128+2, img_shape[2]])
        img_arr_new[:, 0:img_shape[1], :] = img_arr
        img_arr_new[:, img_shape[1]:, :] = img_arr[int(img_shape[0]/2),img_shape[1]-2,int(img_shape[2]/2)]
        img_shape = img_arr_new.shape
        img_arr = img_arr_new

    if img_shape[2] < 128:  # x
        img_arr_new = np.zeros([img_shape[0], img_shape[1], 128+2])
        img_arr_new[:, :, 0:img_shape[2]] = img_arr
        img_arr_new[:, :, img_shape[2]:] = img_arr[int(img_shape[0]/2), int(img_shape[1]/2), img_shape[2]-2]
        img_arr = img_arr_new
    img_arr = np.int16(img_arr)
    new_sitk_img = sitk.GetImageFromArray(img_arr)
    new_sitk_img.SetSpacing(sitk_image.GetSpacing())
    new_sitk_img.SetOrigin(sitk_image.GetOrigin())

    return new_sitk_img


# 去杂
def get_max_area_region(image, number=1):
    # image = image.detach().numpy()
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    _input = sitk.GetImageFromArray(image.astype(np.uint16))
    output_ex = cca.Execute(_input)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(output_ex)
    num_label = cca.GetObjectCount()
    num_list = [i for i in range(1, num_label + 1)]
    area_list = []
    for l in range(1, num_label + 1):
        area_list.append(stats.GetNumberOfPixels(l))
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x - 1])[::-1]
    output = sitk.GetArrayFromImage(output_ex)

    if len(num_list_sorted) == 0:
        output = np.ones_like(output)
        print("全01")
        return torch.tensor(output.astype(np.float32))

    max_area_region = np.zeros_like(output)
    for ii in range(0, number):
        max_area_region[output == num_list_sorted[ii]] = 1

    # max_area_region = torch.tensor(max_area_region.astype(np.float32))
    max_area_region = max_area_region.astype(np.float32)
    return max_area_region


def make_one_hot(input_, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input_: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input_.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input_.cpu(), 1)

    return result


def MyDiceCoeff5(pred, gt):
    # pred = pred.to('cpu').numpy()
    # gt = gt.to('cpu').numpy()
    pred = pred.to('cpu').numpy()
    gt = gt.to('cpu').numpy()
    # if gt is all zero (use inverse to count)
    if np.count_nonzero(gt) == 0:
        gt = gt + 1
        pred = 1 - pred

    return dc(pred[:, 1:, :, :, :], gt[:, 1:, :, :, :])


