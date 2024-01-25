#
# Copyright (c) 2016,2018-2020 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
import argparse
import numpy as np
import os

from PIL import Image

RESIZE_METHOD_ANTIALIAS = "antialias"
RESIZE_METHOD_BILINEAR  = "bilinear"

def __get_img_raw(img_filepath):
    img_filepath = os.path.abspath(img_filepath)
    img = Image.open(img_filepath)
    img_ndarray = np.array(img) # read it
    if len(img_ndarray.shape) != 3:
        raise RuntimeError('Image shape' + str(img_ndarray.shape))
    if (img_ndarray.shape[2] != 3):
        raise RuntimeError('Require image with rgb but channel is %d' % img_ndarray.shape[2])
    # reverse last dimension: rgb -> bgr
    return img_ndarray

def __create_mean_raw(img_raw, mean_rgb):
    if img_raw.shape[2] != 3:
        raise RuntimeError('Require image with rgb but channel is %d' % img_raw.shape[2])
    img_dim = (img_raw.shape[0], img_raw.shape[1])
    mean_raw_r = np.empty(img_dim)
    mean_raw_r.fill(mean_rgb[0])
    mean_raw_g = np.empty(img_dim)
    mean_raw_g.fill(mean_rgb[1])
    mean_raw_b = np.empty(img_dim)
    mean_raw_b.fill(mean_rgb[2])
    # create with c, h, w shape first
    tmp_transpose_dim = (img_raw.shape[2], img_raw.shape[0], img_raw.shape[1])
    mean_raw = np.empty(tmp_transpose_dim)
    mean_raw[0] = mean_raw_r
    mean_raw[1] = mean_raw_g
    mean_raw[2] = mean_raw_b
    # back to h, w, c
    mean_raw = np.transpose(mean_raw, (1, 2, 0))
    return mean_raw.astype(np.float32)

def __create_raw_incv3(img_filepath, mean_rgb, div, req_bgr_raw, save_uint8):
    img_raw = __get_img_raw(img_filepath)
    mean_raw = __create_mean_raw(img_raw, mean_rgb)

    snpe_raw = img_raw - mean_raw
    snpe_raw = snpe_raw.astype(np.float32)
    # scalar data divide
    snpe_raw /= div

    if req_bgr_raw:
        snpe_raw = snpe_raw[..., ::-1]

    if save_uint8:
        snpe_raw = snpe_raw.astype(np.uint8)
    else:
        snpe_raw = snpe_raw.astype(np.float32)

    img_filepath = os.path.abspath(img_filepath)
    filename, ext = os.path.splitext(img_filepath)
    snpe_raw_filename = filename
    snpe_raw_filename += '.raw'
    snpe_raw.tofile(snpe_raw_filename)

    return 0

def __resize_square_to_jpg(src, dst, size,resize_type):
    src_img = Image.open(src)
    # If black and white image, convert to rgb (all 3 channels the same)
    if len(np.shape(src_img)) == 2: src_img = src_img.convert(mode = 'RGB')
    # center crop to square
    width, height = src_img.size
    short_dim = min(height, width)
    crop_coord = (
        (width - short_dim) / 2,
        (height - short_dim) / 2,
        (width + short_dim) / 2,
        (height + short_dim) / 2
    )
    img = src_img.crop(crop_coord)
    # resize to inceptionv3 size
    if resize_type == RESIZE_METHOD_BILINEAR :
        dst_img = img.resize((size, size), Image.BILINEAR)
    else :
        dst_img = img.resize((size, size), Image.ANTIALIAS)
    # save output - save determined from file extension
    dst_img.save(dst)
    return 0

def convert_img(src,dest,size,resize_type):
    print("Converting images for inception v3 network.")

    print("Scaling to square: " + src)
    for root, dirs,files in os.walk(src):
        for jpgs in files:
            src_image=os.path.join(root, jpgs)
            if('.jpg' or ".bmp" in src_image):
                print(src_image)
                dest_image = os.path.join(dest, jpgs)
                __resize_square_to_jpg(src_image,dest_image,size,resize_type)

    print("Image mean: " + dest)
    for root,dirs,files in os.walk(dest):
        for jpgs in files:
            src_image=os.path.join(root, jpgs)
            if('.jpg' or ".bmp" in src_image):
                print(src_image)
                mean_rgb=(128,128,128)
                __create_raw_incv3(src_image,mean_rgb,128,False,False)


def main():
    parser = argparse.ArgumentParser(description="Batch convert jpgs",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dest',type=str, default="/Users/bruce/PycharmProjects/Pytorch_learning/Tools/qnn_output")
    parser.add_argument('-s','--size',type=int, default=224)
    parser.add_argument('-i','--img_folder',type=str, default="/Users/bruce/Downloads/Datasets/calibration_classification_data_bmp_bin_debug")
    parser.add_argument('-r','--resize_type',type=str, default=RESIZE_METHOD_BILINEAR,
                        help='Select image resize type antialias or bilinear. Image resize type should match '
                             'resize type used on images with which model was trained, otherwise there may be impact '
                             'on model accuracy measurement.')

    args = parser.parse_args()

    size = args.size
    src = os.path.abspath(args.img_folder)
    dest = os.path.abspath(args.dest)
    resize_type = args.resize_type

    assert resize_type == RESIZE_METHOD_BILINEAR or resize_type == RESIZE_METHOD_ANTIALIAS, \
           "Image resize method should be antialias or bilinear"

    convert_img(src,dest,size,resize_type)

if __name__ == '__main__':
    exit(main())
