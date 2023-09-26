import os
from PIL import Image


def jpgToBmp(imgFile):
    dst_dir = "/Users/bruce/PycharmProjects/Pytorch_learning/Tools/img_data_ti"

    for fileName in os.listdir(imgFile):
        if os.path.splitext(fileName)[1] == '.JPEG' or os.path.splitext(fileName)[1] == '.jpg':
            name = os.path.splitext(fileName)[0]
            newFileName = name + ".bmp"

            img = Image.open(imgFile + "/" + fileName)
            img.save(dst_dir+"/"+newFileName)


def main():
    imgFile = "/Users/bruce/Downloads/calib_Ti"

    jpgToBmp(imgFile)


if __name__ == '__main__':
    main()
