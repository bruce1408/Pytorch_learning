# # %matplotlib inline
# import matplotlib.pyplot as plt
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# import numpy as np
# import skimage.io as io
# import pylab
# pylab.rcParams['figure.figsize'] = (10.0, 8.0)



# annType = ['segm','bbox','keypoints']
# annType = annType[1]      #specify type here
# prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
# print ('Running demo for *%s* results.'%(annType))


# #initialize COCO ground truth api
# dataDir='/Users/bruce/Downloads/Datasets/COOC'
# dataType='val2014'
# annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
# cocoGt=COCO(annFile)


# #initialize COCO detections api
# resFile='%s/results/%s_%s_fake%s100_results.json'
# resFile = resFile%(dataDir, prefix, dataType, annType)
# cocoDt=cocoGt.loadRes(resFile)

# imgIds=sorted(cocoGt.getImgIds())
# imgIds=imgIds[0:100]
# imgId = imgIds[np.random.randint(100)]



# # running evaluation
# cocoEval = COCOeval(cocoGt,cocoDt,annType)
# cocoEval.params.imgIds  = imgIds
# cocoEval.evaluate()
# cocoEval.accumulate()
# cocoEval.summarize()




from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from tempfile import NamedTemporaryFile
 
# coco格式的json文件，原始标注数据
anno_file = '/Users/bruce/Downloads/Datasets/COOC/annotations/instances_val2014.json'
coco_gt = COCO(anno_file)
 
# 用GT框作为预测框进行计算，目的是得到detection_res
with open(anno_file, 'r') as f:
    json_file = json.load(f)
annotations = json_file['annotations']
detection_res = []
for anno in annotations:
    detection_res.append({
        'score': 1.,
        'category_id': anno['category_id'],
        'bbox': anno['bbox'],
        'image_id': anno['image_id']
    })
 
with NamedTemporaryFile(suffix='.json') as tf:
    # 由于后续需要，先将detection_res转换成二进制后写入json文件
    content = json.dumps(detection_res).encode(encoding='utf-8')
    tf.write(content)
    res_path = tf.name
 
    # loadRes会在coco_gt的基础上生成一个新的COCO类型的instance并返回
    coco_dt = coco_gt.loadRes(res_path)
 
    cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
 
print(cocoEval.stats)
