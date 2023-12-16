from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from tempfile import NamedTemporaryFile
 
# coco格式的json文件，原始标注数据
# anno_file = '/root/annotations/instances_val2014.json'
# coco_gt = COCO(anno_file)
 
# # 用GT框作为预测框进行计算，目的是得到detection_res
# with open(anno_file, 'r') as f:
#     json_file = json.load(f)
# annotations = json_file['annotations']
# detection_res = []
# for anno in annotations:
#     detection_res.append({
#         'score': 1.,
#         'category_id': anno['category_id'],
#         'bbox': anno['bbox'],
#         'image_id': anno['image_id']
#     })
 

def compare_json_mAP():
    with NamedTemporaryFile(suffix='.json') as tf:
        # 由于后续需要，先将detection_res转换成二进制后写入json文件
        content = json.dumps(detection_res).encode("utf-8")
        tf.write(content)
        res_path = tf.name
        print("save the json in path: ", res_path)
        # loadRes会在coco_gt的基础上生成一个新的COCO类型的instance并返回
        coco_dt = coco_gt.loadRes(res_path)
    
        cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
    

def cmp_json_mAP():
    # with NamedTemporaryFile(suffix='.json') as tf:
    with open("./predict.json", "w") as tf:
        # 由于后续需要，先将detection_res转换成二进制后写入json文件
        content = json.dumps(detection_res)
        tf.write(content)
        res_path = tf.name
        res_path = "./predict.json"
        print("save the json in path: ", res_path)
        # loadRes会在coco_gt的基础上生成一个新的COCO类型的instance并返回
        coco_dt = coco_gt.loadRes(res_path)
        
    
        cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
    
    print(cocoEval.stats)


# yolop_gr_path = "/root/yolop_model/yolop/yolop_gt_files.json"
# yolop_gr_path = "/root/yolop_model/yolop/yolop_gt_files_new.json"
yolop_gr_path = "/root/yolop_model/yolop/yolop_gt_coco.json"


coco_gt = COCO(yolop_gr_path)
pred = coco_gt.loadRes("/root/Pytorch_learning/yolop_pred_coord.json")
eval = COCOeval(coco_gt, pred, 'bbox')

eval.evaluate()
eval.accumulate()
eval.summarize()
map, map50 = eval.stats[:2] 
print(eval.stats)
 