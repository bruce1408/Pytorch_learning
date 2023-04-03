res_path = "/Users/bruce/Downloads/chip_onnx_model_evaluation/res_resnet101.txt"
label_path = "/Users/bruce/PycharmProjects/Pytorch_learning/label_name_to_id.txt";


label = dict()
with open(label_path, "r") as f:
    for eachline in f:
        labels = eachline.strip().split(" ")
        label[labels[0]] = labels[1]
        
        
count = 1
acc = 0
with open(res_path, "r") as f:
    for eachline in f:
        path = eachline.strip().split(" ")
        predict = path[1] 
        file_name = path[0].split("/")[-2]
        truth_label = label[file_name]
        if(truth_label == predict) :acc = acc + 1
        count+=1
        
print(acc / count)