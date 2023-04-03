import os

image_label_path = "/Users/bruce/Downloads/chip_onnx_model_evaluation/imagenet2012.label"

imagenet_label_path = "/Users/bruce/Downloads/files_job/ncnn/examples/synset_words.txt"


img_dict = dict()
label = 1
with open(imagenet_label_path) as f:
    for eachline in f:
        n_name = eachline.strip().split(" ")[0]
        img_dict[n_name] = " ".join(i for i in eachline.strip().split(" ")[1:])


print(img_dict["n03126707"])


label_dict = dict()
with open(image_label_path) as f:
    for eachline in f:
        for key, value in img_dict.items():
            # print(value, eachline.strip())
            if(eachline == "crane"):
                print("==============", value, eachline.strip())
            if(eachline.strip() == value):
                # print(eachline.strip(), value)
                label_dict[key] = label
        label = label+1
print(label_dict)
with open("./label_name_to_id.txt", "w") as f:
    for key, value in label_dict.items():
        f.write(str(key) )
        f.write(" ")
        f.write(str(value))
        f.write("\n")
        