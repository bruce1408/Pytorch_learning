import os
import json
# import matplotlib as plt
import matplotlib.pyplot as plt

# from CustomData.dataset import Vocab
from wordcloud import WordCloud


# 读json文件（以训练集为例）：
def read_json(path):
    total_data = []
    with open(path) as input_data:
        json_content = json.load(input_data)
        # 逐条读取记录
        for block in json_content:
            record_id = block['id']
            query1 = block['query1']
            query2 = block['query2']
            label = block['label']
            total_data.append([record_id, query1, query2, label])
    return total_data


def write_json(labels, time, model_merge):
    # 写json文件，本示例代码从测试集KUAKE-QQR_test.json读取数据数据，将预测后的数据写入到KUAKE-QQR_test_pred.json：
    model_merge_str = "multi_models_predict" if model_merge else "single_model_predict"
    with open('./data/KUAKE-QQR_test.json') as input_data, open('./data/outputs/KUAKE-QQR_test_pred_best_' +
                                                                model_merge_str +
                                                                str(time) + '.json', 'w') as output_data:
        json_content = json.load(input_data)
        # 逐条读取记录，并将预测好的label赋值
        i = 0
        for block in json_content:
            query1 = block['query1']
            query2 = block['query2']
            # 此处调用自己的模型来预测当前记录的label，仅做示例用：
            block['label'] = str(labels[i])
            i += 1
        # 写json文件
        json.dump(json_content, output_data, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # a = ["我", "们", "被", "天", "男", "好", "张", "长", "被", "景"]
    # b = Vocab(a)
    # print(b.token_to_idx)
    train_data = read_json("../data/KUAKE-QQR_train.json")
    total_sentence = [eachpair[1] for eachpair in train_data] + [eachpair[1] for eachpair in train_data]
    print(total_sentence)
    # print(train_data.__len__())
    # #
    # # dev_data = read_json("data/KUAKE-QQR_dev.json")
    # # print(dev_data.__len__(), dev_data)
    # #
    # test_data = read_json("data/KUAKE-QQR_test.json")
    # print(test_data.__len__(), test_data)

    import time
    # print(time.localtime())
    import matplotlib as mpl
    mpl.rcParams['font.sans-serif'] = [u'SimHei']  # FangSong/黑体 FangSong/KaiTi
    mpl.rcParams['axes.unicode_minus'] = False
    cloud = WordCloud(width=1440, height=1080, font_path="../../../Reference/simhei.ttf").generate(" ".join(total_sentence))
    plt.figure(figsize=(20, 15))
    plt.imshow(cloud)
    plt.axis('off')
    plt.savefig('./医学问题对中文词云.jpg')

    plt.show()
    # plt.savefig('./医学问题对中文词云.png')

