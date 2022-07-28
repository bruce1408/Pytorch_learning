import os
import json
# from CustomData.dataset import Vocab


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


def write_json(labels, time):
    # 写json文件，本示例代码从测试集KUAKE-QQR_test.json读取数据数据，将预测后的数据写入到KUAKE-QQR_test_pred.json：
    with open('./data/KUAKE-QQR_test.json') as input_data, open('./data/outputs/KUAKE-QQR_test_pred_best_' +
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
    # train_data = read_json("data/KUAKE-QQR_train.json")
    # print(train_data.__len__())
    # #
    # # dev_data = read_json("data/KUAKE-QQR_dev.json")
    # # print(dev_data.__len__(), dev_data)
    # #
    # test_data = read_json("data/KUAKE-QQR_test.json")
    # print(test_data.__len__(), test_data)

    import time
    print(time.localtime())

