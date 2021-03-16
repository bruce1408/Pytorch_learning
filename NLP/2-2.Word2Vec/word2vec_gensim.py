import os
import sys
import jieba
import logging
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences


def seg_words():
    # 定义一些常量值，多次调用的文件路径放在这里，容易修改
    origin_file = "Dataset/Dataset.txt"  # 初代文件
    stop_words_file = "stopwords/cn_stopwords.txt"  # 停用词路径
    user_dict_file = "Dataset/user_dict.txt"  # 用户自定义词典路径
    stop_words = list()
    # 加载停用词
    with open(stop_words_file, 'r', encoding="utf8") as f:
        contents = f.readlines()
        for line in contents:
            line = line.strip()
            stop_words.append(line)

    # 加载用户字典
    jieba.load_userdict(user_dict_file)
    target_file = open("Dataset/douluo_cut_word.txt", 'w', encoding="utf-8")
    with open(origin_file, 'r', encoding="utf-8") as f:
        contents = f.readlines()
        for line in contents:
            line = line.strip()
            out_str = ''
            word_list = jieba.cut(line, cut_all=False)
            for word in word_list:
                if word not in stop_words:
                    if word != "\t":
                        out_str += word
                        out_str += ' '
            target_file.write(out_str.rstrip() + "\n")
    target_file.close()
    print("end")


def train_model():
    # 日志信息输出
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # input为输入语料， outp1为输出模型， outp2位vector格式的模型
    input_dir = 'Dataset/douluo_cut_word.txt'
    outp1 = 'output/words.vector'

    # 训练模型
    # 输入语料目录:PathLineSentences(input_dir)
    # embedding size:256 共现窗口大小:10 去除出现次数5以下的词,多线程运行,迭代10次
    model = Word2Vec(PathLineSentences(input_dir),
                     size=256, window=10, min_count=2,
                     workers=multiprocessing.cpu_count(), iter=10)
    model.save(outp1)  # 存储二进制模型文件
    # model.wv.save_word2vec_format(outp2, binary=False)  # 存储类似于数组的模型文件


if __name__ == "__main__":
    seg_words()
    train_model()
    model = Word2Vec.load("output/words.vector")

    word = "行为"
    print(model[word])
    a = model.most_similar(word)
    for i in a:
        print(i)
    # result = model.most_similar(word)
    #
    # for i in result:
    #     print(i)







