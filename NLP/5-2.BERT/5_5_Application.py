# -*- coding: utf-8 -*-
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from transformers import pipeline

# transformers 自带模型

# 管道里面调用文本分类任务
nlp = pipeline("sentiment-analysis")
print(nlp("I like this book!"))

# ##########################################feature-extraction
import numpy as np

# 特征抽取任务
nlp_features = pipeline('feature-extraction')
output = nlp_features('Code Doctor Studio is a Chinese company based in BeiJing.')
print(np.array(output).shape)  # (Samples, Tokens, Vector Size)(1, 16, 768)

# 掩码语言建模
nlp_fill = pipeline("fill-mask")
# 完型填空任务
print(nlp_fill.tokenizer.mask_token)
print(nlp_fill(
    f"Li Jinhong wrote many {nlp_fill.tokenizer.mask_token} about \
    artificial intelligence technology and helped many people."))

############################抽取式问答
# 问答任务
# nlp_qa = pipeline("question-answering")
# print(nlp_qa(context='Code Doctor Studio is a Chinese company based in BeiJing.',
#              question='Where is Code Doctor Studio?'))


nlp = pipeline("question-answering")
context = "Extractive Question Answering is the task of extracting an answer from a text given a question. An example " \
          "of a question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would " \
          "like to fine-tune a model on a SQuAD task, you may leverage the `run_squad.py`. "

print(nlp(question="What is extractive question answering?", context=context))
print(nlp(question="What is a good example of a question answering dataset?", context=context))


###################################摘要
# 摘要生成
TEXT_TO_SUMMARIZE = '''
In this notebook we will be using the transformer model, first introduced in this paper. Specifically,
we will be using the BERT (Bidirectional Encoder Representations from Transformers) model from this paper.
Transformer models are considerably larger than anything else covered in these tutorials.
As such we are going to use the transformers library to get pre-trained transformers and use them as our
embedding layers.
We will freeze (not train) the transformer and only train the remainder of the model which learns
from the representations produced by the transformer. In this case we will be using a multi-layer bi-directional GRU,
however any model can learn from these representations.
'''
# summarizer = pipeline('summarization')
# print(summarizer(TEXT_TO_SUMMARIZE))


# #################命名实体识别

nlp_token_class = pipeline("ner")
print(nlp_token_class('Code Doctor Studio is a Chinese company based in BeiJing.'))
