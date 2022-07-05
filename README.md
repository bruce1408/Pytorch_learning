
# Pytorch_learning


[![](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)](https://github.com/bruce1408/Pytorch_learning)
![](https://img.shields.io/badge/platform-pytorch-brightgreen.svg)
![](https://img.shields.io/badge/python-3.7-blue.svg)


This repository provides tutorial code for deep learning researchers to learn [PyTorch](https://pytorch.org/)

PyTorch is a Python-based scientific computing package serving two broad purposes:

- A replacement for NumPy to use the power of GPUs and other accelerators.
- An automatic differentiation library that is useful to implement neural networks.

PyTorch, like most other deep learning libraries, supports reverse-mode automatic differentiation of
scalar functions (or vector-Jacobian products of functions with multiple outputs), the most important form 
of automatic differentiation for deep learning applications which usually differentiatea single scalar loss.
You write code as if you were executing tensor operations directly; however, 
instead of operating on Tensors (PyTorch’s equivalentof Numpy’s nd-arrays), 
the user manipulates Variables, which store extra metadata necessary for AD. Variables support a backward() method, 
which computes the gradient of all input Variables involved in computation of this quantity.


This repository contains:

1. [Computer Vision(CV)](/CV/) Implement complex computer vision algorithms.
2. [Natural language processing(NLP)](/NLP/) Examples to show how NLP can tacke real problem.Including the source code,
dataset, state-of-the art in NLP.
3. [Dataset](/Dataset/) All the data you can use in this project are under this directory.
4. [Reference](/Reference) Other reference materials for this project.
### 代码测试环境
* Python: 3.8.5
* PyTorch: 1.8.0
* Transformers: 4.9.0
* NLTK: 3.5
* LTP: 4.0


## Table of Contents

- [Project](#Project)
- [Install](#install)
- [Usage](#usage)
- [Related Efforts](#Related-Efforts)
- [Contributors](#Contributors)

## Project

#### Computer Vision 

- Image Classification 

    - AlexNet
    - VGGNet
    - ResNet
    - Inception-v1
    - Transfer-learning
    
- Generative Adversarial Network

- Open-Set domain adaption

#### Natural language processing
>pip install en_core_web_sm-2.2.5.tar.gz

>pip install de_core_news_sm-2.2.5.tar.gz
- Basic Knowledge
- Bi-LSTM
- Bi-LSTM Attention
- Seq2Seq
- Seq2Seq Attention
- Transformer
- Bert

## Install

This project uses [PyTorch](https://pytorch.org/). Go check them out if you don't have them locally installed and 
thirt-party dependencies.

```sh
$ pip install -r requirements.txt
```

## Usage

All data for this project can be found in [Baidu Cloud Disk](https://pan.baidu.com/s/1TCOPe6PRd6S-SR16yOlZfg) 密码: 39nw

## Related Efforts

- [Sung Kim](https://github.com/hunkim)

- [graykode](https://github.com/graykode/nlp-tutorial) - Learn the code quality.
- [yunjey](https://github.com/yunjey/pytorch-tutorial) - A tutorial code to encourage open-source contributions.



### Contributors

This project exists thanks to all the people who contribute. 
Everyone is welcome to submit code contributions
