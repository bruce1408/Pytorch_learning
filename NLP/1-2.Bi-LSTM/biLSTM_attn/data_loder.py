import numpy as np
import torch
from torch.autograd import Variable
import const
from tqdm import tqdm


class DataLoader(object):
    def __init__(self, src_sents, label, max_len, cuda=True, batch_size=64, shuffle=True, evaluation=False):
        self.cuda = cuda
        self.sents_size = len(src_sents)
        self._step = 0
        self._stop_step = self.sents_size // batch_size
        self.evaluation = evaluation

        self._batch_size = batch_size
        self._max_len = max_len
        self._src_sents = np.asarray(src_sents)
        self._label = np.asarray(label)
        if shuffle:
            self._shuffle()

    def _shuffle(self):
        indices = np.arange(self._src_sents.shape[0])
        np.random.shuffle(indices)  # 对长度src_sects进行打乱顺序
        self._src_sents = self._src_sents[indices]  # 按照index乱序的顺序打乱
        self._label = self._label[indices]

    def __iter__(self):
        return self

    def __next__(self):
        def pad_to_longest(insts, max_len):
            # 不足16长度的补充0, 超过的不做计算
            inst_data = np.array([inst + [const.PAD] * (max_len - len(inst)) for inst in insts])

            inst_data_tensor = torch.from_numpy(inst_data)
            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()
            return inst_data_tensor

        if self._step == self._stop_step:  # 停止进行迭代
            self._step = 0
            raise StopIteration()

        _start = self._step*self._batch_size
        _bsz = self._batch_size
        self._step += 1
        data = pad_to_longest(self._src_sents[_start:_start+_bsz], self._max_len)  # 长度进行补0，都为16长度
        label = torch.from_numpy(self._label[_start:_start+_bsz])
        if self.cuda:
            label = label.cuda()
        return data, label


if __name__ == '__main__':
    batch_size = 16
    cuda_able = True
    max_len = 16
    hidden_size = 32
    bidirectional = True
    weight_decay = 0.001
    attention_size = 16
    sequence_length = 16
    use_cuda = torch.cuda.is_available() and cuda_able
    data = torch.load('./Dataset/corpus.pt')
    training_data = DataLoader(data['train']['src'], data['train']['label'], max_len, batch_size=batch_size,
                               cuda=use_cuda)

    for data, label in tqdm(training_data, mininterval=1, desc='Train Processing', leave=False):
        print(data.shape, label)



