### Attention

#### seq2seq 的 attention 通过阅读论文 Bahadan 和 luong 的两篇论文,得到不同的计算方式.
- Bahadan的论文中attention计算,decoder t时刻的状态 hs 是由 ht-1 和 at;

- Luong的论文计算使用的是当期时刻 hs 和 ht 来计算 at


#### luong Attention:

- 计算 decoder 每一步的 output，即 ht
- 根据 ht 和 hs 计算 at
- 根据 at 和 hs 计算 ct(context vector)
- 拼接 ct 和 ht 得到 h^t (即attention 后的 output)
#### Bahdanau Attention：

- 计算 decoder 每一步的 output，即 ht-1
- 根据 ht-1 和 hs 计算 at
- 根据 at 和 hs 计算 ct
- 根据 ct 和 ht-1 计算得到 ht

> 4-2-1和4-2-4计算attention的思路大致相同,在复现luong的论文时候没有什么本质区别,但是4-2-4在复现Bahadan`s 的 attention 时候有一点瑕疵,计算时间上提前了一步

>4-2-3 和 4-2-2 代码结果相似,都是前一时刻预测下一时刻