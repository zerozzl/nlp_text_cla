# 自然语言处理-文本分类

对比常见模型在文本分类任务上的效果，主要涉及以下几种模型：

- [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)
- [Recurrent Convolutional Neural Networks for Text Classification](http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Recurrent%20Convolutional%20Neural%20Networks%20for%20Text%20Classification.pdf)
- [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf)
- [Deep Pyramid Convolutional Neural Networks for Text Categorization](https://aclanthology.org/P17-1052.pdf)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
- [Hierarchical Attention Networks for Document Classification](http://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)

## 短文本分类

### Char-level 效果

#### TextCNN

|-|Random|Pretrain|Static|Multi Channel|
|----|----|----|----|----|
|ctrip|<b>0.921</b>|0.909|0.899|0.907|
|weibo|<b>0.984</b>|<b>0.984</b>|<b>0.984</b>|<b>0.984</b>|
|shop_emo|<b>0.929</b>|0.926|0.911|0.926|
|shop_cat|<b>0.891</b>|0.871|0.855|0.866|

#### TextRCNN

|-|Random|Pretrain|Static|
|----|----|----|----|
|ctrip|<b>0.914</b>|0.903|0.907|
|weibo|0.985|0.985|<b>0.987<b/>|
|shop_emo|<b>0.918</b>|0.917|<b>0.918</b>|
|shop_cat|<b>0.881</b>|0.83|0.831|

#### FastText

|-|Random|Pretrain|Static|Random + Bigram|Pretrain + Bigram|Static + Bigram|
|----|----|----|----|----|----|----|
|ctrip|0.9|0.898|0.762|0.931|<b>0.934</b>|0.8|
|weibo|0.966|0.967|0.824|<b>0.96</b>|0.957|0.832|
|shop_emo|0.908|0.909|0.751|<b>0.933</b>|0.932|0.804|
|shop_cat|0.888|0.888|0.65|0.904|<b>0.906</b>|0.678|

#### DPCNN

|-|Random|Pretrain|Static|
|----|----|----|----|
|ctrip|<b>0.934</b>|0.931|0.93|
|weibo|<b>0.984</b>|<b>0.984</b>|<b>0.984</b>|
|shop_emo|0.936|<b>0.94</b>|0.932|
|shop_cat|<b>0.9</b>|0.888|0.88|

#### Bert

|-|Simple|Fix Embedding
|----|----|----|
|ctrip|<b>0.944</b>|0.86|
|weibo|<b>0.98</b>|0.897|
|shop_emo|<b>0.95</b>|0.908|
|shop_cat|<b>0.914</b>|0.845|

### Word-level 效果

#### TextCNN

|-|Random|Pretrain|Static|Multi Channel|
|----|----|----|----|----|
|ctrip|<b>0.913</b>|0.909|0.893|0.907|
|weibo|<b>0.984</b>|<b>0.984</b>|0.969|<b>0.984</b>|
|shop_emo|0.926|0.924|0.893|<b>0.928</b>|
|shop_cat|<b>0.88</b>|0.876|0.801|0.877|

#### TextRCNN

|-|Random|Pretrain|Static|
|----|----|----|----|
|ctrip|0.9|<b>0.912</b>|0.909|
|weibo|<b>0.983</b>|<b>0.983</b>|<b>0.983</b>|
|shop_emo|0.919|<b>0.92</b>|0.91|
|shop_cat|0.869|<b>0.876</b>|0.854|

#### FastText

|-|Random|Pretrain|Static|
|----|----|----|----|
|ctrip|0.916|<b>0.917</b>|0.808|
|weibo|<b>0.968</b>|0.967|0.84|
|shop_emo|<b>0.919</b>|0.918|0.831|
|shop_cat|0.882|<b>0.883</b>|0.666|

#### DPCNN

|-|Random|Pretrain|Static|
|----|----|----|----|
|ctrip|0.913|<b>0.925</b>|0.922|
|weibo|<b>0.984</b>|<b>0.984</b>|<b>0.984</b>|
|shop_emo|0.93|<b>0.932</b>|0.921|
|shop_cat|0.885|<b>0.893</b>|0.853|

## 长文本分类

### Char-level 效果

#### HAN

|-|Random|Pretrain|Static|
|----|----|----|----|
|sogou_news|<b>0.93</b>|0.926|0.913|
|fudan_news|0.966|<b>0.97</b>|0.964|

### Word-level 效果

#### HAN

|-|Random|Pretrain|Static|
|----|----|----|----|
|sogou_news|0.924|0.922|<b>0.931</b>|
|fudan_news|0.9|0.946|<b>0.965</b>|
