import os
import random
import codecs
import logging
import jieba
from torch.utils.data import Dataset

TOKEN_PAD = '[PAD]'
TOKEN_UNK = '[UNK]'
TOKEN_CLS = '[CLS]'
TOKEN_SEP = '[SEP]'
TOKEN_EDGES_START = '<s>'
TOKEN_EDGES_END = '</s>'


class TextDataset(Dataset):
    def __init__(self, data_path, char_or_word, tokenizer, max_sent_len=0,
                 do_pad=False, pad_token=TOKEN_PAD, do_to_id=False,
                 add_bigram=False, bigram_tokenizer=None,
                 for_bert=False, do_sort=False, debug=False):
        super(TextDataset, self).__init__()
        self.data = []

        with codecs.open(data_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = line.split('\t')
                if len(line) != 2:
                    continue

                sent = []
                if char_or_word == 'char':
                    sent = [ch for ch in line[1]]
                elif char_or_word == 'word':
                    sent = line[1].split()

                sent_len = len(sent)
                sent_len = max_sent_len if ((max_sent_len > 0) and (sent_len > max_sent_len)) else sent_len

                if for_bert:
                    sent = [TOKEN_CLS] + sent

                if max_sent_len > 0:
                    sent = sent[:max_sent_len]

                bigram = []
                if add_bigram:
                    bigram = [TOKEN_EDGES_START] + sent + [TOKEN_EDGES_END]
                    bigram = [[bigram[i - 1] + bigram[i]] + [bigram[i] + bigram[i + 1]] for i in
                              range(1, len(bigram) - 1)]

                if do_pad and (max_sent_len > 0):
                    mask = [1] * len(sent) + [0] * (max_sent_len - len(sent))

                    if add_bigram:
                        bigram = bigram + [[pad_token, pad_token]] * (max_sent_len - len(sent))

                    sent = sent + [pad_token] * (max_sent_len - len(sent))
                else:
                    mask = [1] * len(sent)

                if do_to_id:
                    sent = tokenizer.convert_tokens_to_ids(sent)

                    if add_bigram:
                        bigram = bigram_tokenizer.convert_tokens_to_ids(bigram)

                example = [sent, int(line[0]), mask, sent_len]
                if add_bigram:
                    example.append(bigram)

                self.data.append(example)

                if debug:
                    if len(self.data) >= 10:
                        break

        if do_sort:
            self.data = sorted(self.data, key=lambda x: x[3], reverse=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def dbc_to_sbc(ustring):
        rstring = ''
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x3000:
                inside_code = 0x0020
            else:
                inside_code -= 0xfee0
            if not (0x0021 <= inside_code and inside_code <= 0x7e):
                rstring += uchar
                continue
            rstring += chr(inside_code)
        return rstring

    @staticmethod
    def save_data(filepath, train_sents, test_sents, postfix):
        logging.info('writing data to: %s' % filepath)
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        with codecs.open('%s/train_%s.txt' % (filepath, postfix), 'w', 'utf-8') as fout:
            for sent in train_sents:
                fout.write('%s\t%s\n' % (sent[0], sent[1]))
        with codecs.open('%s/test_%s.txt' % (filepath, postfix), 'w', 'utf-8') as fout:
            for sent in test_sents:
                fout.write('%s\t%s\n' % (sent[0], sent[1]))

    @staticmethod
    def statistics(datapath):
        sent_len_dict = {128: 0, 192: 0, 208: 0, 300: 0, 400: 0, 500: 0, 1000: 0, 1500: 0, 2000: 0}

        data_granularity = ['char', 'word']
        data_split = ['train', 'test']
        for dg in data_granularity:
            logging.info('%s level data statistics' % dg)
            for ds in data_split:
                with codecs.open('%s/%s/%s.txt' % (datapath, dg, ds), 'r', 'utf-8') as fin:
                    for line in fin:
                        line = line.strip()
                        if line == '':
                            continue

                        line = line.split('\t')
                        if len(line) != 2:
                            continue

                        line = line[1]

                        if dg == 'char':
                            sent_len = len(line)
                        elif dg == 'word':
                            sent_len = len(line.split())
                        for sl in sent_len_dict:
                            if sent_len <= sl:
                                sent_len_dict[sl] = sent_len_dict[sl] + 1
                                break

                logging.info('%s data length: %s' % (ds, str(sent_len_dict)))

                for sl in sent_len_dict:
                    sent_len_dict[sl] = 0


class DocDataset(Dataset):
    def __init__(self, data_path, char_or_word, tokenizer, max_sent_len=0, max_sent_num=0,
                 do_sent_pad=False, pad_token=TOKEN_PAD, do_to_id=False,
                 do_sort=False, debug=False):
        super(DocDataset, self).__init__()
        self.data = []

        label_folders = os.listdir(data_path)
        for label in label_folders:
            file_list = os.listdir('%s/%s' % (data_path, label))
            for filename in file_list:
                sents = []
                masks = []
                with codecs.open('%s/%s/%s' % (data_path, label, filename), 'r', 'utf-8') as fin:
                    for line in fin:
                        line = line.strip()
                        if line == '':
                            continue

                        sent = []
                        if char_or_word == 'char':
                            sent = [ch for ch in line]
                        elif char_or_word == 'word':
                            sent = line.split()

                        if max_sent_len > 0:
                            sent = sent[:max_sent_len]

                        if do_sent_pad and (max_sent_len > 0):
                            mask = [1] * len(sent) + [0] * (max_sent_len - len(sent))
                            sent = sent + [pad_token] * (max_sent_len - len(sent))
                        else:
                            mask = [1] * len(sent)

                        if do_to_id:
                            sent = tokenizer.convert_tokens_to_ids(sent)

                        sents.append(sent)
                        masks.append(mask)

                        if (max_sent_num > 0) and (len(sents) >= max_sent_num):
                            break

                if len(sents) > 0:
                    example = [int(label), sents, masks, len(sents)]
                    self.data.append(example)

                if debug:
                    if len(self.data) >= 10:
                        break

        if do_sort:
            self.data = sorted(self.data, key=lambda x: x[3], reverse=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def statistics(datapath):
        sent_len_dict = {128: 0, 256: 0, 320: 0, 384: 0, 400: 0, 512: 0, 1024: 0, 2048: 0}
        doc_len_dict = {128: 0, 256: 0, 320: 0, 384: 0, 400: 0, 512: 0, 1024: 0, 2048: 0}

        data_granularity = ['char', 'word']
        data_split = ['train', 'test']
        for dg in data_granularity:
            logging.info('%s level data statistics' % dg)
            for ds in data_split:
                cla_list = os.listdir('%s/%s/%s' % (datapath, dg, ds))
                for cla in cla_list:
                    file_list = os.listdir('%s/%s/%s/%s' % (datapath, dg, ds, cla))
                    for filename in file_list:
                        doc_len = 0
                        with codecs.open('%s/%s/%s/%s/%s' % (datapath, dg, ds, cla, filename), 'r', 'utf-8') as fin:
                            for line in fin:
                                line = line.strip()
                                if line == '':
                                    continue

                                if dg == 'char':
                                    sent_len = len(line)
                                elif dg == 'word':
                                    sent_len = len(line.split())

                                for sl in sent_len_dict:
                                    if sent_len <= sl:
                                        sent_len_dict[sl] = sent_len_dict[sl] + 1
                                        break

                                doc_len += 1

                        for dl in doc_len_dict:
                            if doc_len <= dl:
                                doc_len_dict[dl] = doc_len_dict[dl] + 1
                                break

                logging.info('%s data sent length: %s' % (ds, str(sent_len_dict)))
                logging.info('%s data document length: %s' % (ds, str(doc_len_dict)))

                for sl in sent_len_dict:
                    sent_len_dict[sl] = 0
                for dl in doc_len_dict:
                    doc_len_dict[dl] = 0


class CtripDataset:
    @staticmethod
    def transform(src_path, tgt_path, ratio=0.2):
        pos_chars = []
        pos_words = []
        neg_chars = []
        neg_words = []

        with codecs.open('%s/pos.txt' % src_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = TextDataset.dbc_to_sbc(line)
                words = list(jieba.cut(line))

                pos_chars.append(line)
                pos_words.append(words)

        with codecs.open('%s/neg.txt' % src_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = TextDataset.dbc_to_sbc(line)
                words = list(jieba.cut(line))

                neg_chars.append(line)
                neg_words.append(words)

        pos_test_idx = random.sample(range(0, len(pos_chars)), int(len(pos_chars) * ratio))
        neg_test_idx = random.sample(range(0, len(neg_chars)), int(len(neg_chars) * ratio))

        train_sents_char = []
        train_sents_word = []
        test_sents_char = []
        test_sents_word = []

        for i in range(len(pos_chars)):
            if i in pos_test_idx:
                test_sents_char.append([1, pos_chars[i]])
                test_sents_word.append([1, ' '.join(pos_words[i])])
            else:
                train_sents_char.append([1, pos_chars[i]])
                train_sents_word.append([1, ' '.join(pos_words[i])])

        for i in range(len(neg_chars)):
            if i in neg_test_idx:
                test_sents_char.append([0, neg_chars[i]])
                test_sents_word.append([0, ' '.join(neg_words[i])])
            else:
                train_sents_char.append([0, neg_chars[i]])
                train_sents_word.append([0, ' '.join(neg_words[i])])

        logging.info('num of train sents: %s, test sents: %s' % (len(train_sents_char), len(test_sents_char)))
        TextDataset.save_data(tgt_path, train_sents_char, test_sents_char, 'char')
        TextDataset.save_data(tgt_path, train_sents_word, test_sents_word, 'word')
        logging.info('complete transform ctrip dataset')


class WeiboDataset:
    @staticmethod
    def transform(src_path, tgt_path, ratio=0.1):
        pos_chars = []
        pos_words = []
        neg_chars = []
        neg_words = []

        with codecs.open(src_path, 'r', 'utf-8') as fin:
            _ = fin.readline()
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = TextDataset.dbc_to_sbc(line)
                split_idx = line.find(',')
                label = int(line[:split_idx])
                content = line[split_idx + 1:]
                words = list(jieba.cut(content))

                if label == 1:
                    pos_chars.append(content)
                    pos_words.append(words)
                else:
                    neg_chars.append(content)
                    neg_words.append(words)

        pos_test_idx = random.sample(range(0, len(pos_chars)), int(len(pos_chars) * ratio))
        neg_test_idx = random.sample(range(0, len(neg_chars)), int(len(neg_chars) * ratio))

        train_sents_char = []
        train_sents_word = []
        test_sents_char = []
        test_sents_word = []

        for i in range(len(pos_chars)):
            if i in pos_test_idx:
                test_sents_char.append([1, pos_chars[i]])
                test_sents_word.append([1, ' '.join(pos_words[i])])
            else:
                train_sents_char.append([1, pos_chars[i]])
                train_sents_word.append([1, ' '.join(pos_words[i])])

        for i in range(len(neg_chars)):
            if i in neg_test_idx:
                test_sents_char.append([0, neg_chars[i]])
                test_sents_word.append([0, ' '.join(neg_words[i])])
            else:
                train_sents_char.append([0, neg_chars[i]])
                train_sents_word.append([0, ' '.join(neg_words[i])])

        logging.info('num of train sents: %s, test sents: %s' % (len(train_sents_char), len(test_sents_char)))
        TextDataset.save_data(tgt_path, train_sents_char, test_sents_char, 'char')
        TextDataset.save_data(tgt_path, train_sents_word, test_sents_word, 'word')
        logging.info('complete transform weibo dataset')


class ShoppingDataset:
    @staticmethod
    def transform_emotion(src_path, tgt_path, ratio=0.1):
        pos_chars = []
        pos_words = []
        neg_chars = []
        neg_words = []

        with codecs.open(src_path, 'r', 'utf-8') as fin:
            _ = fin.readline()
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = TextDataset.dbc_to_sbc(line)
                split_idx = line.find(',')
                line = line[split_idx + 1:]
                split_idx = line.find(',')
                label = int(line[:split_idx])
                content = line[split_idx + 1:]
                words = list(jieba.cut(content))

                if label == 1:
                    pos_chars.append(content)
                    pos_words.append(words)
                else:
                    neg_chars.append(content)
                    neg_words.append(words)

        pos_test_idx = random.sample(range(0, len(pos_chars)), int(len(pos_chars) * ratio))
        neg_test_idx = random.sample(range(0, len(neg_chars)), int(len(neg_chars) * ratio))

        train_sents_char = []
        train_sents_word = []
        test_sents_char = []
        test_sents_word = []

        for i in range(len(pos_chars)):
            if i in pos_test_idx:
                test_sents_char.append([1, pos_chars[i]])
                test_sents_word.append([1, ' '.join(pos_words[i])])
            else:
                train_sents_char.append([1, pos_chars[i]])
                train_sents_word.append([1, ' '.join(pos_words[i])])

        for i in range(len(neg_chars)):
            if i in neg_test_idx:
                test_sents_char.append([0, neg_chars[i]])
                test_sents_word.append([0, ' '.join(neg_words[i])])
            else:
                train_sents_char.append([0, neg_chars[i]])
                train_sents_word.append([0, ' '.join(neg_words[i])])

        logging.info('num of train sents: %s, test sents: %s' % (len(train_sents_char), len(test_sents_char)))
        TextDataset.save_data(tgt_path, train_sents_char, test_sents_char, 'char')
        TextDataset.save_data(tgt_path, train_sents_word, test_sents_word, 'word')
        logging.info('complete transform shopping dataset')

    @staticmethod
    def transform_category(src_path, tgt_path, ratio=0.1):
        data = {}
        labels = set()

        with codecs.open(src_path, 'r', 'utf-8') as fin:
            _ = fin.readline()
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = TextDataset.dbc_to_sbc(line)
                split_idx = line.find(',')
                label = line[:split_idx]
                line = line[split_idx + 1:]
                split_idx = line.find(',')
                content = line[split_idx + 1:]

                labels.add(label)
                if label in data:
                    data[label].append(content)
                else:
                    data[label] = [content]

        test_idx = {}
        label_dict = {}
        for cat in data:
            test_idx[cat] = random.sample(range(0, len(data[cat])), int(len(data[cat]) * ratio))
        for label in labels:
            label_dict[label] = len(label_dict)

        train_sents_char = []
        train_sents_word = []
        test_sents_char = []
        test_sents_word = []

        for cat in data:
            cat_data = data[cat]
            cat_test_idx = test_idx[cat]
            for i in range(len(cat_data)):
                words = list(jieba.cut(cat_data[i]))
                if i in cat_test_idx:
                    test_sents_char.append([label_dict[cat], cat_data[i]])
                    test_sents_word.append([label_dict[cat], ' '.join(words)])
                else:
                    train_sents_char.append([label_dict[cat], cat_data[i]])
                    train_sents_word.append([label_dict[cat], ' '.join(words)])

        logging.info('num of train sents: %s, test sents: %s' % (len(train_sents_char), len(test_sents_char)))
        TextDataset.save_data(tgt_path, train_sents_char, test_sents_char, 'char')
        TextDataset.save_data(tgt_path, train_sents_word, test_sents_word, 'word')

        with codecs.open('%s/tags.txt' % tgt_path, 'w', 'utf-8') as fout:
            for label in label_dict:
                fout.write('%s\t%s\n' % (label_dict[label], label))

        logging.info('complete transform shopping dataset')


class SogouNewsDataset:
    @staticmethod
    def transform(src_path, tgt_path, ratio=0.1):
        labels = os.listdir(src_path)
        label_dict = {}
        for label in labels:
            label_dict[label] = len(label_dict)

        logging.info('writing data to: %s' % tgt_path)
        if not os.path.exists(tgt_path):
            os.makedirs(tgt_path)

        with codecs.open('%s/tags.txt' % tgt_path, 'w', 'utf-8') as fout:
            for label in label_dict:
                fout.write('%s\t%s\n' % (label_dict[label], label))

        for label in labels:
            docs = os.listdir('%s/%s' % (src_path, label))
            test_idx = random.sample(range(0, len(docs)), int(len(docs) * ratio))

            for doc_idx in range(len(docs)):
                doc_name = docs[doc_idx]
                paragraphs = []
                with codecs.open('%s/%s/%s' % (src_path, label, doc_name), 'r', 'gbk', errors='ignore') as fin:
                    for line in fin:
                        line = line.strip()
                        if line == '':
                            continue

                        line = TextDataset.dbc_to_sbc(line)
                        paragraphs.append(line)

                doc_char_folder = '%s/char/%s/%s' % (
                    tgt_path, 'test' if (doc_idx in test_idx) else 'train', label_dict[label])
                if not os.path.exists(doc_char_folder):
                    os.makedirs(doc_char_folder)
                with codecs.open('%s/%s' % (doc_char_folder, doc_name), 'w', 'utf-8') as fout:
                    for line in paragraphs:
                        fout.write('%s\n' % line)

                doc_word_folder = '%s/word/%s/%s' % (
                    tgt_path, 'test' if (doc_idx in test_idx) else 'train', label_dict[label])
                if not os.path.exists(doc_word_folder):
                    os.makedirs(doc_word_folder)
                with codecs.open('%s/%s' % (doc_word_folder, doc_name), 'w', 'utf-8') as fout:
                    for line in paragraphs:
                        words = list(jieba.cut(line))
                        line = ' '.join(words)
                        fout.write('%s\n' % line)

        logging.info('complete transform sogou news dataset')


class FudanNewsDataset:
    @staticmethod
    def transform(src_path, tgt_path, ratio=0.1):
        labels = os.listdir(src_path)
        label_dict = {}
        for label in labels:
            label_dict[label] = len(label_dict)

        logging.info('writing data to: %s' % tgt_path)
        if not os.path.exists(tgt_path):
            os.makedirs(tgt_path)

        with codecs.open('%s/tags.txt' % tgt_path, 'w', 'utf-8') as fout:
            for label in label_dict:
                fout.write('%s\t%s\n' % (label_dict[label], label))

        for label in labels:
            docs = os.listdir('%s/%s' % (src_path, label))
            test_idx = random.sample(range(0, len(docs)), int(len(docs) * ratio))

            for doc_idx in range(len(docs)):
                doc_name = docs[doc_idx]
                paragraphs = []
                with codecs.open('%s/%s/%s' % (src_path, label, doc_name), 'r', 'gbk', errors='ignore') as fin:
                    for line in fin:
                        line = line.strip()
                        if line == '':
                            continue

                        line = TextDataset.dbc_to_sbc(line)
                        paragraphs.append(line)

                doc_char_folder = '%s/char/%s/%s' % (
                    tgt_path, 'test' if (doc_idx in test_idx) else 'train', label_dict[label])
                if not os.path.exists(doc_char_folder):
                    os.makedirs(doc_char_folder)
                with codecs.open('%s/%s' % (doc_char_folder, doc_name), 'w', 'utf-8') as fout:
                    for line in paragraphs:
                        fout.write('%s\n' % line)

                doc_word_folder = '%s/word/%s/%s' % (
                    tgt_path, 'test' if (doc_idx in test_idx) else 'train', label_dict[label])
                if not os.path.exists(doc_word_folder):
                    os.makedirs(doc_word_folder)
                with codecs.open('%s/%s' % (doc_word_folder, doc_name), 'w', 'utf-8') as fout:
                    for line in paragraphs:
                        words = list(jieba.cut(line))
                        line = ' '.join(words)
                        fout.write('%s\n' % line)

        logging.info('complete transform fudan news dataset')


class Tokenizer:
    def __init__(self, token_to_id):
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}

    def convert_tokens_to_ids(self, tokens, unk_token=TOKEN_UNK):
        ids = []
        for token in tokens:
            if isinstance(token, str):
                ids.append(self.token_to_id.get(token, self.token_to_id[unk_token]))
            else:
                ids.append([self.token_to_id.get(t, self.token_to_id[unk_token]) for t in token])
        return ids

    def convert_ids_to_tokens(self, ids, max_sent_len):
        tokens = [self.id_to_token[i] for i in ids]
        if max_sent_len > 0:
            tokens = tokens[:max_sent_len]
        return tokens


def load_tag_dict(file_path):
    tag_to_id = {}
    with codecs.open(file_path, 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip()
            if line != '':
                line = line.split('\t')
                tag_to_id[line[1]] = int(line[0])

    id_to_tag = {v: k for k, v in tag_to_id.items()}
    return tag_to_id, id_to_tag


def load_pretrain_embedding(filepath, has_meta=False,
                            add_pad=False, pad_token=TOKEN_PAD, add_unk=False, unk_token=TOKEN_UNK, debug=False):
    with codecs.open(filepath, 'r', 'utf-8', errors='ignore') as fin:
        token_to_id = {}
        embed = []

        if has_meta:
            meta_info = fin.readline().strip().split()

        first_line = fin.readline().strip().split()
        embed_size = len(first_line) - 1

        if add_pad:
            token_to_id[pad_token] = len(token_to_id)
            embed.append([0.] * embed_size)

        if add_unk:
            token_to_id[unk_token] = len(token_to_id)
            embed.append([0.] * embed_size)

        token_to_id[first_line[0]] = len(token_to_id)
        embed.append([float(x) for x in first_line[1:]])

        for line in fin:
            line = line.split()

            if len(line) != embed_size + 1:
                continue
            if line[0] in token_to_id:
                continue

            token_to_id[line[0]] = len(token_to_id)
            embed.append([float(x) for x in line[1:]])

            if debug:
                if len(embed) >= 1000:
                    break

    return token_to_id, embed


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    # TextDataset.statistics('../data/datasets/ctrip')
    # TextDataset.statistics('../data/datasets/weibo')
    # TextDataset.statistics('../data/datasets/shop_emo')
    # TextDataset.statistics('../data/datasets/shop_cat')

    # DocDataset.statistics('../data/datasets/fudan_news')
    # DocDataset.statistics('../data/datasets/sogou_news')
