import os
import logging
from argparse import ArgumentParser
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from textcnn.model import TextCNN
from utils.utils import setup_seed
from utils.dataloader import Tokenizer, TextDataset, load_pretrain_embedding, load_tag_dict
from utils.logger import Logger
from utils import modelloader, evaluator


def get_dataset(args, tokenizer):
    train_dataset = TextDataset('%s/%s/%s/train.txt' % (args.data_path, args.task, args.use_char_or_word),
                                args.use_char_or_word, tokenizer, args.max_inp_size,
                                do_pad=True, do_to_id=True, debug=args.debug)
    test_dataset = TextDataset('%s/%s/%s/test.txt' % (args.data_path, args.task, args.use_char_or_word),
                               args.use_char_or_word, tokenizer, args.max_inp_size,
                               do_pad=True, do_to_id=True, debug=args.debug)

    return train_dataset, test_dataset


def data_collate_fn(data):
    data = np.array(data)

    sents = torch.LongTensor(np.array(data[:, 0].tolist()))
    tags = torch.LongTensor(np.array(data[:, 1].tolist()))

    return sents, tags


def train(args, dataset, dataloader, model, optimizer, lr_scheduler):
    model.train()
    loss_sum = 0
    for batch, data in enumerate(dataloader):
        optimizer.zero_grad()
        sents, tags = data

        sents = sents.cpu() if args.use_cpu else sents.cuda()
        tags = tags.cpu() if args.use_cpu else tags.cuda()

        loss = model(sents, decode=False, tags=tags)
        loss = loss.mean()

        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    loss_sum = loss_sum / len(dataset)
    return loss_sum


def evaluate(args, dataloader, model):
    pred_answers = []
    gold_answers = []

    model.eval()

    if args.multi_gpu:
        model = model.module

    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            sents, tags = data

            sents = sents.cpu() if args.use_cpu else sents.cuda()
            tags = tags.cpu() if args.use_cpu else tags.cuda()

            preds = model(sents)
            tags = tags.cpu().numpy()

            pred_answers.extend(preds)
            gold_answers.extend(tags)

    acc, pre, rec, f1 = evaluator.evaluate(gold_answers, pred_answers)
    return acc, pre, rec, f1


def main(args):
    if args.debug:
        args.batch_size = 3

    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    if args.multi_gpu:
        logging.info("run on multi GPU")
        torch.distributed.init_process_group(backend="nccl")

    setup_seed(0)

    output_path = '%s/%s/%s_%s' % (args.output_path, args.task, args.use_char_or_word, args.embed_type)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_logger = Logger(data_path=output_path)

    token_to_id, pretrain_embed = load_pretrain_embedding(args.pretrained_emb_path,
                                                          has_meta=True if (args.use_char_or_word == 'word') else False,
                                                          add_pad=True, add_unk=True, debug=args.debug)
    tokenizer = Tokenizer(token_to_id)
    tag_to_id, _ = load_tag_dict('%s/%s/tags.txt' % (args.data_path, args.task))

    logging.info("loading dataset")
    train_dataset, test_dataset = get_dataset(args, tokenizer)

    if args.multi_gpu:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                      sampler=DistributedSampler(train_dataset))
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                     sampler=DistributedSampler(test_dataset))
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                      shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                     shuffle=False)

    best_f1 = 0
    epoch = 0

    if args.pretrained_model_path is not None:
        logging.info("loading pretrained model")
        model, optimizer, epoch, best_f1 = modelloader.load(args.pretrained_model_path)
        model = model.cpu() if args.use_cpu else model.cuda()
    else:
        logging.info("creating model")
        model = TextCNN(len(tag_to_id), len(token_to_id), args.embed_size, args.kernel_sizes, args.kernel_num,
                        args.input_dropout_rate, args.hidden_dropout_rate, args.embed_type)
        model = model.cpu() if args.use_cpu else model.cuda()

        if args.embed_type in ['pretrain', 'static']:
            model.init_embedding(np.array(pretrain_embed))
        elif args.embed_type == 'multi':
            model.init_embedding_ext(np.array(pretrain_embed))

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.multi_gpu:
        model = DistributedDataParallel(model, find_unused_parameters=True)

    num_train_steps = int(len(train_dataset) / args.batch_size * args.epoch_size)
    num_warmup_steps = int(num_train_steps * args.lr_warmup_proportion)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_warmup_steps, gamma=args.lr_decay_gamma)

    logging.info("begin training")
    while epoch < args.epoch_size:
        epoch += 1

        train_loss = train(args, train_dataset, train_dataloader, model, optimizer, lr_scheduler)

        test_acc, test_pre, test_rec, test_f1 = evaluate(args, test_dataloader, model)

        logging.info('epoch[%s/%s], train loss: %s' % (epoch, args.epoch_size, train_loss))
        logging.info('epoch[%s/%s], test accuracy: %s, precision: %s, recall: %s, f1: %s' % (
            epoch, args.epoch_size, test_acc, test_pre, test_rec, test_f1))
        modelloader.save(output_path, 'last.pth', model, optimizer, epoch, test_f1)

        remark = ''
        if test_f1 > best_f1:
            best_f1 = test_f1
            remark = 'best'
            modelloader.save(output_path, 'best.pth', model, optimizer, epoch, best_f1)

        model_logger.write(epoch, train_loss, test_acc, test_pre, test_rec, test_f1, remark)

    logging.info("complete training")
    model_logger.draw_plot()


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--task', dest='task',
                        default='ctrip')
    parser.add_argument('--data_path', dest='data_path',
                        default='../data/datasets/')
    parser.add_argument('--pretrained_emb_path', dest='pretrained_emb_path',
                        help='gigaword_chn.all.a2b.uni.ite50.vec, news_tensite.pku.words.w2v50',
                        default='../data/embeddings/gigaword_chn.all.a2b.uni.ite50.vec')
    parser.add_argument('--pretrained_model_path', dest='pretrained_model_path',
                        default=None)
    parser.add_argument('--output_path', dest='output_path',
                        default='../runtime/textcnn/')
    parser.add_argument('--use_char_or_word', dest='use_char_or_word',
                        default='char')
    parser.add_argument('--embed_type', dest='embed_type', type=str,
                        help='rand,pretrain,static,multi',
                        default='rand')
    parser.add_argument('--embed_size', dest='embed_size', type=int,
                        default=50)
    parser.add_argument('--kernel_sizes', dest='kernel_sizes', type=str,
                        default='3,4,5')
    parser.add_argument('--kernel_num', dest='kernel_num', type=int,
                        default=100)
    parser.add_argument('--input_dropout_rate', dest='input_dropout_rate', type=float,
                        default=0.5)
    parser.add_argument('--hidden_dropout_rate', dest='hidden_dropout_rate', type=float,
                        default=0.5)
    parser.add_argument('--max_inp_size', dest='max_inp_size', type=int,
                        default=512)
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=32)
    parser.add_argument('--epoch_size', dest='epoch_size', type=int,
                        default=50)
    parser.add_argument('--learning_rate', dest='learning_rate', type=float,
                        default=0.0005)
    parser.add_argument('--lr_warmup_proportion', dest='lr_warmup_proportion', type=float,
                        default=0.1)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', type=float,
                        default=0.9)
    parser.add_argument('--use_cpu', dest='use_cpu', type=bool,
                        default=False)
    parser.add_argument('--multi_gpu', dest='multi_gpu', type=bool, help='run with: -m torch.distributed.launch',
                        default=True)
    parser.add_argument('--local_rank', dest='local_rank', type=int,
                        default=0)
    parser.add_argument('--debug', dest='debug', type=bool,
                        default=False)

    args = parser.parse_args()

    main(args)
