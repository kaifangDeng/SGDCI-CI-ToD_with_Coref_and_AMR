# -*- coding:utf-8 -*-
# -*- coding:utf-8 -*-
import logging
import pprint
import random
import fitlog
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel, AdamW

import models.KBRetriever_DC.base
from models.KBRetriever_DC.GCNLayer import GraphConvolution,HighWay



import utils.tool



class BERTTool(object):
    def init(args):
        BERTTool.bert = BertModel.from_pretrained(args.bert.location, return_dict=False)
        BERTTool.tokenizer = BertTokenizer.from_pretrained(args.bert.location)
        BERTTool.pad = BERTTool.tokenizer.pad_token
        BERTTool.sep = BERTTool.tokenizer.sep_token
        BERTTool.cls = BERTTool.tokenizer.cls_token
        BERTTool.pad_id = BERTTool.tokenizer.pad_token_id
        BERTTool.sep_id = BERTTool.tokenizer.sep_token_id
        BERTTool.cls_id = BERTTool.tokenizer.cls_token_id
        BERTTool.special_tokens = ["[SOK]", "[EOK]", "[SOR]", "[EOR]", "[USR]", "[SYS]", '[MASK1]', '[MASK2]', '[MASK3]']
        # SOK: start of knowledge base
        # EOK: end of knowledge base
        # SOR: start of row
        # EOR: end of row
        # USR: start of user turn
        # SYS: start of system turn


class Model(models.KBRetriever_DC.base.Model):
    def __init__(self, args, DatasetTool, EvaluateTool, inputs):
        np.random.seed(args.train.seed)
        torch.manual_seed(args.train.seed)
        random.seed(args.train.seed)
        super().__init__(args, DatasetTool, EvaluateTool, inputs)
        _, _, _, entities = inputs
        BERTTool.init(self.args)
        self.bert = BERTTool.bert
        self.tokenizer = BERTTool.tokenizer

        special_tokens_dict = {'additional_special_tokens': BERTTool.special_tokens+entities}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.bert.resize_token_embeddings(len(self.tokenizer))

        self.hidden_size = 768

        # GCN
        self.num_layers = 3

        self.gcn = nn.ModuleList()
        for i in range(self.num_layers):
            layer = GraphConvolution(in_features=self.hidden_size,
                                     out_features=self.hidden_size,
                                     edge_types=5, dropout=0.5 if i != self.num_layers - 1 else None,
                                     device=self.device)
            self.gcn.append(layer)

        # HighWay
        self.highway = nn.ModuleList()
        for i in range(self.num_layers):
            hw = HighWay(size=self.hidden_size, dropout_ratio=0.5)
            self.highway.append(hw)

        self.w_hi = nn.Linear(768 * 2, 2)
        self.w_qi = nn.Linear(768 * 2, 2)
        self.w_kbi = nn.Linear(768 * 2, 2)
        self.criterion = nn.BCELoss()

    def set_optimizer(self):
        all_params = set(self.parameters())
        params = [{"params": list(all_params), "lr": self.args.lr.bert}]
        self.optimizer = AdamW(params)

    def run_eval(self, train, dev, test):
        logging.info("Starting evaluation")
        self.load("saved/best_model.pkl")
        self.eval()
        summary = {}
        ds = {"test": test}
        for set_name, dataset in ds.items():
            tmp_summary, pred = self.run_test(dataset)
            self.DatasetTool.record(pred, dataset, set_name, self.args)
            summary.update({"eval_{}_{}".format(set_name, k): v for k, v in tmp_summary.items()})
        logging.info(pprint.pformat(summary))

    def run_train(self, train, dev, test):
        self.set_optimizer()
        iteration = 0
        best = {}
        for epoch in range(self.args.train.epoch):
            self.train()
            logging.info("Starting training epoch {}".format(epoch))
            summary = self.get_summary(epoch, iteration)
            loss, iter = self.run_batches(train, epoch)
            fitlog.add_loss({"train_loss": loss}, step=epoch)
            iteration += iter
            summary.update({"loss": loss})
            ds = {"train": train, "dev": dev, "test": test}
            if not self.args.train.not_eval:
                for set_name, dataset in ds.items():
                    tmp_summary, pred = self.run_test(dataset)
                    self.DatasetTool.record(pred, dataset, set_name, self.args)
                    summary.update({"eval_{}_{}".format(set_name, k): v for k, v in tmp_summary.items()})
                    fitlog.add_metric({"eval_{}_{}".format(set_name, k): v for k, v in tmp_summary.items()}, step=epoch)
            best = self.update_best(best, summary, epoch)
            logging.info(pprint.pformat(best))
            logging.info(pprint.pformat(summary))

    def piece2word(self, constructed_info, last_response, kb_lenth):

        self.batch_size = len(constructed_info)

        piece2word_list, w_lenth_list, t_lenth_list, token_list = [], [], [], []
        sen1, sen2 = [], []
        utt_adj_list = []
        for xx in range(self.batch_size):
            sentence1 =  constructed_info[xx].split(' ')
            sentence1.remove('')
            word_list = ['[CLS]'] + sentence1 + ['[SEP]']
            # word_list.remove([])
            kb_lenth_true = word_list.index('[USR]')
            # if kb_lenth_true != kb_lenth[xx] + 1:
            #     print('error')

            sentence2 = last_response[xx].split(' ')

            word_list += sentence2 + ['[SEP]']
            # word_list.remove('')
            w = np.array(word_list)
            #
            utt_adj = []
            for char in ['[CLS]','[USR]','[SYS]','[SEP]']:
                ww = np.where(w == char)
                utt_adj.extend(ww[0].tolist())

            sen1.append(sentence1)
            sen2.append(sentence2)


            w_lenth = len(word_list)
            # t_lenth = len(pieces)
            #
            mm = np.max(utt_adj)
            assert mm < w_lenth, "index error"
            utt_adj_list.append(utt_adj)

            w_lenth_list.append(w_lenth)
            # t_lenth_list.append(t_lenth)

        # token_2d = pad_sequence(token_list, True)
        w_max = np.max(w_lenth_list)
        # t_max = np.max(t_lenth_list)

        def fill(data, new_data):
            for j, x in enumerate(data):
                x = torch.tensor(x)
                new_data[j, :x.shape[0], :x.shape[1]] = x
            return new_data

        # sub_mat = torch.zeros((self.batch_size, w_max, t_max), dtype=torch.bool)
        # pieces2word = fill(piece2word_list, sub_mat)

        return  w_max, sen1, sen2, w_lenth_list, utt_adj_list

    def get_adj(self, adj_list, core_adj_list, utt_adj_list , w_max, kb_lenth , w_lenth_list ):
        adj_4d = torch.zeros((self.batch_size, 5 , w_max, w_max),dtype=torch.bool)
        for i in range(self.batch_size):
            for j in range(len(adj_list[i][0])):
                if adj_list[i][0][j] + 1 < w_max and adj_list[i][1][j]  + 1 < w_max:
                    adj_4d[i, 0 ,adj_list[i][0][j]  + 1, adj_list[i][1][j] + 1] = 1

            for k in range(w_lenth_list[i]):
                adj_4d[i, 1, k, k] = 1

            for z in range(len(core_adj_list[i][0])):
                if core_adj_list[i][0][z]  + 1 < w_max and core_adj_list[i][1][z]  + 1 < w_max:
                    adj_4d[i, 2 ,core_adj_list[i][0][z]  + 1, core_adj_list[i][1][z]  + 1] = 1

            for q in utt_adj_list[i]:
                for p in utt_adj_list[i]:
                    if p != q:
                        adj_4d[i, 4, q, p] = 1

            for m in range(0, w_lenth_list[i]):
                for n in range(max(m-1,0), min(w_lenth_list[i],m+2)):
                    if m != n:
                        adj_4d[i, 3, m, n] = 1


        return adj_4d



    def get_info(self, batch):
        construced_infos = [item['constructed_info'][1] for item in batch]
        kb_list = [item['constructed_info'][0] for item in batch]
        knowbase = [[],[]]
        for kb in kb_list:
            knowbase[0].extend(kb)
            # 索引
            knowbase[1].append(len(kb))
        for i in range(len(knowbase[1])):
            if i == 0:
                pass
            else:
                knowbase[1][i] += knowbase[1][i-1]
        last_responses = [item['last_response'] for item in batch]
        # sentence_list = [item['constructed_info'] + '[SEP] ' + item['last_response'] for item in batch]
        kb_lenth = [item['kb_lenth'] for item in batch]
        w_max, sen1, sen2, w_lenth_list, utt_adj_list= self.piece2word(construced_infos, last_responses, kb_lenth)
        amr_adj_list = [item['adj'] for item in batch]
        core_adj_list = [item['core_adj'] for item in batch]
        adj_4d = self.get_adj(amr_adj_list,core_adj_list,utt_adj_list, w_max, kb_lenth, w_lenth_list)

        tokenized = self.tokenizer(sen1, sen2, truncation='only_first', padding=True, return_tensors='pt', max_length = self.tokenizer.max_model_input_sizes['bert-base-uncased'], return_token_type_ids=True)
        tokenized = tokenized.data
        tokenized1 = self.tokenizer(knowbase[0], truncation='only_first', padding=True, return_tensors='pt',
                                   max_length=self.tokenizer.max_model_input_sizes['bert-base-uncased'],
                                   return_token_type_ids=True)
        tokenized1 = tokenized1.data
        return tokenized['input_ids'].to(self.device), tokenized['token_type_ids'].to(self.device), tokenized[
            'attention_mask'].to(self.device), w_lenth_list, (tokenized1['input_ids'].to(self.device), \
            tokenized1['token_type_ids'].to(self.device), tokenized1[
            'attention_mask'].to(self.device)), knowbase[1], adj_4d

    def map(self, kb_index, kb_cls):
        max_len = np.max(kb_index)
        m = torch.zeros(len(kb_index),max_len, kb_cls.size(-1)).to(self.device)
        mask = torch.zeros(len(kb_index),max_len).to(self.device)
        for i in range(len(kb_index)):
            if i == 0:
                kb_rep = kb_cls[:kb_index[i],:]
            else:
                kb_rep = kb_cls[kb_index[i-1]:kb_index[i],:]

            j = len(kb_rep)
            mask[i][:j] = 1
            m[i, :j, :] = kb_rep
            # for j in range(len(kb_rep)):
            #     mask[i][j] = 1
            #     for z in range(len(kb_rep[j])):
            #         m[i][j][z] = kb_rep[j][z]


        return m, mask



    def forward(self, batch):
        token_ids, type_ids, mask_ids, w_lenth_list, kb_inputs, kb_index, adj_4d = self.get_info(batch)
        h, utt = self.bert(input_ids=token_ids, token_type_ids=type_ids, attention_mask=mask_ids)

        kb_h, kb_cls = self.bert(input_ids=kb_inputs[0], token_type_ids=kb_inputs[1], attention_mask=kb_inputs[2])

        kb_batch, kb_mask = self.map(kb_index,kb_cls)
        kb_batch_sum = torch.max(kb_batch,dim=1)[0]

        # gcn gcn_layers:3
        for i in range(self.num_layers):
            gcn_output = self.gcn[i](h, adj_4d.to(self.device)) + self.highway[i](h)


        cls = gcn_output[:,0,:]
        # cls = utt

        utt = torch.cat((cls, kb_batch_sum),dim=-1)

        out_qi = self.w_qi(utt)
        out_hi = self.w_hi(utt)
        out_kbi = self.w_kbi(utt)
        loss = torch.Tensor([0])
        if self.training:
            loss =  F.cross_entropy(out_qi,
                                   torch.Tensor(utils.tool.in_each(batch, lambda x: x["consistency"][0])).long().to(
                                       self.device)) \
                   + F.cross_entropy(out_hi,
                                     torch.Tensor(utils.tool.in_each(batch, lambda x: x["consistency"][1])).long().to(
                                         self.device)) \
                   + F.cross_entropy(out_kbi,
                                     torch.Tensor(utils.tool.in_each(batch, lambda x: x["consistency"][2])).long().to(
                                         self.device))

        out = []
        for qi, hi, kbi in zip(out_qi, out_hi, out_kbi):
            out.append([qi.argmax().data.tolist(), hi.argmax().data.tolist(), kbi.argmax().data.tolist()])

        return loss, out

    def load(self, file):
        logging.info("Loading models from {}".format(file))
        state = torch.load(file)
        model_state = state["models"]
        self.load_state_dict(model_state)

    def start(self, inputs):
        train, dev, test, _ = inputs
        if self.args.model.resume is not None:
            self.load(self.args.model.resume)
        if not self.args.model.test:
            self.run_train(train, dev, test)
        self.run_eval(train, dev, test)
