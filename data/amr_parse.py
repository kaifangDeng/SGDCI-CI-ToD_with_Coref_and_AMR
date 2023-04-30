# -*- coding:utf-8 -*-
import re
import spacy
import penman
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import nltk
# nltk.download('wordnet')
nlp = spacy.load('../en_core_web_sm-2.3.0/en_core_web_sm/en_core_web_sm-2.3.0')
nltk.download('omw-1.4')
import string
import pickle as pkl


class AMR_Node():
    def __init__(self):
        self.index = -1                 # word index in the sentence
        self.next_node_relation = []    # the relation of this node and its next node
        self.content = ''               # word
        self.real = False               # is this word real exist in the sentence


def read_amr_file(amr_graph_lines):
    # lines = f.readlines()
    amr = ''                            # store amr forms sentences
    article = []                        # store the whole article as a list of amr sentences
    sentence = []                       # store the original sentences
    for line in amr_graph_lines:
        if line[:7] == '# ::snt':     # used to get the original sentences
            if amr != '':
                article.append(amr)
            amr = ''
            sentence.append(line[8:])
        elif line[0] != '#':            # get amr sentences
            amr = amr + line

    article.append(amr)
    return article, sentence


def load_amr(article):                  # use penman to parser amr sentences
    sentence_node_list = []
    for amr_sentence in article:
        amr_sentence = amr_sentence.replace('\n', '')
        g = penman.decode(amr_sentence)
        sentence_node_list.append(g)    # get all the arm sentences as penman graph
    return sentence_node_list

def load_sentence(sent_list):
    # lines = []
    # for line in raw_file.readlines():
    #     # print(line)
    #     for c in string.punctuation:
    #         line = line.replace(c, '')
    #     lines.append(line)
    word_list = []
    sentence_idx = []
    # article_word = ''
    i = 0
    for line in sent_list:
        i = i + 1
        line = line.replace('\n', '')
        line = line.replace('[','')
        line = line.replace(']','')
        word_line = line.split(' ')
        for word in word_line:
            if word != '':
                word_list.append(word)
                sentence_idx.append(i)
                # article_word = article_word + word + ' '
    # print(word_list)
    # print(sentence_idx)
    return word_list, sentence_idx

def construct_amr_node(article, id, src_node, tgt_node, node_type, edge_type, word_list, sentence_idx):  # construct all the words as AMR_Node
    sentence_node_list = load_amr(article)
    sentence_num = 0
    amr_graph = []                      # store all those words nodes
    dummy_index = id + len(word_list)

    dummy_node_list = []
    for sentence_node in sentence_node_list:        # for every sentence in the article
        sentence_num += 1
        sentence_total = ''
        sentence_list = []
        for i in range(len(sentence_idx)):
            if sentence_idx[i] == sentence_num:
                sentence_total += word_list[i] + ' '
                sentence_list.append(word_list[i])
        sentence_total += '\n'

        nlp_sentence = nlp(sentence_total)
        nlp_list = []

        for i in range(len(sentence_total.split(' '))):
            nlp_list.append(nlp_sentence[i].lemma_)

        if nlp_list[-1] != '\n':
            nlp_list = []
            for word in sentence_list:
                nlp_word = lemmatizer.lemmatize(word)
                nlp_list.append(nlp_word)

        sentence_graph = []             # this sentence graph

        for word in sentence_list:
            id += 1
            node = AMR_Node()
            node.content = word
            node.index = id
            sentence_graph.append(node)
        instance_map = {}

        for instance in sentence_node.instances():
            if instance.target == None:
                break
            content_instance = re.sub(r'[0-9-]', '', instance.target)
            if content_instance in nlp_list:
                index = nlp_list.index(content_instance)
                sentence_graph[index].real = True
                instance_map[instance.source] = sentence_graph[index].index
            else:
                dummy_index += 1
                node = AMR_Node()
                node.content = content_instance
                node.real = False
                dummy_node_list.append(node)
                instance_map[instance.source] = dummy_index
        for edges in sentence_node.edges():
            src_node.append(instance_map[edges.source])
            tgt_node.append(instance_map[edges.target])
            edge_type.append(20)
        amr_graph.append(sentence_graph)            # [[sentence 1 word amr node],...,[sentence N word amr node]]
    for sentence_node_list in amr_graph:
        for sentence_node in sentence_node_list:
            if sentence_node.real == True:
                node_type.append(39)
            else:
                node_type.append(40)
    for _ in dummy_node_list:
        node_type.append(41)
    return src_node, tgt_node, node_type, edge_type, amr_graph


if __name__ == '__main__':
    id = -1
    f = open('output.txt')
    article, sentence = read_amr_file(f)
    raw_f = open('output1.txt', 'r')
    word_list, sentence_idx = load_sentence(raw_f)
    # print(word_list)
    src_node, tgt_node, node_type, edge_type, amr_graph = construct_amr_node(article, id, [], [], [], [], word_list, sentence_idx)
    with open('amr.out.pkl', 'wb') as f:
        pkl.dump((word_list, src_node, tgt_node), f)
    # print(src_node)
    # print(tgt_node)

