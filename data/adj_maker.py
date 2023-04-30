# -*- coding:utf-8 -*-
import json
import amrlib
import os

from amr_parse import load_amr, load_sentence, read_amr_file, construct_amr_node
stog = amrlib.load_stog_model()

SEP = '[SEP] '

file_path = ['KBRetriever_DC/navigate_train.json', 'KBRetriever_DC/calendar_train.json',
             'KBRetriever_DC/weather_new_train.json', 'KBRetriever_DC/navigate_dev.json',
             'KBRetriever_DC/calendar_dev.json', 'KBRetriever_DC/weather_new_dev.json',
             'KBRetriever_DC/navigate_test.json', 'KBRetriever_DC/calendar_test.json',
             'KBRetriever_DC/weather_new_test.json']

output_path = 'amr2adj.json'


def amr_parse(history_sentences, last_reponse, stog=stog):
    sent_list = []
    sent_lenth = 0
    w = []
    s = ''
    for i, sentence in enumerate(history_sentences):
        if i % 2 == 0:
            s += "[USR] " + sentence + ' '
            sent_list.append("[USR] " + sentence + ' ')
            # sent_lenth += len(s.replace('  ',' ').split(' '))
            # w.extend(s.replace('  ',' ').split(' '))
        else:
            s += "[SYS] " + sentence + ' '
            sent_list.append("[SYS] " + sentence + ' ')
            # sent_lenth += len(s.replace('  ',' ').split(' '))
            # w.extend(s.replace('  ',' ').split(' '))
    s += SEP + last_reponse
    sent_list.append(SEP + last_reponse)
    sent_lenth += len(s.replace('  ',' ').split(' '))
    w.extend(s.replace('  ',' ').split(' '))

    graphs = stog.parse_sents(sent_list)


    xxx = []
    for sss in graphs:
        xxx.extend(sss.split('\n'))

    artic, sente =  read_amr_file(xxx)
    word_list, sentence_idx = load_sentence(sent_list)
    src_node, tgt_node, node_type, edge_type, amr_graph \
        = construct_amr_node(artic, -1, [], [], [], [], word_list, sentence_idx)
    head_set, tail_set = [], []
    for i in range(len(src_node)):
        if src_node[i] < sent_lenth and tgt_node[i] < sent_lenth:
            head_set.extend([src_node[i], tgt_node[i]])
            tail_set.extend([tgt_node[i], src_node[i]])

    return head_set, tail_set, sent_lenth


def get_info(dialogue_components_item, domain):
    """
    Transfer a dialogue item from the data
    :param dialogue_components_item: a dialogue(id, dialogue, kb, (qi,hi,kbi)) from data file (json item)
    :param domain: the domain of the data file
    :return: constructed_info: the constructed info which concat the info and format as
    the PhD. Qin mentioned.
            consistency: (qi,hi,kbi)
    """
    dialogue = dialogue_components_item["dialogue"]

    sentences = []
    history_sentences = []
    last_response = ''
    for speak_turn in dialogue:
        sentences.append(speak_turn["utterance"])
    if len(sentences) % 2 == 0:
        history_sentences.extend(sentences[:-1])
        last_response = sentences[-1]
    else:
        history_sentences.extend(sentences)

    head_set, tail_set, sent_lenth = amr_parse(history_sentences, last_response)

    return [head_set, tail_set], sent_lenth


if __name__ == '__main__':
    Data = dict()
    for path in file_path:
        Data[path] = dict()
        with open(path) as f:
            raw_data = json.load(f)
            domain = os.path.split(path)[-1].split("_")[0]

        for dialogue_components_item in raw_data:


            adj, lenth  = get_info(dialogue_components_item,domain)


            Data[path][dialogue_components_item['id']] = dict()
            Data[path][dialogue_components_item['id']]['adj'] = adj
            Data[path][dialogue_components_item['id']]['lenth'] = lenth

            # if i > 2:
            #     break
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([Data], f, indent=2, ensure_ascii=False, )


