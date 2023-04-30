# -*- coding:utf-8 -*-
import os

# Load your usual SpaCy model (one of SpaCy English models)
import spacy
import json
nlp = spacy.load('../en_core_web_sm-2.3.0/en_core_web_sm/en_core_web_sm-2.3.0')

# Add neural coref to SpaCy's pipe
import neuralcoref
neuralcoref.add_to_pipe(nlp)

# You're done. You can now use NeuralCoref as you usually manipulate a SpaCy document annotations.
doc = nlp('i need to find out the date and time for my swimming_activity. i have two which one i have one for the_14th at 6pm and one for the_12th at 7pm')

# print(doc._.has_coref)
# print(doc._.coref_clusters)
# print(doc._.coref_clusters[1].main)
# print(doc._.coref_clusters)
# print(doc._.coref_clusters[1].mentions)
# print(doc._.coref_clusters[1].mentions[-1].start)
# print(doc._.coref_clusters[1].mentions[-1].end)

# print(doc._.coref_clusters[1].mentions[-1]._.coref_cluster.main)
file_path = ['KBRetriever_DC/navigate_train.json', 'KBRetriever_DC/calendar_train.json',
             'KBRetriever_DC/weather_new_train.json', 'KBRetriever_DC/navigate_dev.json',
             'KBRetriever_DC/calendar_dev.json', 'KBRetriever_DC/weather_new_dev.json',
             'KBRetriever_DC/navigate_test.json', 'KBRetriever_DC/calendar_test.json',
             'KBRetriever_DC/weather_new_test.json']

output_path = 'core2adj.json'

SEP = '[SEP] '


def core_parse(history_sentences, last_reponse, nlp=nlp):
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


    doc = nlp(s.replace('[USR]','.').replace('[SYS]','.').replace('[SEP]','.'))
    print(doc._.has_coref)
    print(doc._.coref_clusters)
    print(len(doc._.coref_clusters))
    head_set, tail_set = [], []

    index_tuple_list = []
    for i in range(len(doc._.coref_clusters)):
        index_tuple = []
        for j in range(len(doc._.coref_clusters[i].mentions)):
            if doc._.coref_clusters[i].mentions[j].start < sent_lenth and doc._.coref_clusters[i].mentions[j].end < sent_lenth:
                index_tuple.append((doc._.coref_clusters[i].mentions[j].start,doc._.coref_clusters[i].mentions[j].end))
        index_tuple_list.append(index_tuple)

    for i in range(len(index_tuple_list)):
        for j in range(len(index_tuple_list[i])):
            for k in range(len(index_tuple_list[i])):
                if j != k :
                    head_set.append(index_tuple_list[i][j][0])
                    tail_set.append(index_tuple_list[i][k][0])
                    head_set.append(index_tuple_list[i][j][1])
                    tail_set.append(index_tuple_list[i][k][1])


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

    head_set, tail_set, sent_lenth = core_parse(history_sentences, last_response)

    return [head_set, tail_set], sent_lenth


if __name__ == '__main__':
    Data = dict()
    for path in file_path:
        Data[path] = dict()
        p = os.path.join('../data',path)
        with open(p) as f:
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
