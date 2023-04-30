# Load your usual SpaCy model (one of SpaCy English models)
import spacy
import json
# import pdb

nlp = spacy.load('en_core_web_sm-2.3.0/en_core_web_sm/en_core_web_sm-2.3.0')

# Add neural coref to SpaCy's pipe
import neuralcoref
# conv_dict = json.load(open('./data.coref/entities.json'))
neuralcoref.add_to_pipe(nlp) #conv_dict= conv_dict)

with open('test.txt', 'r', encoding= 'utf-8' )as f:
    text = f.readlines()
    clust = []
    for line in text:
        #pdb.set_trace()
        doc = nlp(line)
        clust.append(doc._.coref_clusters)
     
with open('test.json', 'w', encoding= 'utf-8')as l:
    clust_ = list()
    for i in clust:
        clust__ = dict()
        for j in i:
            key = str(j[0])
            value = [str(j[x]) for x in range(0,len(j))]
            #pdb.set_trace()
            clust__[key] = value
        clust_.append(clust__)
    #pdb.set_trace()
    l.write(json.dumps(clust_))
        
# # You're done. You can now use NeuralCoref as you usually manipulate a SpaCy document annotations.
#         
#   print(doc._.has_coref)
#   print(doc._.coref_clusters)

