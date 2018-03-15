import codecs
from collections import defaultdict
import node2vec

def read(file_path):
    with codecs.open(file_path,'r','utf-8',errors='ignore') as read_f:
        for line in read_f:
            yield line.strip()

import networkx as nx
Gdata=nx.Graph()

glinks=defaultdict(set)
entities=set()
for line in read('./serialized/train'):
    h,r,t=list(map(int,line.strip().split('\t')))
    glinks[t].add(h)
    glinks[h].add(t)
    entities.add(h)
    entities.add(t)
    Gdata.add_edge(h,t)
    Gdata.add_edge(t,h)

for edge in Gdata.edges():
	Gdata[edge[0]][edge[1]]['weight'] = 1
Gdata = Gdata.to_undirected()

# for i in Gdata.neighbors(20974):
# 	print(i)

G=node2vec.Graph(Gdata,True,10,1)
G.preprocess_transition_probs()
walks = G.simulate_walks(1, 16)
print(walks)





