import numpy as np, networkx as nx
from node2vec import Node2Vec
from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
import sys, json, nltk, pickle
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from gensim.models import KeyedVectors as KV

class Node2Vec():
    def __init__(self, input_file, output_file, dimensions=256, walk_length=80, num_walks=20, workers=1, p=0.25, q=1, window_size=15, min_count=1, batch_words=4):
        self.input_file = input_file
        self.output_file = output_file
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        if self.workers > 1:
            print('Warning: parallel computing for over 40K nodes can causes memory issue')
        self.p = p
        self.q = q
        self.window_size = window_size
        self.min_count = min_count
        self.batch_words = batch_words

    def train(self):
        print('input file:', self.input_file, '\noutput file:', self.output_file)
        print('Reading edges...')
        graph = nx.read_edgelist(self.input_file, create_using = nx.Graph())
        sub = graph.subgraph(max(nx.connected_components(graph), key=len))
        print('Number of nodes: ', nx.number_of_nodes(sub))
        print('Number of edges: ', nx.number_of_edges(sub))
        node2vec = Node2Vec(sub, dimensions=self.dimensions, walk_length=self.walk_length, num_walks=self.num_walks, workers=self.workers, p=self.p, q=self.q)
        model = node2vec.fit(window = self.window_size, min_count = self.min_count, batch_words = self.batch_words)
        model.wv.save_word2vec_format(output_file)

class Doc2vec():
    def __init__(self, input_file, output_file, dimensions=256, window_size=15, \
                 min_count=1, sampling_threshold=1e-5, negative_size=5, epochs=100, \
                 dm=1, worker_count=1, alpha=0.025, min_alpha=0.025, seed=1234):
        nltk.download('punkt')
        self.input_file = input_file
        self.output_file = output_file
        self.dimensions = dimensions
        self.window_size = window_size
        self.min_count = min_count
        self.sampling_threshold = sampling_threshold
        self.negative_size = negative_size
        self.epochs = epochs
        self.workers = worker_count
        self.dm = dm
        self.alpha = alpha

        with open(self.input_file, 'r', encoding='UTF8') as f:
            data = json.loads(f.read())
        self.tags = data.keys()
        self.docs = data.values()

        self.sentences = []
        for k in tqdm(data):
            self.sentences.append(TaggedDocument(words=word_tokenize(data[k].lower()), tags=[k]))

    def train(self):
        print('Training doc2vec...')
        d2v_embedder = doc2vec.Doc2Vec(min_count=self.min_count, vector_size=self.dimensions, \
                       alpha=self.alpha, min_alpha=self.min_alpha, seed=self.seed, workers=self.workers)
        d2v_embedder.build_vocab(self.sentences)
        d2v_embedder.train_words = False
        d2v_embedder.train_lbls = True
        d2v_embedder.train(self.sentences, epochs=self.epochs, total_examples=d2v_embedder.corpus_count)
        d2v_embedder.save(self.output_file)

def make_common_index(t_i, i_t, nv, dv):
    nv_keys = list(nv.vocab.keys())
    common_idx = [x for x in tqdm(dv.doctags.keys()) if t_i[x] in nv_keys]
    print('length of common index:', len(common_idx))
    return common_idx

class Hybrid():
    def __init__(self, output_file, alpha, t_i, i_t, nv_file, dv_file):
        self.output_file = output_file
        self.alpha = alpha
        self.t_i = t_i
        self.i_t = i_t
        self.nv = KV.load_word2vec_format(nv_file)
        self.dv = doc2vec.Doc2Vec.load(dv_file).docvecs
        self.common_idx = make_common_index(t_i, i_t, nv, dv)

    def mix_nv_dv(self):
        new_embedding = KV(vector_size = nv.vector_size)
        for key in tqdm(self.common_idx):
            if key not in dv.doctags.keys():
                continue
            new_embedding[key] = alpha * dv[key] + (1-alpha) * nv[t_i[key]]

        new_embedding.save_word2vec_format(self.output_file)
        return new_embedding























#
