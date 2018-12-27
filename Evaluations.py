import numpy as np
from sklearn.svm import SVC
import sys, json
from gensim.models import KeyedVectors
from tqdm import tqdm
from time_check import tic
from time_check import toc

def training_SVM(train_x, train_y):
    clf = SVC(gamma='auto', decision_function_shape='ovo', kernel='rbf')
    print('Traning SVM...')
    clf.fit(train_x, train_y)
    return clf

# input_file: embeddings from Hybrid embedder
def node_classification(input_file, t_i, i_t, t_l, common_idx):
    embeddings = KeyedVectors.load_word2vec_format(input_file)
    data = []
    for title in t_l:
        if title not in common_idx:
            continue
        data.append([embeddings[title], t_l[title]])

    data = np.array(data)
    labels = set(list(t_l.values()))
    data_labels = []
    for label in labels:
        data_labels.append(data[np.where(data[:,1]==label)])

    data_labels = np.array(data_labels)

    train = []
    test = []

    for data_label in data_labels:
        train.append(data_label[:int(len(data_label)*0.8)])
        test.append(data_label[int(len(data_label)*0.8):])

    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(len(train)):
        if train[i].shape[0] < 10:
            continue
        for t in train[i]:
            train_x.append(t[0])
            train_y.append(t[1])
        for t in test[i]:
            test_x.append(t[0])
            test_y.append(t[1])

    np.save('./features/train_x_'+str(alpha), train_x)
    np.save('./features/train_y_'+str(alpha), train_y)

    label_index = {}
    i=0
    for data_label in data_labels:
        if len(data_label) < 1:
            # print(data_label)
            continue
        label = data_label[0][1]
        if label not in set(test_y):
            continue
        else:
            label_index[label] = i
            i += 1

    q=[]
    for ty in test_y:
        q.append(label_index[ty])
    test_y = q

    q=[]
    for ty in train_y:
        q.append(label_index[ty])

    train_y = [float(x) for x in q]
    print('training data size:', len(train_y))
    print('the number of classes:', len(set(label_index)))

    model = training_SVM(train_x, train_y)
    acc = model.score(test_x, test_y)
    print('Testing Acc alpha:'+str(alpha)+':'+str(acc))

def link_prediction(input_file, sample_edges_file, t_i, i_t, t_l, common_idx):
    g = nx.read_edgelist(sample_edges_file, create_using=nx.Graph())
    embeddings = KeyedVectors.load_word2vec_format(input_file)
    accs = []

    for node in tqdm(common_idx):
        neis = [i_t[x] for x in list(g.neighbors(t_i[node]))]
        rec = np.array(embeddings.most_similar(node, topn=len(neis)))[:,0]
        hits = len(set(neis) & set(rec))
        accs.append(hits/len(neis))

    accuracy = np.mean(accs)
    print('Accuracy: (alpha='+str(alpha)+'): '+str(accuracy))



























#
