import json, sys, re, numpy as np
import networkx as nx, random
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool

def make_dictionaries(pages):
    wiki_dic = {}
    i_t = {}
    t_i = {}

    for page in tqdm(pages):
        title = re.findall("<title>(.*?)</title>", page)
        id = re.findall("<id>(.*?)</id>", page)
        if len(title) > 0:
            title = title[0]
        else:
            continue
        if len(id) > 0:
            id = id[0]
        else:
            continue
        if any(s in title for s in self.exclude):
            continue
        title = title.replace(' ', '_')
        title = title.replace('/', '-')
        wiki_dic[title] = page
        i_t[id] = title
        t_i[title] = id
    return wiki_dic, t_i, i_t


class WikiParser():
    # input_file: raw Wikipedia data
    def __init__(self, input_file):
        with open(input_file, 'r') as f:
            self.wiki = f.read()
        self.pages = self.wiki.split('</pages>')
        print('Parsing title:contents, title:index, and index:title into files...')
        self.wiki_dic, self.t_i, self.i_t = make_dictionaries(self.pages)
        self.exclude = ['Wikipedia:', 'Template:', 'Category:', '.png', '.jpg', '.bmp', \
                       '.gif', '.JGP', '.GIF', '.PNG', '.BMP', '.SWF', '.swf','File:', \
                       'Special:', 'Image:', 'User:', 'User talk:']
        with open('parsed/wiki_dic.json', 'w') as w:
            json.dump(self.wiki_dic, w)
        with open('parsed/index_title.json', 'w') as w:
            json.dump(self.i_t, w)
        with open('parsed/title_index.json', 'w') as w:
            json.dump(self.t_i, w)

    # output_file: edge list
    def catch_link(self, output_file):
        link_pattern = re.compile('\[\[(.*?)\]\]')
        edges = ''

        for title in tqdm(wiki_dic):
            if title in t_i:
                s=t_i[title]
            else:
                break
            page = wiki_dic[title]
            links = link_pattern.findall(page)
            to_delete = []
            for link in links:
                if any(s in link for s in self.exclude):
                    to_delete.append(link)
            for link in to_delete:
                if link in links:
                    links.remove(link)
            for link in links:
                link = link.replace(' ', '_')
                if '|' in link:
                    link = link.split('|')[0]
                if '#' in link:
                    link = link.split('#')[0]
                if len(link) < 1:
                    continue
                if link in t_i:
                    t=t_i[link]
                else:
                    continue
                edges += s + ' ' + t + '\n'

        with open(output_file, 'w') as g:
            g.write(edges)

        return edges

    # output: title:label
    def catch_labels(self, output_file, sample_nodes, start_index=2, end_index=50):
        label_pattern = re.compile('\[\[Category:(.*?)\]\]')
        t_l = {}
        for node in tqdm(sample_nodes):
            node = str(node)
            if node not in i_t:
                continue
            page = self.wiki_dic[self.i_t[node]]
            id = re.findall("<id>(.*?)</id>", page)
            if len(id) > 0:
                id = id[0]
            else:
                continue
            if int(id) in sample_nodes:
                labels = label_pattern.findall(page)
                # print(labels)
                t_l[i_t[node]] = labels

        ls = []
        for v in tqdm(t_l.values()):
            for s in v:
                ls.append(s)

        counts = sorted(Counter(ls).items(), reverse=True, key=lambda kv: kv[1])
        np.save('labels', counts)
        counts = counts[start_index:end_index]
        print(counts)
        counts = np.array(counts)[:,0]
        # print(counts)
        print('labeling nodes...')

        t_l = {}
        for node in tqdm(sample_nodes):
            node = str(node)
            if node not in i_t:
                continue
            title = self.i_t[node]
            page = self.wiki_dic[self.i_t[node]]
            id = re.findall("<id>(.*?)</id>", page)
            if len(id) > 0:
                id = id[0]
            else:
                continue
            if int(id) in sample_nodes:
                labels = label_pattern.findall(page)
                for label in labels:
                    if label in counts:
                        t_l[title] = label
                        break
        print(len(t_l))
        with open(output_file, 'w') as g:
            json.dump(t_l, g)

def take_second(e):
    return e[1]

def sample_by_pagerank(edge_file, sample_size):
    g = nx.read_edgelist(edge_file, create_using=nx.DiGrapg())
    print('number of nodes:{}'.format(g.number_of_nodes))

    print('Calculating PageRanks...')
    tic()
    PR = nx.pagerank(g, alpha=0.9)
    toc()
    tic()
    print('sorting by PageRank...')
    pr_sorted = sorted([[k,v] for k,v in PR.items()], reverse=True, key=take_second)
    pr_sorted = [int(x) for x in np.array(pr_sorted)[:,0]]
    np.save('pageranks', pr_sorted)
    sample = pr_sorted[:int(sample_size*1.5)]
    sample = random.sample(sample, sample_size)
    toc()
    return sample

class WikiSampler():
    def __init__(self, i_t, t_i, edge_file, output_file, sample_size, sample_edges_file):
        self.t_i = t_i
        self.i_t = i_t
        self.edge_file = edge_file
        self.output_file = output_file
        self.sample_size = sample_size
        self.sample = self.sample_by_pagerank(self.edge_file, self.sample_size)
        self.sample_edges_file = sample_edges_file

    def find_links_from_sample(self, pool_size=1)
        with open(edge_file, 'r') as f:
            edges = f.readlines()
        edges = [x.split(' ') for x in tqdm(edges)]
        output = open(self.sample_edges_file, 'w')
        output.close()
        pool = Pool(processes = pool_size)
        tic()
        pool.map(write_sample, edges)
        toc()
        print('Sample edges are saved in ' + self.sample_edges_file)

    def write_sample_parallel(self, E):
        output = open(self.sample_edges_file, 'a')
        sample_nodes = self.sample
        s=int(E[0])
        t=int(E[1])
        # print(s, t)
        if s in sample_nodes and t in sample_nodes:
            output.write(str(s)+' '+str(t)+'\n')
        output.close()










#
