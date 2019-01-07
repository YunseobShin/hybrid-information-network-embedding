from WikiParsers import WikiParser, WikiSampler
from Embedders import Node2VecEmbedder, Doc2VecEmbedder, Hybrid
from Evaluations import node_classification
import sys, numpy as np, networkx as nx, argparse, json
from multiprocessing import cpu_count

def main(args):
    # create a wiki parser and save json files
    # wiki_dic.json: {title: raw contents}
    # index_title.json: {index: title}
    # title_index.json: {title:index}
    wiki_parser = WikiParser(args.xml)
    # extract edges between wiki documents in their index
    wiki_parser.catch_link(args.edge)

    process_count = max(1, cpu_count() - 1)

    with open('parsed/wiki_dic.json', 'r', encoding='UTF8') as f:
        wiki_dic = json.loads(f.read())
    with open('parsed/index_title.json', 'r', encoding='UTF8') as f:
        i_t = json.loads(f.read())
    with open('parsed/title_index.json', 'r', encoding='UTF8') as f:
        t_i = json.loads(f.read())

    # sampling
    sample_edge_file = 'sample_edge.txt'
    sampler = WikiSampler(i_t, t_i, edge_file=args.edge, sample_size=args.sample_size, sample_edges_file=sample_edge_file)
    sampler.find_links_from_sample(pool_size=process_count)

    # train node representations with node2vec
    print('Training node2vec...')
    n2v_output = 'embeddings/n2v'
    n2v = Node2VecEmbedder(sample_edge_file, n2v_output)
    n2v.train()

    # train node representations with doc2vec
    print('Training doc2vec...')
    sample = sampler.get_sample()
    title_contents_file = 'parsed/title_contents.json'
    wiki_parser.catch_contents(args.txt, title_contents_file, sample)
    d2v_output = 'embeddings/d2v'
    d2v = Doc2VecEmbedder(title_contents_file, d2v_output)
    d2v.train()

    # combine two embeddings
    print('mixing two embeddings with alpha={}...')
    embedding_output = 'embeddings/hybrid_'+str(args.alpha)
    hybrid = Hybrid(embedding_output, args.alpha, t_i, i_t, n2v_output, d2v_output)
    hybrid.mix_nv_dv()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", help="input wikipedia xml file")
    parser.add_argument("--txt", help="input wikipedia text file")
    parser.add_argument("--edge", help="output edgelist file")
    parser.add_argument("--sample_size", help="size of sample", type=int)
    parser.add_argument("--alpha", help="size of sample", type=float)
    args = parser.parse_args()
    main(args)




#
