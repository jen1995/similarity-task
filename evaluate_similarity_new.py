# coding: utf-8
import os
import io
import glob

from scipy.stats import spearmanr

from hyperboloid_helpers.manifold import *
from hyperboloid_helpers.analogy import *
from hyperboloid_helpers.fasttext import *

SIMILARITY_DIR = 'eval-word-vectors/data/word-sim/'

DATASETS = [
'EN-MC-30.txt', 
'EN-MEN-TR-3k.txt', 
'EN-SimVerb-3500.txt',
'EN-MTurk-287.txt', 
'EN-WS-353-ALL.txt',
'EN-WS-353-REL.txt',
'EN-WS-353-SIM.txt', 
'EN-SIMLEX-999.txt', 
'EN-YP-130.txt']

def normalize(array, l=2, axis=None, return_norm=False):
    div = np.linalg.norm(array, ord=l, axis=axis, keepdims=True)
    if return_norm:
        nrm = div.copy().squeeze()

    div[np.isclose(div, 0)] = 1.
    if return_norm:
        return array / div, nrm
    return array / div


def evaluate_similarity(vec_file, hyperboloid=False, pseudo_eucl=True):
    """
    Computes the Spearman rank correlation between the groundtruth similarity
    data and the similarities computed from word embeddings in 'vec_file'. If
    'hyperboloid=True', the Minkowski dot product is used to rate the
    similarity, otherwise the Euclidean dot product is used.

    Based on the original evaluation script from
    """
    rhos = []
    num_pairs = []
    if hyperboloid:
        word_vecs = load_minkowski_vectors(vec_file)
    elif pseudo_eucl:
        word_vecs = load_fasttext_vectors(vec_file)
        W = np.array(word_vecs)
        metric_pseudo = np.linalg.inv(W.T @ W)
    else:
        word_vecs = load_fasttext_vectors(vec_file)

        # Normalize word vectors to compute the cosine similarity later.
        word_vecs[:] = normalize(np.asarray(word_vecs), l=2, axis=1)
        W = np.array(word_vecs)
        metric_eucl = np.linalg.inv(W.T @ W)

    print('=================================================================================')
    print("%6s" %"Serial", "%20s" % "Dataset", "%15s" % "Num Pairs", "%15s" % "Not found", "%15s" % "Rho")
    print('=================================================================================')

    for i, filename in enumerate(DATASETS):
        file_minc_result = open(os.path.join(SIMILARITY_DIR, filename + "_minkowski_result.txt"), 'w')
        manual_dict, auto_dict = ({}, {})
        not_found, total_size = (0, 0)

        filepath = os.path.join(SIMILARITY_DIR, filename)

        for line in io.open(filepath, 'r', encoding='utf-8'):
            line = line.strip().lower()
            word1, word2, val = line.split()

            if word1 in word_vecs.index and word2 in word_vecs.index:
                manual_dict[(word1, word2)] = float(val)

                if hyperboloid:
                    # Uses the Minkowski dot product to rate the similarity.
                    auto_dict[(word1, word2)] = \
                        minkowski_dot(np.array(word_vecs.loc[word1]),
                                      np.array(word_vecs.loc[word2]))
                    print("{}  {}  {}".format(word1, word2, auto_dict[(word1, word2)]), file=file_minc_result, end='\n')
                elif pseudo_eucl:
                    auto_dict[(word1, word2)] = \
                        np.sum(np.array(word_vecs.loc[word1]) \
                        * np.array(word_vecs.loc[word2]))
                        # np.array(word_vecs.loc[word1]) @ metric_pseudo @ np.array(word_vecs.loc[word2])

                        
                        # np.array(word_vecs.loc[word1]
                else:
                    # Uses the Euclidean dot product to rate the similarity.
                    auto_dict[(word1, word2)] = \
                        np.array(word_vecs.loc[word1]).dot(np.array(word_vecs.loc[word2]))
                        # np.array(word_vecs.loc[word1]) @ metric_eucl @ np.array(word_vecs.loc[word2])
                        
                        

            else:
                not_found += 1
            total_size += 1  
        file_minc_result.close()

        rho = spearmanr(list(manual_dict.values()), list(auto_dict.values()))[0]
        rhos.append(rho)
        num_pairs.append(max(0, total_size-not_found))

        print("%6s" % str(i+1), "%20s" % filename, "%15s" % str(total_size),
              "%15s" % str(not_found), "%15.4f" % rho)
        
    return rhos, num_pairs


def run_similarity_evaluation(hyperbolic_files, euclidean_files, pseudo_eucl_files):
    """
    Evaluate the word similarity task on 13 similarity datasets for hyperbolic
    and Euclidean embeddings given the lists of file names. 
    Returns two dictionaries mapping file names to average Spearman ranks.
    """
    hyperbolic_sim = {}
    euclidean_sim = {}
    pseudo_eucl_sim = {}

    for f in hyperbolic_files:
        print(f)
        rhos, weights = evaluate_similarity(f, True)
        hyperbolic_sim[f] = np.average(rhos, weights=weights)
        print(hyperbolic_sim[f])
        print('\n\n')

    for f in euclidean_files:
        print(f)
        rhos, weights = evaluate_similarity(f, False, False)
        euclidean_sim[f] = np.average(rhos, weights=weights)
        print(euclidean_sim[f])
        print('\n\n')

    for f in pseudo_eucl_files:
        print(f)
        rhos, weights = evaluate_similarity(f, False, True)
        pseudo_eucl_sim[f] = np.average(rhos, weights=weights)
        print(pseudo_eucl_sim[f])
        print('\n\n')

    with open('results_similarity.txt', 'w') as f:
        for k in hyperbolic_sim:
            f.write('{}\t{}\n'.format(k, hyperbolic_sim[k]))
        for k in euclidean_sim:
            f.write('{}\t{}\n'.format(k, euclidean_sim[k]))
        for k in pseudo_eucl_sim:
            f.write('{}\t{}\n'.format(k, pseudo_eucl_sim[k]))

    return hyperbolic_sim, euclidean_sim, pseudo_eucl_sim


if __name__ == '__main__':
    hyperbolic_files = glob.glob('embeddings_.csv')
    euclidean_files = glob.glob('model_new.vec')
    # pseudo_eucl_files = glob.glob(['model_neg_new.vec', 'model_neg_corrected_new.vec'])
    pseudo_eucl_files = ['model_neg_new.vec', 'model_neg_corrected_new.vec']

    hyperbolic_files.sort()
    euclidean_files.sort()
    pseudo_eucl_files.sort()

    print('Evaluating similarity task.')
    hyperbolic_sim, euclidean_sim, pseudo_euclidean_sim  = run_similarity_evaluation(
                                                              hyperbolic_files=hyperbolic_files,
                                                              euclidean_files=euclidean_files,
                                                              pseudo_eucl_files=pseudo_eucl_files)
