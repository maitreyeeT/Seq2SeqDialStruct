import keras
#from variational_layer import VariationalLayer
import numpy as np
import pandas as pd

#from plan_library import PUCRSDatasetGenerator
from collections import deque
import re
import matplotlib.pyplot as plt
from Seq2SeqVAE_dialogueStructuring.vae_LSTM import LSTMVAutoencoder
from Seq2SeqVAE_dialogueStructuring.ae_LSTM import LSTMAutoencoder
from keras.preprocessing.text import Tokenizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import FastICA, PCA
import seaborn as sns
from sklearn.neighbors import kneighbors_graph
import mpl_toolkits.mplot3d.axes3d as p3

import datetime


l = keras.layers
m = keras.models
K = keras.backend
opt = keras.optimizers
losses = keras.losses
cbk = keras.callbacks
reg = keras.regularizers




def check_pattern(q_pattern, comm_func, sender):
    pattern = q_pattern.popleft()
    p_comm_func = '_'.join([comm_func, sender])#, syntax])
    if re.match(pattern, p_comm_func):
        return True
    q_pattern.appendleft(pattern)
    return False


def convert_dataset(size_ngram, pattern=()):
    dir_path = '/withDP.csv'
    data5 = pd.read_csv(dir_path, sep='\t')
    np.random.seed(100)
    dataset = list(zip(
        *[data5['Comfun1'], data5['sender']#,
          ,data5['dialog_policy'],
          data5['dialog_CA'],
         # data5['interjection'].astype(str).str.replace('nan', 'notavailable').str.strip(' '),
          data5['subobjvrb'].astype(str).str.replace('nan', 'notavailable')
          #data5['aux_keywords'].astype(str).str.replace('nan', 'notavailable').str.strip(' ')
          ]))
    labels = list(data5['dial_type'])
    #dataset = list(zip(*[data5['Comfun1'], data5['sender'],
                        #  data5['interjection'].astype(str).str.replace('nan',
                         #                                               'notavailable')
     #                     ]))

    comm_funcs, sender, DP, CA, sov = \
        list(zip(*dataset)) #,, intj, sov, aux
    comm_funcs_dict = list(set(comm_funcs))
    sender_dict = list(set(sender))
    DP_dict = list(set(DP))
    CA_dict = list(set(CA))
  #  aux_dict = list(set(aux))
    sov_dict = list(set(sov))
    def get_one_hot(x, dictionary):
        arr = np.zeros((len(dictionary)))
        arr[dictionary.index(x)] = 1
        return arr

    stack_array = lambda array: \
        np.vstack(list(map
                       (lambda x:
                        np.expand_dims(x, 0)
                        ,array)))


    rows_patterns = []

    encoded_dataset = []
    ngrams_corpus = []
    for i in range(len(dataset) - size_ngram + 1):
        ngram = dataset[i:i + size_ngram]
        ngrams_corpus += [list(ngram)]
        ngram_array = []

        q_patterns = deque(pattern)

        for comm_func, s, dp, ca, sov in ngram:#, dp, ca, in ngram: , intj, aux, sov in ngram
            comm_func_arr = get_one_hot(comm_func, comm_funcs_dict)
            arr_sender = get_one_hot(s, sender_dict)
            arr_DP = get_one_hot(dp, DP_dict)
            arr_CA = get_one_hot(ca, CA_dict)
         #   arr_aux = get_one_hot(aux, aux_dict)
         #   arr_intj = get_one_hot(intj, intj_dict)
            arr_sov = get_one_hot(sov, sov_dict)
            cat_array = np.hstack([comm_func_arr, arr_sender,
                                   arr_DP, arr_CA, arr_sov])#])#, ])#, arr_SOV])#arr_syntax1, arr_syntax2, arr_syntax3])
            ngram_array.append(cat_array)
            #if len(q_patterns) > 0:
             #   check_pattern(q_patterns, comm_funcs, sender, DP, CA, sov)

        #if len(q_patterns) == 0:
         #   rows_patterns.append(i)

        ngram_array = stack_array(ngram_array)
        encoded_dataset.append(ngram_array)

    dataset_array = stack_array(encoded_dataset)

    #rows_patterns = dataset_array[rows_patterns, ...] if len(rows_patterns) > 0 else None

    #rows_patterns = rows_patterns.reshape(-1, rows_patterns.shape[-1], rows_patterns.shape[-2])

    return labels[:-size_ngram+1], \
           ngrams_corpus, dataset_array, \
           np.array(rows_patterns)

# find out the mean value for our weighted input.

def plot_latent(model, X, corpus, cluster_color):

    Y = model.predict(X)
    tsne_model = TSNE(perplexity=50)

    ica = FastICA()
    ica_fit = ica.fit_transform(Y)
    pca = PCA()
    pca_fit = pca.fit_transform(ica_fit)
    Y_ts = tsne_model.fit_transform(Y)
    pca_ts = tsne_model.fit_transform(pca_fit)

    knn_graphpca = kneighbors_graph\
        (pca_fit,20, include_self = False)
    knn_graphY = kneighbors_graph\
        (Y, 20, include_self=False)

   # AggloIca = \
    #    AgglomerativeClustering\
     #       (n_clusters=4,
      #       connectivity=knn_graphica,
       #      linkage='ward').fit(ica_fit)
    AggloPca = \
        AgglomerativeClustering\
            (n_clusters=None,affinity='euclidean',
            connectivity=knn_graphpca,
            linkage='ward',
             distance_threshold=0).fit(pca_fit)

    AggloY = \
        AgglomerativeClustering\
            (n_clusters=None,affinity='euclidean',
             linkage='ward',
             connectivity=knn_graphY,
             distance_threshold=0).fit(Y)
             #
             #linkage='ward')

    #Hclustering_Birch = Birch()

    #distY = euclidean_distances(Y)

    distpca = euclidean_distances(pca_fit)

    #labelPca = hirclusterPca.predict(fit_pca)

    hirclusterPca = AggloPca.fit(distpca)

   # def plot_dendrogram(model, **kwargs):
        # Create linkage matrix and then plot the dendrogram

        # create the counts of samples under each node
    #    counts = np.zeros(model.children_.shape[0])
     #   n_samples = len(model.labels_)
      #  for i, merge in enumerate(model.children_):
       #     current_count = 0
        #    for child_idx in merge:
         #       if child_idx < n_samples:
          #          current_count += 1  # leaf node
           #     else:
            #        current_count +=\
             #         counts[child_idx - n_samples]
            #counts[i] = current_count

       # linkage_matrix = np.column_stack(
        #    [model.children_, model.distances_,counts]).astype(float)
       # dataFrm = pd.DataFrame(linkage_matrix, columns=['child1',
        #                                                'child2',
         #                                               'distance',
          #                                              'count'])
       # print(dataFrm.head())
       # dataFrm.to_csv('/home/maitreyee/Development'
        #               '/autoencoder/'
         #              'dendo_AutoEncod.csv'
          #             , sep='\t')
        # Plot the corresponding dendrogram
        #dendrogram(linkage_matrix, **kwargs)

    #def dendo_plotICA(Agglo):

        #### ica dendogram
     #   plt.figure(figsize=(10, 8))
     #   plt.title(
      #      'Dendogram Agglomerative Clustering'
       #     'for ICA')
       # plot_dendrogram(Agglo, truncate_mode='level',
        #           p=3)#, leaf_rotation=45,
                   #leaf_font_size=15,
                   #show_contracted=True)
   # dendo_plotICA(AggloIca)

  #  def scat_ica(icaAglo):
   #     plt.title(
    #        'Agglomerative Clustering'
     #       'for ICA')
        #cmp = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
       # sns.scatterplot(data = icaAglo,#[:, 0], y=icaAglo[:, 1],
        #                hue=icaAglo.labels_, palette=cmp)
      #  sns.scatterplot(x=ica_fit[:,0],
       #                 y =ica_fit[:,1],
        #            hue=icaAglo.labels_)
       # plt.show()
  #  scat_ica(AggloIca)

    #def dendo_plotPca(Agglox):

        ####pca clustering and dendrogram

     #   plt.figure(figsize=(10, 8))
      #  plt.title(
       #     'Dendogram Agglomerative Clustering'
        #    'for PCA')
   # plot_dendrogram(AggloY)#, truncate_mode='level', p=3)#, leaf_rotation=45,
                   #leaf_font_size=15,
                   #show_contracted=True)
    #dendo_plotPca(AggloPca)


    def scat_pca(pcaAgglo):

        plt.title(
            'Agglomerative Clustering'
            'for PCA')

        #sns.scatterplot(data = hir_pca,#[:, 0], y=hir_pca[:, 1],
         #           hue=hir_pca.labels_)
        sns.scatterplot(x=pca_fit[:,0], y=pca_fit[:,1],
                    hue=pcaAgglo.labels_)
        plt.show()

    #scat_pca(AggloPca)

    def tsne_plot(corp,y,pca):
        "Creates TSNE model and plots it"
        forplt = []
        for word in corp:
            forplt.append\
                (' '.join
                 (['-'.join(p) for p in word]))
        Y_tsne, pca_tsne = y,pca

        tokenize = Tokenizer()
        tokenize.fit_on_texts(cluster_color)
        cls = np.array(tokenize.texts_to_sequences(cluster_color))\
            .astype('int').reshape(-1)
        #print(cls.shape)
        fig,ax= plt.subplots()
        plt.title(
            'tSNE Plot of the latent space for Baseline Autoencoder')
        ax.scatter(Y_tsne[:, 0], Y_tsne[:, 1], s=5, c = cls,
                   cmap='tab10')
        for i, txt in enumerate(forplt):
            ax.annotate(txt, (Y_tsne[i, 0], Y_tsne[i, 1]))
        #sns.scatterplot(Y_tsne[:, 0],
         #               Y_tsne[:, 1],
          #              hue =cluster_color,
          #              palette='hls', s=30)
        plt.show()



        #plt.title(
         #   'tSNE Plot of the '
         #   'latent space for baseline Autoencoder fitted on PCA')
        #ax.scatter(Y_ts[:, 0], Y_ts[:, 1], s=5, c = cls,
         #         cmap='tab10')
        #sns.scatterplot(pca_tsne[:, 0],
         #               pca_tsne[:, 1],
          #             hue=cluster_color,
           #             palette="Set2", s= 30)



        #ax.legend()
        plt.show()

    tsne_plot(corpus, Y_ts, pca_ts)

    def plot_connectICA(data,algo):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        ax.view_init(7, -80)
        label = algo.labels_
        for l in np.unique(label):
            ax.scatter(data[label == l, 0],
                       data[label == l, 1],
                       data[label == l, 2],
                       color=plt.
                       cm.jet(float(l)
                        / np.max(label + 1)),
                       s=40, edgecolor='k')

        plt.show()

    def plot_connectY(dataY, algoY):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        ax.view_init(7, -80)
        label = algoY.labels_
        for l in np.unique(label):
          ax.scatter(dataY[label == l, 0],
                       dataY[label == l, 1],
                       dataY[label == l, 2],
                       color=plt.
                       cm.jet(float(l)
                              / np.max(label + 1)),
                       s=40, edgecolor='k')

        plt.legend()
        plt.show()

    #plot_connectY(Y, AggloY)
    #plot_connectICA(pca_fit, AggloPca)




if __name__ == '__main__':

#    patterns = [('question.*'), ('answer.*'), ('inform.*')]

    dial_type, corpus, X, \
    P = convert_dataset(5)

    _, sequences_length, \
    window_size  = X.shape
    pva = LSTMVAutoencoder(1e-3,
                         window_size,
                         sequences_length)#,dropout=.05

    pa = LSTMAutoencoder(1e-3,
                         window_size,
                         sequences_length)#,dropout=.05

    def plot_callback(logs=None):#e,
       #if e % 100 == 0:
      plot_latent(pa.model_latent,
              X, corpus, dial_type)



    log_dir = "/home/maitreyee" \
              "/Development/autoencoder/logs/pa/" \
              + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = cbk.\
      TensorBoard(log_dir=log_dir, histogram_freq=1)

    pa.fit(X=X,
           epochs=1000,
           callbacks=[cbk.LambdaCallback
                      (on_train_end=plot_callback),
                      tensorboard_callback,#,
],#cbk.EarlyStopping(patience=.0005),
           verbose=1,
           )





