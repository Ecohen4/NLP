import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

def print_topic_top_words(model, feature_names, n):
    '''
    param
    -----
    model            fitted sklearn model (NMF or LDA)
    feature_names    list, e.g. tfidf_vectorizer.get_feature_names()
    n                int

    return
    ------
    topic_words     list of top n words for each topic

    actions
    -------
    print to terminal
    '''
    topic_words = []
    for i, topic_vector in enumerate(model.components_):
        ii = np.argsort(topic_vector)[-n - 1:-1]
        words = [feature_names[i] for i in ii]
        topic_words.append(words)
        out = f'topic {i}: '
        wordstr = ' '.join([x for x in words])
        print(out + wordstr)
    print()

    return topic_words

def pca_plot_3axes(arr, savefig=False, filename=None):
    '''
    for PCA results: plot pairwise components for first 3 components
    3 subplots. Points are labeled as integers.

    param
    -----
    arr         np array, ax0 = features, ax1 = components
    savefig     bool, to save figure
    filename    str, relative path and extension
    '''
    fig, axs = plt.subplots(1,3, figsize=(12,4))
    ax_ii = [(0,1),(1,2),(0,2)]
    for ax, ii in zip(axs, ax_ii):
        xx = arr[:,ii[0]]
        yy = arr[:,ii[1]]
        labels = range(len(xx))

        ax.scatter(xx, yy, s=2)
        for i, x in enumerate(labels):
            ax.text(xx[i], yy[i], x, size=20)

        ax.set_xlabel(f'principal component {ii[0]+1}')
        ax.set_ylabel(f'principal component {ii[1]+1}')

    plt.tight_layout()
    if savefig:
        plt.savefig(filename, dpi=300)
