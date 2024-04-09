import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting capabilities

def tsne_plot(model, dimensions: int=2):
    """
    Visualizes word embeddings using t-SNE (t-Distributed Stochastic Neighbor Embedding) 
    to reduce the dimensionality to either 2D or 3D for plotting. This function extracts 
    word vectors from the given model, applies t-SNE to reduce their dimensions, and 
    plots them using matplotlib.

    Parameters:
    - model (gensim.models.keyedvectors.KeyedVectors): A trained word vector model such as Word2Vec, FastText, 
      or any model loaded with gensim that supports the `wv` attribute for accessing word vectors.
    - dimensions (int, optional): The target dimensionality for t-SNE reduction and plotting. 
      Supports 2 or 3 dimensions. Defaults to 2.

    This function automatically adjusts the perplexity value of t-SNE based on the number 
    of samples to ensure optimal layout. For 3D visualization, it makes use of matplotlib's 
    3D plotting capabilities, while for 2D, it uses standard scatter plots. Each point in the 
    plot represents a word, with its position reflecting its relationship to other words based 
    on the model's embeddings.

    Note:
    - The function plots the first `n_samples` words in the model's vocabulary, where `n_samples`
      is determined by the length of the tokens list. Adjusting the number of words processed may be 
      necessary for large models to manage computational and memory resources.
    - It uses PCA initialization for t-SNE to help with faster convergence and more stable results.

    Example Usage:
    >>> from gensim.models import Word2Vec
    >>> model = Word2Vec.load("path_to_your_model")
    >>> tsne_plot(model, dimensions=3)  # For 3D visualization
    >>> tsne_plot(model)  # Defaults to 2D visualization

    """
    labels = []
    tokens = []
    
    # Extracting words and their vectors from the trained model
    for word in model.wv.index_to_key:
        tokens.append(model.wv[word])
        labels.append(word)
        
    n_samples = len(tokens)
    perplexity_value = min(30, n_samples - 1)
    
    # Train t-SNE based on the specified dimensions
    tsne_model = TSNE(perplexity=perplexity_value, n_components=dimensions, init='pca', n_iter=2500, random_state=42)
    new_values = tsne_model.fit_transform(np.array(tokens))
    
    if dimensions == 3:
        # 3D visualization
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111, projection='3d')
        
        for i, value in enumerate(new_values):
            ax.scatter(value[0], value[1], value[2])
            ax.text(value[0], value[1], value[2], labels[i], size=10, zorder=1, color='k')
        
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")
    else:
        # Default to 2D visualization if dimensions are not set to 3
        plt.figure(figsize=(16, 16))
        for i, value in enumerate(new_values):
            plt.scatter(value[0], value[1])
            plt.annotate(labels[i],
                         xy=(value[0], value[1]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")

    plt.show()

def tsne_plot_similar_words(model, labels: list, perplexity: int=15, top_n: int=30):
    """
    Visualizes clusters of similar words from a word embedding model using t-SNE.

    This function takes a pre-trained word embedding model, a list of seed words (labels),
    and visualizes a scatter plot where each point represents a word. Words that are similar
    are clustered together in the plot. t-SNE is used to reduce the dimensionality of the
    word vectors to two dimensions.

    Parameters:
    - model (gensim.models.KeyedVectors): A pre-trained word embedding model from Gensim.
      This model should have the `.wv` attribute to access word vectors.
    - labels (list of str): A list of seed words to visualize along with their most similar words.
      Each seed word from this list will form a cluster in the plot.
    - perplexity (int, optional): The perplexity parameter for the t-SNE model. Perplexity is a
      measure of how to balance attention between local and global aspects of your data. The default
      is 15, but this can be adjusted based on your dataset size and density. The optimal value usually
      lies between 5 and 50.
    - top_n (int, optional): The number of top similar words to fetch for each seed word in `labels`.
      The default value is 30, but this can be adjusted based on the desired granularity of the visualization.

    Returns:
    - This function does not return any value. It directly shows a matplotlib scatter plot.

    Note:
    - The function plots words in 2D space for visual inspection of similarity and clustering.
    - The colors of the clusters are automatically generated to distinguish between different seed words.
    - Adjusting `perplexity` and `top_n` can significantly affect the visualization's quality and interpretability.

    Example Usage:
    >>> from gensim.models import Word2Vec
    >>> model = Word2Vec.load("your_model_path")
    >>> seed_words = ['king', 'computer', 'paris']
    >>> tsne_plot_similar_words(model, seed_words, perplexity=20, top_n=25)
    """
    embedding_clusters = []
    word_clusters = []
    for word in labels:
        similar_words = model.wv.most_similar(word, topn=top_n)
        words = [word for word, _ in similar_words]
        embeddings = [model.wv[word] for word in words]
        
        embedding_clusters.append(embeddings)
        word_clusters.append(words)

    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    tsne_model_en_2d = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=3500, random_state=42)
    embeddings_en_2d = tsne_model_en_2d.fit_transform(embedding_clusters.reshape(-1, k)).reshape(n, m, 2)

    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    
    for label, embeddings, words, color in zip(labels, embeddings_en_2d, word_clusters, colors):
        x, y = embeddings[:, 0], embeddings[:, 1]
        plt.scatter(x, y, c=[color], alpha=0.7, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', fontsize=8)
            
    plt.legend(loc='best')
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()
