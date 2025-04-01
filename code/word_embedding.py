import numpy as np
import sklearn
from staticvectors import StaticVectors # Library for handling static word embeddings (like Word2Vec, GloVe)

# Import custom preprocessing functions
from preprocessing import downsample_word_vectors, make_delayed

# %% Define a Bag-of-Words (BoW) model class
class BagofWords:
    """
    Creates a Bag-of-Words representation for text data.
    It builds a vocabulary based on word frequencies and transforms text
    into one-hot encoded vectors.
    """
    def __init__(self):
        """Initializes the BagofWords model with an empty vocabulary."""
        self.word_to_index = None # Dictionary to map words to unique integer indices

    def fit(self, texts):
        """
        Builds the vocabulary from a list of texts (list of lists of words).

        Args:
            texts (list): A list where each element is a list of words (a document/text).

        Returns:
            self: The fitted BagofWords instance.
        """
        word_freq = {} # Dictionary to store word frequencies
        # Count word frequencies across all texts
        for text in texts:
            for word in text:
                word = word.lower().strip() # Normalize word (lowercase, remove whitespace)
                if word not in word_freq:
                    word_freq[word] = 0
                word_freq[word] += 1

        # Build the word_to_index mapping, ignoring words that occur only once
        self.word_to_index = {}
        for text in texts:
            for word in text:
                word = word.lower().strip()
                # Add word to vocabulary only if it appears more than once
                if word_freq[word] > 1 and word not in self.word_to_index:
                    self.word_to_index[word] = len(self.word_to_index) # Assign next available index

        return self # Return the fitted object

    def transform(self, text):
        """
        Transforms a single text (list of words) into a one-hot encoded matrix.
        Unknown words (or words ignored during fit) are mapped to an extra "unknown" index.

        Args:
            text (list): A list of words representing a single document.

        Returns:
            np.ndarray: A 2D NumPy array where each row corresponds to a word
                        and is a one-hot vector representing that word's index
                        in the vocabulary (or the unknown index).
        """
        # Initialize an array to store indices for each word in the text
        one_hot_idx = np.zeros(len(text))
        for i, word in enumerate(text):
            word = word.lower().strip() # Normalize word
            # Assign the word's index if it's in the vocabulary, otherwise assign the "unknown" index
            if word in self.word_to_index:
                one_hot_idx[i] = self.word_to_index[word]
            else:
                # The unknown index is one greater than the last known word index
                one_hot_idx[i] = len(self.word_to_index)

        # Create the one-hot matrix
        # Rows = number of words in text, Columns = vocab size + 1 (for unknown)
        one_hot = np.zeros((len(text), len(self.word_to_index) + 1))
        # Set the appropriate index to 1 for each word (row)
        one_hot[np.arange(len(text)), one_hot_idx.astype(int)] = 1
        return one_hot

    @property
    def vocab_size(self):
        """Returns the size of the vocabulary (excluding the unknown word)."""
        return len(self.word_to_index)

    def __repr__(self):
        """Provides a string representation of the BagofWords object."""
        return f'BagofWords(vocab_size={self.vocab_size})'

# %% Define a wrapper class for static word embedding models (like Word2Vec, GloVe)
class StaticVectorsModel:
    """
    A wrapper class for the StaticVectors library to provide a consistent
    fit/transform interface similar to scikit-learn models.
    """
    def __init__(self, model_path):
        """
        Initializes the model by loading pre-trained static vectors.

        Args:
            model_path (str): The path to the pre-trained static word embedding model file.
        """
        self.model = StaticVectors(model_path) # Load the static vectors

    def fit(self):
        """
        Placeholder fit method (static vectors are pre-trained, so no fitting needed).
        """
        pass # No operation needed for fitting pre-trained models

    def transform(self, text):
        """
        Transforms a text (list of words) into a matrix of word embeddings.

        Args:
            text (list): A list of words.

        Returns:
            np.ndarray: A 2D NumPy array where each row is the embedding vector for a word.
                        Words not found in the model's vocabulary might be handled
                        by the underlying StaticVectors library (e.g., zero vector).
        """
        return self.model.embeddings(text) # Get embeddings for the words in the text

    def fit_transform(self, text):
        """
        Combines fit and transform (calls transform directly as fit does nothing).

        Args:
            text (list): A list of words.

        Returns:
            np.ndarray: A 2D NumPy array of word embeddings.
        """
        return self.transform(text) # Since fit is a no-op, just call transform

# %% Define a function to generate word embeddings for multiple stories and downsample them
def embed_sentences(stories, wordseqs, model):
    """
    Generates word embeddings for each story using the provided model and
    then downsamples them to match a target timing (e.g., fMRI TR).

    Args:
        stories (list): A list of story identifiers.
        wordseqs (dict): A dictionary mapping story identifiers to word sequence objects
                         (containing word data and timings).
        model: An embedding model instance with a `transform` method (e.g., BagofWords, StaticVectorsModel).

    Returns:
        dict: A dictionary mapping story identifiers to the downsampled embedding matrices.
    """
    word_vectors = {} # Dictionary to store raw word embeddings per story
    # Generate embeddings for each word in each story
    for story in stories:
        word_vectors[story] = model.transform(wordseqs[story].data) # Get embeddings for the word list

    # Downsample the word vectors to align with desired time points (e.g., fMRI TRs)
    # This typically involves averaging embeddings within time bins.
    embeddings = downsample_word_vectors(stories, word_vectors, wordseqs)
    return embeddings

# %% Define a class to aggregate, delay, and standardize embeddings across stories
class embeddings_aggregator:
    """
    Aggregates embeddings from multiple stories, applies time delays,
    optionally standardizes the features, and trims time points.
    """
    def __init__(self, delays=None, standardize=True, trim_range=(5, -10)):
        """
        Initializes the aggregator.

        Args:
            delays (list or range, optional): Time delays (in TRs) to apply to the features.
                                              If None, no delays are applied. Defaults to None.
            standardize (bool, optional): Whether to standardize the features (zero mean, unit variance).
                                          Defaults to True.
            trim_range (tuple, optional): A tuple (start, end) specifying the range of time points
                                          (TRs) to keep after aggregation. Uses Python slicing.
                                          Defaults to (5, -10).
        """
        self.delays = delays # List of delays to apply
        self.standardize = standardize # Flag for standardization
        # Initialize a StandardScaler if standardization is requested
        self.scaler = sklearn.preprocessing.StandardScaler() if standardize else None
        self.trim_range = trim_range # Tuple for trimming start/end TRs

    def _concatenate_embeddings(self, stories, embeddings):
        """
        Helper method to concatenate embeddings for a list of stories,
        apply trimming, and optionally apply delays.

        Args:
            stories (list): List of story identifiers to process.
            embeddings (dict): Dictionary mapping story identifiers to embedding matrices.

        Returns:
            np.ndarray: A single 2D NumPy array containing the concatenated,
                        trimmed, and potentially delayed embeddings.
        """
        all_embeddings = []
        # Iterate through the specified stories
        for story in stories:
            embedding = embeddings[story] # Get the embedding matrix for the story
            trimmed_embedding = embedding # Default to untrimmed
            # Apply trimming if specified
            if self.trim_range:
                start, end = self.trim_range
                trimmed_embedding = embedding[start:end] # Slice the time dimension
            all_embeddings.append(trimmed_embedding) # Add trimmed embedding to the list

        # Concatenate embeddings from all stories along the time axis (axis 0)
        all_embeddings = np.concatenate(all_embeddings, axis=0)

        # Apply time delays if specified
        if self.delays:
            # make_delayed likely creates lagged versions of features and concatenates them
            all_embeddings = make_delayed(all_embeddings, self.delays)

        return all_embeddings

    def fit(self, stories, embeddings):
        """
        Fits the aggregator, primarily fitting the StandardScaler if enabled.

        Args:
            stories (list): List of story identifiers for fitting (e.g., training stories).
            embeddings (dict): Dictionary mapping story identifiers to embedding matrices.

        Returns:
            self: The fitted embeddings_aggregator instance.
        """
        if self.standardize:
            # Concatenate embeddings for fitting the scaler
            all_embeddings = self._concatenate_embeddings(stories, embeddings)
            # Fit the StandardScaler on the concatenated data
            self.scaler = sklearn.preprocessing.StandardScaler().fit(all_embeddings)
        return self # Return the fitted object

    def transform(self, stories, embeddings):
        """
        Transforms embeddings by concatenating, trimming, delaying, and standardizing (if fitted).

        Args:
            stories (list): List of story identifiers to transform (e.g., test stories).
            embeddings (dict): Dictionary mapping story identifiers to embedding matrices.

        Returns:
            np.ndarray: The transformed embedding matrix.
        """
        # Concatenate, trim, and delay the embeddings
        all_embeddings = self._concatenate_embeddings(stories, embeddings)
        # Apply standardization using the fitted scaler
        if self.standardize:
            all_embeddings = self.scaler.transform(all_embeddings)
        return all_embeddings

    def fit_transform(self, stories, embeddings):
        """
        Combines fitting and transforming in one step.

        Args:
            stories (list): List of story identifiers to fit and transform (e.g., training stories).
            embeddings (dict): Dictionary mapping story identifiers to embedding matrices.

        Returns:
            np.ndarray: The fitted and transformed embedding matrix.
        """
        # Concatenate, trim, and delay the embeddings
        all_embeddings = self._concatenate_embeddings(stories, embeddings)
        # Fit the scaler and transform the data simultaneously
        if self.standardize:
            self.scaler = sklearn.preprocessing.StandardScaler() # Re-initialize scaler before fit_transform
            all_embeddings = self.scaler.fit_transform(all_embeddings)
        return all_embeddings