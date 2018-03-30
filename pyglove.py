import logging
import numpy as np
from os import path
from os import makedirs
from scipy.spatial.distance import cosine
from os import remove
from subprocess import check_call


class GloVe(object):
    """
    Class to learn and load GloVe vectors. You can learn them from a corpus (stored in a
    file or a Python iterator), or load them from a a text file.

    """
    def __init__(self, data, output_file, glove_installation_dir='',
                 tmp_data_loc="pyglove_tmp", verbose=2, memory=4.0,
                 vocab_min_count=5, vector_size=50, max_iter=15, window_size=15,
                 binary=2, num_threads=8, x_max=10, run=True,
                 padding_token='`'):
        """

        Parameters
        ----------
        data: str or list
            Either the full path to a corpus file with a single line, with each token
            separated by a space, or an iterable of tokens in order (eg ["the", "dog",
            "ran", "I", "saw", "it"]).

        output_file: str
            Full path to the output file that will store the vectors, with one word and
            its vector per line.

        glove_installation_dir: str
            Path to GloVe directory.

        tmp_data_loc: str
            Full path to the directory for storing intermediate files.

        verbose: int

        memory: float

        vocab_min_count: int
            Minimum frequency required to include a token in the model.

        vector_size: int
            Number of dimensions in the output vectors.

        max_iter: int

        window_size: int
            Size of context window in tokens.

        binary: int

        num_threads: int

        x_max: int

        run: bool
            Whether or not to build the vectors on initialization.

        padding_token: str or None
            Token to be used for padding sentences. Will be excluded from
            to results. If None, no padding will be added and contexts will
            span sentence boundaries. (like the original GloVe examples).
        """
        logging.basicConfig(format=('%(asctime)s - %(name)s - '
                                    '%(levelname)s - %(message)s'),
                            level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.data = data  # iterable of sentence
        self.build_dir = path.join(glove_installation_dir, "build")
        self.data_location = tmp_data_loc
        makedirs(self.data_location, exist_ok=True)

        if isinstance(data, str):  # `data` is a filepath
            self.corpus_loc = data
        else:
            self.corpus_loc = path.join(self.data_location, "corpus.tmp")

        self.vocab_loc = path.join(self.data_location, "vocab.txt")
        self.coocurrence_loc = path.join(self.data_location, "coocurrence.bin")
        self.coocurrence_shuf_loc = path.join(self.data_location,
                                              "coocurrence.shuf.bin")
        self.verbose = verbose
        self.memory = memory
        self.min_count = vocab_min_count
        self.max_iter = max_iter
        self.window_size = window_size
        self.binary = binary
        self.num_threads = num_threads
        self.x_max = x_max
        self.vector_size = vector_size
        self.vectors = {}
        self.vector_file = (output_file
                            if output_file.endswith(".txt")
                            else output_file + ".txt")
        self.padding_token = padding_token

        if run == True:
            # Build Vectors
            self._run(self.vector_size, self.vector_file)

    @classmethod
    def load_from_vector_file(cls, filename):
        """
        Alternate constructor. Load word vectors from a file containing one
        token and its vector (separated by a space) per line.

        Parameters
        ----------
        filename: str
            Full path to vector files.

        Returns
        -------
        GloVe object
            A initialized GloVe object with loaded token vectors.

        """
        glove = cls(None, filename)
        glove._load_vectors(filename)
        return glove

    def __getitem__(self, token):
        """Allow vector lookup using obj[<word>]"""
        return self.vectors[token]

    def _write_tmp_data_file(self):
        """Writes temporary data files for GloVe to use."""
        with open(self.corpus_loc, 'w') as fout:
            for sentence in self.data:
                if self.padding_token is not None:
                    pads = [self.padding_token] * (self.window_size - 1)
                    sentence = pads + sentence + pads
                for token in sentence:
                    if token == "<unk>":
                        continue
                    fout.write("{} ".format(token))

    def _clean_tmp_dir(self):
        """Remove temporary data files"""
        for intermediate_file in (self.vocab_loc,
                                  self.corpus_loc,
                                  self.coocurrence_loc,
                                  self.coocurrence_shuf_loc):
            remove(intermediate_file)

    def _build_vocab(self):
        check_call("{build}/vocab_count -min-count {vocab_min_count} "
                   "-verbose {verbose} < {corpus} > {vocab_file}"
                   .format(build=self.build_dir,
                           vocab_min_count=self.min_count,
                           verbose=self.verbose, corpus=self.corpus_loc,
                           vocab_file=self.vocab_loc), shell=True)

    def _build_coocurrences(self):
        check_call("{build}/cooccur -memory {memory} -vocab-file {vocab_file} "
                   "-verbose {verbose} -window-size {window_size} < {corpus} "
                   "> {coocurrence_file}"
                   .format(build=self.build_dir,
                           memory=self.memory,
                           vocab_file=self.vocab_loc,
                           verbose=self.verbose,
                           window_size=self.window_size,
                           corpus=self.corpus_loc,
                           coocurrence_file=self.coocurrence_loc),
                   shell=True)

    def _shuffle_coocurrences(self):
        check_call("{build}/shuffle -memory {memory} -verbose {verbose} < "
                   "{cooc_file} > {cooc_shuf}"
                   .format(build=self.build_dir, memory=self.memory,
                           verbose=self.verbose,
                           cooc_file=self.coocurrence_loc,
                           cooc_shuf=self.coocurrence_shuf_loc),
                   shell=True)

    def _build_vectors(self, vector_size, outfile):
        check_call("{build}/glove -save-file {outfile} -threads {threads} "
                   "-input-file {cooc_shuf} -x-max {xmax} -iter {maxiter} "
                   "-vector-size {vector_size} -binary {binary} -vocab-file "
                   "{vocab_file} -verbose {verbose}"
                   .format(build=self.build_dir, outfile=outfile.split('.')[0],
                           threads=self.num_threads,
                           cooc_shuf=self.coocurrence_shuf_loc,
                           xmax=self.x_max, maxiter=self.max_iter,
                           vector_size=vector_size,
                           binary=self.binary,
                           vocab_file=self.vocab_loc,
                           verbose=self.verbose),
                   shell=True)

    def _run(self, vector_size, output_file):
        """
        Run the full GloVe vector pipeline.

        Parameters
        ----------
        vector_size: int
            How many dimensions the word vectors will contain.

        output_file: str
            Path to txt file where GloVe vectors will be written.

        Returns
        -------
        None

        """
        self._write_tmp_data_file()
        self._build_vocab()
        self._build_coocurrences()
        self._shuffle_coocurrences()
        self._build_vectors(vector_size, output_file)
        self._load_vectors(output_file)

    def _load_vectors(self, vector_file):
        """
        Loads word vectors from a txt file containing one word and its
        corresponding vector (space-separated) per line.

        Parameters
        ----------
        vector_file: str
            Path to txt file containing words and their corresponding vectors.

        Returns
        -------
        None

        """
        with open(vector_file) as f:
            for line in f:
                token, vector = line.split(' ', 1)
                self.vectors[token] = np.array([float(d) for d
                                                in vector.split() if d])
        if self.padding_token is not None:
            del self.vectors[self.padding_token]

    def most_similar(self, token, n=10):
        """
        Given a token, generate tokens that are most similar to it in the
        current model.

        Parameters
        ----------
        token: str
            Token to be compared.

        n: int
            Number of results to return.

        Returns
        -------
        list of tuples
            The top n most-similar tokens and their cosine similarity scores.

        """
        vec = self.vectors[token]
        return sorted([(t, 1 - cosine(vec, v)) for t, v
                       in  self.vectors.items() if t != token],
                      key=lambda tup: tup[1], reverse=True)[:n]