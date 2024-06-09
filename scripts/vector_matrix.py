#!/usr/bin/env python3


import argparse
import os
import re
import string
import sys
import pandas
import numpy
import rse.utils.file as utils
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

here = os.path.abspath(os.path.dirname(__file__))

from sklearn.feature_extraction.text import TfidfVectorizer


def get_parser():
    parser = argparse.ArgumentParser(description="word2vec")
    parser.add_argument(
        "--vectors",
        help="vectors.tsv file",
        default=os.path.join(here, "scripts", "data", "combined", "vectors.tsv"),
    )
    parser.add_argument(
        "--metadata",
        help="metadata.tsv file",
        default=os.path.join(here, "scripts", "data", "combined", "metadata.tsv"),
    )
    parser.add_argument(
        "--output",
        help="Output data directory",
        default=os.path.join(here, "data", "combined"),
    )
    return parser


def main():
    """
    Generate cosine distance matrix from vectors.
    """

    parser = get_parser()
    args, _ = parser.parse_known_args()

    for path in [args.vectors, args.metadata]:
        if not os.path.exists(path):
            sys.exit(f"{path} does not exist.")

    vectors = pandas.read_csv(args.vectors, header=None, index_col=None, sep="\t")

    # These are just the column names
    metadata = pandas.read_csv(args.metadata, header=None, index_col=0)
    vectors.index = metadata.index

    # Save them together for next time
    combined_path = f"{args.vectors}.df.csv"
    if not os.path.exists(combined_path):
        vectors.to_csv(combined_path)

    # Now generate cosine matrix - too big to save
    cosines = cosine_similarity(vectors)
    cosines = pandas.DataFrame(cosines)
    cosines.index = metadata.index
    cosines.columns = metadata.index

    # Get "meaningful" terms based on tf-idf
    data_file = os.path.join(args.output, "combined-jobspecs.txt")
    corpus = [x.strip() for x in utils.read_file(data_file)]

    # Get rid of entire numbers - first space is optional
    corpus = [re.sub("\w?(\d)+ ", " ", x) for x in corpus]

    # Each line is a jobspec
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names_out()

    # Filter to shared set
    features = [x for x in features if x in cosines.index]

    # Figure out where to filter...
    print("Generating cosine histogram...")
    plt.figure(figsize=(12, 12))
    flat = cosines.to_numpy().flatten()
    sns.histplot(flat)
    plt.title(f"Distribution of values for cosine matrix")
    plt.savefig(
        os.path.join(args.output, f"costines-histogram.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()

    # I decided to use tf-IDF instead
    # Let's choose an arbitrary mean to get rid of terms by
    # We can come up with a better method for this, but we would generally want to remove
    # the entire terms that are less connected to others.
    # This is rough but filters down to about 2.3k from 4.5k
    # print('Filtering to mean values not in range -0.1 to 0.1...')
    # keepers = []
    # for idx, value in enumerate(cosines.mean()):
    #    if (value == 0) or (value > 0 and value < 0.1) or (value < 0 and value > -0.1):
    #        continue
    #    keepers.append(cosines.index[idx])

    # This is the smaller filtered matrix
    filtered = cosines.loc[features, features]
    plt.figure(figsize=(36, 36))
    g = sns.clustermap(filtered)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=5)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=5)
    plt.title(f"Filtered Word2Vec Term Cosine Matrix")
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.output, f"filtered-cosines-heatmap.pdf"), bbox_inches="tight"
    )
    plt.clf()


if __name__ == "__main__":
    main()
