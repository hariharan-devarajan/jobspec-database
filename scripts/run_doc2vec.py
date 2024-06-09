#!/usr/bin/env python3


import argparse
import fnmatch
import hashlib
import os
import re
import sys

import numpy as np
import rse.utils.file as utils
import tensorflow as tf

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(here, "doc2vec"))

from dataset import Doc2VecDataset
from doc2vec import Doc2VecTrainer


def get_parser():
    parser = argparse.ArgumentParser(description="word2vec")
    parser.add_argument(
        "input",
        help="Input directory",
        default=os.path.join(here, "data", "jobs"),
    )
    parser.add_argument(
        "--output",
        help="Output data directory",
        default=os.path.join(here, "data", "doc2vec"),
    )
    parser.add_argument(
        "--arch",
        help="Architecture (DBOW or DM).",
        default="PV-DBOW",
    )
    parser.add_argument(
        "--algm",
        help="Training algorithm (negative_sampling or hierarchical_softmax).",
        default="negative_sampling",
    )
    parser.add_argument(
        "--epochs",
        help="Num of epochs to iterate training data.",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size",
        default=64,
        type=int,
    )
    parser.add_argument(
        "--max-vocab-size",
        help="Maximum vocabulary size. If > 0, the top `max_vocab_size` most frequent words are kept in vocabulary.",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--min-count",
        help="Words whose counts < `min_count` are not" " included in the vocabulary.",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--sample",
        help="Sub-sampling rate",
        default=1e-3,
        type=float,
    )
    parser.add_argument(
        "--window-size",
        help="Window size",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--dbow-train-words",
        help="Whether to train the word vectors in DBOW architecture.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--dm-concat",
        help="Whether to concatenate word and document vectors or compute their mean in DM architecture.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--embed-size",
        help="Length of word vector.",
        default=300,
        type=int,
    )
    parser.add_argument(
        "--negatives",
        help="Num of negative words to sample.",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--power",
        help="Distortion for negative sampling.",
        default=0.75,
        type=float,
    )
    parser.add_argument(
        "--alpha",
        help="Initial learning rate.",
        default=0.025,
        type=float,
    )
    parser.add_argument(
        "--min-alpha",
        help="Final learning rate.",
        default=0.0001,
        type=float,
    )
    parser.add_argument(
        "--no-add-bias",
        help="Do not add bias term to dotproduct between syn0 and syn1 vectors (this means by default we add bias)",
        default=False,
        action="store_true",
    )
    return parser


def content_hash(filename):
    with open(filename, "rb", buffering=0) as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def recursive_find(base, pattern="*"):
    for root, _, filenames in os.walk(base):
        for filename in fnmatch.filter(filenames, pattern):
            yield os.path.join(root, filename)


def combine_data(data_file, args, files):
    seen = set()
    flags = {}
    with open(data_file, "w") as fd:
        for file in files:
            print(f"Writing file {file}")
            # 1. Step 1, ensure we don't duplicate files
            digest = content_hash(file)
            if digest in seen:
                print(f"Found duplicate file {file}")
                continue
            seen.add(digest)
            content = utils.read_file(file)
            tokens, new_flags = tokenize(content)

            # Write each "document" on a single line
            fd.write(" ".join(tokens) + "\n")
            flags[file] = new_flags
    return flags


# directives and special punctuation
directive_regex = "#(SBATCH|PBATCH|COBALT|PBS|OAR|BSUB|FLUX)"
punctuation = "!\"#$%&'()*+,.:;<=>?@[\\]^`{|}~\n"


def tokenize(lines):
    """
    Special tokenize for scripts, and parsing of flag directives
    """
    # Get rid of hash bangs
    lines = [x for x in lines if "#!" not in x]
    directives = [x for x in lines if re.search(directive_regex, x)]

    new_flags = {}
    for directive in directives:
        directive = directive.split(" ", 1)[-1].strip()
        if "=" in directive:
            key, value = directive.split("=", 1)
            value = value.strip()
        elif directive.count(" ") == 0:
            key = directive
            value = True
        else:
            key, value = directive.split(" ", 1)
            value = value.strip()
        key = key.replace("--", "").strip("-").strip()
        new_flags[key] = value

    content = " ".join(lines)
    content = re.sub("(_|-|\/)", " ", content)
    lowercase = tf.strings.lower(content)
    content = tf.strings.regex_replace(lowercase, "[%s]" % re.escape(punctuation), "")

    # Convert from EagerTensor back to string
    content = content.numpy().decode("utf-8")
    tokens = content.split()

    # Replace underscore - hyphen with space (treat like token / word)
    return tokens, new_flags


def main():
    """
    doc2vec for jobspecs

    See: https://github.com/chao-ji/tf-doc2vec

    You need to clone each of doc2vec and word2vec for this to work.
    """
    parser = get_parser()
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.input):
        sys.exit("An input directory is required.")

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Read in text, and as we go, generate content hash.
    # We don't want to use duplicates (forks are disabled, but just being careful)
    files = list(recursive_find(args.input))

    # We need to combine all files into one - this time parsed by token with
    # a space between. We are also going to parse flags associated with files here
    data_file = os.path.join(args.output, "jobspec-docs.txt")
    directives_file = os.path.join(args.output, "jobspec-directives.json")
    if not os.path.exists(data_file):
        flags = combine_data(data_file, args, files)
        utils.write_json(flags, directives_file)

    dataset = Doc2VecDataset(
        arch=args.arch,
        algm=args.algm,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_vocab_size=args.max_vocab_size,
        min_count=args.min_count,
        sample=args.sample,
        window_size=args.window_size,
        dbow_train_words=args.dbow_train_words,
        dm_concat=args.dm_concat,
    )
    dataset.build_vocab([data_file])
    doc2vec = Doc2VecTrainer(
        arch=args.arch,
        algm=args.algm,
        embed_size=args.embed_size,
        batch_size=args.batch_size,
        negatives=args.negatives,
        power=args.power,
        alpha=args.alpha,
        min_alpha=args.min_alpha,
        add_bias=not args.no_add_bias,
        random_seed=0,
        dm_concat=args.dm_concat,
        window_size=args.window_size,
    )

    to_be_run_dict = doc2vec.train(dataset, [data_file])
    save_list = doc2vec.get_save_list()

    sess = tf.Session()
    sess.run(dataset.iterator_initializer)
    sess.run(tf.tables_initializer())
    sess.run(tf.global_variables_initializer())

    average_loss = 0.0
    step = 0
    while True:
        try:
            result_dict = sess.run(to_be_run_dict)
        except tf.errors.OutOfRangeError:
            break
        average_loss += result_dict["loss"].mean()
        if step % 10000 == 0:
            if step > 0:
                average_loss /= 10000
            print(
                "step",
                step,
                "average_loss",
                average_loss,
                "learning_rate",
                result_dict["learning_rate"],
            )
            average_loss = 0.0
        step += 1

    saver = tf.train.Saver(var_list=save_list)
    saver.save(sess, os.path.join(FLAGS.out_dir, "doc2vec.ckpt"))

    syn0_w_final = sess.run(doc2vec.syn0_w)
    syn0_d_final = sess.run(doc2vec.syn0_d)

    np.save(os.path.join(FLAGS.out_dir, "word_embed"), syn0_w_final)
    np.save(os.path.join(FLAGS.out_dir, "train_doc_embed"), syn0_d_final)

    with open(os.path.join(FLAGS.out_dir, "vocab.txt"), "w") as fid:
        for w in dataset.table_words:
            fid.write(w + "\n")

    print("Word embeddings saved to", os.path.join(FLAGS.out_dir, "word_embed.npy"))
    print(
        "Train doc embeddings saved to",
        os.path.join(FLAGS.out_dir, "train_doc_embed.npy"),
    )
    print("Vocabulary saved to", os.path.join(FLAGS.out_dir, "vocab.txt"))


if __name__ == "__main__":
    main()
