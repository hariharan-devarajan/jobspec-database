#!/usr/bin/env python3


import tensorflow as tf
import shlex
import hashlib
import fnmatch
import os
import sys
import re

import argparse
import rse.utils.file as utils

here = os.path.abspath(os.path.dirname(__file__))


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
        default=os.path.join(here, "data", "combined"),
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
    duplicates = 0
    print(f"Assessing {len(files)} conteder jobscripts...")
    with open(data_file, "w") as fd:
        for file in files:
            # 1. Step 1, ensure we don't duplicate files
            digest = content_hash(file)
            if digest in seen:
                duplicates += 1
                continue
            seen.add(digest)
            content = utils.read_file(file)
            tokens, new_flags = tokenize(content)

            # Write each "document" on a single line
            fd.write(" ".join(tokens) + "\n")
            flags[file] = new_flags

    print(f"Found (and skipped) {duplicates} duplicates.")
    return flags


# directives and special punctuation
directive_regex = "#(SBATCH|PBATCH|COBALT|PBS|OAR|BSUB|FLUX)"
punctuation = "!\"#$%&'()*+,.:;<=>?@[\\]^`{|}~\n"


# Keep track of those we skip for false positives
skips = set()


def tokenize(lines):
    """
    Special tokenize for scripts, and parsing of flag directives
    """
    global skips

    # Get rid of hash bangs
    lines = [x for x in lines if "#!" not in x]
    directives = [x for x in lines if re.search(directive_regex, x)]

    new_flags = {}

    def add_flag(key, value):
        """
        shared function to add a flag
        """
        key = key.replace("--", "").strip("-").strip("—").strip()
        new_flags[key] = value

    for directive in directives:
        # Get rid of the beginning of the line
        directive = directive.split("#", 1)[-1]

        # Assume that flags are required
        if "-" not in directive and "--" not in directive and "—" not in directive:
            skips.add(directive)
            continue

        # Any comments?
        if "#" in directive:
            directive = directive.rsplit("#", 1)[0]

        # Split by tab OR space
        directive = re.split("( |\t)", directive, 1)[-1].strip()

        # Get rid of quotations, and escapes that are left.
        directive = re.sub("('|\")", "", directive)
        directive = directive.replace("\\", "")

        # If it's empty!
        if not directive.strip():
            continue

        if "=" in directive:
            key, value = directive.split("=", 1)
            value = value.strip()
            add_flag(key, value)

        elif directive.count(" ") == 0:
            key = directive
            value = True
            add_flag(key, value)

        # This can be an arbitrary listing of 1+ arguments
        else:
            parts = shlex.split(directive)
            while parts:
                key = parts.pop(0).strip()
                next_value = None
                if parts:
                    next_value = parts[0]

                # If it start with a dash, etc., we found a flag with argument
                # if the next value exists and isn't a flag, this is the value
                if (
                    re.search("^(--|-|—)", key)
                    and next_value
                    and not re.search("^(--|-|—)", next_value)
                ):
                    # Pop it so we don't parse it next!
                    value = parts.pop(0).strip()
                    add_flag(key, value)

                # This means we found a boolean or lone flag
                elif (
                    re.search("^(--|-|—)", key)
                    and not next_value
                    or (next_value is not None and re.search("^(--|-|—)", next_value))
                ):
                    add_flag(key, True)

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
    jobspec feature parsing
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

    # Keep track of those we've skipped
    skips_file = os.path.join(args.output, "skipped-directives.txt")

    flags = combine_data(data_file, args, files)
    utils.write_json(flags, directives_file)
    utils.write_file("\n".join(list(skips)), skips_file)


if __name__ == "__main__":
    main()
