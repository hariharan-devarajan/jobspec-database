# Job Specification Database

This database is 🚧️ under developement! 🚧️

It will eventually be added to 🥑️ [Dinosaur Datasets](https://vsoch.github.io/datasets/). 🥑️

## Usage

The data files are organized by repository in [data](data). These instructions are for generation. Create a python environment and install dependencies:

```bash
pip install -r requirements.txt
```

You'll need to make a "drivers" directory and download the chromedriver (matching your browser) to it inside of scripts. Then, run the parsing script, customizing the matrix of search terms. You should have a chromedriver installed, all browsers closed, and be prepared to login to GitHub.

```bash
cd scripts/
python search.py
```

Then download files, from the root, targeting the output file of interest.

```bash
python scripts/get_jobspecs.py ./scripts/data/raw-links-may-23.json --outdir ./data
```

Note that the data now is just a trial run! For the first run, we had 11k+ unique results from just a trial run.
For the second run, that went up to `19544`. When I added more applications, for half of the run it was 25k.
The current total is `31932` scripts. I didn't add the last run of flux because I saw what I thought were false positives.

## Analysis

### 1. Word2Vec

Word2Vec is a little old, and I think a flaw is that it is combining jobspecs. But if we have the window the correct size, we can make associations between close terms.
The space I'm worried about is the beginning of one script and the end of another, and maybe a different approach or strategy could help with that.
To generate the word2vec embeddings you can run:

```bash
python scripts/word2vec.py --input ./data
```

Updates to the above on June 9th:

- Better parsing to tokenize 
  - we combine by space instead of empty space so words at end are not combined (this was a bug)
  - punctuation that should be replaced by space instead of empty space honored (dashes, underscore, etc)
  - hash bangs for shell parsed out
  - better tokenization and recreation of content
  - each script is on one line (akin to how done for word2vec)

I think it would be reasonable to create a similarity matrix, specifically cosine distance between the vectors.
This will read in the metadata.tsv and vectors.tsv we just generated.

```bash
python scripts/vector_matrix.py --vectors ./scripts/data/combined/vectors.tsv --metadata ./scripts/data/combined/metadata.tsv
```

The above does the following:

1. We start with our jobspecs that are tokenized according to the above.
2. We further remove anything that is purely numerical
3. We use TF-IDF to reduce the feature space to 300 terms
4. We do a clustering of these terms to generate the resulting plot.

The hardest thing is just seeing all the terms. I messed with JavaScript for a while but gave up for the time being, the data is too big for the browser
and likely we need to use canvas.

### 2. Directive Counts

I thought it would be interesting to explicitly parse the directives. That's a bit hard, but I took a first shot:

```bash
python scripts/parse_directives.py --input ./data
```
```console
Assessing 33851 conteder jobscripts...
Found (and skipped) 535 duplicates.
```

You can find tokenized lines (with one jobspec per line), the directive counts, and the dictionary and skips in [scripts/data/combined/](scripts/data/combined/)

### 3. Adding Topics or More Structure

I was thinking about adding doc2vec, because word2vec is likely making associations between terms in different documents,
but I don't think anyone is using doc2vec anymore, because the examples I'm finding using a deprecated version of tensorflow that
has functions long removed. We could use the old gensim version, but I think it might be better to think of a more modern approach.
Note that I'm currently writing this - will push the final result when I finish.


## License

HPCIC DevTools is distributed under the terms of the MIT license.
All new contributions must be made under this license.

See [LICENSE](https://github.com/converged-computing/cloud-select/blob/main/LICENSE),
[COPYRIGHT](https://github.com/converged-computing/cloud-select/blob/main/COPYRIGHT), and
[NOTICE](https://github.com/converged-computing/cloud-select/blob/main/NOTICE) for details.

SPDX-License-Identifier: (MIT)

LLNL-CODE- 842614

