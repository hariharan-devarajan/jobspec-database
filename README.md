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

You can run:

```bash
python scripts/word2vec.py --input ./data
```

Updates to the above on June 9th:

- Better parsing to tokenize 
  - we combine by space instead of empty space so words at end aren't combined (this was a bug)
  - punctuation that should be replaced by space instead of empty space honored (dashes, underscore, etc)
  - hash bangs for shell parsed out
  - better tokenization and recreation of content
  - each script is on one line (akin to how done for word2vec)
  
I realize that we probably want doc2vec, because word2vec is likely making associations between terms in different documents.
We want to maintain the level of the script, and further, to be able to associate job parameters with specific ones.
We are going to use [tf-doc2vec](https://github.com/chao-ji/tf-doc2vec) and we will only need to prepare our data.

```bash
cd ./scripts
git clone git@github.com:chao-ji/tf-doc2vec.git doc2vec
git clone git@github.com:chao-ji/tf-word2vec.git word2vec
cd ../
python scripts/run_doc2vec.py --input ./data
```

Note that I'm currently writing this - will push the final result when I finish.

## License

HPCIC DevTools is distributed under the terms of the MIT license.
All new contributions must be made under this license.

See [LICENSE](https://github.com/converged-computing/cloud-select/blob/main/LICENSE),
[COPYRIGHT](https://github.com/converged-computing/cloud-select/blob/main/COPYRIGHT), and
[NOTICE](https://github.com/converged-computing/cloud-select/blob/main/NOTICE) for details.

SPDX-License-Identifier: (MIT)

LLNL-CODE- 842614

