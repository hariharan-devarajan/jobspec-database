#!/usr/bin/env python3


import argparse
import json
import os
import sys

import requests

# Note that we are doing with the web UI because the programmatic rest API
# limits queries per day
here = os.path.abspath(os.path.dirname(__file__))


def get_parser():
    parser = argparse.ArgumentParser(description="Job Specification Downloader")
    parser.add_argument(
        "--input",
        help="Input json file",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join(here, "data", "jobs"),
        help="output data directory for results",
    )
    return parser


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    if not args.input:
        sys.exit("An input directory is required.")

    with open(args.input, "r") as fd:
        links = json.loads(fd.read())

    for _, listing in links.items():
        for item in listing:
            repo = os.sep.join(item.strip("/").split("/")[0:2])
            outdir = os.path.join(args.outdir, repo)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            _, filename = item.split("blob/")
            blob, filename = filename.split("/", 1)
            filename = filename.rsplit("#", 1)[0]
            outfile = os.path.join(outdir, filename)
            if os.path.exists(outfile):
                continue

            # Assemble the raw GitHub URL
            url = f"https://raw.githubusercontent.com/{repo}/{blob}/{filename}"
            response = requests.get(url)
            if response.status_code == 404:
                url = f"https://raw.githubusercontent.com/{repo}/master/{filename}"
                response = requests.get(url)

            if response.status_code == 404:
                continue

            if response.status_code != 200:
                print(f"Issue with response {response}")
                import IPython

                IPython.embed()

            dirname = os.path.dirname(outfile)
            print(f"Writing job file {outfile}")
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            with open(outfile, "w") as fd:
                fd.write(response.text)

    print('Sanity check...')
    import IPython

    IPython.embed()
    sys.exit()


if __name__ == "__main__":
    main()
