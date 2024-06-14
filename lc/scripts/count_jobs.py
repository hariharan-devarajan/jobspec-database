#!/usr/bin/env python3

import argparse
import hashlib
import os
import json
import sys
import fnmatch
import rse.utils.file as utils
import tarfile

here = os.path.abspath(os.path.dirname(__file__))


def recursive_find(base, pattern="*"):
    for root, _, filenames in os.walk(base):
        for filename in fnmatch.filter(filenames, pattern):
            yield os.path.join(root, filename)


def get_parser():
    parser = argparse.ArgumentParser(description="count jobs")
    parser.add_argument(
        "--input",
        help="jobspec json input directory",
        default=os.path.join(here, "jobdata_json"),
    )
    return parser


def content_hash(content):
    h = hashlib.new("sha256")
    try:
        h.update(content.encode())
    except:
        h.update(content)
    return h.hexdigest()


def main():
    """
    Count individual jobspec json files

    There seem to be a combination of json AND archive, so keep track of
    unique by ids.
    """
    parser = get_parser()
    args, _ = parser.parse_known_args()
    if not os.path.exists(args.input):
        sys.exit(f"{args.input} does not exist.")

    # Keep a count!
    seen = set()
    has_batch_script = set()
    digests = set()
    duplicates = 0

    def update_counts(content):
        if "BatchScript" in content["scontrol"]:
            has_batch_script.add(jobid)
            print(content["scontrol"]["BatchScript"])
        seen.add(jobid)

    for filename in recursive_find(args.input, "*.json"):
        jobid = os.path.basename(filename).replace(".json", "")
        content = utils.read_json(filename)
        digest = content_hash(json.dumps(content))
        if digest in digests:
            duplicates += 1
            continue
        update_counts(content)

    # Now read the tars
    for filename in recursive_find(args.input, "*.tar"):
        tar = tarfile.open(filename, "r")
        print(f"Reading {filename}")

        # This is a tar info
        for member in tar.getmembers():
            jobid = os.path.basename(member.name).replace(".json", "")
            if member.isdir() or not member.name.endswith("json"):
                continue
            content = tar.extractfile(member).read()
            digest = content_hash(content)
            if digest in digests:
                duplicates += 1
                continue

            if not content:
                continue
            content = json.loads(content)
            update_counts(content)
        tar.close()

    print(f"Total duplicates: {duplicates}")
    print(f"Total unique jobspec jsons: {len(seen)}")
    print(f"Total with BatchScript: {len(has_batch_script)}")


if __name__ == "__main__":
    main()
