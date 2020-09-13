# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import re
from tokenize import tokenize, untokenize, COMMENT, STRING, NEWLINE, ENCODING, ENDMARKER, NL, INDENT, NUMBER
from io import BytesIO

def process_string(token):
    str_quote_options = ["'''", '"""', "'", '"']
    start_quote = ""
    end_quote = ""
    qualifier_regex = r"^[a-z]+"
    qualifier_match = re.search(qualifier_regex, token)
    # string qualifiers like 'r' for regex, 'f' for formatted string, 'b' for bytes, 'u' for unicode, etc (or combination of them)
    qualifier = "" if not qualifier_match else qualifier_match[0]
    # token string without qualifiers
    token_string = re.sub(qualifier_regex, "", token)
    # string literal without quotes
    str_lit = token_string
    for q in str_quote_options:
        if token_string.startswith(q):
            start_quote = q
            str_lit = str_lit[len(q) :]
            if token_string.endswith(q):
                end_quote = q
                str_lit = str_lit[: -len(q)]
            break
    if start_quote in str_quote_options[:2]:
        return ""
    return (
        f"{qualifier}{start_quote}{str_lit}{end_quote}"
        if len(str_lit) < 15 and "\n" not in str_lit and "</s>" not in str_lit and "<s>" not in str_lit and "<pad>" not in str_lit and "<EOL>" not in str_lit
        else f"{qualifier}{start_quote}{end_quote}"
    )

def py_tokenize(args, file_name, file_type):
    file_paths = open(os.path.join(args.base_dir, file_name)).readlines()
    wf = open(os.path.join(args.output_dir, f"{file_type}.txt"), 'w')
    for ct,path in enumerate(file_paths):
        try:
            code = open(os.path.join(args.base_dir, path.strip())).read()
            token_gen = tokenize(BytesIO(bytes(code, "utf8")).readline)
            out_tokens = []
            prev_eol = False
            for toknum, tokval, _, _, _ in token_gen:
                tokval = " ".join(tokval.split())
                if len(tokval) > 100:
                    continue
                if toknum == STRING:
                    add_token = process_string(tokval)
                    if len(add_token) > 0:
                        out_tokens.append(add_token)
                        prev_eol = False
                elif toknum == NUMBER:
                    if len(tokval) < 50:
                        out_tokens.append(tokval)
                        prev_eol = False
                elif toknum in [NEWLINE, NL]:
                    if not prev_eol:
                        out_tokens.append("<EOL>")
                        prev_eol = True
                elif toknum in [COMMENT, INDENT, ENCODING, ENDMARKER] or len(tokval) == 0:
                    continue
                else:
                    out_tokens.append(tokval)
                    prev_eol = False
            if out_tokens[0] == "<EOL>":
                out_tokens = out_tokens[1:]
            if out_tokens[-1] == "<EOL>":
                out_tokens = out_tokens[:-1]
        except Exception:
            out_tokens = []
        out_tokens = ["<s>"] + out_tokens + ["</s>"]
        out = " ".join(out_tokens)
        wf.write(out+"\n")

        if ct % 10000 == 0:
            print(f"{file_type}: {ct} are done")
    wf.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="py150_files", type=str, 
                        help="The downloaded data path")
    parser.add_argument("--output_dir", default="token_completion", type=str, 
                        help="The output directory")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dev_paths = open(os.path.join(args.base_dir, "python50k_eval.txt")).readlines()[:5000]
    wf = open(os.path.join(args.base_dir, "python5k_dev.txt"), "w")
    for path in dev_paths:
        wf.write(path)
    wf.close()

    py_tokenize(args, file_name="python100k_train.txt", file_type="train")
    py_tokenize(args, file_name="python5k_dev.txt", file_type="dev")
    py_tokenize(args, file_name="python50k_eval.txt", file_type="test")

if __name__ == "__main__":
    main()
