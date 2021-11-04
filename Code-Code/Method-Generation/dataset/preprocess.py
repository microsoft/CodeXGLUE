# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import re
from tokenize import tokenize, untokenize, COMMENT, STRING, NEWLINE, ENCODING, ENDMARKER, NL, INDENT, NUMBER, DEDENT
from io import BytesIO
import json
from tqdm import tqdm

lits = json.load(open("literals.json"))

def process_string(token, special_chars={" ": "U+0020", ",": "U+002C"}):
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
    # if start_quote in str_quote_options[:2]:
    #     return ""
    for sc in special_chars:
        str_lit = str_lit.replace(sc, special_chars[sc])
    return (
        f"{qualifier}{start_quote}<STR_LIT:{str_lit}>{end_quote}"
        if str_lit in lits['str']
        else f"{qualifier}{start_quote}<STR_LIT>{end_quote}"
    )

def preprocess(code):
    if code["signature"][0] == " ":
        indent_size = len(code["signature"]) - len(code["signature"].lstrip(" "))
        code["signature"] = code["signature"].lstrip()
        body = ""
        lines = code["body"].split("\n")
        for line in lines:
            if len(line.strip()):
                body += line[indent_size*2:] + "\n"
        code["body"] = body
        code["docstring"] = code["docstring"].strip()
    else:
        indent_size = len(code["body"]) - len(code["body"].lstrip(" "))
        code["signature"] = code["signature"].lstrip()
        body = ""
        lines = code["body"].split("\n")
        for line in lines:
            if len(line.strip()):
                body += line[indent_size:] + "\n"
        code["body"] = body
        code["docstring"] = code["docstring"].strip()
    # indent_size = len(code["body"]) - len(code["body"].lstrip(" "))
    return indent_size

def py_tokenize(args, file_name, file_type):
    codes = open(os.path.join(args.base_dir, file_name)).readlines()
    wf = open(os.path.join(args.output_dir, f"{file_type}.jsonl"), 'w')
    for ct,code in enumerate(tqdm(codes)):
        code = json.loads(code)
        indent_size = preprocess(code)
        for name in ["signature", "body"]:
            out_str = ""
            try:
                token_gen = tokenize(BytesIO(bytes(code[name], "utf8")).readline)
                prev_eol = False
                last_ed = (0, 0)
                for toknum, tokval, st, ed, _ in token_gen:
                    add_token = ""
                    tokval = " ".join(tokval.split())
                    if toknum == STRING:
                        add_token = process_string(tokval)
                        prev_eol = False
                    elif toknum == NUMBER:
                        if tokval in lits['num']:
                            add_token = f"<NUM_LIT:{tokval}>"
                        else:
                            add_token = f"<NUM_LIT>"
                        prev_eol = False
                    elif toknum in [NEWLINE, NL]:
                        if not prev_eol:
                            add_token = "<EOL>"
                            prev_eol = True
                    elif toknum in [COMMENT, ENCODING, ENDMARKER]:
                        pass
                    elif toknum == INDENT:
                        add_token = "<INDENT>"
                        prev_eol = False
                    elif toknum == DEDENT:
                        add_token = "<DEDENT>"
                        prev_eol = False
                    else:
                        add_token = tokval
                        prev_eol = False
                    if st[0] == last_ed[0]:
                        out_str += " "*(st[1]-last_ed[1]) + add_token
                    else:
                        out_str += add_token
                    last_ed = ed
            except Exception:
                pass
            code[name] = out_str 
        wf.write(json.dumps(code)+"\n")
        # if ct % 10000 == 0:
        #     print(f"{file_type}: {ct} are done")
    wf.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="py150_files", type=str, 
                        help="The downloaded data path")
    parser.add_argument("--output_dir", default="processed", type=str, 
                        help="The output directory")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    py_tokenize(args, file_name="train.json", file_type="train")
    py_tokenize(args, file_name="valid.json", file_type="dev")
    py_tokenize(args, file_name="test.json", file_type="test")

if __name__ == "__main__":
    main()