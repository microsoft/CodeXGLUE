# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import re
import javalang
import json

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
    use_char = False
    if len(str_lit) == 1 and start_quote == "'":
        use_char = True
    for sc in special_chars:
        str_lit = str_lit.replace(sc, special_chars[sc])
    if not use_char:
        ret = (
            f"{qualifier}{start_quote}<STR_LIT:{str_lit}>{end_quote}"
            if str_lit in lits['str']
            else f"{qualifier}{start_quote}<STR_LIT>{end_quote}"
        )
    else:
        ret = (
            f"{qualifier}{start_quote}<CHAR_LIT:{str_lit}>{end_quote}"
            if str_lit in lits['char']
            else f"{qualifier}{start_quote}<CHAR_LIT>{end_quote}"
        )
    return ret

def preprocess(args, file_name, file_type):
    contents = open(os.path.join(args.base_dir, file_name)).readlines()
    wf = open(os.path.join(args.output_dir, f"{file_type}.txt"), 'w')

    for content in contents:
        content = content.strip().lstrip("<s>").rstrip("</s>")
        new_data = []
        try:
            for tok in list(javalang.tokenizer.tokenize(content)):
                if "String" in str(type(tok)) or "Character" in str(type(tok)):
                    token = process_string(tok.value)
                elif "Integer" in str(type(tok)) or "FloatingPoint" in str(type(tok)):
                    if tok.value in lits['num']:
                        token = f"<NUM_LIT:{tok.value}>"
                    else:
                        token = "<NUM_LIT>"
                else:
                    token = tok.value
                new_data.append(token)
        except Exception:
            continue
        if len(new_data) == 0:
            continue
        data = "<s> " + " ".join(new_data) + " </s>"
        wf.write(data+"\n")
    
    print(f"{file_type} are done")
    wf.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="token_completion", type=str, 
                        help="The downloaded data path")
    parser.add_argument("--output_dir", default="token_completion", type=str, 
                        help="The output directory")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    preprocess(args, file_name="train.txt", file_type="train")
    preprocess(args, file_name="dev.txt", file_type="dev")
    preprocess(args, file_name="test.txt", file_type="test")

if __name__ == "__main__":
    main()