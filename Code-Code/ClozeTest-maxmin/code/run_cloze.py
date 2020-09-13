# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch

from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizer
import argparse
import json
import os

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)}


def get_cloze_words(filename, tokenizer):
    with open(filename, 'r', encoding='utf-8') as fp:
        words = fp.read().split('\n')
    idx2word = {tokenizer.encoder[w]: w for w in words}
    return idx2word


def test_single(text, model, idx2word, tokenizer, device):
    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))[:510]
    inputs = tokenizer.build_inputs_with_special_tokens(tokenized_text)
    index = inputs.index(tokenizer.mask_token_id)

    inputs = torch.tensor([inputs])
    inputs = inputs.to(device)

    with torch.no_grad():
        scores = model(inputs)[0]
        score_list = scores[0][index]
        word_index = torch.LongTensor(list(idx2word.keys())).to(device)
        word_index = torch.zeros(score_list.shape[0]).to(device).scatter(0, word_index, 1)
        score_list = score_list + (1-word_index) * -1e6
        predict_word_id = torch.argmax(score_list).data.tolist()

    return predict_word_id


def cloze_test(args, lang, model, tokenizer, device):
    cloze_words_file = os.path.join('../data', 'cloze-'+args.cloze_mode, 'cloze_test_words.txt')
    file_path = os.path.join('../data', 'cloze-'+args.cloze_mode, lang, 'clozeTest.json')

    idx2word = get_cloze_words(cloze_words_file, tokenizer)
    lines = json.load(open(file_path))

    results = []
    for line in lines:
        text = ' '.join(line['nl_tokens'] + line['pl_tokens'])
        predict_id = test_single(text, model, idx2word, tokenizer, device)
        results.append({'idx': line['idx'],
                        'prediction': idx2word[predict_id]})
    with open(os.path.join(args.output_dir, lang, 'predictions.txt'), 'w', encoding='utf-8') as fp:
        for inst in results:
            fp.write(inst['idx']+'<CODESPLIT>'+inst['prediction']+'\n')
    print("ClozeMaxmin for {} finished".format(lang))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='roberta-base', help='"roberta-base" or "microsoft/codebert-base-mlm" or model path(pytorch_model.bin)')
    parser.add_argument('--cloze_mode', default='maxmin', help='"all" or "maxmin" mode')
    parser.add_argument('--output_dir', default='../evaluator/predictions/', help='directory to save output predictions')
    args = parser.parse_args()

    config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
    config = config_class.from_pretrained('roberta-base')
    tokenizer = tokenizer_class.from_pretrained('roberta-base')
    model = RobertaForMaskedLM.from_pretrained(args.model, from_tf=bool('.ckpt' in args.model), config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print('cloze test mode: {}'.format(args.cloze_mode))
    cloze_results = []
    for lang in ['ruby', 'javascript', 'go', 'python', 'java', 'php']:
        cloze_results.extend(cloze_test(args, lang, model, tokenizer, device))


if __name__ == '__main__':
    main()
