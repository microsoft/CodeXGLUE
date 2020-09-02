# -*- coding: utf-8 -*-
import torch

from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizer
import argparse
import json
import os

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)}


def get_cloze_words(filename, tokenizer):
    with open(filename, 'r', encoding='utf-8') as fp:
        words = fp.read().split('\n')
    words_id_map = {w: tokenizer.encoder[w] for w in words}
    return words_id_map


def test_single(text, real_token, model, words_id_map, tokenizer, device):
    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))[:510]
    inputs = tokenizer.build_inputs_with_special_tokens(tokenized_text)

    try:
        index = inputs.index(tokenizer.mask_token_id)
    except ValueError:
        return False, True
    inputs = torch.tensor([inputs])
    inputs = inputs.to(device)

    with torch.no_grad():
        scores = model(inputs)[0]
        score_list = scores[0][index]
        word_index = torch.LongTensor(list(words_id_map.values())).to(device)
        word_index = torch.zeros(score_list.shape[0]).to(device).scatter(0, word_index, 1)
        score_list = score_list + (1-word_index) * -1e6
        predict_word_id = torch.argmax(score_list).data

    is_correct = False
    if words_id_map[real_token] == predict_word_id:
        is_correct = True
    return is_correct, False


def cloze_test(args, lang, model, tokenizer, device):
    # TODO file name print variable 'selectBan'
    cloze_words_file = os.path.join('./data', 'cloze-'+args.cloze_mode, 'cloze_test_words.txt')
    file_path = os.path.join('./data', 'cloze-'+args.cloze_mode, lang, 'clozeTest.json')

    words_id_map = get_cloze_words(cloze_words_file, tokenizer)
    lines = json.load(open(file_path))

    results = []
    results_statistics_all = {w:0 for w in words_id_map.keys()}
    results_statistics_correct = {w:0 for w in words_id_map.keys()}
    for line in lines:
        real_token = line['word']
        text = ' '.join(line['nl_tokens'] + line['pl_tokens'])
        is_correct, is_break = test_single(text, real_token, model, words_id_map, tokenizer, device)
        if is_break:
            continue
        results_statistics_all[real_token] += 1
        if is_correct:
            results.append(1)
            results_statistics_correct[real_token] += 1
        else:
            results.append(0)
    with open(os.path.join('./data', 'cloze-'+args.cloze_mode, lang, 'cloze_test_statistics.txt'), 'a', encoding='utf-8') as fp:
        fp.write("\n".join([w+'\t'+str(results_statistics_correct[w])+'\t'+str(a)+'\t'+str(results_statistics_correct[w]*1.0/(a+0.1)) for w, a in results_statistics_all.items()]))
        fp.write("\n<ALL>\t{}\t{}\t{}".format(len(results), sum(results), sum(results) * 1.0 / len(results)))
    print('{} cloze test-{}, example num: {}, correct num: {}, acc: {}'.format(
            lang, args.cloze_mode, len(results), sum(results), sum(results) * 1.0 / len(results)))
    return results


def cloze_test_simple(args, lang, model, tokenizer, device):
    cloze_words_file = os.path.join('./data', 'cloze-' + args.cloze_mode, 'cloze_test_words.txt')
    file_path = os.path.join('./data', 'cloze-' + args.cloze_mode, lang, 'clozeTest.json')

    words_id_map = get_cloze_words(cloze_words_file, tokenizer)
    lines = json.load(open(file_path))

    results = []
    for line in lines:
        real_token = line['word']
        text = ' '.join(line['nl_tokens'] + line['pl_tokens'])
        is_correct, is_break = test_single(text, real_token, model, words_id_map, tokenizer, device)
        if is_break:
            continue
        if is_correct:
            results.append(1)
        else:
            results.append(0)
    print('{} cloze test-{}, example num: {}, correct num: {}, acc: {}'.format(
            lang, args.cloze_mode, len(results), sum(results), sum(results) * 1.0 / len(results)))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='roberta-base', help='"roberta-base" or "microsoft/codebert-base-mlm" or model path(pytorch_model.bin)')
    parser.add_argument('--cloze_mode', default='all', help='"all" or "maxmin" mode')
    args = parser.parse_args()

    config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
    config = config_class.from_pretrained('roberta-base')
    tokenizer = tokenizer_class.from_pretrained('roberta-base')
    model = RobertaForMaskedLM.from_pretrained(args.model, from_tf=bool('.ckpt' in args.model), config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # print('cloze test mode: {}, words number: {}'.format(args.cloze_mode, len(cloze_words_id_map)))
    print('cloze test mode: {}'.format(args.cloze_mode))
    cloze_results = []
    for lang in ['ruby', 'javascript', 'go', 'python', 'java', 'php']:
        cloze_results.extend(cloze_test(args, lang, model, tokenizer, device))

    print('Cloze test-{}, examples number: {}, correct number: {}, accuracy: {}'.format(args.cloze_mode,
                    len(cloze_results), sum(cloze_results), sum(cloze_results) * 1.0 / len(cloze_results)))


if __name__ == '__main__':
    main()
