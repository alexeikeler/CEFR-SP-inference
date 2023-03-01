import itertools
from itertools import takewhile, repeat
from pathlib import Path
from typing import Union

import torch
from transformers import AutoTokenizer
from argparse import ArgumentParser
from model import LevelEstimaterClassification, LevelEstimaterContrastive
from tqdm import tqdm
from jsonlines import open as open_jsonl


# https://stackoverflow.com/a/27518377/4243650
def fast_linecount(filename: Union[str, Path]) -> int:
    with open(filename, 'rb') as f:
        bufgen = takewhile(lambda x: x, (f.raw.read(1024 * 1024) for _ in repeat(None)))
        linecount = sum(buf.count(b'\n') for buf in bufgen)

    return linecount


def main():
    parser = ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--encoder', default='bert-base-cased')
    parser.add_argument('--model_type', choices=['regression', 'metric'], default='regression')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--output', type=Path, default=Path('cefr_scores.jsonl'))

    args = parser.parse_args()
    model_class = {
        'regression': LevelEstimaterClassification,
        'metric': LevelEstimaterContrastive
    }[args.model_type]
    print('Loading model and tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    model = model_class.load_from_checkpoint(args.model)
    if torch.cuda.is_available():
        model.cuda()

    input_count = fast_linecount(args.input)

    with open_jsonl(args.output, 'w') as outfile:
        with open_jsonl(args.input, 'r') as infile:
            for _ in tqdm(range(0, input_count, args.batch_size), desc='Assining difficulties'):
                text_input = [x['text'] for x in itertools.islice(infile, args.batch_size)]
                inputs = tokenizer(text_input, return_tensors='pt', padding=True)
                with torch.no_grad():
                    outputs = model(inputs.to(model.device), return_logits=True).squeeze().tolist()

                assert len(text_input) == len(outputs)
                for text, difficulty in zip(text_input, outputs):
                    outfile.write({
                        'text': text,
                        'diff': difficulty
                    })

    print('Done')


if __name__ == '__main__':
    main()
