import itertools
from itertools import takewhile, repeat
from pathlib import Path
from typing import Union
import numpy as np
import torch
from transformers import AutoTokenizer
from argparse import ArgumentParser
from model import LevelEstimaterClassification, LevelEstimaterContrastive
from tqdm import tqdm
from util import convert_numeral_to_six_levels

#from jsonlines import open as open_jsonl

CEFR_LEVELS = {
    0: "A1",
    1: "A2",
    2: "B1",
    3: "B2",
    4: "C1",
    5: "C2"
}


# https://stackoverflow.com/a/27518377/4243650
def fast_linecount(filename: Union[str, Path]) -> int:
    with open(filename, 'rb') as f:
        bufgen = takewhile(lambda x: x, (f.raw.read(1024 * 1024) for _ in repeat(None)))
        linecount = sum(buf.count(b'\n') for buf in bufgen)

    return linecount


def main():
    parser = ArgumentParser()
    parser.add_argument('--model', required=True)
    #parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--encoder', default='bert-base-cased')
    parser.add_argument('--model_type', choices=['regression', 'metric'], default='regression')
    parser.add_argument('--batch_size', type=int, default=20)
    #Sparser.add_argument('--output', type=str, required=True)

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

    with open("/home/oleksiik/Omelas_sentences.txt", "r") as file:
        sentences = file.read().split("\n")

    #sentences = [
    #    "They leave Omelas, they walk ahead into the darkness, and they do not come back.",
    #    "Each alone, they go west or north, towardsthe mountains.",
    #    "Often the young people go home in tears, or in a tearless rage, when they have seen the childand faced this terrible paradox."
    #]
    
    inputs = tokenizer(sentences, return_tensors='pt', padding=True)
    with torch.no_grad():
        outputs = model(inputs.to(model.device), return_logits=True).cpu().detach().clone().numpy()
    
    num_levels = convert_numeral_to_six_levels(np.array(outputs))
    num_levels = np.concatenate(num_levels)
    print(num_levels)
    cefr_levels = [CEFR_LEVELS[sent_level] for sent_level in num_levels] 
    print(len(sentences), len(cefr_levels))

    for sent, diff in zip(sentences, cefr_levels):
        print(
            f"Sentece:\n{sent}\nCEFR level: {diff}\n\n"
        )

#    print(outputs)


    #input_count = fast_linecount(args.input)

    # with open_jsonl(args.output, 'w') as outfile:
    #     with open_jsonl(args.input, 'r') as infile:
    #         for _ in tqdm(range(0, input_count, args.batch_size), desc='Assining difficulties'):
    #             text_input = [x['text'] for x in itertools.islice(infile, args.batch_size)]
    #             inputs = tokenizer(text_input, return_tensors='pt', padding=True)
    #             with torch.no_grad():
    #                 outputs = model(inputs.to(model.device), return_logits=True).squeeze().tolist()

    #             assert len(text_input) == len(outputs)
    #             for text, difficulty in zip(text_input, outputs):
    #                 outfile.write({
    #                     'text': text,
    #                     'diff': difficulty
    #                 })

    # print('Done')


    # with open(args.input, "r") as infile:
    #     text = infile.read()
    #     sentences = text.split("\n")

    # with open(args.output, "w") as outfile:

    #     for _ in tqdm(range(0, len(sentences), args.batch_size), desc="Assigning difficulties"):
    #         inputs = tokenizer(sentences, return_tensors="pt", padding=True)
    #         with torch.no_grad():
    #             outputs = model(inputs.to(model.device), return_logits=True).squeeze().tolist()
    #         assert len(sentences) == len(outputs)

    #         for text, difficulty in zip(sentences, outputs):
    #             outfile.write(
    #                 f"Sentence:\n{text}\nLevel:{difficulty}\n\n"
    #             )

    # print('Done')


if __name__ == '__main__':
    main()
