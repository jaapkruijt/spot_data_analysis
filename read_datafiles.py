from collections import Counter
import os
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import spacy
import torch

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

def prepare_tsv_datafile(path):
    df = pd.read_csv(path, sep='\t')
    df['character'] = df['character'].fillna(0)
    df['transaction_unit'] = df['transaction_unit'].fillna(0)
    df['mention'] = df['mention'].fillna(0)
    df['character'] = df['character'].astype(str)
    df['transaction_unit'] = df['transaction_unit'].astype(str)
    df['round'] = df['round'].astype(str)

    return df


def organize_mentions(datafile, experiment):
    mention_dict_per_tu = {}
    mention_dict_per_round = {}

    if experiment == 'AK':
        for i in range(16):
            mention_dict_per_tu[str(i)] = {}
    elif experiment == 'EN':
        for i in range(17):
            mention_dict_per_tu[str(i)] = {}

    for ind in datafile.index:
        if datafile['speaker'][ind] == 'robot' or datafile['round'][ind] == '0':
            continue
        else:
            if datafile['mention'][ind] != 0:
                separate_mentions = datafile['mention'][ind].split(';')
                separate_characters = datafile['character'][ind].split(';')
                separate_tu = datafile['transaction_unit'][ind].split(';')
                game_round = datafile['round'][ind]
                tu_relation = datafile['tu_relation'][ind]
                for i, mention in enumerate(separate_mentions):
                    character = str(separate_characters[i])
                    if len(separate_tu) != len(separate_mentions):
                        tu = str(separate_tu[0])
                    else:
                        tu = separate_tu[i]
                    mention_tuple = (mention, tu_relation)
                    if game_round in mention_dict_per_tu[character].keys():
                        if tu in mention_dict_per_tu[character][game_round]:
                            mention_dict_per_tu[character][game_round][tu].append(mention_tuple)
                        else:
                            mention_dict_per_tu[character][game_round][tu] = [mention_tuple]
                    else:
                        mention_dict_per_tu[character][game_round] = {tu: [mention_tuple]}

    for character, game_rounds in mention_dict_per_tu.items():
        for game_round, units in game_rounds.items():
            for unit, mentions in units.items():
                new_mentions = []
                for i, mention in enumerate(mentions):
                    if i == 0:
                        new_mentions.append(mention[0])
                    else:
                        if mention[1] == 'continue-description':
                            new_mentions[i - 1] = ' '.join([new_mentions[i - 1], mention[0]])
                        else:
                            new_mentions.append(mention[0])
                mention_dict_per_tu[character][game_round][unit] = new_mentions

    for character, game_rounds in mention_dict_per_tu.items():
        mention_dict_per_round[character] = {}
        for game_round, units in game_rounds.items():
            mentions = [mention for mentions in units.values() for mention in mentions]
            mention_dict_per_round[character][game_round] = mentions

    return mention_dict_per_tu, mention_dict_per_round


def linguistic_analysis(characters):
    nlp = spacy.load('nl_core_news_sm')
    character_lemma_counts = {}
    character_desc_length = {}
    character_similarity_scores = {}

    for character, game_rounds in characters.items():
        lemma_count = Counter()
        character_desc_length[character] = {}
        character_similarity_scores[character] = {}
        for game_round, mentions in game_rounds.items():
            desc_lengths = []
            for mention in mentions:
                desc_length = len(mention.split(' '))
                desc_lengths.append(desc_length)
                doc = nlp(mention)
                for token in doc:
                    if token.pos_ != 'PUNCT':
                        lemma_count[token.lemma_] += 1
            avg_desc_length = sum(desc_lengths)/len(desc_lengths)
            character_desc_length[character][game_round] = avg_desc_length
            if game_rounds.get(str(int(game_round)-1)) is not None:
                mentions_previous = game_rounds[str(int(game_round)-1)]
                embeddings_current = model.encode(mentions, convert_to_tensor=True)
                embeddings_previous = model.encode(mentions_previous, convert_to_tensor=True)
                sim_score = util.cos_sim(embeddings_previous, embeddings_current)
                mean = torch.mean(sim_score)
                character_similarity_scores[character][game_round] = mean

        character_lemma_counts[character] = lemma_count.most_common()

    return character_lemma_counts, character_desc_length, character_similarity_scores


if __name__ == "__main__":
    data = prepare_tsv_datafile('/Users/jaapkruijt/Documents/ALANI/SPOT/SPOT_pilotadata/SPOT-mentions/EN_12.tsv')
    per_tu, per_round = organize_mentions(data, 'EN')
    for char, g_round in per_tu.items():
        print(f'CHARACTER: {char}')
        for r, unit in g_round.items():
            print(f'ROUND: {r}')
            for u, ment in unit.items():
                print(f'UNIT: {u}')
                print(ment)
    for char, g_round in per_round.items():
        print(f'CHARACTER: {char}')
        for r, mentions in g_round.items():
            print(f'ROUND: {r}')
            for ment in mentions:
                print(ment)
    counts, length, similarity = linguistic_analysis(per_round)
    print(counts)
    print(length)
    print(similarity)
