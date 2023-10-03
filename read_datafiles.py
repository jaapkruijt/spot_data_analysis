from collections import Counter
import os
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import spacy
import torch
import seaborn as sns
import matplotlib.pyplot as plt

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')


def prepare_tsv_datafile(path):
    df = pd.read_csv(path, sep='\t')
    df['character'] = df['character'].fillna(0)
    df['transaction_unit'] = df['transaction_unit'].fillna(0)
    df['mention'] = df['mention'].fillna(0)
    try:
        df['character'] = df['character'].astype(int)
        df['character'] = df['character'].astype(str)
    except ValueError:
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
                            new_mentions[-1] = ' '.join([new_mentions[-1], mention[0]])
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
    # character_desc_length = {}
    # character_similarity_scores = {}
    character_data = {}

    for character, game_rounds in characters.items():
        lemma_count = Counter()
        # character_desc_length[character] = {}
        # character_similarity_scores[character] = {}
        character_data[character] = {}
        for game_round, mentions in game_rounds.items():
            desc_lengths = []
            character_data[character][game_round] = {}
            for mention in mentions:
                desc_length = len(mention.split(' '))
                desc_lengths.append(desc_length)
                doc = nlp(mention)
                for token in doc:
                    if token.pos_ != 'PUNCT':
                        lemma_count[token.lemma_] += 1
            avg_desc_length = sum(desc_lengths)/len(desc_lengths)
            # character_desc_length[character][game_round] = avg_desc_length
            character_data[character][game_round]['desc_length'] = avg_desc_length
            if game_rounds.get(str(int(game_round)-1)) is not None:
                mentions_previous = game_rounds[str(int(game_round)-1)]
                embeddings_current = model.encode(mentions, convert_to_tensor=True)
                embeddings_previous = model.encode(mentions_previous, convert_to_tensor=True)
                sim_scores = util.cos_sim(embeddings_previous, embeddings_current)
                sim_scores = torch.flatten(sim_scores)
                highest = sorted(sim_scores, reverse=True)[0]
                # character_similarity_scores[character][game_round] = float(highest)
                character_data[character][game_round]['sim_score'] = float(highest)

        character_lemma_counts[character] = lemma_count.most_common()

    return character_lemma_counts, character_data


def combine_data(participant_number, experiment, character_data, combined_data):
    for character, rounds in character_data.items():
        for round, values in rounds.items():
            if not values:
                continue
            else:
                combined_data['experiment'].append(experiment)
                combined_data['participant'].append(participant_number)
                combined_data['character'].append(character)
                combined_data['round'].append(round)
                if character in ['1','2','3']:
                    combined_data['circle'].append('inner')
                else:
                    combined_data['circle'].append('outer')
                combined_data['utt_len'].append(values['desc_length'])
                if 'sim_score' in values:
                    combined_data['sim_score'].append(values['sim_score'])
                else:
                    combined_data['sim_score'].append(None)


def main():
    directory = '/Users/jaapkruijt/Documents/ALANI/SPOT/SPOT_pilotadata/SPOT-mentions'
    data = {'experiment': [], 'participant': [], 'character': [], 'round': [], 'circle': [], 'utt_len': [],
                    'sim_score': []}
    filenames = [f for f in os.listdir(directory) if not f.startswith('.')]
    for filename in filenames:
        f = os.path.join(directory, filename)
        experiment = os.path.splitext(filename)[0].split('_')[0]
        participant = os.path.splitext(filename)[0].split('_')[1]
        mentions = prepare_tsv_datafile(f)
        mentions_per_tu, mentions_per_round = organize_mentions(mentions, experiment)
        word_counts, character_data = linguistic_analysis(mentions_per_round)
        combine_data(participant, experiment, character_data, data)

    dataframe = pd.DataFrame.from_dict(data)
    sns.set_theme()
    sns.catplot(dataframe, kind='bar', x='round', y='sim_score', hue='circle', col='experiment')
    plt.show()
    return dataframe





if __name__ == "__main__":
    main()
    # for char, g_round in per_tu.items():
    #     print(f'CHARACTER: {char}')
    #     for r, unit in g_round.items():
    #         print(f'ROUND: {r}')
    #         for u, ment in unit.items():
    #             print(f'UNIT: {u}')
    #             print(ment)
    # for char, g_round in per_round.items():
    #     print(f'CHARACTER: {char}')
    #     for r, mentions in g_round.items():
    #         print(f'ROUND: {r}')
    #         for ment in mentions:
    #             print(ment)
    # print(counts)
    # print(data)
    # print(similarity)

