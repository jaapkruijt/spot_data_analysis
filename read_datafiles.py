import statistics
from collections import Counter
import os
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import spacy
import torch
import seaborn as sns
import matplotlib.pyplot as plt

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
nlp = spacy.load('nl_core_news_sm')


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
    try:
        df['transaction_unit'] = df['transaction_unit'].astype(int)
        df['transaction_unit'] = df['transaction_unit'].astype(str)
    except ValueError:
        df['transaction_unit'] = df['transaction_unit'].astype(str)
    df['round'] = df['round'].astype(str)

    return df


def organize_mentions(datafile, experiment):
    mention_dict_per_tu = {}
    mention_dict_per_round = {}
    transaction_units = {}

    char_amount = 15 if experiment == 'AK' else 16
    for i in range(char_amount+1):
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
                    # if tu in transaction_units.keys():
                    #     transaction_units[tu]['characters'].append(character)
                    # else:
                    #     da_count = Counter()
                    #     transaction_units[tu] = {'characters': [character], 'length': 0, 'repair_amt': 0, 'da': da_count}
                    mention_tuple = (mention, tu_relation)
                    if game_round in mention_dict_per_tu[character].keys():
                        if tu in mention_dict_per_tu[character][game_round]:
                            mention_dict_per_tu[character][game_round][tu].append(mention_tuple)
                        else:
                            mention_dict_per_tu[character][game_round][tu] = [mention_tuple]
                    else:
                        mention_dict_per_tu[character][game_round] = {tu: [mention_tuple]}
    #
    # for ind in datafile.index:
    #     if datafile['transaction_unit'][ind] != '0' and datafile['round'][ind] != '0':
    #         separate_tu = datafile['transaction_unit'][ind].split(';')
    #         tu_relation = datafile['tu_relation'][ind]
    #         for tu in separate_tu:
    #             transaction_units[tu]['length'] += 1
    #             dialog_act = datafile['da'][ind]
    #             if tu_relation in ['req-repair', 'clar-repair']:
    #                 transaction_units[tu]['repair_amt'] += 1
    #             transaction_units[tu]['da'][dialog_act] += 1

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


def organize_transaction_units(datafile):
    transaction_units = {}

    for ind in datafile.index:
        if datafile['mention'][ind] != 0:
            separate_mentions = datafile['mention'][ind].split(';')
            separate_characters = datafile['character'][ind].split(';')
            separate_tu = datafile['transaction_unit'][ind].split(';')
            # game_round = datafile['round'][ind]
            # tu_relation = datafile['tu_relation'][ind]
            for i, mention in enumerate(separate_mentions):
                character = str(separate_characters[i])
                if len(separate_tu) != len(separate_mentions):
                    tu = str(separate_tu[0])
                else:
                    tu = separate_tu[i]
                if tu in transaction_units.keys():
                    transaction_units[tu]['characters'].append(character)
                else:
                    da_count = Counter()
                    transaction_units[tu] = {'characters': [character], 'length': 0, 'repair_amt': 0, 'da': da_count}
                # mention_tuple = (mention, tu_relation)
                # if game_round in mention_dict_per_tu[character].keys():
                #     if tu in mention_dict_per_tu[character][game_round]:
                #         mention_dict_per_tu[character][game_round][tu].append(mention_tuple)
                #     else:
                #         mention_dict_per_tu[character][game_round][tu] = [mention_tuple]
                # else:
                #     mention_dict_per_tu[character][game_round] = {tu: [mention_tuple]}

    for ind in datafile.index:
        if datafile['transaction_unit'][ind] != '0':
            separate_tu = datafile['transaction_unit'][ind].split(';')
            tu_relation = datafile['tu_relation'][ind]
            for tu in separate_tu:
                transaction_units[tu]['length'] += 1
                dialog_act = datafile['da'][ind]
                if tu_relation in ['req-repair', 'clar-repair']:
                    transaction_units[tu]['repair_amt'] += 1
                transaction_units[tu]['da'][dialog_act] += 1

    return transaction_units


def participant_linguistic_analysis(experiment):
    directory = '/Users/jaapkruijt/Documents/ALANI/SPOT/SPOT_pilotadata/SPOT-mentions'
    filenames = [f for f in os.listdir(directory) if not f.startswith('.')]
    mc_mentions = {}
    sc_mentions = {}
    char_amount = 15 if experiment == 'AK' else 13
    round_amounts = ['1','2','3','4','5','6']
    character_mean_sim = {'character': [], 'round': [], 'sim_score': []}
    for i in ['1','2','3']:
        mc_mentions[i] = {}
        for j in round_amounts:
            mc_mentions[i][j] = []
    for i in range(4, char_amount+1):
        sc_mentions[str(i)] = {}
        for j in round_amounts:
            sc_mentions[str(i)][str(j)] = []
    for filename in filenames:
        exp = os.path.splitext(filename)[0].split('_')[0]
        if exp != experiment:
            continue
        f = os.path.join(directory, filename)
        data = prepare_tsv_datafile(f)
        mentions_per_tu, mentions_per_round = organize_mentions(data, experiment)
        for character, rounds in mentions_per_round.items():
            if character == '0':
                continue
            if character not in ['1', '2', '3']:
                for game_round, mentions in rounds.items():
                    if game_round == '0':
                        continue
                    sc_mentions[character][game_round].extend(mentions)
            else:
                for game_round, mentions in rounds.items():
                    if game_round == '0':
                        continue
                    mc_mentions[character][game_round].extend(mentions)
    inner_mentions_count = Counter()
    outer_mentions_count = Counter()
    mentions_round_6 = {'inner': inner_mentions_count, 'outer': outer_mentions_count}
    for character, rounds in mc_mentions.items():
        for game_round, mentions in rounds.items():
            for mention in mentions:
                doc = nlp(mention)
                root = [token for token in doc if token.head == token][0]
                mentions_round_6['inner'][root.pos_] += 1
            mention_embeddings = model.encode(mentions)
            sim_scores = util.cos_sim(mention_embeddings, mention_embeddings)
            mean = torch.mean(sim_scores)
            character_mean_sim['character'].append(character)
            character_mean_sim['round'].append(game_round)
            character_mean_sim['sim_score'].append(float(mean))
    for character, rounds in sc_mentions.items():
        for game_round, mentions in rounds.items():
            for mention in mentions:
                doc = nlp(mention)
                root = [token for token in doc if token.head == token][0]
                mentions_round_6['outer'][root.pos_] += 1
    dataframe = pd.DataFrame.from_dict(character_mean_sim)
    return dataframe, mc_mentions, mentions_round_6


def dataset_details(experiment, directory):
    details = {'utterances': 0, 'mentions': 0, 'turns': 1, 'tu': [], 'repair': 0, 'da': Counter(), 'avg_length': []}
    filenames = [f for f in os.listdir(directory) if not f.startswith('.')]
    for filename in filenames:
        exp = os.path.splitext(filename)[0].split('_')[0]
        if exp != experiment:
            continue
        f = os.path.join(directory, filename)
        dataset = prepare_tsv_datafile(f)
        for ind in dataset.index:
            if ind > 0:
                if dataset['speaker'][ind] != dataset['speaker'][ind-1]:
                    details['turns'] += 1
            details['utterances'] += 1
            if dataset['mention'][ind] != 0:
                details['mentions'] += 1
            if dataset['transaction_unit'][ind] != '0':
                if dataset['tu_relation'][ind] in ['clar-repair', 'req-repair', 'correction']:
                    details['repair'] += 1
            dialog_act = dataset['da'][ind]
            details['da'][dialog_act] += 1
            transaction_units = organize_transaction_units(dataset)
            units = sorted([int(key) for key in transaction_units.keys()])[-1]
            details['tu'].append(units)
            for tu, tu_details in transaction_units.items():
                details['avg_length'].append(tu_details['length'])

    return details


def character_tu_analysis(characters):
    character_tu_data = {}
    for character, game_rounds in characters.items():
        character_tu_data[character] = {}
        for game_round, tus in game_rounds.items():
            character_tu_data[character][game_round] = {'tu_amt': len(tus)}

    return character_tu_data


def character_linguistic_analysis(characters):
    character_lemma_counts = {}
    character_data = {}

    for character, game_rounds in characters.items():
        lemma_count = Counter()
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
            num_mentions = len(mentions)
            avg_desc_length = sum(desc_lengths)/len(desc_lengths)
            # character_desc_length[character][game_round] = avg_desc_length
            character_data[character][game_round]['desc_length'] = avg_desc_length
            character_data[character][game_round]['num_mentions'] = num_mentions
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


def combine_character_data(participant_number, experiment, character_data, tu_data, combined_data):
    for character, rounds in character_data.items():
        for game_round, values in rounds.items():
            if not values:
                continue
            else:
                if experiment == 'EN':
                    combined_data['experiment'].append('1')
                    combined_data['character'].append(f'1.{character}')
                elif experiment == 'AK':
                    combined_data['experiment'].append('2')
                    combined_data['character'].append(f'2.{character}')
                combined_data['participant'].append(participant_number)
                combined_data['round'].append(game_round)
                if character in ['1','2','3']:
                    combined_data['circle'].append('inner')
                else:
                    combined_data['circle'].append('outer')
                combined_data['utt_len'].append(values['desc_length'])
                combined_data['num_mentions'].append(values['num_mentions'])
                if 'sim_score' in values:
                    combined_data['sim_score'].append(values['sim_score'])
                else:
                    combined_data['sim_score'].append(None)
    for character, rounds in tu_data.items():
        for game_round, values in rounds.items():
            if not values:
                continue
            else:
                combined_data['num_tus'].append(values['tu_amt'])


def combine_experiments():
    directory = '/Users/jaapkruijt/Documents/ALANI/SPOT/SPOT_pilotadata/SPOT-mentions'
    character_data = {'experiment': [], 'participant': [], 'character': [], 'round': [], 'circle': [], 'utt_len': [],
                    'sim_score': [], 'num_mentions': [], 'num_tus': []}
    transaction_data = {'experiment': [], 'participant': [], 'transaction_unit': [], 'length': [],
                        'repair_relations': [], 'dialog act': []}
    filenames = [f for f in os.listdir(directory) if not f.startswith('.')]
    for filename in filenames:
        f = os.path.join(directory, filename)
        experiment = os.path.splitext(filename)[0].split('_')[0]
        participant = os.path.splitext(filename)[0].split('_')[1]
        mentions = prepare_tsv_datafile(f)
        mentions_per_tu, mentions_per_round= organize_mentions(mentions, experiment)
        word_counts, mention_data = character_linguistic_analysis(mentions_per_round)
        tu_data = character_tu_analysis(mentions_per_tu)
        combine_character_data(participant, experiment, mention_data, tu_data, character_data)

    dataframe = pd.DataFrame.from_dict(character_data)
    return dataframe


def show_dataset_details():
    en_dataset_details = dataset_details('EN', '/Users/jaapkruijt/Documents/ALANI/SPOT/SPOT_pilotadata/SPOT-mentions')
    ak_dataset_details = dataset_details('AK', '/Users/jaapkruijt/Documents/ALANI/SPOT/SPOT_pilotadata/SPOT-mentions')
    print('Experiment 1:')
    print('utterances:')
    print(en_dataset_details['utterances'])
    print('mentions:')
    print(en_dataset_details['mentions'])
    print('turns:')
    print(en_dataset_details['turns'])
    print('median no transaction units:')
    print(statistics.median(en_dataset_details['tu']))
    print('repair:')
    print(en_dataset_details['repair'])
    print('dialog acts:')
    print(en_dataset_details['da'].most_common())
    print('median tu length:')
    print(statistics.median(en_dataset_details['avg_length']))
    print('Experiment 2:')
    print('utterances:')
    print(ak_dataset_details['utterances'])
    print('mentions:')
    print(ak_dataset_details['mentions'])
    print('turns:')
    print(ak_dataset_details['turns'])
    print('median no transaction units:')
    print(statistics.median(ak_dataset_details['tu']))
    print('repair:')
    print(ak_dataset_details['repair'])
    print('dialog acts:')
    print(ak_dataset_details['da'].most_common())
    print('median tu length')
    print(statistics.median(ak_dataset_details['avg_length']))


if __name__ == "__main__":
    frame = combine_experiments()
    # per_tu, per_round = organize_mentions()
    frame.to_pickle('data.pkl')
    en_mean_sims, en_mentions, en_round6 = participant_linguistic_analysis('EN')
    ak_mean_sims, ak_mentions, ak_round6 = participant_linguistic_analysis('AK')
    print('Experiment 1:')
    for gameround, mentions in en_mentions['2'].items():
        print(mentions)
    print('Experiment 2:')
    for gameround, mentions in ak_mentions['3'].items():
        print(mentions)
    # print(en_round6)
    # print(ak_round6)
    # print(ak_mean_sims)
    # en_mean_sims.to_pickle('en_sim.pkl')
    # ak_mean_sims.to_pickle('ak_sim.pkl')
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




