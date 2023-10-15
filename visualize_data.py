import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
import pandas as pd

if __name__ == "__main__":
    frame = pd.read_pickle('data.pkl')
    sns.set_theme()
    # sns.set_theme(rc={'figure.figsize':(12,3)})
    sns.set_palette('colorblind')
    g = sns.catplot(frame,  kind='bar', x='round', y='utt_len', hue='circle', col='experiment', col_order=['1','2'],
                    height=4, aspect=1.3)
    # plt.ylabel('Mean utterance length')
    g.set_ylabels('Mean utterance length')
    g.set_titles('Experiment {col_name}')
    # rcParams['figure.figsize'] = 12,3
    plt.savefig('utt_len.png', bbox_inches='tight', dpi=200)
    h = sns.catplot(frame, kind='bar', x='round', y='num_mentions', hue='circle', col='experiment', col_order=['1','2'],
                    height=4, aspect=1.3)
    # plt.ylabel('Mean number of mentions')
    h.set_ylabels('Mean number of mentions')
    h.set_titles('Experiment {col_name}')
    plt.savefig('num_mentions.png', bbox_inches='tight', dpi=200)
    sns.catplot(frame, kind='bar', x='round', y='num_tus', hue='circle', col='experiment', col_order=['1','2'])
    plt.show()
    sim_frame = frame[(frame['circle'] == 'inner') & ~(frame['round'] == '1')]
    sim_frame_en = frame[(frame['circle'] == 'inner') & ~(frame['round'] == '1') & (frame['experiment'] == '1')]
    sim_frame_ak = frame[(frame['circle'] == 'inner') & ~(frame['round'] == '1') & (frame['experiment'] == '2')]
    fig, (ax1,ax2) = plt.subplots(1, 2, sharey=True)
    ax1.set_title('Experiment 1')
    ax2.set_title('Experiment 2')
    ax1.set(ylabel='Cosine similarity score')
    plt.subplots_adjust(wspace=0.1)
    fig.set_figwidth(12)
    fig.set_figheight(4)
    palette = sns.color_palette()
    sns.barplot(sim_frame_en, ax=ax1, x='round', y='sim_score', hue='character')
    sns.barplot(sim_frame_ak, ax=ax2, x='round', y='sim_score', hue='character', palette=palette[3:6])
    xtick_labels = ['1-2', '2-3', '3-4', '4-5', '5-6']
    ax1.set_xticks([0,1,2,3,4])
    ax2.set_xticks([0,1,2,3,4])
    ax1.set_xticklabels(xtick_labels)
    ax2.set_xticklabels(xtick_labels)
    # ax1.get_legend().set_visible(False)
    # ax2.get_legend().set_visible(False)
    plt.savefig('sim_score.png', bbox_inches='tight', dpi=200)
    ak_sim = pd.read_pickle('ak_sim.pkl')
    en_sim = pd.read_pickle('en_sim.pkl')
    # print(ak_sim)
    fig2, (ax3,ax4) = plt.subplots(1, 2, sharey=True)
    sns.barplot(en_sim, ax=ax3, x='round', y='sim_score', hue='character')
    sns.barplot(ak_sim, ax=ax4, x='round', y='sim_score', hue='character')
    plt.show()
