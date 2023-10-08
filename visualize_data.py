import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

if __name__ == "__main__":
    frame = pd.read_pickle('data.pkl')
    sns.set_theme()
    sns.set_palette('colorblind')
    sns.catplot(frame,  kind='bar', x='round', y='utt_len', hue='circle', col='experiment', col_order=['1', '2'])
    # plt.show()
    # plt.gcf()
    plt.savefig('utt_len.png', bbox_inches='tight')
    sns.catplot(frame, kind='bar', x='round', y='num_mentions', hue='circle', col='experiment', col_order=['1','2'])
    plt.savefig('num_mentions.png', bbox_inches='tight')
    sim_frame = frame[(frame['circle'] == 'inner') & ~(frame['round'] == '1')]
    sim_frame_en = frame[(frame['circle'] == 'inner') & ~(frame['round'] == '1') & (frame['experiment'] == '1')]
    sim_frame_ak = frame[(frame['circle'] == 'inner') & ~(frame['round'] == '1') & (frame['experiment'] == '2')]
    # g = sns.FacetGrid(sim_frame, col='experiment', col_order=['1', '2'], hue='character')
    # g.map(sns.catplot, 'round', 'sim_score')
    fig, (ax1,ax2) = plt.subplots(1, 2, sharey=True)
    fig.set_figwidth(10)
    palette = sns.color_palette()
    sns.barplot(sim_frame_en, ax=ax1, x='round', y='sim_score', hue='character')
    sns.barplot(sim_frame_ak, ax=ax2, x='round', y='sim_score', hue='character', palette=palette[3:6])
    plt.savefig('sim_score.png', bbox_inches='tight')
    ak_sim = pd.read_pickle('ak_sim.pkl')
    en_sim = pd.read_pickle('en_sim.pkl')
    # print(ak_sim)
    fig2, (ax3,ax4) = plt.subplots(1, 2, sharey=True)
    sns.barplot(en_sim, ax=ax3, x='round', y='sim_score', hue='character')
    sns.barplot(ak_sim, ax=ax4, x='round', y='sim_score', hue='character')
    plt.show()