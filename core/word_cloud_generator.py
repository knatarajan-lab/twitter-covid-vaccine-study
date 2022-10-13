import matplotlib.pyplot as plt
from pandas import read_csv
from wordcloud import WordCloud

from resources.constants import data_path


def main(frequencies):
    wc = WordCloud(width=1600, height=800)
    wc.generate_from_frequencies(frequencies=frequencies)
    plt.figure(figsize=(20, 10), facecolor='k')
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(data_path / 'wordcloud.png')


if __name__ == '__main__':
    freq_path = data_path / 'wordcloud.csv'
    df = read_csv(freq_path, names=['ne', 'count'], header=0)
    freqs = dict(zip(df['ne'], df['count']))
    main(freqs)
