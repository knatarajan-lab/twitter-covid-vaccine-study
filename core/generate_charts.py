# Python imports
from datetime import datetime

# Third-party imports
import matplotlib as mpl
from pandas import read_csv, merge, date_range

# Project imports
from resources.constants import data_path

mpl.rcParams.update({'font.size': 20})


def read_bq_out():
    vax_trends_path = data_path / 'vax_trends_updated.csv'
    vax_trends_df = read_csv(
        vax_trends_path,
        delimiter=',',
        names=['ne', 'count', 'created_at'],
        header=1,
        parse_dates=['created_at'],
        date_parser=lambda x: datetime.strptime(x.strip(), '%Y-%m-%d'))
    nes = vax_trends_df['ne'].unique().tolist()

    new_df = vax_trends_df['created_at'].drop_duplicates(keep='first').reset_index(drop=True)

    for ne in nes:
        ne_df = vax_trends_df[vax_trends_df['ne'] == ne][[
            'created_at', 'count'
        ]]
        ne_df = ne_df.rename(
            columns={'count': f'{ne.replace(" ", "_")}'})
        new_df = merge(new_df, ne_df, on='created_at', how='left')

    new_df.set_index(keys=['created_at'], inplace=True)
    order = new_df[new_df.index == '2021-05-01'].to_dict('records')[0]
    nes.sort(key=lambda x: -order[x])
    plot = new_df.plot(logy=True, linewidth=3.0)

    events_df = read_csv(
        data_path / 'event_dates.csv',
        delimiter=',',
        names=['event', 'date'],
        header=1,
        parse_dates=['date'],
        date_parser=lambda x: datetime.strptime(x.strip(), '%Y-%m-%d'))

    events = events_df['event'].to_list()
    event_dates = events_df['date'].to_list()

    for i, event_date in enumerate(event_dates):
        plot.vlines(x=event_date,
                    ymin=0,
                    ymax=10 ** 4,
                    linestyles='dotted',
                    colors='orange',
                    linewidth=2.0)
        plot.text(event_date,
                  10 ** 4.1,
                  events[i],
                  rotation=90,
                  horizontalalignment='center',
                  verticalalignment='top',
                  bbox={
                      'facecolor': 'white',
                      'pad': 4
                  })
    plot.set_xlabel("Date of Tweets")
    plot.set_ylabel("Number of Tweets")
    dates = date_range(start='11/30/2020', end='1/15/2022', freq='MS')
    dates = dates.insert(len(dates), datetime.fromisoformat('2022-01-15'))
    plot.axes.set_xticks(dates)
    date_str = [datetime.strftime(date, '%Y-%m-%d') for date in dates.to_list()[:-1]] + [""]
    plot.axes.set_xticklabels(date_str)

    handles, labels = plot.get_legend_handles_labels()

    # specify order of items in legend
    curr_order = {label: i for i, label in enumerate(labels)}
    order = [curr_order[label] for label in nes]

    # add legend to plot
    plot.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    fig = plot.get_figure()
    fig.set_size_inches(50, 20)
    fig.tight_layout(pad=3)

    fig.savefig(data_path / 'vax_trends_new.png')


if __name__ == '__main__':
    read_bq_out()
