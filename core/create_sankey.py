import plotly.graph_objects as go

from resources.constants import data_path


def main():
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="blue", width=0.5),
            label=["Tweets from keyword search", "Relevant tweets", "Irrelevant tweets",
                   "Self-reported with side-effects", "Self-reported with no side-effect info", "COVID-related tweets",
                   "Not self-reported/other", "CLAMP input", "All excluded tweets"],
            color="green"
        ),
        link=dict(
            source=[0, 0, 1, 1, 1, 1, 3, 4, 5, 6, 7],
            target=[1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8],
            value=[24, 4, 6, 4.5, 0.7, 13, 6, 4.5, 0.7, 13, 6]
        ))])

    fig.update_layout(title_text="Tweets flow", font_size=10)
    fig.write_image(data_path / 'sankey.png')


if __name__ == '__main__':
    main()
