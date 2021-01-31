"""
Module: LibUNTREF.B_Audio
"""

import librosa
import soundfile as sf
import IPython.display as ipd
import pandas as pd


def audio_player_list(signals, rates, width=270, height=40, columns=None, column_align='center'):
    """Generate list of audio player
    Args:
        signals: List of audio signals
        rates: List of sample rates
        width: Width of player (either number or list)
        height: Height of player (either number or list)
        columns: Column headings
        column_align: Left, center, right
    """
    pd.set_option('display.max_colwidth', -1)

    if isinstance(width, int):
        width = [width] * len(signals)
    if isinstance(height, int):
        height = [height] * len(signals)

    audio_list = []
    for cur_x, cur_Fs, cur_width, cur_height in zip(signals, rates, width, height):
        audio_html = ipd.Audio(data=cur_x, rate=cur_Fs)._repr_html_()
        audio_html = audio_html.replace('\n', '').strip()
        audio_html = audio_html.replace('<audio ', f'<audio style="width: {cur_width}px; height: {cur_height}px" ')
        audio_list.append([audio_html])

    df = pd.DataFrame(audio_list, index=columns).T
    table_html = df.to_html(escape=False, index=False, header=bool(columns))
    table_html = table_html.replace('<th>', f'<th style="text-align: {column_align}">')
    ipd.display(ipd.HTML(table_html))
