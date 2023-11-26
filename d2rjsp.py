import streamlit as st
import logging
import requests
import pandas as pd
import re
import ast

import plotly.express as px

from urllib.parse import urljoin
from bs4 import BeautifulSoup

# ============================================================
# Constants
# ============================================================
LADDER_SEASON_3 = 99167800      # Starting Post ID for Ladder Season 3
LADDER_SEASON_4 = 100753157     # Starting Post ID for Ladder Season 4
LADDER_SEASON_5 = 101872270     # Starting Post ID for Ladder Season 5
SEASONS = ["Season 3", "Season 4", "Season 5"]
FORUMS = {SEASONS[0]: "D2:R Softcore Ladder 2022+", SEASONS[1]: "D2:R Softcore Ladder 2022+", SEASONS[2]: "D2:R Softcore Ladder Trading"}
SEASONS_STARTING_POST_ID = {SEASONS[0]: LADDER_SEASON_3, SEASONS[1]: LADDER_SEASON_4, SEASONS[2]: LADDER_SEASON_5}
RUNES = ['el', 'eld', 'tir', 'nef', 'eth', 'ith', 'tal', 'ral', 'ort', 'thul', 'amn', 'sol', 'shael', 'dol', 'hel', 'io', 'lum', 'ko', 'fal', 'lem', 'pul', 'um', 'mal', 'ist', 'gul', 'vex', 'ohm', 'lo', 'sur', 'ber', 'jah', 'cham', 'zod']
LOW_RUNES = RUNES[:16]
MED_RUNES = RUNES[17:24]
HIGH_RUNES = RUNES[25:]
RUNES_KEYWORDS = ['rune', 'runes'] + RUNES
DEFAULT_NB_THREADS = 30000

# ============================================================
# Functions
# ============================================================
def get_threads_df(start_id, start_id_offset=0, nb_thread=1000, season="Season 3", progress_bar=None):
    threads = []
    nb_threads_to_fetch = (start_id + start_id_offset + nb_thread) - (start_id + start_id_offset)
    nb_threads_fetched = 0
    for cur_id in range(start_id + start_id_offset, start_id + start_id_offset + nb_thread):
        page = requests.get(f"https://forums.d2jsp.org/topic.php?t={cur_id}")
        soup = BeautifulSoup(page.text)
        topic = soup.find("div", {"class": "c p3"})

        if topic and FORUMS[season] in topic.text:
            anchors = topic.find_all('a')

            # Extract the title and category
            title = topic.text.split('>')[-1].strip()
            category = anchors[-1].get_text(strip=True)

            posts = []
            posts_in_thread = soup.find_all("dl")
            for post in posts_in_thread:
                post_section_text = post.find('div', class_='ppc')
                if post_section_text is not None:
                    post_time = post_section_text.find('div', {'class': 'desc cl rc'}).text.strip().split('\n')[1]
                    post_text = post_section_text.find('div', {'class': 'bts'}).text
                    post_author = post.find('a').text
                    posts.append({'time': post_time, 'text': post_text, 'author': post_author})

            threads.append({'id': cur_id, 'title': title, 'category': category, 'time': posts[0]['time'], 'text': posts[0]['text'], 'posts': posts})

        nb_threads_fetched += 1
        if progress_bar:
            my_bar.progress(nb_threads_fetched/nb_threads_to_fetch, f"{nb_threads_fetched}/{nb_threads_to_fetch}")
        elif cur_id % 100 == 0:
            st.write(f"Processed {cur_id - start_id} threads so far...")

        if cur_id % 1000 == 0:
            df_threads_temp = pd.DataFrame(threads)
            df_threads_temp.to_csv(f'd2r_threads_{season}_{(start_id + start_id_offset)}_{cur_id}.csv', index=False)

    df_threads = pd.DataFrame(threads)
    return df_threads


def see_post(df, id):
    thread = df[df['id'] == id].iloc[0]
    st.write("========================================")
    st.write(thread['id'], "|", thread['time'])
    st.write(thread['title'], ":", thread['text'])
    for post in thread['posts']:
        st.write('----------------')
        st.write(f"{post['time']} -- {post['author']}")
        st.write(post['text'])
    st.write("========================================")


def get_df_runes(df, verbose=False):
    id_with_runes = []
    for i, thread in df.iterrows():
        if any(re.search(r'\b' + re.escape(rune) + r'\b', thread['title'].lower()) for rune in RUNES_KEYWORDS):
            id_with_runes.append(thread['id'])

            if verbose:
                st.write(thread['id'], "|", thread['time'])
                st.write(thread['title'], ":", thread['text'])
                st.write('---')
                for post in thread['posts']:
                    st.write(post['text'])
                st.write("--------------------------------------")

    return df[df['id'].isin(id_with_runes)]

@st.cache_data
def get_df_runes_freq(df, runes, hours=48):
    """Returns a long df with the frequency of each rune in the last n hours.

    Args:
        df (df): Dataframe of threads
        runes (list of str): List of runes
        hours (int, optional): Nb of hours to get frequency for. Defaults to 48.

    Returns:
        df: Dataframe with columns ['hour, 'rune', 'frequency']
    """
    df_runes_hour = None
    for cur_hour in range(1, hours+1):
        df_hour = df[(df['time'] >= df.iloc[0]['time'].floor('H') + pd.Timedelta(hours=cur_hour-1)) & (df['time'] <= df.iloc[0]['time'].floor('H') + pd.Timedelta(hours=cur_hour))]

        thread_titles = [title.lower() for title in df_hour['title'].values.tolist()]

        runes_freq = {}
        for rune in runes:
            runes_freq[rune] = 0
            for title in thread_titles:
                if re.search(r'\b' + re.escape(rune) + r'\b', title):
                    runes_freq[rune] += 1

        cur_hour_df = pd.DataFrame(list(runes_freq.items()), columns=['rune', 'frequency'])
        cur_hour_df['hour'] = cur_hour
        cur_hour_df.sort_values(by='frequency', ascending=False, ignore_index=True, inplace=True)

        df_runes_hour = pd.concat([df_runes_hour, cur_hour_df]) if df_runes_hour is not None else cur_hour_df

    return df_runes_hour


def get_df_runes_freq_nback_hours(df, runes, hour=10, cumul=False, remove_zero_freq=True):
    df_runes_hour = get_df_runes_freq(df, runes)

    if cumul:
        df_runes_hour = df_runes_hour[df_runes_hour['hour'] <= hour]
        df_runes_hour = df_runes_hour.groupby(['rune']).sum().reset_index().sort_values(by='frequency', ascending=False, ignore_index=True)
    else:
        df_runes_hour = df_runes_hour[df_runes_hour['hour'] == hour]

    if remove_zero_freq:
        df_runes_hour = df_runes_hour[df_runes_hour['frequency'] > 0]

    return df_runes_hour


def print_runeword(df, runeword="spirit"):
    for i, thread in df.iterrows():
        if runeword.lower() in thread['title'].lower():
            st.write("============================================================")
            st.write(thread['id'], "|", thread['time'])
            st.write(thread['title'], ":", thread['text'])
            #posts = ast.literal_eval(thread['posts'])
            #for post in posts[1:]:
            #    st.write('----')
            #    st.write(f"{post['time']} -- {post['author']}")
            #    st.write(post['text'])


# ============================================================
# Streamlit App
# ============================================================
st.title("D2R JSP: Rune Trader")

if 'df_threads' not in st.session_state:
    st.session_state['df_threads'] = pd.DataFrame()

df_threads = st.session_state['df_threads']

# Sidebar - Config
with st.sidebar:
    st.title("Rune Trader - Config")
    season_selected = st.selectbox("Season", SEASONS)

    nb_threads = st.number_input("Number of Threads", min_value=1, max_value=100000, value=DEFAULT_NB_THREADS)

    from_file = st.checkbox("From File", value=True)
    if st.button("Fetch Data"):
        my_bar = st.progress(0, text='Fetching threads...')
        if from_file:
            try:
                df_threads = pd.read_csv(f'd2r_threads_{season_selected}.csv', nrows=nb_threads, index_col=None)
            except:
                st.error("Can't open the file.")
        else:
            df_threads = get_threads_df(start_id=SEASONS_STARTING_POST_ID[season_selected], start_id_offset=0, nb_thread=nb_threads, season=season_selected, progress_bar=my_bar)
            df_threads.to_csv(f'd2r_threads_{season_selected}.csv', index=False)

        if 'time' in df_threads.columns:
            df_threads['time'] = pd.to_datetime(df_threads['time'])  # , format='%b %d %Y %I:%M%p')

        if 'title' in df_threads.columns:
            df_threads['title'] = df_threads['title'].apply(str)

        st.session_state['df_threads'] = df_threads

        my_bar.progress(100)
        if len(df_threads):
            st.success(f"Successfully fetched {len(df_threads)} related threads.")

    st.markdown("---")
    st.markdown("ToDo:")
    st.markdown("0. Profile the Fetching of Threads.")
    st.markdown("1. Add threads up to 48h.")
    st.markdown("2. Customize Posts Display.")
    st.markdown("3. Add season 1 & 2.")
    st.markdown("4. Spin a thread for Processing.")

# Main - Display
if len(st.session_state['df_threads']):
    df_threads = st.session_state['df_threads']
    df_threads_iso = df_threads[df_threads['category'] == "ISO"]
    df_threads_ft = df_threads[df_threads['category'] == "FT"]
    df_threads_service = df_threads[df_threads['category'] == "Service"]

    # ====================================
    # Main - Animation Runes Frequency
    # ====================================
    st.subheader("Runes Frequency")
    col1, col2, col3, col4 = st.columns(4)
    animation_rune_type = col1.selectbox("Rune Type", ["Low", "Medium", "High"])
    animation_duration = col2.number_input("Animation Speed (ms):", min_value=500, max_value=10000, value=1500)
    animation_max_freq = col3.number_input("Freq Max (x-axis):", min_value=1, max_value=10000, value=150)
    animation_cumul = col4.checkbox("Cumulative", value=True, key="cumul_animation")
    runes = LOW_RUNES if animation_rune_type == "Low" else MED_RUNES if animation_rune_type == "Medium" else HIGH_RUNES
    runes_freqs_iso = get_df_runes_freq(df_threads_iso, runes)
    runes_freqs_ft = get_df_runes_freq(df_threads_ft, runes)

    runes_freqs_iso['cumulative_frequency'] = runes_freqs_iso.groupby('rune')['frequency'].cumsum()
    runes_freqs_ft['cumulative_frequency'] = runes_freqs_ft.groupby('rune')['frequency'].cumsum()

    runes_freqs_iso['category'] = "ISO"
    runes_freqs_ft['category'] = "FT"
    df_runes_freqs_combined = pd.concat([runes_freqs_iso, runes_freqs_ft])

    #runes_freqs_iso = runes_freqs_iso[runes_freqs_iso['cumulative_frequency'] > 0]
    fig_iso = px.bar(df_runes_freqs_combined,
                    x='cumulative_frequency' if animation_cumul else 'frequency',
                    y='rune',
                    color='category',
                    #color_discrete_map={"ISO": "green", "FT": "red"},
                    orientation='h',
                    animation_frame='hour',
                    category_orders={"rune": runes}) # category_orders={"rune": runes})  # Sort rune names alphabetically)
    fig_iso.update_layout(xaxis=dict(range=[0, animation_max_freq]))
    fig_iso.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = animation_duration
    fig_iso.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 300
    st.write(fig_iso)

    # Group by 'hour' and 'rune' and sum 'cumulative_frequency'
    grouped_iso = df_runes_freqs_combined[df_runes_freqs_combined['category'] == 'ISO'].groupby(['hour', 'rune'])['frequency'].sum().reset_index()
    grouped_ft = df_runes_freqs_combined[df_runes_freqs_combined['category'] == 'FT'].groupby(['hour', 'rune'])['frequency'].sum().reset_index()

    # Merge the grouped data on 'hour' and 'rune'
    merged_df = pd.merge(grouped_iso, grouped_ft, on=['hour', 'rune'], suffixes=('_iso', '_ft'))

    # Calculate the ratio
    merged_df['iso_vs_ft'] = merged_df['frequency_iso'] - merged_df['frequency_ft']

    fig_sad = px.line(merged_df, x='hour', y='iso_vs_ft', color='rune', range_x=(1, 24))
    st.write(fig_sad)

    with st.expander("Break Down"):
        # ====================================
        # Main - Runes Frequency: ISO
        # ====================================
        st.subheader("Runes Frequency - ISO")
        cumul_iso = st.checkbox("Cumulative", value=True, key="cumul_iso")
        slider_iso = st.slider("Hours after launch:", 1, 10, 1, 1, key="slider_iso")

        runes_freqs_iso = get_df_runes_freq_nback_hours(df_threads_iso, RUNES, slider_iso, cumul_iso, remove_zero_freq=True)
        runes_freqs_iso.sort_values(by='frequency', ascending=False, ignore_index=True, inplace=True)
        fig_iso = px.bar(runes_freqs_iso[20:0:-1], x='frequency', y='rune', orientation='h', animation_frame='hour', animation_group='rune')
        st.write(fig_iso)

        # ====================================
        # Main - Runes Frequency: FT
        # ====================================
        st.subheader("Runes Frequency - FT")
        cumul_ft = st.checkbox("Cumulative", value=True, key="cumul_ft")
        slider_ft = st.slider("Hours after launch:", 1, 10, 1, 1, key="slider_ft")

        runes_freqs_ft = get_df_runes_freq_nback_hours(df_threads_ft, RUNES, slider_ft, cumul_ft, remove_zero_freq=True)
        runes_freqs_ft.sort_values(by='frequency', ascending=False, ignore_index=True, inplace=True)
        fig_ft = px.bar(runes_freqs_ft[::-1], x='frequency', y='rune', orientation='h')
        st.write(fig_ft)


    # ====================================
    st.markdown("---")
    # ====================================

    # Main - Runes ISO
    st.subheader("Runes - ISO")
    st.write(get_df_runes(df_threads_iso))

    # Main - Runes FT
    st.subheader("Runes - FT")
    st.write(get_df_runes(df_threads_ft))

    # Main - Runewords
    st.subheader("Runewords")
    #print_runeword(df_threads_iso, "spirit")