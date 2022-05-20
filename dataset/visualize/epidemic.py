import streamlit as st
import pandas as pd
import pandas.io.json as pjson
import json
from datetime import datetime, timedelta
import pydeck as pdk


def basic_analyze(city):
    raw = pjson.loads(open(f'data/{city}.json').read())
    raw = pjson.json_normalize(raw, record_path='data')
    raw['dateId'] = pd.to_datetime(raw['dateId'], format='%Y%m%d')
    raw = raw.set_index(['dateId'])
    return raw


def time_agg(raw: pd.DataFrame, freq, only22):
    if only22:
        raw = raw[raw.index >= pd.to_datetime('20220210', format='%Y%m%d')]
    return raw.groupby(pd.Grouper(freq=freq[0])).sum()


def sum_city(city, tt):
    df = time_agg(basic_analyze(city), 'M', False)
    df = df[str(tt)[:7]]
    return df.iloc[0]['confirmedCount']


"""
## Comparison of Epidemic Development in Cities with Different Economic Status
> Sample Cities: Guangdong, Shanghai, Jilin, Yunnan
"""

rng_sel = st.select_slider(
    'Data aggregate by',
    options=['Day', 'Month', 'Year'])

col1, col2 = st.columns(2)

with col1:
    only2022 = st.checkbox('Recent Erupt Only')

with col2:
    cnt_or_rate = st.checkbox('Count / Increase')

gd_raw = time_agg(basic_analyze('广东')['confirmedIncr' if cnt_or_rate else 'confirmedCount'], rng_sel, only2022)
sh_raw = time_agg(basic_analyze('上海')['confirmedIncr' if cnt_or_rate else 'confirmedCount'], rng_sel, only2022)
jl_raw = time_agg(basic_analyze('吉林')['confirmedIncr' if cnt_or_rate else 'confirmedCount'], rng_sel, only2022)
yn_raw = time_agg(basic_analyze('云南')['confirmedIncr' if cnt_or_rate else 'confirmedCount'], rng_sel, only2022)

merge = pd.merge(gd_raw, sh_raw, how='inner', left_index=True, right_index=True)
merge = pd.merge(merge, jl_raw, how='inner', left_index=True, right_index=True)
merge = pd.merge(merge, yn_raw, how='inner', left_index=True, right_index=True)
merge.columns = ['Guangdong', 'Shanghai', 'Jilin', 'Yunnan']

st.line_chart(merge)

if cnt_or_rate and only2022:
    incr = pd.DataFrame()
    gd_tmp = time_agg(basic_analyze('广东'), rng_sel, only2022)
    sh_tmp = time_agg(basic_analyze('上海'), rng_sel, only2022)
    jl_tmp = time_agg(basic_analyze('吉林'), rng_sel, only2022)
    yn_tmp = time_agg(basic_analyze('云南'), rng_sel, only2022)
    incr['Guangdong'] = gd_tmp['confirmedCount'] / (gd_tmp['confirmedCount'] - gd_tmp['confirmedIncr'])
    incr['Shanghai'] = sh_tmp['confirmedCount'] / (sh_tmp['confirmedCount'] - sh_tmp['confirmedIncr'])
    incr['Jilin'] = jl_tmp['confirmedCount'] / (jl_tmp['confirmedCount'] - jl_tmp['confirmedIncr'])
    incr['Yunnan'] = yn_tmp['confirmedCount'] / (yn_tmp['confirmedCount'] - yn_tmp['confirmedIncr'])
    st.line_chart(incr)

to_time = st.slider(
    "From start to",
    value=datetime(2022, 1, 1),
    min_value=datetime(2020, 2, 1),
    max_value=datetime(2022, 2, 1),
    step=timedelta(days=30),
    format="YYYY/MM")

cities = pd.DataFrame()

raw_geo = json.loads(open('china.json').read())
cities['name'] = [r['properties']['name'] for r in raw_geo]
cities['lat'] = [r['properties']['cp'][1] for r in raw_geo]  # N
cities['lon'] = [r['properties']['cp'][0] for r in raw_geo]  # E
cities['num'] = [sum_city(r, to_time) for r in cities['name']]

st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(
        latitude=30.19,
        longitude=102.91,
        zoom=3,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
            'ColumnLayer',
            data=cities,
            get_position='[lon, lat]',
            get_elevation='num',
            radius=120000,
            elevation_scale=4,
            elevation_range=[0, 100000],
            get_fill_color=["num", "lon", "lat", 150],
            # extruded=True,
            pickable=True,
        ),
        # pdk.Layer(
        #     'ScatterplotLayer',
        #     data=cities,
        #     get_elevation='num',
        #     get_position='[lon, lat]',
        #     get_color='[200, 30, 0, 160]',
        #     get_radius=150000,
        # ),
    ],
))
