import streamlit as st
import numpy as np
import pandas as pd
import re, io, sys

from prediction import (predict_top5_per_genre, genre_name_to_id)

# ─── Style ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp {
        background-color: #111111;
        color: #f5f5f5;
    }
    .stButton>button {
        color: #fff;
        background: #222;
        border-radius: 6px;
        border: 1px solid #444;
    }
    .stTextInput>div>input, .stNumberInput>div>input, .stSelectbox>div>div>div {
        background: #222;
        color: #fff;
        border-radius: 6px;
        border: 1px solid #444;
    }
    th, td {
        background: #222 !important;
        color: #fff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title(":clapper: BERT Movie Recommendation")

st.markdown("""
<div style='background-color:#222;padding:16px;border-radius:10px;'>
<h4 style='color:#fff;'>Enter User and Movie Details</h4>
</div>
<style>
label, .stTextInput label, .stNumberInput label, .stSelectbox label {
    color: #fff !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
}
.stTextInput>div>label, .stNumberInput>div>label, .stSelectbox>div>label {
    color: #fff !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    user_id = st.number_input(
        "User ID", min_value=1, value=100, step=1, format="%d", key="user_id"
    )
with col2:
    movie_id = st.number_input(
        "Movie ID", min_value=1, value=1000, step=1, format="%d", key="movie_id"
    )

interaction_seq = st.text_input(
    "Interaction Sequence (comma-separated Movie IDs)",
    value="1826, 2296, 1605, 925, 1754",
    key="interaction_seq",
)

all_genres = list(genre_name_to_id.keys())
genre_inputs = []
st.markdown("<br><b style='color:#fff;'>Select 5 genres:</b>", unsafe_allow_html=True)
cols = st.columns(5)
for i in range(5):
    with cols[i]:
        genre = st.selectbox(
            f"Genre {i+1}", all_genres, key=f"genre_{i}"
        )
        genre_inputs.append(genre)
st.markdown("<br>", unsafe_allow_html=True)

# ─── Generic parser for any 2-line “User ID … | …” table ────────────────────
def parse_genre_table(text_block: str) -> pd.DataFrame:
    """
    Parses a two-line block:
      Line1: headers, e.g. 'User ID Horror Adventure ... Romance | Comedy'
      Line2: all ints in column-major order (first user IDs, then genre blocks)
    Returns DataFrame in row-major form.
    """
    # Split header vs data
    hdr, data = text_block.strip().split("\n", 1)
    toks, cols = hdr.split(), []
    i = 0
    # Merge User+ID and any 'A | B'
    while i < len(toks):
        if toks[i] == "User" and i + 1 < len(toks) and toks[i+1] == "ID":
            cols.append("User ID"); i += 2; continue
        if i + 2 < len(toks) and toks[i+1] == "|":
            cols.append(f"{toks[i]} | {toks[i+2]}"); i += 3; continue
        cols.append(toks[i]); i += 1

    # Extract all integers in appearance order
    all_ints = list(map(int, re.findall(r"\d+", data)))
    n_cols = len(cols)
    if len(all_ints) % n_cols != 0:
        raise ValueError(f"{len(all_ints)} ints for {n_cols} columns")
    n_rows = len(all_ints) // n_cols

    # Reconstruct rows from column-major ordering
    rows = []
    for r in range(n_rows):
        row = [all_ints[r + n_rows*c] for c in range(n_cols)]
        rows.append(row)

    return pd.DataFrame(rows, columns=cols)

# ─── Main Callback ───────────────────────────────────────────────────────────
if st.button("Get Recommendations"):
    # 1) Parse interaction seq
    try:
        interaction_seq_list = [
            int(x.strip()) for x in interaction_seq.split(",") if x.strip()
        ]
    except Exception as e:
        st.error(f"Invalid interaction sequence: {e}")
        st.stop()

    # 2) Capture raw two-line output for selected genres
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    predict_top5_per_genre(user_id, interaction_seq_list, genre_inputs)
    sys.stdout = old_stdout

    raw_lines = buf.getvalue().strip().splitlines()
    if len(raw_lines) < 2:
        st.error("No data returned from predict_top5_per_genre()")
        st.stop()

    # 3) Parse into DataFrame
    text_block = "\n".join(raw_lines[:2])
    try:
        df = parse_genre_table(text_block)
    except Exception as e:
        st.error(f"Failed to parse recommendations table: {e}")
        st.stop()

    # 4) Insert Rank and render
    df.insert(0, "Rank", range(1, len(df) + 1))
    st.subheader(":sparkles: Recommendation Table")
    st.table(df)

    # 5) Optional raw output
    st.subheader(":page_facing_up: Raw Output")
    st.text("\n".join(raw_lines))