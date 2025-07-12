import streamlit as st
import numpy as np

import streamlit as st
import numpy as np
from prediction import predict_user_genre_top5, predict_top5_genres, predict_top5_per_genre, genre_name_to_id, genre_id_to_name

# Set background color to black and style
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

st.title(":clapper: BERT Movie Recommendation Demo")


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
    user_id = st.number_input("User ID", min_value=1, value=1, step=1, format="%d", key="user_id", label_visibility="visible")
with col2:
    movie_id = st.number_input("Movie ID", min_value=1, value=1, step=1, format="%d", key="movie_id", label_visibility="visible")

interaction_seq = st.text_input("Interaction Sequence (comma-separated Movie IDs)", value="15,25,35,45,55", key="interaction_seq", label_visibility="visible")

# Genre dropdowns
all_genres = list(genre_name_to_id.keys())
genre_inputs = []
st.markdown("<br><b style='color:#fff;'>Select 5 genres:</b>", unsafe_allow_html=True)
cols = st.columns(5)
for i in range(5):
    with cols[i]:
        genre = st.selectbox(
            f"Genre {i+1}",
            all_genres,
            key=f"genre_{i}",
            label_visibility="visible",
            format_func=lambda x: x
        )
        genre_inputs.append(genre)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Get Recommendations"):
    # Parse interaction sequence
    try:
        interaction_seq_list = [int(x.strip()) for x in interaction_seq.split(",") if x.strip()]
    except Exception as e:
        st.error(f"Invalid interaction sequence: {e}")
        st.stop()

    # Predict top 5 movies for user and genres
    user_result = predict_user_genre_top5(user_id, movie_id, interaction_seq_list, genre_inputs)
    # Predict top 5 genres for the user
    top5_genres = predict_top5_genres(user_id, interaction_seq_list)
    top5_genre_names = [genre_id_to_name.get(g, str(g)) for g in top5_genres]

    # Get table output (User ID, 5 genres, 1 multi-genre)
    st.subheader(":sparkles: Recommendations Table")
    # Use the same logic as predict_top5_per_genre to get the table
    import io
    import sys as _sys
    buf = io.StringIO()
    _stdout = _sys.stdout
    _sys.stdout = buf
    predict_top5_per_genre(user_id, interaction_seq_list, top5_genres)
    _sys.stdout = _stdout
    table_str = buf.getvalue()
    # Parse table
    lines = table_str.strip().split("\n")
    if len(lines) >= 2:
        header = lines[0].split("\t")
        row = lines[1].split("\t")
        # Each genre column is a comma-separated list of 5 movie IDs
        genre_cols = header[5:-1]
        multi_col = header[-1]
        genre_movies = [row[5+i].split(", ") for i in range(len(genre_cols))]
        multi_movies = row[-1].split(", ")
        # Build table: 5 rows, each row is [User ID, Genre1 Movie, ..., Genre5 Movie, Multi-Genre Movie]
        table_data = []
        for i in range(5):
            row_data = [str(user_id)]
            for g in genre_movies:
                row_data.append(g[i] if i < len(g) else "")
            row_data.append(multi_movies[i] if i < len(multi_movies) else "")
            table_data.append(row_data)
        # Table headers: User ID, Genre1, Genre2, ..., Multi-Genre
        table_headers = ["User ID"] + genre_cols + [multi_col]
        st.table(
            {h: [row[i] for row in table_data] for i, h in enumerate(table_headers)}
        )
    else:
        st.write(table_str)

    st.subheader(":page_facing_up: Raw Output")
    st.text(table_str)
