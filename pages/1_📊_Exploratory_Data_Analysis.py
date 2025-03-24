import streamlit as st

from source import (df, df_describe, Obs_read, num_obs1, fig_num, cat, cat_obs,
                    fig_heatmap,obs_heatmap, obs_heatmap2, eda_concl, fig_target)

st.set_page_config(page_title="Exploratory Data Analysis", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    body, p, div {
        font-size: 13px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Exploratory Data Analysis (EDA)')

st.subheader('1.Read Data')
with st.expander("Dataset, description"):
    st.dataframe(df, use_container_width=True)
    st.dataframe(df_describe, use_container_width=True)

col1, col2 = st.columns(2)
col1.write(Obs_read)

placeholder = st.empty()
st.subheader('2. Academic Success Feature \'Target\'')

st.plotly_chart(fig_target)

st.subheader('3. Numerical Features vs. \'Target\'')

col1, col2, col3 = st.columns([10,1,20])
col1.markdown(num_obs1)
col3.pyplot(fig_num)

st.subheader('4. Categorical Features vs. \'Target\'')


col1, col2, col3 = st.columns([20,1,10])
col1.pyplot(cat)
col3.markdown(cat_obs)

st.subheader('5. Heatmap & Correlation Analysis')
col1, col2, col3 = st.columns([20,1,10])
with col1:
    st.plotly_chart(fig_heatmap)
with col3:
    st.markdown(obs_heatmap)


col1, col2 = st.columns(2)
col1.markdown(obs_heatmap2)
col2.markdown(eda_concl)


