import requests
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie

st.set_page_config(page_title="My Webpage", layout="wide")

selected = option_menu(
    menu_title=None,
    options=["HOME", "NBA", "PREMIER LEAGUE"],
    default_index=0,
    orientation="horizontal",

styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "red", "font-size": "20px"},
        "nav-link": {
            "font-size": "20px",
            "text-align": "middle",
            "margin": "0px",
            "--hover-color": "#eee",
        },
        "nav-link-selected": {"background-color": "black"},
    },
)

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#LOAD ASSETS
lottie_coding = load_lottieurl("https://assets3.lottiefiles.com/private_files/lf30_ef3beoqa.json")
lottie_coding1 = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_yLCHY6q3aU.json")

#HEADER SECTION
with st.container():
    image = Image.open('sportix.png')
    st.image(image)

#BODYPART
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("NBA")
        st.write("##")
        st.write("The National Basketball Association (NBA) is a professional basketball league in North America composed of 30 teams (29 in the United States and 1 in Canada). It is one of the major professional sports leagues in the United States and Canada and is considered the premier professional basketball league in the world.The league was founded in New York City on June 6, 1946, as the Basketball Association of America (BAA).[1] It changed its name to the National Basketball Association on August 3, 1949, after merging with the competing National Basketball League (NBL).[4] In 1976, the NBA and the American Basketball Association (ABA) merged, adding four franchises to the NBA. The NBA's regular season runs from October to April, with each team playing 82 games. The league's playoff tournament extends into June. As of 2020, NBA players are the world's best paid athletes by average annual salary per player.")
        st.write("[Latest News >](https://www.sportingnews.com/sg/nba?gr=www)")
    with right_column:
        st_lottie(lottie_coding, height=350, key="coding")
        st.subheader("National Basketball Association")
        result = st.button("NBA")
        if result:
            st.write("🏀")

with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        image = Image.open('premierl.png')
        st.image(image)
        st.subheader("Premier League")
        result = st.button("PREMIER LEAGUE")
        if result:
            st.write("⚽")
    with right_column:
        st.header("Premier League")
        st.write("##")
        st.write("The Premier League (legal name: The Football Association Premier League Limited) is the highest level of the English football league system. Contested by 20 clubs, it operates on a system of promotion and relegation with the English Football League (EFL). Seasons typically run from August to May with each team playing 38 matches against all other teams both home and away. Most games are played on Saturday and Sunday afternoons, with occasional weekday evening fixtures. The competition was founded as the FA Premier League on 20 February 1992 following the decision of First Division (top-tier league from 1888 until 1992) clubs to break away from the English Football League. However, teams may still be relegated into and promoted from the EFL Championship. The Premier League takes advantage of a lucrative television rights sale to Sky: from 2019 to 2020, accumulated television rights were worth around £3.1 billion a year, with Sky and BT Group securing the domestic rights to broadcast 128 and 32 games respectively.")
        st.write("[Latest News >](https://www.premierleague.com/)")


