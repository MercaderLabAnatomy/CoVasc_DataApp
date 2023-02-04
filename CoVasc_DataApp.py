import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import pydeck as pdk
import numpy as np#Load and Cache the data
from skimage.io import imread, imshow 
from PIL import Image
from glob import glob

st.set_page_config(  layout="wide")

st.title('Interactive Data App - Covid Box Drug Screening')


st.subheader('Workflow Scheme')

image = Image.open('Images/Design/Online_Short workflow.png')