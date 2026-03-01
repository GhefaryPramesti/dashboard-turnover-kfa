import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import datetime
from datetime import timedelta

#page configuration
st.set_page_config(
    page_title="Dashboard Turnover Karyawan | PT Kimia Farma Apotek",
    page_icon="💊",
    layout="wide"
)

#load model
@st.cache_resource
def load_model_resources():
    model = joblib.load('turnover_model.pkl')
    features = joblib.load('model_features.pkl')
    return model, features

model, model_features = load_model_resources()
