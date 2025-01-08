import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings('ignore')

PLOT_WIDTH = 875
PLOT_HEIGHT = 750

@st.cache_data
def load_data():
  data_folder = "data/"

  cic1 = pd.read_csv(f"{data_folder}Cicalino1.csv", parse_dates=True)
  cic1["DateTime"] = pd.to_datetime(cic1["DateTime"], format="%Y-%m-%d %H:%M:%S")
  cic1["DateTime"] = cic1["DateTime"].dt.date
  cic1.set_index('DateTime', inplace=True)

  cic2 = pd.read_csv(f"{data_folder}Cicalino2.csv", parse_dates=True)
  cic2["DateTime"] = pd.to_datetime(cic2["DateTime"], format="%Y-%m-%d %H:%M:%S")
  cic2["DateTime"] = cic2["DateTime"].dt.date
  cic2.set_index('DateTime', inplace=True)

  im1 = pd.read_csv(f"{data_folder}Imola1.csv", parse_dates=True)
  im1["DateTime"] = pd.to_datetime(im1["DateTime"], format="%Y-%m-%d %H:%M:%S")
  im1["DateTime"] = im1["DateTime"].dt.date
  im1.set_index('DateTime', inplace=True)

  im2 = pd.read_csv(f"{data_folder}Imola2.csv", parse_dates=True)
  im2["DateTime"] = pd.to_datetime(im2["DateTime"], format="%Y-%m-%d %H:%M:%S")
  im2["DateTime"] = im2["DateTime"].dt.date
  im2.set_index('DateTime', inplace=True)

  im3 = pd.read_csv(f"{data_folder}Imola3.csv", parse_dates=True)
  im3["DateTime"] = pd.to_datetime(im3["DateTime"], format="%Y-%m-%d %H:%M:%S")
  im3["DateTime"] = im3["DateTime"].dt.date
  im3.set_index('DateTime', inplace=True)
  
  cic = pd.read_csv(f"{data_folder}Cicalino-Merged.csv", parse_dates=True)
  cic["DateTime"] = pd.to_datetime(cic["DateTime"], format="%Y-%m-%d")
  cic.set_index('DateTime', inplace=True)

  im = pd.read_csv(f"{data_folder}Imola-Merged.csv", parse_dates=True)
  im["DateTime"] = pd.to_datetime(im["DateTime"], format="%Y-%m-%d")
  im.set_index('DateTime', inplace=True)

  return [cic1,cic2,cic,im1,im2,im3,im]


def plot_accorpamento(dfs:tuple):
  cic,im = dfs

  fig = make_subplots(rows=2, cols=2, subplot_titles=('Catture zona Cicalino', 'Meteo zona Cicalino', 'Catture zona Imola', 'Meteo zona Imola'))
  
  fig.add_trace(go.Scatter(x=cic.index, y=cic['Nuove catture (per evento)'], mode='lines', name='Catture (Cicalino)'), row=1, col=1)
  fig.add_trace(go.Scatter(x=cic.index, y=cic['Media Temperatura'], mode='lines', name='Temperatura (Cicalino)'), row=1, col=2)
  fig.add_trace(go.Scatter(x=cic.index, y=cic['Media Umidità'], mode='lines', name='Umidità (Cicalino)'), row=1, col=2)
  
  fig.add_trace(go.Scatter(x=im.index, y=im['Nuove catture (per evento)'], mode='lines', name='Catture (Imola)'), row=2, col=1)
  fig.add_trace(go.Scatter(x=im.index, y=im['Media Temperatura'], mode='lines', name='Temperatura (Imola)'), row=2, col=2)  
  fig.add_trace(go.Scatter(x=im.index, y=im['Media Umidità'], mode='lines', name='Umidità (Imola)'), row=2, col=2)

  fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT, showlegend=True)
  return fig

def plot_distribuzione_meteo(dfs:tuple):
  cic,im = dfs
  
  labels = ['Cicalino', 'Imola']
  columns = ['Media Temperatura', 'Media Umidità']

  fig = make_subplots(rows=2, cols=2, subplot_titles=[f'Distribuzione di {col}' for col in columns for _ in range(2)])

  for i, col in enumerate(columns):
      for df, label in (zip([cic, im], labels)):
          fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=f'{label} - {col}'), row=1, col=i+1)

  for i, col in enumerate(columns):
      for df, label in zip([cic, im], labels):
         fig.add_trace(go.Histogram(x=df[col], name=f'{label} - {col}', histnorm='probability density', opacity=0.6), row=2, col=i+1)

  fig.update_layout(height=PLOT_HEIGHT, width=PLOT_WIDTH, barmode='overlay', showlegend=True)
  return fig

def plot_relazione_variabili(dfs:tuple):
  cic,im = dfs
  labels = ['Cicalino', 'Imola']

  fig = go.Figure()

  for df, label in zip([cic, im], labels):
    fig.add_trace(go.Scatter(
      x=df['Media Temperatura'], 
      y=df['Media Umidità'],
      customdata=df['Nuove catture (per evento)'].to_numpy(),
      mode='markers', 
      marker=dict(
        size=df['Nuove catture (per evento)'], 
        sizemode='diameter', 
        sizeref=max(df['Nuove catture (per evento)']) / 40, 
        sizemin=10
      ),
      hovertemplate='<b>Nuove catture</b>: %{customdata}<br>' +
                    '<b>Temperatura</b>: %{x:,.2f} C<br>' +
                    '<b>Umidità</b>: %{y:,.2f}%' +
                    '<extra></extra>',
      name=label
    ))

  fig.update_layout(xaxis_title="Media Temperatura",yaxis_title="Media Umidità",height=PLOT_HEIGHT//1.5, width=PLOT_WIDTH)
  return fig


def plot_violin_correlazione(dfs:tuple):
  cic,im = dfs

  labels = ['Cicalino', 'Imola']
  fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{label} - Violin Plot" for label in labels] + [f"{label} - Heatmap" for label in labels])

  for i, df in enumerate([cic, im]):
    for col in df.columns:
      if col != 'DateTime':
        fig.add_trace(go.Violin(y=df[col], name=f'{col}', box_visible=True, meanline_visible=True), row=1, col=i+1)

      corr = df.corr()
      fig.add_trace(go.Heatmap(x=corr.columns, y=corr.index, z=np.array(corr), text=corr.values, texttemplate='%{text:.2f}', colorbar=dict(title='Correlazione'), colorscale='viridis'), row=2, col=i+1)

  fig.update_layout(height=PLOT_HEIGHT, width=PLOT_WIDTH, showlegend=False)
  return fig


def rmse(y_true, y_pred):
  return np.sqrt(mean_squared_error(y_true, y_pred))


def create_multivariate_lstm_sequences(features, target, window_size=3):
  X, y = [], []
  for i in range(len(target) - window_size):
      X.append(features[i : i + window_size])
      y.append(target[i + window_size])

  X = np.array(X)  # (num_samples, window_size, num_features)
  y = np.array(y)  # (num_samples,)
  return X, y


def build_lstm_model(window_size=3, num_features=2, hidden_units=50, learning_rate=0.001):
  model = Sequential()
  model.add(LSTM(hidden_units, input_shape=(window_size, num_features)))
  model.add(Dense(1))  # un singolo valore in output

  optimizer = Adam(learning_rate=learning_rate)
  model.compile(optimizer=optimizer, loss='mse')
  return model

@st.cache_data(show_spinner=False)
def train_models(dfs, names):
  results = {}

  for i,df in enumerate(dfs):
    st.subheader(f"Analisi del dataframe: {names[i]}")

    try:
        df.set_index('DateTime', inplace=True)
    except:
        pass

    target_column = 'Nuove catture (per evento)'
    if target_column not in df.columns:
        raise ValueError(f"Colonna '{target_column}' non trovata in {names[i]}.")

    possible_features = ['Media Temperatura', 'Media Umidità']
    feature_columns = [col for col in possible_features if col in df.columns]

    df.dropna(subset=[target_column] + feature_columns, inplace=True)

    X = df[feature_columns].values   # shape (N, num_features)
    y = df[target_column].values     # shape (N,)

    N = len(df)
    train_size = int(N * 0.8)

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # --------------------------------------------------------
    # A) DECISION TREE
    # --------------------------------------------------------
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)

    # RMSE Decision Tree
    dt_rmse_val = rmse(y_test, dt_preds)
    st.write(f"[Decision Tree] RMSE: {dt_rmse_val:.4f}")

    # --------------------------------------------------------
    # B) auto_ARIMA
    # --------------------------------------------------------
    arima_model = auto_arima(
        y_train,
        exogenous=X_train,
        seasonal=False,
        trace=False,
        error_action='ignore',
        suppress_warnings=True
    )

    forecast_period = len(y_test)
    arima_forecasts = arima_model.predict(
        n_periods=forecast_period,
        exogenous=X_test
    )

    arima_forecasts_rounded = np.round(arima_forecasts)

    arima_rmse_val = rmse(y_test, arima_forecasts_rounded)
    st.write(f"[Auto ARIMA] RMSE: {arima_rmse_val:.4f}")

    # --------------------------------------------------------
    # C) LSTM
    # --------------------------------------------------------
    window_size = 3
    X_seq, y_seq = create_multivariate_lstm_sequences(X, y, window_size)

    seq_len = len(X_seq)
    seq_train_size = int(seq_len * 0.8)

    X_seq_train, X_seq_test = X_seq[:seq_train_size], X_seq[seq_train_size:]
    y_seq_train, y_seq_test = y_seq[:seq_train_size], y_seq[seq_train_size:]

    num_features = X.shape[1]
    lstm_model = build_lstm_model(
        window_size=window_size,
        num_features=num_features,
        hidden_units=50,
        learning_rate=0.001
    )

    lstm_model.fit(
        X_seq_train,
        y_seq_train,
        epochs=20,
        batch_size=4,
        validation_split=0.1,
        verbose=0
    )

    lstm_preds = lstm_model.predict(X_seq_test).flatten()
    lstm_preds_rounded = np.round(lstm_preds)

    lstm_rmse_val = rmse(y_seq_test, lstm_preds_rounded)
    st.write(f"[LSTM] RMSE: {lstm_rmse_val:.4f}")

    full_index = df.index

    dt_arima_test_index = full_index[train_size:]

    lstm_full_index = full_index[window_size:]  # dal window_size in poi
    lstm_test_index = lstm_full_index[seq_train_size:]

    forecast_fig = go.Figure()

    forecast_fig.add_trace(go.Scatter(x=full_index, y=y, mode='lines', name='Serie Reale'))
    forecast_fig.add_trace(go.Scatter(x=dt_arima_test_index, y=dt_preds, mode='lines', name='Decision Tree'))
    forecast_fig.add_trace(go.Scatter(x=dt_arima_test_index, y=arima_forecasts_rounded, mode='lines', name='ARIMA'))
    forecast_fig.add_trace(go.Scatter(x=lstm_test_index, y=lstm_preds_rounded, mode='lines', name='LSTM'))

    forecast_fig.update_layout(title=f"Confronto previsioni sul test - {names[i]}", xaxis_title="Data", yaxis_title="Nuove catture (per evento)", legend_title="Legenda", height=500, width=PLOT_WIDTH)

    st.plotly_chart(forecast_fig)

    results[names[i]] = {
        "DecisionTree": dt_rmse_val,
        "ARIMA_exog": arima_rmse_val,
        "LSTM_multivar": lstm_rmse_val}
  
  return results

# COMPOSIZIONE PAGINA

st.title('ISBI - Homework 2')
st.subheader('Dashboard Interattiva')
st.write('Marra Leonardo Maria, Russo Antonio')
st.write('---')

st.header('Caricamento Dati')

with st.spinner('Caricamento ...'):
  cic1,cic2,cic,im1,im2,im3,im = load_data()

with st.expander('Dati forniti'):
    col1, col2 = st.columns(2)

    with col1:
      st.write('Cicalino 1')
      st.dataframe(cic1)

      st.write('Imola 1')
      st.dataframe(im1)

      st.write('Imola 3')
      st.dataframe(im3)

    with col2:
      st.write('Cicalino 2')
      st.dataframe(cic2)

      st.write('Imola 2')
      st.dataframe(im2)
    
with st.expander('Accorpamento'):
    st.dataframe(cic)

st.subheader('Visualizzazione')
st.plotly_chart(plot_accorpamento((cic,im)))

st.write('---')
st.header('Analisi esplorativa')

st.subheader('Distribuzione e Line Plot di Media Temperatura e Umidità')
st.plotly_chart(plot_distribuzione_meteo((cic,im)))

st.subheader('Media Temperatura vs Media Umidità, dimensione punti Nuove catture (per evento)')
st.plotly_chart(plot_relazione_variabili((cic,im)))

st.subheader('Violin plots e Matrici di Correlazione')
st.plotly_chart(plot_violin_correlazione((cic,im)))

st.write('---')
st.header('Forecast')

dfs = [cic1, cic2, cic, im1, im2, im3, im]
names = ['Cicalino1', 'Cicalino2', 'Cicalino-Merged', 'Imola1', 'Imola2', 'Imola3', 'Imola-Merged']

with st.spinner('Addestramento'):
  results = train_models(dfs, names)

st.success('Addestramento completato')

st.write('---')
st.header('Risultati')
st.subheader("Risultati (RMSE)")
st.dataframe(results)
