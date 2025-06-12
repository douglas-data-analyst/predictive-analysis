"""
Utilitários para modelagem preditiva e análise de séries temporais.

Este módulo contém funções para preparação de dados, engenharia de features,
treinamento de modelos e avaliação de desempenho para análise preditiva.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import joblib

from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb


def load_and_prepare_data(base_path='../data/processed/'):
    """
    Carrega e prepara dados para modelagem preditiva.
    
    Parameters:
    -----------
    base_path : str
        Caminho base para os arquivos de dados processados
        
    Returns:
    --------
    tuple
        (df_daily_sales, df_model) - DataFrames para modelagem
    """
    try:
        # Carregando dados processados
        fact_sales = pd.read_parquet(f'{base_path}fact_sales.parquet')
        dim_date = pd.read_parquet(f'{base_path}dim_date.parquet')
        dim_product = pd.read_parquet(f'{base_path}dim_product.parquet')
        
        # Juntando tabelas para criar o dataset de modelagem
        df_model = pd.merge(fact_sales, dim_date, left_on='date_id', right_on='id', suffixes=('', '_date'))
        df_model = pd.merge(df_model, dim_product, left_on='product_id', right_on='id', suffixes=('', '_product'))
        
        # Selecionando colunas relevantes e renomeando
        df_model = df_model[[
            'date', 'price', 'freight_value', 'review_score', 'year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend',
            'product_category_name_english', 'product_weight_g', 'product_photos_qty'
        ]].copy()
        
        df_model.rename(columns={'product_category_name_english': 'category'}, inplace=True)
        
        # Agregando dados por dia para prever vendas diárias
        df_daily_sales = df_model.groupby('date').agg(
            total_sales=('price', 'sum'),
            avg_freight=('freight_value', 'mean'),
            avg_review_score=('review_score', 'mean'),
            year=('year', 'first'),
            month=('month', 'first'),
            day=('day', 'first'),
            dayofweek=('dayofweek', 'first'),
            quarter=('quarter', 'first'),
            is_weekend=('is_weekend', 'first')
        ).reset_index()
        
        # Ordenando por data
        df_daily_sales = df_daily_sales.sort_values('date').set_index('date')
        
        return df_daily_sales, df_model
    
    except FileNotFoundError as e:
        print(f"Erro ao carregar dados: {e}")
        return None, None


def create_time_features(df):
    """
    Cria features temporais para análise de séries temporais.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame com índice de data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame com features temporais adicionadas
    """
    df_copy = df.copy()
    
    # Criando features de lag (vendas passadas)
    for lag in [1, 7, 14, 30]:
        df_copy[f'sales_lag_{lag}'] = df_copy['total_sales'].shift(lag)
    
    # Criando features de média móvel
    for window in [7, 14, 30]:
        df_copy[f'sales_rolling_mean_{window}'] = df_copy['total_sales'].shift(1).rolling(window=window).mean()
        df_copy[f'sales_rolling_std_{window}'] = df_copy['total_sales'].shift(1).rolling(window=window).std()
    
    # Removendo linhas com NaN geradas pelas features de lag e média móvel
    df_copy = df_copy.dropna()
    
    return df_copy


def split_time_series_data(X, y, test_size=0.2):
    """
    Divide dados de séries temporais em conjuntos de treino e teste.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Features
    y : pandas.Series
        Target
    test_size : float
        Proporção do conjunto de teste
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    # Divisão cronológica dos dados
    split_point = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    
    return X_train, X_test, y_train, y_test


def create_preprocessing_pipeline(numerical_features, categorical_features=None):
    """
    Cria um pipeline de pré-processamento para features numéricas e categóricas.
    
    Parameters:
    -----------
    numerical_features : list
        Lista de nomes de features numéricas
    categorical_features : list, optional
        Lista de nomes de features categóricas
        
    Returns:
    --------
    sklearn.compose.ColumnTransformer
        Pipeline de pré-processamento
    """
    transformers = [('num', StandardScaler(), numerical_features)]
    
    if categorical_features and len(categorical_features) > 0:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features))
    
    return ColumnTransformer(transformers=transformers, remainder='passthrough')


def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor, models=None):
    """
    Treina e avalia múltiplos modelos de regressão.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Features de treino
    X_test : pandas.DataFrame
        Features de teste
    y_train : pandas.Series
        Target de treino
    y_test : pandas.Series
        Target de teste
    preprocessor : sklearn.compose.ColumnTransformer
        Pipeline de pré-processamento
    models : dict, optional
        Dicionário de modelos para treinar
        
    Returns:
    --------
    tuple
        (results_df, best_model_name, best_pipeline)
    """
    if models is None:
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42, n_jobs=-1),
            'LightGBM': lgb.LGBMRegressor(random_state=42, n_jobs=-1)
        }
    
    results = {}
    best_r2 = -float('inf')
    best_model_name = None
    best_pipeline = None
    
    for model_name, model in models.items():
        print(f"\nTreinando {model_name}...")
        
        # Criando o pipeline completo
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        
        # Treinando o modelo
        pipeline.fit(X_train, y_train)
        
        # Fazendo previsões
        y_pred = pipeline.predict(X_test)
        
        # Avaliando o modelo
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[model_name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        
        print(f"{model_name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
        
        # Verificando se é o melhor modelo até agora
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = model_name
            best_pipeline = pipeline
    
    # Criando DataFrame de resultados
    results_df = pd.DataFrame(results).T.sort_values(by='R2', ascending=False)
    
    return results_df, best_model_name, best_pipeline


def plot_predictions(X_test, y_test, y_pred, model_name, save_path=None):
    """
    Plota previsões vs. valores reais.
    
    Parameters:
    -----------
    X_test : pandas.DataFrame
        Features de teste
    y_test : pandas.Series
        Target de teste
    y_pred : numpy.ndarray
        Previsões do modelo
    model_name : str
        Nome do modelo
    save_path : str, optional
        Caminho para salvar a figura
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figura com o gráfico
    """
    # Criando DataFrame para visualização
    predictions_df = pd.DataFrame({
        'Data': X_test.index,
        'Vendas Reais': y_test,
        'Vendas Previstas': y_pred
    }).set_index('Data')
    
    # Visualizando previsões vs. valores reais
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(predictions_df.index, predictions_df['Vendas Reais'], label='Vendas Reais', color='blue', alpha=0.7)
    ax.plot(predictions_df.index, predictions_df['Vendas Previstas'], label='Vendas Previstas', color='red', linestyle='--')
    ax.set_title(f'Previsões de Vendas vs. Valores Reais ({model_name})', fontsize=16)
    ax.set_xlabel('Data', fontsize=12)
    ax.set_ylabel('Vendas Totais (R$)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance(pipeline, X, model_name, top_n=10, save_path=None):
    """
    Plota importância das features para modelos baseados em árvores.
    
    Parameters:
    -----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline com modelo treinado
    X : pandas.DataFrame
        Features
    model_name : str
        Nome do modelo
    top_n : int, optional
        Número de features mais importantes para mostrar
    save_path : str, optional
        Caminho para salvar a figura
        
    Returns:
    --------
    matplotlib.figure.Figure or None
        Figura com o gráfico ou None se o modelo não suportar importância de features
    """
    if not hasattr(pipeline.named_steps['regressor'], 'feature_importances_'):
        print(f"O modelo {model_name} não suporta diretamente a análise de importância de features.")
        return None
    
    try:
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    except (AttributeError, ValueError):
        feature_names = X.columns  # Aproximação, pode não ser precisa com OneHotEncoder
    
    importances = pipeline.named_steps['regressor'].feature_importances_
    
    # Garantindo que temos o mesmo número de nomes e importâncias
    if len(feature_names) != len(importances):
        print("Aviso: Número de nomes de features não corresponde ao número de importâncias.")
        feature_names = [f'Feature {i}' for i in range(len(importances))]
    
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis', ax=ax)
    ax.set_title(f'Top {top_n} Features Mais Importantes ({model_name})', fontsize=16)
    ax.set_xlabel('Importância', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_model(pipeline, save_path):
    """
    Salva o modelo treinado.
    
    Parameters:
    -----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline com modelo treinado
    save_path : str
        Caminho para salvar o modelo
        
    Returns:
    --------
    bool
        True se o modelo foi salvo com sucesso, False caso contrário
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(pipeline, save_path)
        print(f"Modelo salvo com sucesso em: {save_path}")
        return True
    except Exception as e:
        print(f"Erro ao salvar modelo: {e}")
        return False


def load_model(model_path):
    """
    Carrega um modelo salvo.
    
    Parameters:
    -----------
    model_path : str
        Caminho do modelo salvo
        
    Returns:
    --------
    sklearn.pipeline.Pipeline
        Pipeline com modelo carregado
    """
    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return None


def make_future_predictions(model, last_data, periods=30):
    """
    Faz previsões futuras usando o modelo treinado.
    
    Parameters:
    -----------
    model : sklearn.pipeline.Pipeline
        Pipeline com modelo treinado
    last_data : pandas.DataFrame
        Últimos dados disponíveis
    periods : int, optional
        Número de períodos futuros para prever
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame com previsões futuras
    """
    # Cria um DataFrame para armazenar previsões futuras
    future_dates = pd.date_range(start=last_data.index[-1] + pd.Timedelta(days=1), periods=periods)
    future_df = pd.DataFrame(index=future_dates)
    
    # Copia as últimas features disponíveis como ponto de partida
    last_features = last_data.iloc[-1:].copy()
    
    predictions = []
    
    # Para cada data futura
    for i, date in enumerate(future_dates):
        # Atualiza features temporais
        future_row = last_features.copy()
        future_row.index = [date]
        future_row['year'] = date.year
        future_row['month'] = date.month
        future_row['day'] = date.day
        future_row['dayofweek'] = date.dayofweek
        future_row['quarter'] = (date.month - 1) // 3 + 1
        future_row['is_weekend'] = 1 if date.dayofweek >= 5 else 0
        
        # Faz a previsão
        pred = model.predict(future_row)[0]
        predictions.append(pred)
        
        # Atualiza as features de lag para a próxima previsão
        if i + 1 < len(future_dates):
            if 'sales_lag_1' in future_row.columns:
                future_row['sales_lag_1'] = pred
            
            # Atualiza outras features de lag e médias móveis conforme necessário
            # Isso depende das features específicas do seu modelo
    
    # Cria DataFrame com as previsões
    future_predictions = pd.DataFrame({
        'data': future_dates,
        'vendas_previstas': predictions
    }).set_index('data')
    
    return future_predictions


if __name__ == "__main__":
    # Exemplo de uso do módulo
    print("Módulo de modelagem preditiva carregado.")
