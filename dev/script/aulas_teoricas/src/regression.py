# Importando modelos
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import seaborn as sns
# Visualização dos dados
import matplotlib.pyplot as plt

# Edição das databases
import pandas as pd
import numpy as np

# Seleção de hiperparâmetros e validação cruzada
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

# Estruturação dos dados e pré-processamento
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Organização do fluxo de trabalho (Pipeline)
from sklearn.pipeline import Pipeline

# Seleção de variáveis
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# Métricas da regressão
from sklearn.metrics import mean_squared_error, r2_score


class Regression:
    '''
        Regressor genérico que encapsula diferentes modelos de 
        aprendizado de máquina.
            Parâmetros
            ----------
            df : pd.DataFrame
                DataFrame contendo as features e a coluna alvo 'target'.

            Métodos
            -------
            regress(modelo)
                Treina e avalia o modelo especificado.
    '''
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.X = df.drop('IFN-γ', axis=1)
        self.y = df['IFN-γ']

        return None

    def __obter_metricas(self, grid_search: GridSearchCV, modelo: str) -> dict:
        '''
            Calcula métricas de avaliação do modelo.
                Parâmetros
                ----------
                grid_search : GridSearchCV
                    Objeto GridSearchCV já ajustado com os dados.
                
                Retorna
                -------
                metrics : dict
                    Dicionário contendo as métricas de avaliação.
        '''
        y_true = self.y
        y_pred = grid_search.predict(self.X)

        df_test = pd.DataFrame()
        df_test['y_true'] = y_true
        df_test['y_pred'] = y_pred

        r_cal, r_cal_p = spearmanr(df_test['y_true'], df_test['y_pred'])
        r2_cal = r2_score(df_test['y_true'], df_test['y_pred'])
        mae_cal = mean_absolute_error(df_test['y_true'], df_test['y_pred'])
        rmse_cal = mean_squared_error(df_test['y_true'], df_test['y_pred'])
        mape_cal = mean_absolute_percentage_error(
                                                        df_test['y_true'],
                                                        df_test['y_pred']
                                                )

        mensagem = f'''
    R Spearman: {r_cal:.4f}
    R p-value: {r_cal_p:.4f}
    R2: {r2_cal:.4f}
    MAE: {mae_cal:.4f}
    RMSE: {rmse_cal:.4f}
    MAPE: {mape_cal:.4f}
        '''
        
        p = sns.jointplot(
            x='y_true',
            y='y_pred',
            data=df_test,
            kind='reg',
        )
        p.fig.suptitle(f"Correlação predito / observado - {modelo}", y=1.02)
        plt.text(
            0.05,
            0.99,
            mensagem,
            transform=p.ax_joint.transAxes,
            fontsize=12,
            ha='left',
            va='top'
        )
        plt.xlabel('Observado')
        plt.ylabel('Predito')
        plt.show()

        return None

    def __variaveis_selecionadas(
            self,
            grid_search: GridSearchCV) -> pd.DataFrame:
        '''
            Exibe as variáveis selecionadas pelo melhor modelo encontrado pelo 
            GridSearchCV.
                Parâmetros
                ----------
                grid_search : GridSearchCV
                    Objeto GridSearchCV já ajustado com os dados.
                
                Retorna
                -------
                df_summary : pd.DataFrame
                    DataFrame contendo o status de cada atributo no pipeline.
        '''
        
        # 1. Extrai o melhor pipeline ajustado pelo GridSearchCV
        best_pipeline = grid_search.best_estimator_

        # 2. Recupera os nomes de TODAS as colunas geradas após a 
        # codificação/escala (ColumnTransformer)
        feat_names_orig = best_pipeline['preprocessor'].get_feature_names_out()

        # 3. Etapa 1: Aplica a máscara do VarianceThreshold
        mask_variance = best_pipeline['var_threshold'].get_support()
        features_after_variance = feat_names_orig[mask_variance]

        # 4. Etapa 2: Aplica a máscara do SelectKBest sobre as colunas 
        # sobressalentes
        mask_kbest = best_pipeline['feature_selection'].get_support()
        selected_features = features_after_variance[mask_kbest]

        # 5. Imprime o resultado final de forma amigável
        m = "Total de variáveis "
        print(f"{m}originais pré-processadas: {len(feat_names_orig)}")
        print(f"{m}após VarianceThreshold:    {len(features_after_variance)}")
        print(f"{m}selecionadas no modelo:    {len(selected_features)}")

        print("\n--- Lista das Variáveis Selecionadas ---")
        for i, feature in enumerate(selected_features, 1):
            print(f"{i}. {feature}")

        # Cria uma máscara final combinando as duas seleções
        final_mask = mask_variance.copy()
        final_mask[mask_variance] = mask_kbest

        # Constrói o relatório em DataFrame
        df_summary = pd.DataFrame({
            'Atributo_Preprocessado': feat_names_orig,
            'Passou_Variancia': mask_variance,
            'Selecionado_Final': final_mask
        })

        print("\n--- Status de Cada Atributo no Pipeline ---")
        print(df_summary.to_string(index=False))

        return df_summary

    def __preprocessador(self) -> ColumnTransformer:
        '''
            Cria um pré-processador que padroniza variáveis numéricas e 
            aplica One-Hot Encoding em variáveis categóricas.
                Retorna
                -------
                preprocessor : ColumnTransformer
                    Objeto ColumnTransformer para pré-processamento.
        '''
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    'num',
                    StandardScaler(),
                    (
                        self.df
                        .drop('IFN-γ', axis=1)
                        .select_dtypes(include=['number'])
                        .columns
                    )
                ),
                (
                    'cat',
                    OneHotEncoder(drop='first', handle_unknown='ignore'),
                    (
                        self.df
                        .select_dtypes(include=['object'])
                        .columns
                    )
                )
            ]
        )
        return preprocessor

    def __regressor_classify(self) -> tuple[Pipeline, dict]:
        '''
            Cria um pipeline para o regressor genérico e define a grade de 
            hiperparâmetros para busca.
                Retorna
                -------
                pipeline : Pipeline
                    Objeto Pipeline configurado com pré-processamento e regressão linear.
                param_grid : dict
                    Dicionário contendo a grade de hiperparâmetros para busca.
        '''
        # 4. Pipeline Principal
        pipeline = Pipeline([
            ('preprocessor', self.__preprocessador()),
            ('var_threshold', VarianceThreshold(threshold=1e-4)),
            ('feature_selection', SelectKBest(
                                                score_func=f_regression,
                                                k=min(10, self.X.shape[1])
                                            )), # Camada de Feature Selection
            ('regressor', LinearRegression())
        ])
        # 5. Definição da Grade de Hiperparâmetros
        param_grid = {
            'var_threshold__threshold': [1e-4, 0.01, 0.05],
            'feature_selection__k': [1, 2, 'all'],
            'regressor__fit_intercept': [True, False],
            'regressor__positive': [True, False]
        }

        return pipeline, param_grid

    def regress(
            self,
            modelo: str,
            n_splits: int = 5) -> tuple[pd.DataFrame, np.ndarray]:
        '''
            Treina e avalia o modelo especificado.
                Parâmetros
                ----------
                modelo : str
                    Nome do modelo a ser treinado. Opções: 
                                    'regressor'.
                n_splits : int, opcional
                    Número de divisões para a validação cruzada (default é 5).
                Retorna
                    return retorno, grid_search.predict(self.X)
                    pd.DataFrame
        '''
        if modelo == 'regressor':
            # 7. Configuração do GridSearch para Regressor genérico
            estimator, param_grid = self.__regressor_classify()
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=kf,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            # 5. Treinando o modelo para encontrar os melhores hiperparâmetros e grupos
            print("Iniciando o Grid Search com Cross Validation...")
            grid_search.fit(self.X, self.y)

        self.__variaveis_selecionadas(grid_search=grid_search)
        self.__obter_metricas(grid_search=grid_search, modelo=modelo)


        retorno = grid_search.predict(self.X)

        return retorno
