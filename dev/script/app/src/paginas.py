import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
from joblib import load

from src.acesso_data import (
                                acesso_data_test,
                                acesso_data_banco,
                                acesso_data_icon
                            )
from src.modelos.clustering import Clustering
from src.modelos.classifier import Classifier


class Paginas:
    def pagina_estrutura(self) -> str:
        '''
            Função de estruturação da página inicial.
            Parametros:
                self: Referência para a própria classe.
            Retorna:
                str: Mensagem de sucesso.
        '''
        st.set_page_config(
            page_title="FAcPyML",
            layout="wide",
            page_icon=acesso_data_icon()
        )
        st.markdown("""
            <style>
                .block-container {
                        padding-top: 0rem;
                        padding-bottom: 0rem;
                        padding-left: 3rem;
                        padding-right: 3rem;
                    }
            </style>
            """, unsafe_allow_html=True)

        title = "Métricas de Modelos de Machine Learning"
        st.markdown(
            f"<br><h3 style='text-align: center; '>{title}</h3>",
            unsafe_allow_html=True
        )

        return "Ajustado"

    def busca_dados_cluster(
            self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''
            Função de busca dos dados de treino e teste.
            Retorna:
                tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                DataFrames dos datasets de teste, treino e validação.
        '''
        # Importando os dados de treino e teste
        banco = acesso_data_banco()
        link = banco+"df_estruturada/"
        datasets = [link+i for i in os.listdir(link) if '.csv' in i]
        df_test = pd.read_csv(datasets[0])
        df_train = pd.read_csv(datasets[1])
        df_valid = pd.read_csv(datasets[2])

        return df_train, df_valid, df_test

    def busca_dados_class_regr(
            self) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        '''
            Função de busca dos dados de treino e teste.
            Retorna:
                tuple[dict[str, pd.DataFrame], pd.DataFrame]:
                DataFrames dos datasets de teste, treino e validação.
        '''
        # Importando os dados de treino e teste
        banco = acesso_data_banco()
        link = banco+"df_resultados/"
        datasets = {i[25:]: load(link+i) for i in os.listdir(link)}

        df_test = load(banco+'test_resultados_classif')

        return datasets, df_test

    def clustering(
            self,
            variavel_1: str,
            variavel_2: str,
            df_train: pd.DataFrame,
            df_test: pd.DataFrame) -> tuple[dict, str, str, str]:
        '''
            Função de teste dos modelos de clustering.
            Parametros:
                self: Referência para a própria classe.
                variavel_1: Primeira variável para o clustering.
                variavel_2: Segunda variável para o clustering.
                df_train: DataFrame de treino.
                df_test: DataFrame de teste.
            Retorna:
                tuple[dict, str, str, str]: Resultados dos modelos e
                    mensagens de sucesso.
        '''
        # K-means ------------------------------------------------
        i = variavel_1
        j = variavel_2

        cluster = Clustering(
            c1=i,
            c2=j,
            df=df_train,
            modelo='kmeans'
        )
        resultados = {}
        resultados["K-means"], msn_kmenans = cluster.testar_modelo(
            best_param={
                            'n_clusters': 3,
                            'init': 'k-means++',
                            'n_init': 10
                        },
            df_test=df_test,
        )

        # DBSCAN -------------------------------------------------
        i = variavel_1
        j = variavel_2

        cluster = Clustering(
            c1=i,
            c2=j,
            df=df_train,
            modelo='dbscan'
        )
        resultados["DBSCAN"], msn_dbscan = cluster.testar_modelo(
            best_param={
                            'eps': 0.5,
                            'min_samples': 10,
                            'metric': 'euclidean'
                        },
            df_test=df_test,
        )

        # Agglomerative ------------------------------------------
        i = variavel_1
        j = variavel_2

        cluster = Clustering(
            c1=i,
            c2=j,
            df=df_train,
            modelo='agglomerative'
        )
        resultados["Agg"], msn_agg = cluster.testar_modelo(
            best_param={
                            'n_clusters': 2,
                            'metric': 'euclidean',
                            'linkage': 'average'
                        },
            df_test=df_test,
        )

        return resultados, msn_kmenans, msn_dbscan, msn_agg

    def metricas(self) -> str:
        '''
            Função de cálculo das métricas de clustering.
            Parametros:
                self: Referência para a própria classe.
            Retorna:
                str: Mensagem de sucesso.
        '''
        df_train, df_valid, df_test = self.busca_dados_cluster()

        abas = st.tabs([
                            "Clustering",
                            'Classificação',
                            'Regressão',
                            'Datasets',
                        ])
        with abas[0]:
            col = st.columns((2, 2, .05, 2, .05, 2))
            with col[0]:
                # Create a dropdown menu
                variavel_1 = st.radio(
                    "Primeira variável:",
                    ['PPD_log', 'IFN-γ', 'CD3_2']
                )
                variavel_2 = st.radio(
                    "Segunda variável:",
                    ['CD3_2', 'PPD_log', 'IFN-γ']
                )

            resultados, msn_kmenans, msn_dbscan, msn_agg = self.clustering(
                                                        variavel_1=variavel_1,
                                                        variavel_2=variavel_2,
                                                        df_train=df_train,
                                                        df_test=df_test
                                                    )

            with col[1]:
                # Testando os modelos de clustering via K-means
                st.markdown(f"<p>{msn_kmenans}</p>", unsafe_allow_html=True)
                st.pyplot(fig=resultados["K-means"])

            with col[3]:
                # Testando os modelos de clustering via DBSCAN
                st.markdown(f"<p>{msn_dbscan}</p>", unsafe_allow_html=True)
                st.pyplot(fig=resultados["DBSCAN"])

            with col[5]:
                # Testando os modelos de clustering via Agglomerative
                st.markdown(f"<p>{msn_agg}</p>", unsafe_allow_html=True)
                st.pyplot(fig=resultados["Agg"])

        with abas[1]:
            dados, df_test_class = self.busca_dados_class_regr()

            col = st.columns((5, .5, 5, .5, 5))
            dados_metricas = {
                0: "GBM",
                2: "k-NN",
                4: "NB",
            }
            for k, v in dados_metricas.items():
                with col[k]:
                    st.write('='*69)
                    st.markdown(
                        f"<h4 style='text-align: center; '>{v}</h4>",
                        unsafe_allow_html=True
                    )
                    st.write('='*69)
                    classificador = Classifier(df=dados[v])
                    classificador.testar_hipotese()
                    st.divider()
                    classificador.matrix_confusao(
                        df=df_test_class,
                        obs='classe',
                        pred=v
                    )

            st.divider()

            col = st.columns((5, .5, 5, .5, 5))
            dados_metricas = {
                0: "NN",
                2: "RF",
                4: "SVM"
            }
            for k, v in dados_metricas.items():
                with col[k]:
                    st.write('='*69)
                    st.markdown(
                        f"<h4 style='text-align: center; '>{v}</h4>",
                        unsafe_allow_html=True
                    )
                    st.write('='*69)
                    classificador = Classifier(df=dados[v])
                    classificador.testar_hipotese()
                    st.divider()
                    classificador.matrix_confusao(
                        df=df_test_class,
                        obs='classe',
                        pred=v
                    )

        with abas[2]:
            st.write("Métricas de Regressão:")
            st.write("Em desenvolvimento...")

        with abas[3]:
            col = st.columns((2, .05, 2, .05, 2))
            with col[0]:
                # Importando os dados de treino
                st.markdown(
                    "<h4 style='text-align: center; '>Dados de Treino</h4>",
                    unsafe_allow_html=True
                )
                st.dataframe(df_train)

            with col[2]:
                # Importando os dados de validação
                st.markdown(
                    "<h4 style='text-align: center; '>Dados de Validação</h4>",
                    unsafe_allow_html=True
                )
                st.dataframe(df_valid)

            with col[4]:
                # Importando os dados de teste
                st.markdown(
                    "<h4 style='text-align: center; '>Dados de Teste</h4>",
                    unsafe_allow_html=True
                )
                st.dataframe(df_test)

            col = st.columns((2, 10))
            with col[0]:
                metr = pd.DataFrame({
                    "Métrica": ["Treino", "Validação", "Teste"],
                    "Tamanho": [len(df_train), len(df_valid), len(df_test)]
                })
                st.write("Tamanho dos datasets:")
                st.dataframe(metr)

        return "Métricas Calculadas"

    def pagina_inicio(self) -> str:
        '''
            Função de inicialização da página inicial.
            Parametros:
                self: Referência para a própria classe.
            Retorna:
                str: Mensagem de sucesso.
        '''
        return "Diga lá loco"

    @st.cache_data
    def logo_datalab(_self) -> str:
        '''
            Função de inserção do logo do DataLab.
            Parametros:
                _self: Referência para a própria classe.
            Retorna:
                str: Mensagem de sucesso.
        '''
        # Plota a imagem
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(np.asarray(Image.open(acesso_data_test())))
        plt.axis("off")

        # Edita as colunas e insere os dados
        col_img = st.columns((15, 1))
        with col_img[1]:
            st.pyplot(fig=fig)

        return "Logo Inserido"
