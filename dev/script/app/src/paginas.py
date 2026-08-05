import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os

from src.acesso_data import acesso_data_test, acesso_data_banco
from src.modelos.clustering import Clustering


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
            page_title="Alunos",
            layout="wide",
        )
        st.markdown("""
            <style>
                .block-container {
                        padding-top: 1rem;
                        padding-bottom: 0rem;
                        padding-left: 3rem;
                        padding-right: 3rem;
                    }
            </style>
            """, unsafe_allow_html=True)

        title = "Página Inicial"
        st.markdown(
            f"<h1 style='text-align: center; '>{title}</h1>",
            unsafe_allow_html=True
        )

        return "Ajustado"

    def busca_dados(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''
            Função de busca dos dados de treino e teste.
            Retorna:
                tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                DataFrames dos datasets de teste, treino e validação.
        '''
        # Importando os dados de treino e teste
        banco = acesso_data_banco()
        link = banco+"df_estruturada/"
        datasets = [link+i for i in os.listdir(link)]
        df_train = pd.read_csv(datasets[1])
        df_valid = pd.read_csv(datasets[2])
        df_test = pd.read_csv(datasets[0])

        return df_train, df_valid, df_test

    def metricas_clustering(self) -> str:
        '''
            Função de cálculo das métricas de clustering.
            Parametros:
                self: Referência para a própria classe.
            Retorna:
                str: Mensagem de sucesso.
        '''
        df_train, df_valid, df_test = self.busca_dados()

        abas = st.tabs([
                            "Métricas de Clustering",
                            'Métricas de Classificação',
                            'Métricas de Regressão',
                            'Dados de Treino, Validação e Teste',
                        ])
        with abas[0]:
            col = st.columns((2, 2, .05, 2, .05, 2))
            with col[0]:
                # Create a dropdown menu
                st.write("Métricas Clusterização")
                variavel_1 = st.radio(
                    "Primeira variável:",
                    ['PPD_log', 'IFN-γ', 'CD3_2']
                )
                variavel_2 = st.radio(
                    "Segunda variável:",
                    ['PPD_log', 'IFN-γ', 'CD3_2']
                )

            with col[1]:
                # Testando os modelos de clustering via K-means
                st.write("Métricas de Clustering via K-means:")
                i = variavel_1
                j = variavel_2
                st.write("-"*50)
                st.write(i, j)

                cluster = Clustering(
                    c1=i,
                    c2=j,
                    df=df_train,
                    modelo='kmeans'
                )
                resultados = {}
                resultados["K-means"] = cluster.testar_modelo(
                    best_param={
                                    'n_clusters': 3,
                                    'init': 'k-means++',
                                    'n_init': 10
                                },
                    df_test=df_test,
                )

            with col[3]:
                # Testando os modelos de clustering via DBSCAN
                st.write("Métricas de Clustering via DBSCAN:")
                i = variavel_1
                j = variavel_2
                st.write("-"*50)
                st.write(i, j)

                cluster = Clustering(
                    c1=i,
                    c2=j,
                    df=df_train,
                    modelo='dbscan'
                )
                resultados["DBSCAN"] = cluster.testar_modelo(
                    best_param={
                                    'eps': 0.5,
                                    'min_samples': 10,
                                    'metric': 'euclidean'
                                },
                    df_test=df_test,
                )

            with col[5]:
                # Testando os modelos de clustering via Agglomerative
                st.write("Métricas de Clustering via Agglomerative:")
                i = variavel_1
                j = variavel_2
                st.write("-"*50)
                st.write(i, j)

                cluster = Clustering(
                    c1=i,
                    c2=j,
                    df=df_train,
                    modelo='agglomerative'
                )
                resultados["Agg"] = cluster.testar_modelo(
                    best_param={
                                    'n_clusters': 2,
                                    'metric': 'euclidean',
                                    'linkage': 'average'
                                },
                    df_test=df_test,
                )

                st.dataframe(
                            pd.DataFrame(
                                columns=['K-means', 'DBSCAN', 'Agglomerative'],
                                index=[''],
                                data=[[
                                    resultados['K-means'],
                                    resultados['DBSCAN'],
                                    resultados['Agg']
                                ]]
                            )
                        )
        with abas[1]:
            st.write("Métricas de Classificação:")
            st.write("Em desenvolvimento...")

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
        data = acesso_data_test()
        data_lab_logo = data
        # Importa a imagem
        img = np.asarray(Image.open(data_lab_logo))

        # Plota a imagem
        plt.imshow(img)
        plt.axis("off")

        # Edita as colunas e insere os dados
        col_img = st.columns((15, 1))
        with col_img[1]:
            st.write("DataLab()")
            st.pyplot(fig=plt)

        return "Logo Inserido"
