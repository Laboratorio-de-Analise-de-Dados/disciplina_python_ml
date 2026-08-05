import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os

from src.acesso_data import acesso_data_test, acesso_data_banco
# from src.modelos.clustering import Clustering


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

        col = st.columns((2, .05, 2, .05, 2))
        with col[0]:
            # Importando os dados de treino
            df_train = pd.read_csv(datasets[1])
            st.markdown(
                "<h4 style='text-align: center; '>Dados de Treino</h4>",
                unsafe_allow_html=True
            )
            st.dataframe(df_train)

        with col[2]:
            # Importando os dados de validação
            df_valid = pd.read_csv(datasets[2])
            st.markdown(
                "<h4 style='text-align: center; '>Dados de Validação</h4>",
                unsafe_allow_html=True
            )
            st.dataframe(df_valid)

        with col[4]:
            # Importando os dados de teste
            df_test = pd.read_csv(datasets[0])
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

        return df_train, df_valid, df_test

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
