import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
from joblib import load

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

        # Importando os dados de treino e teste
        banco = acesso_data_banco()
        link = banco+"df_estruturada/"
        datasets = [link+i for i in os.listdir(link)]

        col = st.columns((2, .05, 2))
        with col[0]:
            # Importando os dados de treino
            df_train = pd.concat([
                load(datasets[1]),
                load(datasets[4])
            ], axis=1)
            title = "Dados de Treino"
            st.markdown(
                f"<h4 style='text-align: center; '>{title}</h4>",
                unsafe_allow_html=True
            )
            st.dataframe(df_train)

        with col[2]:
            # Importando os dados de teste
            df_test = pd.concat([
                load(datasets[0]),
                load(datasets[3])
            ], axis=1)
            title = "Dados de Teste"
            st.markdown(
                f"<h4 style='text-align: center; '>{title}</h4>",
                unsafe_allow_html=True
            )
            st.dataframe(df_test)

        return "Ajustado"

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
