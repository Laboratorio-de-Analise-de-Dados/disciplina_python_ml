import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.acesso_data_test import acesso_data_test

class Paginas:
    def pagina_estrutura(self) -> str:
        '''
            Função de inicialização da estrutura da página.
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

    def pagina_inicio(self) -> str:
        '''
            Função de inicialização do pagina_inicial.
        '''
        return "Diga lá loco"

    @st.cache_data
    def logo_datalab(_self) -> str:
        '''
        Função que implementa o Logo do DataLab()
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
