import os

from src.paginas import Paginas
from src.acesso_data import acesso_data_test, acesso_data_banco


def test_acesso_data_img():
    # Given
    teste_local = "./app/data/img/DataLab_Logo_i.jpg"

    # When
    teste_saida = acesso_data_test()
    if os.path.isfile(teste_saida):
        teste_local = teste_saida

    # Then
    assert teste_local == teste_saida


def test_acesso_data_banco():
    # Given
    teste_local = "./app/data/"

    # When
    teste_saida = acesso_data_banco()
    if os.path.isdir(teste_saida):
        teste_local = teste_saida

    # Then
    assert teste_local == teste_saida


def test_busca_dados():
    # Given
    teste_entrada = 3

    # When
    iniciando = Paginas()
    df_train, df_valid, df_test = iniciando.busca_dados()
    teste_saida = len([df_train, df_valid, df_test])

    # Then
    assert teste_entrada == teste_saida


def test_metricas_clustering():
    # Given
    teste_entrada = "Métricas Calculadas"

    # When
    iniciando = Paginas()
    teste_saida = iniciando.metricas_clustering()

    # Then
    assert teste_entrada == teste_saida


def test_estrutura_pagina():
    # Given
    teste_entrada = "Ajustado"

    # When
    iniciando = Paginas()
    teste_saida = iniciando.pagina_estrutura()

    # Then
    assert teste_entrada == teste_saida


def test_iniciando_sistema():
    # Given
    teste_entrada = "Diga lá loco"

    # When
    iniciando = Paginas()
    teste_saida = iniciando.pagina_inicio()

    # Then
    assert teste_entrada == teste_saida


def test_logo_datalab():
    # Given
    teste_entrada = "Logo Inserido"

    # When
    iniciando = Paginas()
    teste_saida = iniciando.logo_datalab()

    # Then
    assert teste_entrada == teste_saida
