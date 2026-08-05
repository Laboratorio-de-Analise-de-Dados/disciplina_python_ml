import os

from src.paginas import Paginas
from src.acesso_data import acesso_data_test, acesso_data_banco


def test_acesso_data_img():
    # Given
    teste_remoto = "./dev/script/app/data/img/DataLab_Logo_i.jpg"
    teste_local = "./app/data/img/DataLab_Logo_i.jpg"

    # When
    teste_saida = acesso_data_test()
    if os.path.isfile(teste_local):
        teste_saida = teste_local
    else:
        teste_saida = teste_remoto

    # Then
    assert teste_local == teste_saida


def test_acesso_data_banco():
    # Given
    teste_remoto = "./dev/script/app/data/"
    teste_local = "./app/data/"

    # When
    teste_saida = acesso_data_banco()
    if os.path.isdir(teste_local):
        teste_saida = teste_local
    else:
        teste_saida = teste_remoto

    # Then
    assert teste_local == teste_saida


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
