from src.paginas import pagina_inicial


def test_pagina_inicial():
    # Given
    teste_entrada = "Página inicial carregada com sucesso."

    # When
    teste_saida = pagina_inicial()

    # Then
    assert teste_entrada == teste_saida
