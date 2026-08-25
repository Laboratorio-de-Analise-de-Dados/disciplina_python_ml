from src.iniciar import Iniciar


if __name__ == '__main__':
    mensagem = 'Alguma coisa'
    inicia = Iniciar(mensagem=mensagem)
    inicia.mostra_mensagem()

    print('-'*50)

    print(inicia.testo)
