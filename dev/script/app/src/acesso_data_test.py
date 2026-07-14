import os


def acesso_data_test() -> str:
    '''
        Função para acessar o banco de dados de treino.
    '''
    path_banco = "./app/data/img/DataLab_Logo_i.jpg"
    database = "./dev/script/app/data/img/DataLab_Logo_i.jpg"

    if os.path.isfile(path_banco):
        database = path_banco

    return database
