# Importando modelos
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Visualização dos dados
import matplotlib.pyplot as plt

# Edição das databases
import pandas as pd
import numpy as np

# Seleção de hiperparâmetros e validação cruzada
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Extruturação dos dados e pré-processamento
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Organização do fluxo de trabalho (Pipeline)
from sklearn.pipeline import Pipeline

# Métricas da classificação
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay


class Classifier:
    '''
    Classificador genérico que encapsula diferentes modelos de aprendizado de máquina.
        Parâmetros
        ----------
        df : pd.DataFrame
            DataFrame contendo as features e a coluna alvo 'classe'.

        Métodos
        -------
        classify(modelo)
            Treina e avalia o modelo especificado.
    '''
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.X = df.drop('classe', axis=1)
        self.y = df['classe']

        return None
    
    def __matrix_confusao(self, grid_search: GridSearchCV) -> None:
        '''
        Exibe a matriz de confusão do melhor modelo encontrado pelo GridSearchCV.
            Parâmetros
            ----------
            grid_search : GridSearchCV
                Objeto GridSearchCV já ajustado com os dados.
                
        '''
        grid_search.fit(self.X, self.y)

        # 10. Avaliação no conjunto de Teste
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X)

        accuracy = accuracy_score(self.y, y_pred)
        cm = confusion_matrix(self.y, y_pred)
        disp = ConfusionMatrixDisplay(
                        confusion_matrix=cm,
                        display_labels=self.y.value_counts().index,
        )
        disp.plot(cmap=plt.cm.Greens)
        plt.title(f"Accuracy {accuracy:.2f}")
        plt.show()
        print("\nClassification Report:")
        print(classification_report(self.y, y_pred))
        return None

    def __metricas_pontuais(self, grid_search: GridSearchCV) -> None:
        '''
            Exibe os melhores hiperparâmetros e a acurácia do melhor modelo encontrado pelo GridSearchCV.
                Parâmetros
                ----------
                grid_search : GridSearchCV
                    Objeto GridSearchCV já ajustado com os dados.
        '''
        # 10. Treinamento Final no Dataset Completo
        # Após validar a capacidade de generalização via Nested CV, ajustamos o GridSearch nos dados totais

        print("--- Treinando modelo com Pipeline de pré-processamento ---")
        grid_search.fit(self.X, self.y)

        # 9. Exibição dos Resultados
        print("\n--- Melhores Resultados do Grid Search ---")
        print(f"Melhores hiperparâmetros: {grid_search.best_params_}")
        print(f"Melhor acurácia (CV): {grid_search.best_score_:.4f}")


        # 10. Avaliação no conjunto de Teste
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X)

        print("\n--- Desempenho no Conjunto de Teste ---")
        print(f"Acurácia final: {accuracy_score(self.y, y_pred):.4f}\n")

        return None

    def __preprocessador(self) -> ColumnTransformer:
        '''
            Cria um pré-processador que padroniza variáveis numéricas e aplica One-Hot Encoding em variáveis categóricas.
                Retorna
                -------
                preprocessor : ColumnTransformer
                    Objeto ColumnTransformer configurado para pré-processamento.
        '''
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    'num',
                    StandardScaler(),
                    (
                        self.df
                        .select_dtypes(include=['number'])
                        .columns
                    )
                ),
                (
                    'cat',
                    OneHotEncoder(drop='first', handle_unknown='ignore'),
                    (
                        self.df
                        .drop('classe', axis=1)
                        .select_dtypes(include=['object'])
                        .columns
                    )
                )
            ]
        )
        return preprocessor

    def __knn_classify(self) -> tuple[Pipeline, dict]:
        '''
            Cria um pipeline para o classificador KNN e define a grade de hiperparâmetros para busca.
                Retorna
                -------
                pipeline : Pipeline
                    Objeto Pipeline configurado com pré-processamento e KNN.
                param_grid : dict
                    Dicionário contendo a grade de hiperparâmetros para busca.
        '''
        # 4. Pipeline Principal
        pipeline = Pipeline([
            ('preprocessor', self.__preprocessador()),
            ('knn', KNeighborsClassifier())
        ])
        # 5. Definição da Grade de Hiperparâmetros
        param_grid = {
            'knn__n_neighbors': [1, 3, 5],
            'knn__weights': ['uniform', 'distance'],
            'knn__metric': ['euclidean', 'manhattan']
        }

        return pipeline, param_grid

    def __svm_classify(self) -> tuple[Pipeline, dict]:
        '''
            Cria um pipeline para o classificador SVM e define a grade de hiperparâmetros para busca.
                Retorna
                -------
                pipeline : Pipeline
                    Objeto Pipeline configurado com pré-processamento e SVM.
                param_grid : dict
                    Dicionário contendo a grade de hiperparâmetros para busca.
        '''
        # 4. Pipeline Principal com o Classificador SVM (SVC)
        pipeline = Pipeline([
            ('preprocessor', self.__preprocessador()),
            ('svm', SVC(random_state=42))
        ])
        # 5. Definição da Grade de Hiperparâmetros para o SVM
        # Note o prefixo 'svm__' para acessar os parâmetros do SVC dentro do Pipeline
        param_grid = {
            'svm__C': [0.1, 1, 10, 100],               # Parâmetro de regularização
            'svm__kernel': ['linear', 'rbf'],           # Tipo de kernel (Linear ou Radial Basis Function)
            'svm__gamma': ['scale', 'auto', 0.01, 0.1]  # Coeficiente do kernel RBF
        }

        return pipeline, param_grid

    def __rf_classify(self) -> tuple[Pipeline, dict]:
        '''
            Cria um pipeline para o classificador Random Forest e define a grade de hiperparâmetros para busca.
                Retorna
                -------
                pipeline : Pipeline
                    Objeto Pipeline configurado com pré-processamento e Random Forest.
                param_grid : dict
                    Dicionário contendo a grade de hiperparâmetros para busca.
        '''
        # 4. Pipeline Principal com Random Forest
        pipeline = Pipeline([
            ('preprocessor', self.__preprocessador()),
            ('rf', RandomForestClassifier(random_state=42))
        ])

        # 5. Definição da Grade de Hiperparâmetros para o Random Forest
        # Note o prefixo 'rf__' para acessar os parâmetros do modelo dentro do Pipeline
        param_grid = {
            'rf__n_estimators': [50, 100, 200],          # Número de árvores na floresta
            'rf__max_depth': [None, 5, 10],               # Profundidade máxima de cada árvore
            'rf__min_samples_split': [2, 5],              # Mínimo de amostras para dividir um nó
            'rf__criterion': ['gini', 'entropy']          # Critério de medição de qualidade da divisão
        }

        return pipeline, param_grid

    def __gbm_classify(self) -> tuple[Pipeline, dict]:
        '''
            Cria um pipeline para o classificador Gradient Boosting e define a grade de hiperparâmetros para busca.
                Retorna
                -------
                pipeline : Pipeline
                    Objeto Pipeline configurado com pré-processamento e Gradient Boosting.
                param_grid : dict
                    Dicionário contendo a grade de hiperparâmetros para busca.
        '''
        # 4. Pipeline Principal com Gradient Boosting
        pipeline = Pipeline([
            ('preprocessor', self.__preprocessador()),
            ('gb', GradientBoostingClassifier(random_state=42))
        ])

        # 5. Definição da Grade de Hiperparâmetros para o Gradient Boosting
        # Note o prefixo 'gb__' para acessar os parâmetros do modelo dentro do Pipeline
        param_grid = {
            'gb__n_estimators': [50, 100, 150],       # Número de estágios de boosting (árvores)
            'gb__learning_rate': [0.01, 0.1, 0.2],     # Taxa de aprendizado (encolhimento do impacto de cada árvore)
            'gb__max_depth': [3, 5],                  # Profundidade máxima dos estimadores individuais
            'gb__subsample': [0.8, 1.0]                # Fração de amostras usadas para ajustar os estimadores base
        }

        return pipeline, param_grid

    def __nb_classify(self) -> tuple[Pipeline, dict]:
        '''
            Cria um pipeline para o classificador Gaussian Naive Bayes e define a grade de hiperparâmetros para busca.
                Retorna
                -------
                pipeline : Pipeline
                    Objeto Pipeline configurado com pré-processamento e Gaussian Naive Bayes.
                param_grid : dict
                    Dicionário contendo a grade de hiperparâmetros para busca.
        '''
        pipeline = Pipeline([
            ('preprocessor', self.__preprocessador()),
            ('nb', GaussianNB())
        ])

        # 5. Definição da Grade de Hiperparâmetros para o GaussianNB
        # Note o prefixo 'nb__' para acessar os parâmetros do modelo dentro do Pipeline
        param_grid = {
            'nb__var_smoothing': np.logspace(0, -9, num=10) # Suavização de variância para estabilidade numérica
        }

        return pipeline, param_grid

    def __nn_classify(self) -> tuple[Pipeline, dict]:
        '''
            Cria um pipeline para o classificador Rede Neural (MLP) e define a grade de hiperparâmetros para busca.
                Retorna
                -------
                pipeline : Pipeline
                    Objeto Pipeline configurado com pré-processamento e Rede Neural (MLP).
                param_grid : dict
                    Dicionário contendo a grade de hiperparâmetros para busca.
        '''
        # 4. Pipeline Principal com Rede Neural (MLP)
        # max_iter expandido para garantir convergência durante a otimização
        pipeline = Pipeline([
            ('preprocessor', self.__preprocessador()),
            ('mlp', MLPClassifier(max_iter=1000, random_state=42))
        ])

        # 5. Definição da Grade de Hiperparâmetros para a Rede Neural
        # Note o prefixo 'mlp__' para acessar os parâmetros do modelo dentro do Pipeline
        param_grid = {
            'mlp__hidden_layer_sizes': [(10,), (20, 10), (50,)], # Arquiteturas: 1 camada com 10/50 neurônios ou 2 camadas (20, 10)
            'mlp__activation': ['relu', 'tanh'],                  # Funções de ativação
            'mlp__solver': ['adam', 'lbfgs'],                     # Otimizadores (lbfgs costuma performar muito bem em datasets pequenos)
            'mlp__alpha': [0.0001, 0.01]                          # Termo de regularização L2 (penalty)
        }

        return pipeline, param_grid

    def classify(self, modelo) -> None:
        '''
            Treina e avalia o modelo especificado.
                Parâmetros
                ----------
                modelo : str
                    Nome do modelo a ser treinado. Opções: 'knn', 'svm', 'rf', 'gbm', 'nb', 'nn'.
        '''
        if modelo == 'knn':
            # 7. Configuração da Validação Cruzada Interna (Inner CV) via GridSearch
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                estimator=self.__knn_classify()[0],
                param_grid=self.__knn_classify()[1],
                cv=inner_cv,
                scoring='accuracy',
                n_jobs=-1
            )
        elif modelo == 'svm':
            # 7. Configuração da Validação Cruzada Interna (Inner CV) via GridSearch
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                estimator=self.__svm_classify()[0],
                param_grid=self.__svm_classify()[1],
                cv=inner_cv,
                scoring='accuracy',
                n_jobs=-1
            )
        elif modelo == 'rf':
            # 7. Configuração da Validação Cruzada Interna (Inner CV) via GridSearch
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                estimator=self.__rf_classify()[0],
                param_grid=self.__rf_classify()[1],
                cv=inner_cv,
                scoring='accuracy',
                n_jobs=-1
            )
        elif modelo == 'gbm':
            # 7. Configuração da Validação Cruzada Interna (Inner CV) via GridSearch
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                estimator=self.__gbm_classify()[0],
                param_grid=self.__gbm_classify()[1],
                cv=inner_cv,
                scoring='accuracy',
                n_jobs=-1
            )
        elif modelo == 'nb':
            # 7. Configuração da Validação Cruzada Interna (Inner CV) via GridSearch
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                estimator=self.__nb_classify()[0],
                param_grid=self.__nb_classify()[1],
                cv=inner_cv,
                scoring='accuracy',
                n_jobs=-1
            )
        elif modelo == 'nn':
            # 7. Configuração da Validação Cruzada Interna (Inner CV) via GridSearch
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                estimator=self.__nn_classify()[0],
                param_grid=self.__nn_classify()[1],
                cv=inner_cv,
                scoring='accuracy',
                n_jobs=-1
            )

        self.__metricas_pontuais(grid_search=grid_search)
        self.__matrix_confusao(grid_search=grid_search)

        return None
