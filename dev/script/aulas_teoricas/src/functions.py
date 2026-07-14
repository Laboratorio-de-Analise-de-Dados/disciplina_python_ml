class Functions:
    def __init__(self, dados: str, coluna_x: str, coluna_y: str) -> None:
        self.dados = dados
        self.coluna_x = coluna_x
        self.coluna_y = coluna_y

    def __colunas_base(self) -> tuple:
        x_base = self.dados.loc[:, self.coluna_x]
        y_base = self.dados.loc[:, self.coluna_y]
        a = (x_base.min(), y_base.min())
        b = (x_base.max(), y_base.max())
        return a, b

    def __coeficiente_angular(self) -> float:
        '''
            m = ya - yb
            ---------
                xa - xb

            m = delta y / delta x
        '''
        a = self.__colunas_base()[0]
        b = self.__colunas_base()[1]

        delta_x = a[0] - b[0]
        delta_y = a[1] - b[1]

        if delta_x < 0:
            delta_x = delta_x * -1

        if delta_y < 0:
            delta_y = delta_y * -1

        return (delta_y, delta_x)
    
    def equacao_da_reta_calculo(
            self,
            varia_coeficiente: int = None,
            new_y: float = None
        ) -> tuple:
        '''
            y - y0 = m(x - x0)
        '''
        a = self.__colunas_base()[0]
        b = self.__colunas_base()[1]
        
        x = b[0]
        if new_y != None:
            x = new_y
        ponto = a

        m = self.__coeficiente_angular()
        x0 = m[0] * ponto[0]
        y0 = m[1] * ponto[1]
        c = - x0 + y0

        m_perc = 0
        if varia_coeficiente != None:
            m_perc = (m[0]/10)*varia_coeficiente

        y = (((m[0] + m_perc) * x) + c)/m[1]

        return y

    def reta(
            self,
            x: float,
            intercept: int = 5,
            slope: int = 5,
            bias: int = 0
        ) -> float:
        y = intercept + x * slope + bias
        return y