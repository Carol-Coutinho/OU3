# Coleção de funções para a disciplina de OU3
import numpy as np
from scipy.optimize import fsolve
import pandas as pd

def f_Pvap_Antoine_db(Temp, i_comp, dados):
    #import numpy as np
    ''' Função que calcula a pressão de vapor, segundo a equação de Antoine, para o componente
      i_comp presente no databank_properties.pickle.
      Equação de Antoine: Pvap = exp(A - B /(Temp + C)), com: 
      [Temp] = K
      [Pvap] = mmHg
      Entrada (argumentos da função)
      Temp   = temperatura em K para a qual será calculada a Pvap
      i_comp = inteiro que corresponde ao número do componente no banco de dados
      dados  = pandas dataframe com os dados lidos do arquivo
      Saida: tupla
      Pvap - pressão de vapor do i_comp em mmHg
      par = dicionário com os parâmetros A, B e C da equação de Antoine
    '''
    # param <- as.numeric(param)
    par_array = np.array(dados[dados['num'] == i_comp][['pvap_a','pvap_b','pvap_c']])[0]
    par = {'a': par_array[0], 'b': par_array[1], 'c': par_array[2]}
    a = par['a']
    b = par['b']
    c = par['c']
    Pvap = np.exp(a - b/(Temp + c))
    # attr(x = Pvap, which = "units") <- "mmHg"
    return Pvap, par

def f_K_Raoult_db(T_eq, P_eq, lista_componentes, dados):
    # import numpy as np
    ''' Função para o cálculo da volatilidade segundo a Lei de Raoult:
        - fase vapor -> mistura de gás ideal
        - fase líquida -> solução normal
        K = P_vap(Teq) / P_eq
        Entrada (argumentos da função)
        T_eq - temperatura de equilíbrio em K
        P_eq - pressão de equilíbrio em mmHg
        lista_componentes - lista com os números inteiro dos componentes no databank
        dados - pandas dataframe com os dados do databank_properties.pickle
        Saida: tupla
        K_comp - np.array com os valores da volatilidade na ordem da lista_componentes
        P_vap_comp - np.array com os valores de P_vap segundo a equação de Antoine e os parâmetros
                    do databank_properties.pickle
    '''
    nc = len(lista_componentes)
    P_vap_comp = np.empty(nc)
    K_comp = np.empty(nc)
    k = 0
    for i_comp in lista_componentes:
        P_vap_comp[k], par = f_Pvap_Antoine_db(T_eq, i_comp, dados)
        K_comp[k] = P_vap_comp[k] / P_eq
        k += 1
    return K_comp, P_vap_comp

def f_Pb_T(Temp,P,z,lista_componentes,dados):
  ''' Função que retorna o resíduo para o cálculo da Temperatura do ponto de bolha
      Entrada:
      Temp - temperaura de equilíbrio em K - variável implícita da equação
      P - pressão de equilíbrio em mmHg
      lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
      dados - dataframe com os dados do databank
      Saida:
      f - resíduo da função (f = 0 -> solução)
  '''
  if (type(Temp) == float):
    Temp = np.array([Temp])
  nc = len(z)
  nr = len(Temp)
  MP = np.empty((nr,nc))
  x = z
  for i, T_vez in enumerate(Temp):
    K_comp = f_K_Raoult_db(T_vez, P, lista_componentes, dados)[0]
    MP[i,:] = K_comp * x
  f = 1 - np.sum(MP, axis=1)
  return f

def f_Po_T(Temp,P,z,lista_componentes,dados):
  ''' Função que retorna o resíduo para o cálculo da Temperatura do ponto de orvalho
      Entrada:
      Temp - temperaura de equilíbrio em K - variável implícita da equação
      P - pressão de equilíbrio em mmHg
      lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
      dados - dataframe com os dados do databank
      Saida:
      f - resíduo da função (f = 0 -> solução)
  '''
  if (type(Temp) == float):
    Temp = np.array([Temp])
  nc = len(z)
  nr = len(Temp)
  MP = np.empty((nr,nc))
  y = z
  for i, T_vez in enumerate(Temp):
    K_comp = f_K_Raoult_db(T_vez, P, lista_componentes, dados)[0]
    MP[i,:] = y / K_comp
  f = 1 - np.sum(MP, axis=1)
  return f

def f_Pb_P(P,Temp,z,lista_componentes,dados):
  ''' Função que retorna o resíduo para o cálculo da Pressão do ponto de bolha
      Entrada:
      P - pressão de equilíbrio em mmHg - variável implícita da equação
      Temp - temperaura de equilíbrio em K 
      lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
      dados - dataframe com os dados do databank
      Saida:
      f - resíduo da função (f = 0 -> solução)
  '''
  if (type(P) == float):
    P = np.array([P])
  nc = len(z)
  nr = len(P)
  MP = np.empty((nr,nc))
  x = z
  for i, P_vez in enumerate(P):
    K_comp = f_K_Raoult_db(Temp, P_vez, lista_componentes, dados)[0]
    MP[i,:] = K_comp * x
  f = 1 - np.sum(MP, axis=1)
  return f

def f_Po_P(P,Temp,z,lista_componentes,dados):
  ''' Função que retorna o resíduo para o cálculo da Pressão do ponto de orvalho
      Entrada:
      P - pressão de equilíbrio em mmHg - variável implícita da equação
      Temp - temperaura de equilíbrio em K 
      lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
      dados - dataframe com os dados do databank
      Saida:
      f - resíduo da função (f = 0 -> solução)
  '''
  if (type(P) == float):
    P = np.array([P])
  nc = len(z)
  nr = len(P)
  MP = np.empty((nr,nc))
  y = z
  for i, P_vez in enumerate(P):
    K_comp = f_K_Raoult_db(Temp, P_vez, lista_componentes, dados)[0]
    MP[i,:] = y / K_comp
  f = 1 - np.sum(MP, axis=1)
  return f

def f_calculo_PbPo_db(vp, x_pot, z, lista_componentes, dados):
    ''' Função para o cálculo das temperatura ou pressões do ponto de bolha 
          e do ponto de orvalho ( [T] em K e [P] em mmHg)
        Entradas:
        vp - variável do problema 'T' ou 'P' - string
        x_pot - valor de pressão ou temperatura dado
        z - composição da carga em fração molar
        lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
        dados - dataframe com os dados do databank
        Saidas:
        Se vp == 'T' -> T_Pb, T_Po, T_eb_comp = lista com as temperaturas de
                        ebulição normal dos componentes
        Se vp == 'P' -> P_Pb, P_Po, M_P_vap = matriz com as pressões de 
                        vapor dos componentes nas T_eb_comp
    '''
    #from scipy.optimize import fsolve
    nc = len(lista_componentes)
    if (vp == 'T'):
        P_eq = x_pot
        T_eb_comp = dados[dados['num'].isin(lista_componentes)]['boiling_point']
        T_eb_comp = T_eb_comp.tolist()
        T_guest = (min(T_eb_comp) + max(T_eb_comp) )/2
        T_Pb = fsolve(f_Pb_T, T_guest, args=(P_eq, z, lista_componentes, dados))[0]
        T_Po = fsolve(f_Po_T, T_guest, args=(P_eq, z, lista_componentes, dados))[0]
        return (T_Pb, T_Po, T_eb_comp)
    if (vp == 'P'):
        T_eq = x_pot
        T_eb_comp = dados[dados['num'].isin(lista_componentes)]['boiling_point']
        T_eb_comp = T_eb_comp.tolist()
        P_vap_eb_comp = np.empty(nc)
        k = 0
        for i_comp in lista_componentes:
          P_vap_eb_comp[k] = f_Pvap_Antoine_db(T_eq, i_comp, dados)[0]
          k += 1
        P_guest = (np.min(P_vap_eb_comp) + np.max(P_vap_eb_comp))/2
        P_Pb = fsolve(f_Pb_P, P_guest, args=(T_eq, z, lista_componentes, dados))[0]
        P_Po = fsolve(f_Po_P, P_guest, args=(T_eq, z, lista_componentes, dados))[0]
        return (P_Pb, P_Po, P_vap_eb_comp)

def f_res_RR_flash_db(fv, z, P, Temp, lista_componentes, dados):
    ''' Função que determina o resíduo da equação de Rachford-Rice para o flash
        multicomponente na solução para encontrar fv (fração vaporizada da carga)
      Entrada:
      fv - fração vaporizada da carga - variável implícita
      z - composição da carga em fração molar
      P - pressão do flash em mmHg
      T - temperatura do flash em K
      lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
      dados - dataframe com os dados do databank
      Saidas:
      res - resíduo na busca da solução - res = 0 -> solução
    '''
    nc = len(lista_componentes)
    if (type(fv) == float):
      fv = np.array([fv])
    nr = len(fv)
    K_comp = f_K_Raoult_db(Temp, P, lista_componentes, dados)[0]
    M_parc = np.empty((nr, nc))
    num = z * K_comp
    for i, fv_vez in enumerate(fv):
        den = 1.0 + fv_vez*(K_comp - 1.0)
        M_parc[i,:] = num / den
    res = 1.0 - np.sum(M_parc, axis=1)
    return res

def f_sol_RR_flash_db(z, P, Temp, lista_componentes, dados):
    ''' Função que resolve a equação de Rachford-Rice e encontra a fv 
        (fração vaporizada da carga)
        Entrada:
        z - composição da carga em fração molar
        P - pressão do flash em mmHg
        T - temperatura do flash em K
        lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
        dados - dataframe com os dados do databank
        Saidas: {dicionário}
        fv_flash - fração vaporizada - solução do flash
        x_eq - composição do líquido no equilíbrio
        y_eq - composição do vapor no equilíbrio
        K_comp - volatilidade dos componentes
        alpha_comp - volatilidade relativa em relação ao componente chave pesado (i_chk)
    '''
    fv_guest = 0.5
    fv_flash = fsolve(f_res_RR_flash_db, fv_guest, args=(z, P, Temp, lista_componentes, dados))[0]
    K_comp = f_K_Raoult_db(Temp, P, lista_componentes, dados)[0]
    num = z * K_comp
    den = 1.0 + fv_flash*(K_comp - 1.0)
    y_eq = num / den
    x_eq = y_eq / K_comp
    i_chk = np.argmin(K_comp)
    alpha_comp = K_comp/K_comp[i_chk]
    return {'fv_flash': fv_flash, 'x_eq': x_eq, 'y_eq': y_eq, 'K_comp': K_comp,
            'alpha_comp':alpha_comp}

def f_sol_ELV_2c_db(Temp, P, lista_componentes, dados):
    ''' Função para o cálculo do ELV em um sistema binário ideal (Lei de Raoult)
      Entrada:
      P - pressão do flash em mmHg
      T - temperatura do flash em K
      lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
      dados - dataframe com os dados do databank
      Saidas: tupla de vetores
      x_eq - concentrações do componentes no ELV na fase líquida
      y_eq - concentrações do componentes no ELV na fase vapor
    '''
    nc = len(lista_componentes)
    P_vap_comp = f_K_Raoult_db(Temp, P, lista_componentes, dados)[1]
    v_rhs = np.array([1,1,0,0])
    A_elv = np.array([[1,1,0,0],
                      [0,0,1,1],
                      [P_vap_comp[0],0,-P,0],
                      [0, P_vap_comp[1], 0, -P]])
    x_sol = np.linalg.inv(A_elv) @ v_rhs
    x_eq = np.empty(nc)
    y_eq = np.empty(nc)
    x_eq[0] = x_sol[0]
    x_eq[1] = x_sol[1]
    y_eq[0] = x_sol[2]
    y_eq[1] = x_sol[3]
    return (x_eq, y_eq)

def f_gerar_dados_elv_2c_bd(P_eq, n_pontos, lista_componentes, dados):
    ''' Função para gerar um pandas.dataframe com n_pontos instâncias de dados
          do ELV de um sistema binário ideal
        Entradas:
        P_eq - pressão de equilíbrio
        n_pontos - número de instâncias geradas
        lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
        dados - dataframe com os dados do databank
        Saida: pandas.dataframe
        dados_elv - com as seguintes series: 'T', 'x1' e 'y1'
    '''
    T_eb_comp = dados[dados['num'].isin(lista_componentes)]['boiling_point']
    T_eb_comp = T_eb_comp.tolist()
    T_faixa = np.linspace(T_eb_comp[0], T_eb_comp[1], n_pontos)
    dados_elv = pd.DataFrame({'T': T_faixa})
    for i, T in enumerate(dados_elv['T']):
        x_eq, y_eq = f_sol_ELV_2c_db(T, P_eq, lista_componentes, dados)
        dados_elv.loc[i,'x1'] = x_eq[0]
        dados_elv.loc[i,'y1'] = y_eq[0]
    return dados_elv
