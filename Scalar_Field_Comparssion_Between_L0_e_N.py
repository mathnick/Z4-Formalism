import numpy as np
import matplotlib.pyplot as plt
import math

# =========================================================================
# 0. FUNÇÃO PARA CONFIGURAR AS BASES ESPECTRAIS PARA DIFERENTES L0 E N
# =========================================================================

V=0

def configurar_bases_espectrais(L0_valor, N_valor):
    """
    Configura as bases espectrais para valores específicos de L0 e N
    """
    col = np.cos(np.arange(2*N_valor + 4)*math.pi /(2*N_valor + 3))  # collocation points
    colr = col[1:N_valor+2]

    r1 = L0_valor * colr/(np.sqrt(1-colr**2))       # physical domain
    r = np.flip(r1)

    # Base Matrix (Tchebyshev Polinomials):
    SB = np.zeros([N_valor+3,N_valor+1])
    rSB = np.zeros([N_valor+3,N_valor+1])
    rrSB = np.zeros([N_valor+3,N_valor+1])

    for i in range(N_valor+1+1+1):
        SB[i,] = np.sin((2*i+1)*np.arctan(L0_valor/r))

    for i in range(N_valor+1+1+1):
        rSB[i,] = -np.cos((2*i+1)*np.arctan(L0_valor/r))*(2*i+1)*L0_valor/(r**2*(1+L0_valor**2/r**2))

    for i in range(N_valor+1+1+1):
        rrSB[i,] = -np.sin((2*i+1)*np.arctan(L0_valor/r))*(2*i+1)**2*L0_valor**2/(r**4*(1+L0_valor**2/r**2)**2)+2*np.cos((2*i+1)*np.arctan(L0_valor/r))*(2*i+1)*L0_valor/(r**3*(1+L0_valor**2/r**2))-2*np.cos((2*i+1)*np.arctan(L0_valor/r))*(2*i+1)*L0_valor**3/(r**5*(1+L0_valor**2/r**2)**2)

    psi = SB[0:N_valor+1,:]       # Base function
    rpsi = rSB[0:N_valor+1,:]
    rrpsi = rrSB[0:N_valor+1,:]

    inv_psi = np.linalg.inv(psi)

    # Bases para Krr
    SB1 = 1/2*(SB[1:(N_valor+2),:] + SB[0:(N_valor+1),:])
    rSB1 = 1/2*(rSB[1:(N_valor+2),:] + rSB[0:(N_valor+1),:])

    # Bases para Beta e Z
    SB2 = np.zeros([N_valor+1,N_valor+1])
    rSB2 = np.zeros([N_valor+1,N_valor+1])
    rrSB2 = np.zeros([N_valor+1,N_valor+1])

    for i in range(N_valor+1):
        SB2[i,] = np.sin((2*(i+1/2)+1)*np.arctan(L0_valor/r))

    for i in range(N_valor+1):
        rSB2[i,] = -np.cos((2*i+2)*np.arctan(L0_valor/r))*(2*i+2)*L0_valor/(r**2*(1+L0_valor**2/r**2))

    for i in range(N_valor+1):
        rrSB2[i,] = -np.sin((2*i+2)*np.arctan(L0_valor/r))*(2*i+2)**2*L0_valor**2/(r**4*(1+L0_valor**2/r**2)**2)+2*np.cos((2*i+2)*np.arctan(L0_valor/r))*(2*i+2)*L0_valor/(r**3*(1+L0_valor**2/r**2))-2*np.cos((2*i+2)*np.arctan(L0_valor/r))*(2*i+2)*L0_valor**3/(r**5*(1+L0_valor**2/r**2)**2)

    inv_SB2 = np.linalg.pinv(SB2)

    # Quadrature setup (para restrições)
    Nq = int(3/2*N_valor)
    gauss_quadrature = np.polynomial.legendre.leggauss(Nq + 1)
    new_col = gauss_quadrature[0]
    wq_col = gauss_quadrature[1] # Pesos
    rq = L0_valor*(1+new_col)/(1-new_col) # Physical quadrature domain

    # Bases para Quadrature
    qSB = np.zeros([Nq+3,Nq+1])
    qrSB = np.zeros([Nq+3,Nq+1])
    qrrSB = np.zeros([Nq+3,Nq+1])

    for i in range(Nq+1+1+1):
        qSB[i,] = np.sin((2*i+1)*np.arctan(L0_valor/rq))

    for i in range(Nq+1+1+1):
        qrSB[i,] = -np.cos((2*i+1)*np.arctan(L0_valor/rq))*(2*i+1)*L0_valor/(rq**2*(1+L0_valor**2/rq**2))

    for i in range(Nq+1+1+1):
        qrrSB[i,] = -np.sin((2*i+1)*np.arctan(L0_valor/rq))*(2*i+1)**2*L0_valor**2/(rq**4*(1+L0_valor**2/rq**2)**2)+2*np.cos((2*i+1)*np.arctan(L0_valor/rq))*(2*i+1)*L0_valor/(rq**3*(1+L0_valor**2/rq**2))-2*np.cos((2*i+1)*np.arctan(L0_valor/rq))*(2*i+1)*L0_valor**3/(rq**5*(1+L0_valor**2/rq**2)**2)

    qpsi = qSB[0:N_valor+1,:]
    rqpsi = qrSB[0:N_valor+1,:]
    rrqpsi = qrrSB[0:N_valor+1,:]

    qSB1 = 1/2*(qSB[1:(N_valor+2),:] + qSB[0:(N_valor+1),:])
    rqSB1 = 1/2*(qrSB[1:(N_valor+2),:] + qrSB[0:(N_valor+1),:])

    qSB2 = np.zeros([N_valor+1, Nq+1])
    rqSB2 = np.zeros([N_valor+1, Nq+1])

    for i in range(N_valor+1):
        qSB2[i,] = np.sin((2*(i+1/2)+1)*np.arctan(L0_valor/rq))

    for i in range(N_valor+1):
        rqSB2[i,] = -np.cos((2*i+2)*np.arctan(L0_valor/rq))*(2*i+2)*L0_valor/(rq**2*(1+L0_valor**2/rq**2))

    # Alpha na origem
    psi_0 = np.zeros(N_valor+1)
    for i in range(N_valor+1):
        psi_0[i,] = np.sin((2*i+1)*math.pi/2)

    return {
        'r': r, 'psi': psi, 'rpsi': rpsi, 'rrpsi': rrpsi, 'inv_psi': inv_psi,
        'SB1': SB1, 'rSB1': rSB1, 'SB2': SB2, 'rSB2': rSB2, 'rrSB2': rrSB2, 'inv_SB2': inv_SB2,
        'rq': rq, 'wq_col': wq_col, 'qpsi': qpsi, 'rqpsi': rqpsi, 'rrqpsi': rrqpsi,
        'qSB1': qSB1, 'rqSB1': rqSB1, 'qSB2': qSB2, 'rqSB2': rqSB2, 'psi_0': psi_0,
        'L0': L0_valor, 'N': N_valor
    }

# =========================================================================
# 1. FUNÇÃO PARA CRIAR CONDIÇÕES INICIAIS
# =========================================================================

def criar_condicoes_iniciais(A0_valor, bases):
    """
    Cria condições iniciais para um valor específico de A0 usando as bases fornecidas
    """
    r = bases['r']
    psi = bases['psi']
    rpsi = bases['rpsi']
    rrpsi = bases['rrpsi']
    inv_psi = bases['inv_psi']
    N_valor = bases['N']
    
    r0 = 2
    sigma = 1
    V = 0 # Potencial

    # Phi e Pi
    Phi_0 = A0_valor*r**2*(np.exp(-(r-r0)**2/sigma**2)+np.exp(-(r+r0)**2/sigma**2))
    a0 = np.dot(Phi_0, inv_psi)
    Pi_0 = np.zeros(N_valor+1)
    b0 = np.dot(Pi_0, inv_psi) 

    # Chi (Solução da Restrição Hamiltoniana) - Newton Raphson
    c0 = 0.001 * np.ones([N_valor+1]) # Chute inicial
    Phi = np.dot(a0, psi)
    rPhi= np.dot(a0, rpsi)

    tol = 1e-18
    n = 0
    nf = 50

    while n <= nf:
        Chi=np.dot(c0,psi)
        rChi=np.dot(c0,rpsi)
        rrChi=np.dot(c0,rrpsi)
        H0 = 4*rChi**2 + 4*rrChi + 8/r*rChi + 1/2*(rPhi)**2
        JH = 8*np.dot(c0,rpsi)*rpsi + 4*rrpsi + 8/r*rpsi
        inv_JH = np.linalg.inv(JH)
        cnew = c0 - np.dot(H0, inv_JH)
        if min(abs(cnew-c0)) < tol:
            break
        c0 = cnew
        n = n + 1

    # Theta e Z (Z4)
    theta0 = np.zeros(N_valor+1)
    z0 = np.zeros(N_valor+1)

    return c0, a0, b0, theta0, z0

# =========================================================================
# 2. DEFINIÇÕES DAS FUNÇÕES DE EVOLUÇÃO
# =========================================================================

def calcular_taxas_z4(c_coefs, a_coefs, b_coefs, theta_coefs, z_coefs, t, kappa1, kappa2, bases):
    """
    Calcula as taxas de variação (rhs) para o estado atual.
    """
    # Recuperar bases
    psi = bases['psi']; rpsi = bases['rpsi']; rrpsi = bases['rrpsi']; inv_psi = bases['inv_psi']
    SB1 = bases['SB1']; rSB1 = bases['rSB1']; SB2 = bases['SB2']; rSB2 = bases['rSB2']; inv_SB2 = bases['inv_SB2']
    psi_0 = bases['psi_0']; r = bases['r']
    
    # ---------------------------------------------
    # 1. Recuperar Campos dos Coeficientes
    # ---------------------------------------------
    Chi = np.dot(c_coefs, psi)
    rChi = np.dot(c_coefs, rpsi)
    rrChi = np.dot(c_coefs, rrpsi)
    Phi = np.dot(a_coefs, psi)
    rPhi = np.dot(a_coefs, rpsi)
    rrPhi = np.dot(a_coefs, rrpsi)
    Pi = np.dot(b_coefs, psi)
    rPi = np.dot(b_coefs, rpsi) 
    Theta = np.dot(theta_coefs, psi)
    rTheta = np.dot(theta_coefs, rpsi)
    Z = np.dot(z_coefs, SB2)
    rZ = np.dot(z_coefs, rSB2)

    # ---------------------------------------------
    # 2. Solução Elíptica para Krr (Restrição de Momento)
    # ---------------------------------------------
    Matrix_Krr = 2 * rChi * SB1 + rSB1 + 3 / r * SB1
    inv_matrix_krr = np.linalg.inv(Matrix_Krr)
    rhsk = - Pi * rPhi * np.exp(4 * Chi)
    ck0 = np.dot(rhsk, inv_matrix_krr)
    Krr = np.dot(ck0, SB1)
    rKrr = np.dot(ck0, rSB1)

    # ---------------------------------------------
    # 3. Solução Elíptica para Alpha (Lapse) - Z4
    # ---------------------------------------------
    Matrix_Alpha = (rrpsi + 2*(1/r + rChi)*rpsi -
                    (3/2)*np.exp(-4*Chi)*Krr**2*psi -
                    np.exp(4*Chi)*(Pi**2 - V)*psi -
                    2*np.exp(4*Chi)*rZ*psi -
                    2*np.exp(4*Chi)*(6*rChi + 2/r)*Z*psi)
    
    inv_matrix_alpha = np.linalg.inv(Matrix_Alpha)
    
    rhsal = ((3/2)*np.exp(-4*Chi)*Krr**2 +
             np.exp(4*Chi)*(Pi**2 - V) +
             2*np.exp(4*Chi)*rZ +
             2*np.exp(4*Chi)*(6*rChi + 2/r)*Z)
             
    al0 = np.dot(rhsal, inv_matrix_alpha)
    Alpha = 1 + np.dot(al0, psi)
    rAlpha = np.dot(al0, rpsi)
    
    # Lapse no centro (r=0)
    Alpha_central = 1 + np.dot(al0, psi_0)

    # ---------------------------------------------
    # 4. Solução Elíptica para Beta (Shift)
    # ---------------------------------------------
    Matrix_Beta = rSB2/r - SB2/r**2
    inv_matrix_beta = np.linalg.inv(Matrix_Beta)
    rhsbe = 3/2 * Alpha * np.exp(-4*Chi) * Krr / r
    be0 = np.dot(rhsbe, inv_matrix_beta)
    Beta = np.dot(be0, SB2)
    rBeta = np.dot(be0, rSB2)
    
    # ---------------------------------------------
    # 5. Calcular as Taxas de Evolução (d/dt)
    # ---------------------------------------------
    
    # Taxa para Chi (dc/dt) - ADM
    dChi_dt = np.dot(Beta * rChi + Beta / (2*r) + Alpha / 4 * np.exp(-4*Chi) * Krr, inv_psi)
    
    # Taxa para Phi (da/dt) - ADM
    dPhi_dt = np.dot(Alpha * Pi + Beta * rPhi, inv_psi)
    
    # Taxa para Pi (db/dt) - ADM
    dPi_dt_expr = (Beta*rPi + 
                   np.exp(-4*Chi)*(2*Alpha/r + rAlpha + 2*rChi*Alpha)*rPhi + 
                   np.exp(-4*Chi)*Alpha*rrPhi - Alpha*V)
    dPi_dt = np.dot(dPi_dt_expr, inv_psi)
    
    # Termo R (Ricci Escalar)
    R = -8*np.exp(-4*Chi)*(rrChi + rChi**2 + 2*rChi/r)

    # Taxa para Theta (dtheta/dt) - Z4
    dTheta_dt_expr = (Beta*rTheta + 0.5*Alpha*(R - 1.5*np.exp(-8*Chi)*Krr**2 - Pi**2 - np.exp(-4*Chi)*rPhi**2)
                      - Z*rAlpha + Alpha*(rZ + (6*rChi + 2/r)*Z) - 0.5*Alpha*kappa1*(kappa2 + 2)*Theta)
    dtheta_dt = np.dot(dTheta_dt_expr, inv_psi)

    # Taxa para Z (dz/dt) - Z4
    dZ_dt_expr = (Beta*rZ - Z*rBeta +
                  Alpha*np.exp(-4*Chi)*(rKrr + 3*Krr/r + 2*rChi*Krr + np.exp(4*Chi)*Pi*rPhi) +
                  np.exp(-4*Chi)*(Alpha*rTheta - Theta*rAlpha) - kappa1*Alpha*Z)
    dz_dt = np.dot(dZ_dt_expr, inv_SB2)
    
    # Retorna as 5 taxas de variação
    return dChi_dt, dPhi_dt, dPi_dt, dtheta_dt, dz_dt, Alpha_central, ck0

def passo_rk4_z4(c0, a0, b0, theta0, z0, t, h, kappa1, kappa2, filter1, bases):
    """
    Executa um único passo de Runge-Kutta de 4ª ordem (RK4)
    """
    # Estágio 1 (K1)
    K1_rc, K1_ra, K1_rb, K1_rtheta, K1_rz, _, _ = calcular_taxas_z4(c0, a0, b0, theta0, z0, t, kappa1, kappa2, bases)
    K1 = h * K1_rc; L1 = h * K1_ra; N1 = h * K1_rb; dtheta1 = h * K1_rtheta; dz1 = h * K1_rz

    # Estágio 2 (K2)
    c_temp2 = c0 + K1/2; a_temp2 = a0 + L1/2; b_temp2 = b0 + N1/2
    theta_temp2 = theta0 + dtheta1/2; z_temp2 = z0 + dz1/2
    K2_rc, K2_ra, K2_rb, K2_rtheta, K2_rz, _, _ = calcular_taxas_z4(c_temp2, a_temp2, b_temp2, theta_temp2, z_temp2, t + h/2, kappa1, kappa2, bases)
    K2 = h * K2_rc; L2 = h * K2_ra; N2 = h * K2_rb; dtheta2 = h * K2_rtheta; dz2 = h * K2_rz

    # Estágio 3 (K3)
    c_temp3 = c0 + K2/2; a_temp3 = a0 + L2/2; b_temp3 = b0 + N2/2
    theta_temp3 = theta0 + dtheta2/2; z_temp3 = z0 + dz2/2 
    K3_rc, K3_ra, K3_rb, K3_rtheta, K3_rz, _, _ = calcular_taxas_z4(c_temp3, a_temp3, b_temp3, theta_temp3, z_temp3, t + h/2, kappa1, kappa2, bases)
    K3 = h * K3_rc; L3 = h * K3_ra; N3 = h * K3_rb; dtheta3 = h * K3_rtheta; dz3 = h * K3_rz

    # Estágio 4 (K4)
    c_temp4 = c0 + K3; a_temp4 = a0 + L3; b_temp4 = b0 + N3
    theta_temp4 = theta0 + dtheta3; z_temp4 = z0 + dz3
    K4_rc, K4_ra, K4_rb, K4_rtheta, K4_rz, _, _ = calcular_taxas_z4(c_temp4, a_temp4, b_temp4, theta_temp4, z_temp4, t + h, kappa1, kappa2, bases)
    K4 = h * K4_rc; L4 = h * K4_ra; N4 = h * K4_rb; dtheta4 = h * K4_rtheta; dz4 = h * K4_rz

    # Atualização Final
    c_novo = filter1 * (c0 + (K1 + 2*K2 + 2*K3 + K4)/6)
    a_novo = filter1 * (a0 + (L1 + 2*L2 + 2*L3 + L4)/6)
    b_novo = filter1 * (b0 + (N1 + 2*N2 + 2*N3 + N4)/6)
    theta_novo = filter1 * (theta0 + (dtheta1 + 2*dtheta2 + 2*dtheta3 + dtheta4)/6)
    z_novo = filter1 * (z0 + (dz1 + 2*dz2 + 2*dz3 + dz4)/6)
    
    return c_novo, a_novo, b_novo, theta_novo, z_novo

def executar_simulacao_L0N(c0, a0, b0, theta0, z0, t_init, h, It, kappa1, kappa2, bases):
    """
    Executa a simulação completa para configurações específicas de L0 e N
    """
    # Recuperar bases para cálculo das restrições
    qpsi = bases['qpsi']; rqpsi = bases['rqpsi']; rrqpsi = bases['rrqpsi']
    qSB1 = bases['qSB1']; rqSB1 = bases['rqSB1']; wq_col = bases['wq_col']; rq = bases['rq']
    N_valor = bases['N']
    
    # O filtro (filter1) é mantido como np.ones(N+1)
    filter1 = np.ones(N_valor+1)
    
    # Listas para armazenar os dados de monitoramento
    Time_data = [] 
    Alpha_data = []
    L2HC_data = [] # L2 Hamiltonian Constraint
    L2MC_data = [] # L2 Momentum Constraint

    t_atual = t_init
    c_atual = c0.copy()
    a_atual = a0.copy()
    b_atual = b0.copy()
    theta_atual = theta0.copy()
    z_atual = z0.copy()
    
    for i in range(It):

        # --- 1. MONITORAMENTO (ESTADO ATUAL) ---
        _, _, _, _, _, Alpha_central, ck0 = calcular_taxas_z4(c_atual, a_atual, b_atual, theta_atual, z_atual, t_atual, kappa1, kappa2, bases)
        
        # Armazenar dados
        Time_data.append(t_atual) 
        Alpha_data.append(Alpha_central) 
        
        # Restrição Hamiltoniana
        qPhi = np.dot(a_atual, qpsi)
        rqPhi = np.dot(a_atual, rqpsi)
        qPi = np.dot(b_atual, qpsi)
        qChi = np.dot(c_atual, qpsi)
        rqChi = np.dot(c_atual, rqpsi)
        rrqChi = np.dot(c_atual, rrqpsi)
        qKrr = np.dot(ck0, qSB1)
        rqKrr = np.dot(ck0, rqSB1)

        H = 4*rqChi**2 + 4*rrqChi + 8*rqChi/rq + 3/4*np.exp(-4*qChi)*qKrr**2 + np.exp(4*qChi)*(1/2*qPi**2 + np.exp(-4*qChi)/2*rqPhi**2)
        L2HC_data.append(np.sqrt(1/2*np.dot(H**2, wq_col)))

        M = 2*rqChi*qKrr + rqKrr + 3/rq*qKrr + qPi*rqPhi*np.exp(4*qChi)
        L2MC_data.append((1/2*np.dot(M**2,wq_col))**1/2)

        # --- 2. EXECUÇÃO DO PASSO RK4 ---
        c_novo, a_novo, b_novo, theta_novo, z_novo = passo_rk4_z4(
            c_atual, a_atual, b_atual, theta_atual, z_atual, t_atual, h, kappa1, kappa2, filter1, bases
        )
        
        # --- 3. ATUALIZAÇÃO DO ESTADO ---
        t_atual += h
        c_atual = c_novo
        a_atual = a_novo
        b_atual = b_novo
        theta_atual = theta_novo
        z_atual = z_novo
        
        # Condição de parada (se Alpha atingir zero)
        if Alpha_central < 1e-6:
             print(f"Colapso detectado em t = {t_atual:.4f}. Parando simulação.")
             break

    return Time_data, Alpha_data, L2HC_data, L2MC_data

# =========================================================================
# 3. EXECUÇÃO DA SIMULAÇÃO COM DIFERENTES PARES (L0, N)
# =========================================================================

# 1. Definições dos Cenários de L0 e N
cenarios_L0N = {
    # Etiqueta: (L0_valor, N_valor, estilo_linha)
    r"$L_0=5, N=50$": (5, 50, 'b-'),        # Azul, linha cheia
    r"$L_0=5, N=100$": (5, 100, 'r--'),     # Vermelho, linha tracejada  
    r"$L_0=10, N=50$": (10, 50, 'g:'),      # Verde, linha pontilhada
    r"$L_0=10, N=100$": (10, 100, 'm-.')    # Magenta, linha traço-ponto
}

# 2. Parâmetros FIXOS
A0_FIXO = 0.05
KAPPA1_FIXO = 10
KAPPA2_FIXO = -0.5

# 3. Inicialização dos parâmetros de simulação
h = 0.0002
tf = 15
It = int(tf/h)
t = 0.0 

resultados_alpha = {}
resultados_hc = {}
resultados_mc = {}

print(f"Iniciando múltiplas simulações com A0={A0_FIXO}, kappa1={KAPPA1_FIXO}, kappa2={KAPPA2_FIXO}")
print(f"Iterações por rodada: {It}")

# 4. Loop de Execução para diferentes pares (L0, N)
for label, (L0_valor, N_valor, linestyle) in cenarios_L0N.items():
    print(f"\nConfigurando bases para {label}...")
    
    # Configurar bases espectrais para este par (L0, N)
    bases = configurar_bases_espectrais(L0_valor, N_valor)
    
    print(f"Criando condições iniciais para {label}...")
    # Criar condições iniciais específicas para estas bases
    c0_temp, a0_temp, b0_temp, theta0_temp, z0_temp = criar_condicoes_iniciais(A0_FIXO, bases)
    
    print(f"Executando simulação para {label}...")
    Time_data, Alpha_data, L2HC_data, L2MC_data = executar_simulacao_L0N(
        c0_temp, a0_temp, b0_temp, theta0_temp, z0_temp, t, h, It, 
        KAPPA1_FIXO, KAPPA2_FIXO, bases
    )
    
    # Armazena os resultados no dicionário
    resultados_alpha[label] = {'Time': np.array(Time_data), 'Alpha': np.array(Alpha_data), 'Style': linestyle}
    resultados_hc[label] = {'Time': np.array(Time_data), 'L2HC': np.array(L2HC_data), 'Style': linestyle}
    resultados_mc[label] = {'Time': np.array(Time_data), 'L2MC': np.array(L2MC_data), 'Style': linestyle}

print("\nTodas as simulações concluídas.")

# =========================================================================
# 4. PLOTAGEM FINAL - COMPARAÇÃO ENTRE DIFERENTES (L0, N)
# =========================================================================

# 4.1 PLOTAGEM FINAL (Lapse Central)
plt.figure(figsize=(12, 7)) 
for label, data in resultados_alpha.items():
    plt.plot(data['Time'], data['Alpha'], data['Style'], label=label, linewidth=2)

plt.title(f"Evolução do Lapse Central $\\alpha(t,0)$ para diferentes $(L_0, N)$\n" + 
          f"($A_0={A0_FIXO}$, $\\kappa_1={KAPPA1_FIXO}$, $\\kappa_2={KAPPA2_FIXO}$)", fontsize=14)
plt.ylabel(r"$\alpha(t,0)$", fontsize=12)
plt.xlabel("t", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title="Configurações $(L_0, N)$", fontsize=10)
plt.show() 

# 4.2 PLOTAGEM FINAL (Restrição Hamiltoniana)
plt.figure(figsize=(12, 7)) 
for label, data in resultados_hc.items():
    plt.plot(data['Time'], data['L2HC'], data['Style'], label=label, linewidth=2)

plt.title(f"Violação da Restrição Hamiltoniana $L_2$ para diferentes $(L_0, N)$\n" + 
          f"($A_0={A0_FIXO}$, $\\kappa_1={KAPPA1_FIXO}$, $\\kappa_2={KAPPA2_FIXO}$)", fontsize=14)
plt.ylabel(r"$\log_{10}(\|\mathcal{H}\|)$", fontsize=12)
plt.xlabel("t", fontsize=12)
plt.yscale('log')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title="Configurações $(L_0, N)$", fontsize=10)
plt.show()

# 4.3 PLOTAGEM FINAL (Restrição de Momento)
plt.figure(figsize=(12, 7)) 
for label, data in resultados_mc.items():
    plt.plot(data['Time'], data['L2MC'], data['Style'], label=label, linewidth=2)

plt.title(f"Violação da Restrição de Momento $L_2$ para diferentes $(L_0, N)$\n" + 
          f"($A_0={A0_FIXO}$, $\\kappa_1={KAPPA1_FIXO}$, $\\kappa_2={KAPPA2_FIXO}$)", fontsize=14)
plt.ylabel(r"$\log_{10}(\|\mathcal{M}\|)$", fontsize=12)
plt.xlabel("t", fontsize=12)
plt.yscale('log')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title="Configurações $(L_0, N)$", fontsize=10)
plt.show()
