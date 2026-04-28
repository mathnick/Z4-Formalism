import numpy as np
import matplotlib.pyplot as plt
import math

# =========================================================================
# 1. CONFIGURAÇÃO DAS BASES ESPECTRAIS (MULTIDOMÍNIO - Eq. 39)
# =========================================================================
def configurar_bases_multidominio(L0, N1, N2):
    def gerar_matrizes_dominio(N, x_min, x_max, L0):
        j = np.arange(1, N + 2)
        xi = np.cos(j * np.pi / (N + 2)) 
        
        x_global = x_min + (x_max - x_min) * (xi + 1.0) / 2.0
        r = L0 * (1.0 + x_global) / (1.0 - x_global)
        
        r = np.flip(r)
        xi_flip = np.flip(xi)
        
        psi = np.zeros([N+1, N+1])
        rpsi = np.zeros([N+1, N+1])
        rrpsi = np.zeros([N+1, N+1])
        
        dr_dx = (r + L0)**2 / (2.0 * L0)
        dx_dxi = (x_max - x_min) / 2.0
        dxi_dr = 1.0 / (dr_dx * dx_dxi)
        d2xi_dr2 = - (2.0 / (r + L0)) * dxi_dr

        for i in range(N+1):
            theta = np.arccos(xi_flip)
            T_i = np.cos(i * theta)
            
            if i == 0:
                dT_i = np.zeros_like(xi_flip)
                d2T_i = np.zeros_like(xi_flip)
            else:
                sin_t = np.sin(theta)
                dT_i = i * np.sin(i * theta) / sin_t
                d2T_i = -i**2 * np.cos(i * theta) / (sin_t**2) + \
                         i * np.sin(i * theta) * xi_flip / (sin_t**3)
            
            psi[i, :] = T_i
            rpsi[i, :] = dT_i * dxi_dr
            rrpsi[i, :] = d2T_i * (dxi_dr**2) + dT_i * d2xi_dr2

        inv_psi = np.linalg.inv(psi)
        return {'r': r, 'psi': psi, 'rpsi': rpsi, 'rrpsi': rrpsi, 'inv_psi': inv_psi, 'N': N}

    D1 = gerar_matrizes_dominio(N1, -1.0, 0.0, L0)
    D2 = gerar_matrizes_dominio(N2, 0.0, 1.0, L0)
    return D1, D2

# =========================================================================
# 2. CONDIÇÕES INICIAIS
# =========================================================================
def criar_condicoes_iniciais_fccz4(bases, M=1.0):
    r = bases['r']
    inv_psi = bases['inv_psi']
    
    psi_bh = 1.0 + M / (2.0 * r)
    X_0 = psi_bh**(-4)
    a_0 = np.ones_like(r)
    b_0 = np.ones_like(r)
    alpha_0 = psi_bh**(-2)  
    zeros = np.zeros_like(r)
    
    c_a = np.dot(a_0 - 1.0, inv_psi)
    c_b = np.dot(b_0 - 1.0, inv_psi)
    c_X = np.dot(X_0 - 1.0, inv_psi)
    c_alpha = np.dot(alpha_0 - 1.0, inv_psi)
    c_K = np.dot(zeros, inv_psi)
    c_Aa = np.dot(zeros, inv_psi)
    c_Theta = np.dot(zeros, inv_psi)
    c_Lambda = np.dot(zeros, inv_psi)
    c_beta = np.dot(zeros, inv_psi)
    c_B = np.dot(zeros, inv_psi)
    
    return np.array([c_a, c_b, c_X, c_K, c_Aa, c_Theta, c_Lambda, c_alpha, c_beta, c_B])

# =========================================================================
# 3. EVOLUÇÃO fCCZ4
# =========================================================================
def calcular_taxas_fccz4(state, kappa1, kappa2, eta_param, bases):
    c_a, c_b, c_X, c_K, c_Aa, c_Theta, c_Lambda, c_alpha, c_beta, c_B = state
    psi = bases['psi']; rpsi = bases['rpsi']; rrpsi = bases['rrpsi']; inv_psi = bases['inv_psi']
    r = bases['r']
    
    a = 1.0 + np.dot(c_a, psi); da = np.dot(c_a, rpsi); dda = np.dot(c_a, rrpsi)
    b = 1.0 + np.dot(c_b, psi); db = np.dot(c_b, rpsi); ddb = np.dot(c_b, rrpsi)
    X = 1.0 + np.dot(c_X, psi); dX = np.dot(c_X, rpsi); ddX = np.dot(c_X, rrpsi)
    alpha = 1.0 + np.dot(c_alpha, psi); dalpha = np.dot(c_alpha, rpsi); ddalpha = np.dot(c_alpha, rrpsi)
    beta = np.dot(c_beta, psi); dbeta = np.dot(c_beta, rpsi); ddbeta = np.dot(c_beta, rrpsi)
    B_shift = np.dot(c_B, psi)
    Lambda = np.dot(c_Lambda, psi); dLambda = np.dot(c_Lambda, rpsi)
    K = np.dot(c_K, psi); dK = np.dot(c_K, rpsi)
    Aa = np.dot(c_Aa, psi); dAa = np.dot(c_Aa, rpsi)
    Theta = np.dot(c_Theta, psi); dTheta = np.dot(c_Theta, rpsi)
    
    eps_sq = 1e-24
    X_reg = np.sqrt(X**2 + eps_sq)
    a_reg = np.sqrt(a**2 + eps_sq)
    b_reg = np.sqrt(b**2 + eps_sq)
    Ab = - (b_reg / (2.0 * a_reg)) * Aa  
    
    div_beta = dbeta + beta * (db / b_reg + da / (2.0 * a_reg) + 2.0 / r)
    d_div_beta = ddbeta + dbeta * (db / b_reg + da / (2.0 * a_reg) + 2.0 / r) + \
                 beta * ((ddb * b_reg - db**2) / (b_reg**2) + (dda * a_reg - da**2) / (2.0 * a_reg**2) - 2.0 / r**2)
                 
    bar_Lambda = (1.0 / a_reg) * (da / (2.0 * a_reg) - db / b_reg - (2.0 / r) * (1.0 - a_reg / b_reg))
    Zr = (a_reg / 2.0) * (Lambda - bar_Lambda)
    Zr_up = Zr / a_reg
    c_Zr = np.dot(Zr, inv_psi)
    dZr = np.dot(c_Zr, rpsi)
    dZr_up = (1.0 / a_reg) * dZr - (Zr / a_reg**2) * da
    
    Dm_Zm = dZr_up + Zr_up * (da / (2.0 * a_reg) + db / b_reg + 2.0 / r - 1.5 * dX / X_reg)
    Dr_Zr = dZr_up + Zr_up * (da / (2.0 * a_reg) - 0.5 * dX / X_reg)

    bar_R_rr = - ddb / b_reg + (db**2) / (2.0 * b_reg**2) + (da * db) / (2.0 * a_reg * b_reg) + 2.0 * da / (r * a_reg)
    bar_R_tt = - (r**2 * ddb) / (2.0 * a_reg) - (3.0 * r * db) / (2.0 * a_reg) + (r**2 * da * db) / (4.0 * a_reg**2) + (r * da) / (2.0 * a_reg) + 1.0 - a_reg / b_reg
    R_rr = bar_R_rr + (1.0 / (2.0 * X_reg)) * ddX - (1.0 / (4.0 * X_reg**2)) * dX**2 - (da * dX) / (4.0 * a_reg * X_reg) + (dX / X_reg) * (1.0 / r + db / (2.0 * b_reg))
    R_tt = bar_R_tt + (r**2 * b_reg / (2.0 * a_reg * X_reg)) * ddX - (r**2 * b_reg / (4.0 * a_reg * X_reg**2)) * dX**2 + (r**2 * b_reg * dX) / (a_reg * X_reg) * (1.0 / r + db / (2.0 * b_reg) - da / (4.0 * a_reg))
    Ricci = (X_reg / a_reg) * R_rr + 2.0 * (X_reg / (r**2 * b_reg)) * R_tt

    D2_alpha = (X_reg / a_reg) * ddalpha + (X_reg / a_reg) * dalpha * (2.0 / r + db / b_reg - da / (2.0 * a_reg) + dX / (2.0 * X_reg))
    DrDr_alpha = (X_reg / a_reg) * (ddalpha - dalpha * (da / (2.0 * a_reg) + dX / (2.0 * X_reg)))
    
    dt_a = beta * da + 2.0 * a_reg * dbeta - (2.0 / 3.0) * a_reg * div_beta - 2.0 * alpha * a_reg * Aa
    dt_b = beta * db + 2.0 * b_reg * beta / r - (2.0 / 3.0) * b_reg * div_beta - 2.0 * alpha * b_reg * Ab
    dt_X = beta * dX - (1.0 / 3.0) * X_reg * div_beta + (1.0 / 3.0) * X_reg * alpha * K
    dt_K = - D2_alpha + alpha * (Ricci + 2.0 * Dm_Zm + K**2 - 2.0 * Theta * K) + beta * dK - 3.0 * alpha * kappa1 * (1.0 + kappa2) * Theta
    dt_Theta = beta * dTheta + 0.5 * alpha * (Ricci + 2.0 * Dm_Zm - (Aa**2 + 2.0 * Ab**2) + (2.0 / 3.0) * K**2 - 2.0 * Theta * K) - Zr_up * dalpha - alpha * kappa1 * (2.0 + kappa2) * Theta
    dt_Aa = beta * dAa - (DrDr_alpha - (1.0 / 3.0) * D2_alpha) + alpha * ((X_reg / a_reg) * R_rr - (1.0 / 3.0) * Ricci) + alpha * (2.0 * Dr_Zr - (2.0 / 3.0) * Dm_Zm) + alpha * Aa * (K - 2.0 * Theta)
    
    t1 = beta * dLambda - Lambda * dbeta + (1.0 / a_reg) * ddbeta + (2.0 / b_reg) * (dbeta / r - beta / r**2)
    t2 = (1.0 / 3.0) * ((1.0 / a_reg) * d_div_beta + 2.0 * bar_Lambda * div_beta)
    t3 = - (2.0 / a_reg) * (Aa * dalpha + alpha * dAa)
    t4 = 2.0 * alpha * (Aa * bar_Lambda - (2.0 / (r * b_reg)) * (Aa - Ab))
    t5 = (2.0 * alpha / a_reg) * (dAa - (2.0 / 3.0) * dK - 1.5 * Aa * dX / X_reg + (Aa - Ab) * (2.0 / r + db / b_reg))
    t6 = (2.0 / a_reg) * (alpha * dTheta - Theta * dalpha - (2.0 / 3.0) * alpha * K * Zr)
    t7 = (2.0 / a_reg) * ((2.0 / 3.0) * Zr * div_beta - Zr * dbeta) - (2.0 / a_reg) * kappa1 * Zr
    dt_Lambda = t1 + t2 + t3 + t4 + t5 + t6 + t7                                                 
    dt_alpha = - (alpha**2 + (2.0 / 3.0)) * (K - 2.0 * Theta)
    dt_beta = B_shift
    dt_B = 0.75 * dt_Lambda - eta_param * B_shift

    return np.array([np.dot(dt_a, inv_psi), np.dot(dt_b, inv_psi), np.dot(dt_X, inv_psi),
                     np.dot(dt_K, inv_psi), np.dot(dt_Aa, inv_psi), np.dot(dt_Theta, inv_psi),
                     np.dot(dt_Lambda, inv_psi), np.dot(dt_alpha, inv_psi), 
                     np.dot(dt_beta, inv_psi), np.dot(dt_B, inv_psi)])

# =========================================================================
# 4. CONDIÇÕES DE TRANSMISSÃO E FILTRO
# =========================================================================
def aplicar_condicoes_transmissao(rhs_D1, rhs_D2, N1, N2):
    k1 = np.arange(N1 + 1); k2 = np.arange(N2 + 1)
    T1 = np.ones(N1 + 1); T2 = (-1)**k2                     
    dT1 = k1**2; dT2 = (((-1)**(k2 + 1)) * (k2**2))

    rhs_D1_new = np.copy(rhs_D1)
    rhs_D2_new = np.copy(rhs_D2)
    denom = float(N1**2 + N2**2)

    for var in range(len(rhs_D1)):
        V1 = np.dot(rhs_D1[var], T1); V2 = np.dot(rhs_D2[var], T2)
        D1 = np.dot(rhs_D1[var], dT1); D2 = np.dot(rhs_D2[var], dT2)
        delta_V = V2 - V1
        delta_D = D2 - D1
        x = (delta_D + (N2**2) * delta_V) / denom
        y = (delta_D - (N1**2) * delta_V) / denom
        rhs_D1_new[var][-1] += x
        rhs_D2_new[var][-1] += y * ((-1)**N2)

    return rhs_D1_new, rhs_D2_new                             

def aplicar_filtro_espectral(state, N, p): 
    k = np.arange(N + 1)
    sigma = np.exp(-36.0 * (k / N)**(2 * p))
    state_filtrado = np.copy(state)
    for i in [3, 4, 5, 6]:
        state_filtrado[i] = state[i] * sigma
    return state_filtrado                                           

# =========================================================================
# 5. INTEGRADOR RK4 MULTIDOMÍNIO
# =========================================================================
def passo_rk4_multidominio(state_D1, state_D2, h, kappa1, kappa2, eta, bases_D1, bases_D2):
    N1 = bases_D1['N']; N2 = bases_D2['N']
    k1_1 = calcular_taxas_fccz4(state_D1, kappa1, kappa2, eta, bases_D1)
    k1_2 = calcular_taxas_fccz4(state_D2, kappa1, kappa2, eta, bases_D2)
    k1_1, k1_2 = aplicar_condicoes_transmissao(k1_1, k1_2, N1, N2)
    
    s2_1 = state_D1 + 0.5 * h * k1_1; s2_2 = state_D2 + 0.5 * h * k1_2
    k2_1 = calcular_taxas_fccz4(s2_1, kappa1, kappa2, eta, bases_D1)
    k2_2 = calcular_taxas_fccz4(s2_2, kappa1, kappa2, eta, bases_D2)
    k2_1, k2_2 = aplicar_condicoes_transmissao(k2_1, k2_2, N1, N2)                                
    
    s3_1 = state_D1 + 0.5 * h * k2_1; s3_2 = state_D2 + 0.5 * h * k2_2
    k3_1 = calcular_taxas_fccz4(s3_1, kappa1, kappa2, eta, bases_D1)
    k3_2 = calcular_taxas_fccz4(s3_2, kappa1, kappa2, eta, bases_D2)
    k3_1, k3_2 = aplicar_condicoes_transmissao(k3_1, k3_2, N1, N2)
    
    s4_1 = state_D1 + h * k3_1; s4_2 = state_D2 + h * k3_2
    k4_1 = calcular_taxas_fccz4(s4_1, kappa1, kappa2, eta, bases_D1)
    k4_2 = calcular_taxas_fccz4(s4_2, kappa1, kappa2, eta, bases_D2)
    k4_1, k4_2 = aplicar_condicoes_transmissao(k4_1, k4_2, N1, N2)
    
    new_D1 = state_D1 + (h / 6.0) * (k1_1 + 2.0*k2_1 + 2.0*k3_1 + k4_1)
    new_D2 = state_D2 + (h / 6.0) * (k1_2 + 2.0*k2_2 + 2.0*k3_2 + k4_2)
    return new_D1, new_D2                         

import time
import matplotlib.pyplot as plt
import numpy as np

# =========================================================================
# 6. FUNÇÃO DE VARREDURA (INSTÂNCIA ÚNICA)
# =========================================================================
def simular_instancia(kappa1, eta, p_filtro):
    L0, N, tf, h = 5.0, 110, 10.0, 0.00004 # tf reduzido para 10M para a varredura não durar dias
    b1, b2 = configurar_bases_multidominio(L0, N, N)
    s1, s2 = criar_condicoes_iniciais_fccz4(b1) , criar_condicoes_iniciais_fccz4(b2)
    
    t_atual = 0.0
    for step in range(int(tf/h)):
        # Condição de parada (Crash por NaN ou Overflow detectado precocemente)
        if np.isnan(s1).any() or np.max(np.abs(s1)) > 1e15:
            return t_atual 
        
        def calc(ss1, ss2):
            r1 = calcular_taxas_fccz4(ss1, kappa1, 0.0, eta, b1) # kappa2 = 0.0 fixo
            r2 = calcular_taxas_fccz4(ss2, kappa1, 0.0, eta, b2)
            return aplicar_condicoes_transmissao(r1, r2, N, N)
        
        # RK4
        k11, k12 = calc(s1, s2)
        k21, k22 = calc(s1+0.5*h*k11, s2+0.5*h*k12)
        k31, k32 = calc(s1+0.5*h*k21, s2+0.5*h*k22)
        k41, k42 = calc(s1+h*k31, s2+h*k32)
        
        s1 += (h/6.0)*(k11 + 2*k21 + 2*k31 + k41)
        s2 += (h/6.0)*(k12 + 2*k22 + 2*k32 + k42)
        
        # Aplica o filtro campeão
        s1 = aplicar_filtro_espectral(s1, N, p_filtro)
        s2 = aplicar_filtro_espectral(s2, N, p_filtro)
        
        t_atual += h
        
    return tf # Sobreviveu até o fim!

# =========================================================================
# 7. MOTOR DO HEATMAP
# =========================================================================
p_ideal = 4.0 # O vencedor do seu teste anterior

# Grade de busca (Podemos refinar depois se acharmos uma região promissora)
lista_kappa = [1.0, 0.1]
lista_eta = [5.0, 5.5, 6.0, 6.5, 7.0]

resultados = np.zeros((len(lista_kappa), len(lista_eta)))

print("==================================================")
print(f" INICIANDO MAPA DE CALOR (Filtro p={p_ideal})")
print("==================================================")

start_time = time.time()

for i, k in enumerate(lista_kappa):
    for j, e in enumerate(lista_eta):
        print(f"Testando [kappa={k:.2f}, eta={e:.2f}]...", end=" ")
        
        # O try/except ignora os avisos de overflow para não poluir o terminal
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tempo_sobrevivencia = simular_instancia(k, e, p_ideal)
            
        resultados[i, j] = tempo_sobrevivencia
        if tempo_sobrevivencia >= 9.9:
            print(f"ESTÁVEL! (Passou de {tempo_sobrevivencia:.2f}M)")
        else:
            print(f"Crash em {tempo_sobrevivencia:.2f}M")

print(f"\nVarredura concluída em {(time.time() - start_time)/60:.2f} minutos.")

# Plotando o Mapa de Calor
plt.figure(figsize=(8, 6))
plt.imshow(resultados, extent=[min(lista_eta), max(lista_eta), min(lista_kappa), max(lista_kappa)], 
           origin='lower', aspect='auto', cmap='plasma')

# Anotações no gráfico
for i in range(len(lista_kappa)):
    for j in range(len(lista_eta)):
        plt.text(lista_eta[j], lista_kappa[i], f'{resultados[i, j]:.1f}', 
                 ha='center', va='center', color='black' if resultados[i,j] > 5.0 else 'white')

plt.colorbar(label="Tempo de Sobrevivência (M)")
plt.xlabel(r"Parâmetro de Gauge ($\eta$)", fontsize=12)
plt.ylabel(r"Amortecimento Z4 ($\kappa_1$)", fontsize=12)
plt.title(f"Estabilidade do fCCZ4 (Filtro $p={p_ideal}$)", fontsize=14)
plt.grid(False)
plt.show()
