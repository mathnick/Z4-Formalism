import numpy as np
import time
import math

# =========================================================================
# 1. ARQUITETURA DE DOMÍNIO E BASES DE PARIDADE 
# =========================================================================
def configurar_bases_fccz4(L0, N):
    # A sacada de mestre: ignoramos o índice 0 para nunca tocar no r=0 exato
    col = np.cos(np.arange(2*N + 4) * math.pi / (2*N + 3))
    colr = col[1:N+2] 
    r1 = L0 * colr / (np.sqrt(1 - colr**2))
    r = np.flip(r1)

    # Matrizes para Variáveis PARES (alpha, chi, a, b, K, Aa, Theta, B)
    SB = np.zeros([N+1, N+1])
    rSB = np.zeros([N+1, N+1])
    rrSB = np.zeros([N+1, N+1])

    # Matrizes para Variáveis ÍMPARES (beta, Lambda)
    SB2 = np.zeros([N+1, N+1])
    rSB2 = np.zeros([N+1, N+1])
    rrSB2 = np.zeros([N+1, N+1])

    for i in range(N+1):
        # BASE PAR
        SB[i, :] = np.sin((2*i + 1) * np.arctan(L0/r))
        rSB[i, :] = -np.cos((2*i + 1) * np.arctan(L0/r)) * (2*i + 1) * L0 / (r**2 * (1 + L0**2 / r**2))
        rrSB[i, :] = -np.sin((2*i + 1) * np.arctan(L0/r)) * (2*i + 1)**2 * L0**2 / (r**4 * (1 + L0**2 / r**2)**2) \
                     + 2 * np.cos((2*i + 1) * np.arctan(L0/r)) * (2*i + 1) * L0 / (r**3 * (1 + L0**2 / r**2)) \
                     - 2 * np.cos((2*i + 1) * np.arctan(L0/r)) * (2*i + 1) * L0**3 / (r**5 * (1 + L0**2 / r**2)**2)

        # BASE ÍMPAR: 
        SB2[i, :] = np.sin((2*i + 2) * np.arctan(L0/r))
        rSB2[i, :] = -np.cos((2*i + 2) * np.arctan(L0/r)) * (2*i + 2) * L0 / (r**2 * (1 + L0**2 / r**2))
        rrSB2[i, :] = -np.sin((2*i + 2) * np.arctan(L0/r)) * (2*i + 2)**2 * L0**2 / (r**4 * (1 + L0**2 / r**2)**2) \
                      + 2 * np.cos((2*i + 2) * np.arctan(L0/r)) * (2*i + 2) * L0 / (r**3 * (1 + L0**2 / r**2)) \
                      - 2 * np.cos((2*i + 2) * np.arctan(L0/r)) * (2*i + 2) * L0**3 / (r**5 * (1 + L0**2 / r**2)**2)

    # Inversas
    inv_psi = np.linalg.pinv(SB)
    inv_psi2 = np.linalg.pinv(SB2)

    # FIltro
    erfc_vec = np.vectorize(math.erfc)
    eta1 = np.arange(1, N + 2) / (N + 1)
    n1 = 10.0
    u = eta1 - 0.5
    u_sq = u**2
    arg = np.clip(1.0 - 4.0 * u_sq, 1e-14, 1.0)
    denom = np.clip(4.0 * u_sq, 1e-14, 1.0)
    sqrt_term = np.sqrt(-np.log(arg) / denom)
    sqrt_term[np.abs(u) < 1e-8] = 1.0
    filtro_erfc = 0.5 * erfc_vec(2.0 * np.sqrt(n1) * u * sqrt_term)

    return {
        'r': r, 'N': N, 'filtro': filtro_erfc,
        'psi': SB, 'rpsi': rSB, 'rrpsi': rrSB, 'inv_psi': inv_psi,
        'psi2': SB2, 'rpsi2': rSB2, 'rrpsi2': rrSB2, 'inv_psi2': inv_psi2
    }

# =========================================================================
# 2. CONDIÇÕES INICIAIS (Atenção à Projeção das Ímpares)
# =========================================================================
def criar_condicoes_iniciais_fccz4(b, M=1.0):
    r = b['r']
    psi_bh = 1.0 + M / (2.0 * r)
    chi_0 = psi_bh**(-2)
    alpha_0 = psi_bh**(-2)
    zeros = np.zeros_like(r)
    
    inv_psi = b['inv_psi']   # Base Par
    inv_psi2 = b['inv_psi2'] # Base Ímpar

    # Indices: 0:a, 1:b, 2:chi, 3:K, 4:Aa, 5:Theta, 6:Lambda(Ímpar), 7:alpha, 8:beta(Ímpar), 9:B
    return np.array([
        np.dot(zeros, inv_psi),          # c_a (Par)
        np.dot(zeros, inv_psi),          # c_b (Par)
        np.dot(chi_0 - 1.0, inv_psi),    # c_chi (Par)
        np.dot(zeros, inv_psi),          # c_K (Par)
        np.dot(zeros, inv_psi),          # c_Aa (Par)
        np.dot(zeros, inv_psi),          # c_Theta (Par)
        np.dot(zeros, inv_psi2),         # c_Lambda (ÍMPAR) <--- NOVO
        np.dot(alpha_0 - 1.0, inv_psi),  # c_alpha (Par)
        np.dot(zeros, inv_psi2),         # c_beta (ÍMPAR) <--- NOVO
        np.dot(zeros, inv_psi)           # c_B (Par)
    ])

# =========================================================================
# 3. EVOLUÇÃO fCCZ4 
# =========================================================================
def calcular_taxas_fccz4(state, kappa1, kappa2, eta_param, b):
    # Desempacotamento
    c_a, c_b, c_chi = state[0], state[1], state[2]
    c_K, c_Aa, c_Theta = state[3], state[4], state[5]
    c_Lambda = state[6] # ÍMPAR
    c_alpha = state[7]
    c_beta = state[8]   # ÍMPAR
    c_B = state[9]

    # Matrizes
    psi, rpsi, rrpsi, inv_psi = b['psi'], b['rpsi'], b['rrpsi'], b['inv_psi']
    psi2, rpsi2, rrpsi2, inv_psi2 = b['psi2'], b['rpsi2'], b['rrpsi2'], b['inv_psi2']
    r = b['r']

    # RECONSTRUÇÃO VARIÁVEIS PARES
    a = 1.0 + np.dot(c_a, psi); da = np.dot(c_a, rpsi); dda = np.dot(c_a, rrpsi)
    b_met = 1.0 + np.dot(c_b, psi); db = np.dot(c_b, rpsi); ddb = np.dot(c_b, rrpsi)
    alpha = 1.0 + np.dot(c_alpha, psi); dalpha = np.dot(c_alpha, rpsi); ddalpha = np.dot(c_alpha, rrpsi)
    chi = 1.0 + np.dot(c_chi, psi); dchi = np.dot(c_chi, rpsi); ddchi = np.dot(c_chi, rrpsi)
    B_shift = np.dot(c_B, psi)
    K = np.dot(c_K, psi); dK = np.dot(c_K, rpsi)
    Aa = np.dot(c_Aa, psi); dAa = np.dot(c_Aa, rpsi)
    Theta = np.dot(c_Theta, psi); dTheta = np.dot(c_Theta, rpsi)

    # RECONSTRUÇÃO VARIÁVEIS ÍMPARES
    beta = np.dot(c_beta, psi2); dbeta = np.dot(c_beta, rpsi2); ddbeta = np.dot(c_beta, rrpsi2)
    Lambda = np.dot(c_Lambda, psi2); dLambda = np.dot(c_Lambda, rpsi2)

    # Regularização Suave Original
    eps_sq = 1e-24
    chi_reg = np.sqrt(chi**2 + eps_sq)
    a_reg = np.sqrt(a**2 + eps_sq)
    b_reg = np.sqrt(b_met**2 + eps_sq)
    alpha_reg = np.sqrt(alpha**2 + eps_sq)
    chi_sq = chi_reg**2
    dchi_chi = dchi / chi_reg
    ddchi_chi = ddchi / chi_reg
    Ab = - (b_reg / (2.0 * a_reg)) * Aa

    # O termo beta/r agora é matematicamente limpo porque beta é nulo na origem!
    div_beta = dbeta + beta * (db / b_reg + da / (2.0 * a_reg) + 2.0 / r)
    d_div_beta = ddbeta + dbeta * (db / b_reg + da / (2.0 * a_reg) + 2.0 / r) + beta * ((ddb * b_reg - db**2) / (b_reg**2) + (dda * a_reg - da**2) / (2.0 * a_reg**2) - 2.0 / (r**2))

    
    kappa1_local = kappa1 
    kappa2 = 0

    bar_Lambda = (1.0 / a_reg) * (da / (2.0 * a_reg) - db / b_reg - (2.0 / r) * (1.0 - a_reg / b_reg))
    Zr = (a_reg / 2.0) * (Lambda - bar_Lambda)
    Zr_up = Zr / a_reg

    term1 = 0.5 * (da * Lambda + a_reg * dLambda)
    term2 = -0.25 * (dda / a_reg - (da**2) / (a_reg**2))
    term3 = 0.5 * (ddb / b_reg - (db**2) / (b_reg**2))
    term4 = -1.0 / (r**2)
    term5 = - (da / (r * b_reg) - a_reg / (r**2 * b_reg) - (a_reg * db) / (r * b_reg**2))
    dZr = term1 + term2 + term3 + term4 + term5
    dZr_up = (1.0 / a_reg) * dZr - (Zr / a_reg**2) * da

    Dm_Zm = dZr_up + Zr_up * (da / (2.0 * a_reg) + db / b_reg + 2.0 / r - 3.0 * dchi_chi)
    Dr_Zr = dZr_up + Zr_up * (da / (2.0 * a_reg) - 1.0 * dchi_chi)

    bar_R_rr = - ddb / b_reg + (db**2) / (2.0 * b_reg**2) + (da * db) / (2.0 * a_reg * b_reg) + 2.0 * da / (r * a_reg)
    bar_R_tt = - (r**2 * ddb) / (2.0 * a_reg) - (3.0 * r * db) / (2.0 * a_reg) + (r**2 * da * db) / (4.0 * a_reg**2) + (r * da) / (2.0 * a_reg) + 1.0 - a_reg / b_reg

    R_rr = bar_R_rr + 2.0 * ddchi_chi + (2.0 / r + db / b_reg - da / a_reg) * dchi_chi - 3.0 * dchi_chi**2
    R_tt = bar_R_tt + (r**2 * b_reg / a_reg) * (ddchi_chi + (3.0 / r + 1.5 * db / b_reg - 0.5 * da / a_reg) * dchi_chi - 2.0 * dchi_chi**2)
    Ricci = (chi_sq / a_reg) * R_rr + 2.0 * (chi_sq / (r**2 * b_reg)) * R_tt

    D2_alpha = (chi_sq / a_reg) * (ddalpha + dalpha * (2.0 / r + db / b_reg - da / (2.0 * a_reg) - dchi_chi))
    DrDr_alpha = (chi_sq / a_reg) * (ddalpha - dalpha * (da / (2.0 * a_reg) - dchi_chi))

    dt_a = beta * da + 2.0 * a_reg * dbeta - (2.0 / 3.0) * a_reg * div_beta - 2.0 * alpha_reg * a_reg * Aa
    dt_b = beta * db + 2.0 * b_reg * beta / r - (2.0 / 3.0) * b_reg * div_beta - 2.0 * alpha_reg * b_reg * Ab
    dt_chi = beta * dchi - (1.0 / 3.0) * chi_reg * div_beta + (1.0 / 6.0) * chi_reg * alpha_reg * K
    
    dt_K = - D2_alpha + alpha_reg * (Ricci + 2.0 * Dm_Zm + K**2 - 2.0 * Theta * K) + beta * dK - 3.0 * alpha_reg * kappa1_local * (1.0 + kappa2) * Theta
    dt_Theta = beta * dTheta + 0.5 * alpha_reg * (Ricci + 2.0 * Dm_Zm - (Aa**2 + 2.0 * Ab**2) + (2.0 / 3.0) * K**2 - 2.0 * Theta * K) - Zr_up * dalpha - alpha_reg * kappa1_local * (2.0 + kappa2) * Theta
    dt_Aa = beta * dAa - (DrDr_alpha - (1.0 / 3.0) * D2_alpha) + alpha_reg * ((chi_sq / a_reg) * R_rr - (1.0 / 3.0) * Ricci) + alpha_reg * (2.0 * Dr_Zr - (2.0 / 3.0) * Dm_Zm) + alpha_reg * Aa * (K - 2.0 * Theta)

    t1 = beta * dLambda - Lambda * dbeta + (1.0 / a_reg) * ddbeta + (2.0 / b_reg) * (dbeta / r - beta / (r**2))
    t2 = (1.0 / 3.0) * ((1.0 / a_reg) * d_div_beta + 2.0 * Lambda * div_beta)
    t3 = - (2.0 / a_reg) * (Aa * dalpha + alpha_reg * dAa)
    t4 = 2.0 * alpha_reg * (Aa * Lambda - (2.0 / (r * b_reg)) * (Aa - Ab))
    t5 = (2.0 * alpha_reg / a_reg) * (dAa - (2.0 / 3.0) * dK - 3.0 * Aa * dchi_chi + (Aa - Ab) * (2.0 / r + db / b_reg))
    t6 = (2.0 / a_reg) * (alpha_reg * dTheta - Theta * dalpha - (2.0 / 3.0) * alpha_reg * K * Zr)
    t7 = (2.0 / a_reg) * ((2.0 / 3.0) * Zr * div_beta - Zr * dbeta) - (2.0 / a_reg) * kappa1_local * Zr
    dt_Lambda = t1 + t2 + t3 + t4 + t5 + t6 + t7

    # GAUGE 
    dt_alpha = - 2.0 * alpha * (K - 2.0 * Theta)
    dt_beta = B_shift
    eta_local = eta_param
    dt_B = 0.75 * dt_Lambda - eta_local * B_shift

    # PROJEÇÃO PARES E ÍMPARES
    return np.array([
        np.dot(dt_a, inv_psi),       # Par
        np.dot(dt_b, inv_psi),       # Par
        np.dot(dt_chi, inv_psi),     # Par
        np.dot(dt_K, inv_psi),       # Par
        np.dot(dt_Aa, inv_psi),      # Par
        np.dot(dt_Theta, inv_psi),   # Par
        np.dot(dt_Lambda, inv_psi2), # ÍMPAR
        np.dot(dt_alpha, inv_psi),   # Par
        np.dot(dt_beta, inv_psi2),   # ÍMPAR
        np.dot(dt_B, inv_psi)        # Par
    ])

# =========================================================================
# 4. INTEGRADOR RK4, SONDA E FILTRO
# =========================================================================

def passo_rk4(s, h, k1, k2, eta, b):
    k_1 = calcular_taxas_fccz4(s, k1, k2, eta, b)
    k_2 = calcular_taxas_fccz4(s + 0.5*h*k_1, k1, k2, eta, b)
    k_3 = calcular_taxas_fccz4(s + 0.5*h*k_2, k1, k2, eta, b)
    k_4 = calcular_taxas_fccz4(s + h*k_3, k1, k2, eta, b)
    return s + (h/6.0) * (k_1 + 2*k_2 + 2*k_3 + k_4)


def simular_parametros_com_sonda(L0, N, tf, h, k1, eta_param):
    b = configurar_bases_fccz4(L0, N)
    s = criar_condicoes_iniciais_fccz4(b)
    k2 = 0.0

    r_grid = b['r']
    # Referências para Reconstrução na Sonda
    psi_mat, rpsi_mat, rrpsi_mat = b['psi'], b['rpsi'], b['rrpsi']
    psi2_mat, rpsi2_mat = b['psi2'], b['rpsi2']

    passos_totais = int(tf/h)
    passos_salvar = 500
    tempo_sobrevivido = 0.0
    erro_H_final = 0.0
    erro_M_final = 0.0

    # Cria e abre o arquivo txt com o nome dos parâmetros
    nome_arquivo = f"evolucao_L2_k{k1:.2f}_eta{eta_param:.2f}.txt"
    
    with open(nome_arquivo, "w") as arquivo_dados:
        # Escreve o cabeçalho das colunas no txt
        arquivo_dados.write("Tempo_M, L2_H, L2_Mr\n")

        for i in range(passos_totais):
            s_anterior = np.copy(s)
            
            # Dá o passo no tempo com RK4
            s = passo_rk4(s, h, k1, k2, eta_param, b)

            # ===========================================================
            # FILTRO CIRÚRGICO EXPANDIDO
            # ===========================================================
            s[2] *= b['filtro'] # chi (Fator Conforme)
            s[3] *= b['filtro'] # K (Traço da Curvatura)
            s[5] *= b['filtro'] # Theta (Escalar Z4)
            s[6] *= b['filtro'] # Lambda (Vetor Z4)
            s[7] *= b['filtro'] # alpha (Lapso)

            if i % passos_salvar == 0 or i == passos_totais - 1:
                a_at = 1.0 + np.dot(s[0], psi_mat)
                b_at = 1.0 + np.dot(s[1], psi_mat)
                chi_at = 1.0 + np.dot(s[2], psi_mat)
                K_at = np.dot(s[3], psi_mat)
                Aa_at = np.dot(s[4], psi_mat)
                Ab_at = -0.5 * Aa_at

                da_at = np.dot(s[0], rpsi_mat)
                db_at = np.dot(s[1], rpsi_mat)
                dchi_at = np.dot(s[2], rpsi_mat)
                dK_at = np.dot(s[3], rpsi_mat)
                dAa_at = np.dot(s[4], rpsi_mat)
                ddb_at = np.dot(s[1], rrpsi_mat)
                ddchi_at = np.dot(s[2], rrpsi_mat)

                eps_sq = 1e-12
                a_reg = np.sqrt(a_at**2 + eps_sq)
                b_reg = np.sqrt(b_at**2 + eps_sq)
                chi_reg = np.sqrt(chi_at**2 + eps_sq)
                chi_sq = chi_reg**2
                dchi_chi = dchi_at / chi_reg
                ddchi_chi = ddchi_at / chi_reg
                db_b = db_at / b_reg

                r_reg = np.sqrt(r_grid**2 + 1e-8)
                r_reg_sq = r_reg**2

                bar_R_rr = - ddb_at / b_reg + (db_at**2) / (2.0 * b_reg**2) + (da_at * db_at) / (2.0 * a_reg * b_reg) + 2.0 * da_at / (r_reg * a_reg)
                bar_R_tt = - (r_reg_sq * ddb_at) / (2.0 * a_reg) - (3.0 * r_reg * db_at) / (2.0 * a_reg) + (r_reg_sq * da_at * db_at) / (4.0 * a_reg**2) + (r_reg * da_at) / (2.0 * a_reg) + 1.0 - a_reg / b_reg
                R_rr = bar_R_rr + 2.0 * ddchi_chi + (2.0 / r_reg + db_at / b_reg - da_at / a_reg) * dchi_chi - 3.0 * dchi_chi**2
                R_tt = bar_R_tt + (r_reg_sq * b_reg / a_reg) * (ddchi_chi + (3.0 / r_reg + 1.5 * db_at / b_reg - 0.5 * da_at / a_reg) * dchi_chi - 2.0 * dchi_chi**2)
                R_fisico = (chi_sq / a_reg) * R_rr + 2.0 * (chi_sq / (r_reg_sq * b_reg)) * R_tt

                M_r = dAa_at - (2.0/3.0)*dK_at - 3.0*Aa_at*dchi_chi + (Aa_at - Ab_at)*(2.0/r_reg + db_b)
                H_const = R_fisico - (Aa_at**2 + 2.0*Ab_at**2) + (2.0/3.0)*K_at**2

                # Isola a região externa (r > 1.0) para análise
                mascara = r_grid > 1.0
                r_ext = r_grid[mascara]
                H_ext = H_const[mascara]
                M_ext = M_r[mascara]

                # Tamanho do domínio de integração
                delta_R = r_ext[-1] - r_ext[0]

                # Integração L2 real usando a regra do trapézio espacial
                integral_H2 = np.trapezoid(H_ext**2, x=r_ext)
                integral_M2 = np.trapezoid(M_ext**2, x=r_ext)

                erro_H_final = np.sqrt(integral_H2 / delta_R)
                erro_M_final = np.sqrt(integral_M2 / delta_R)

                # Atualiza o tempo perfeitamente para print e txt
                tempo_sobrevivido = i * h

                if i % (passos_salvar * 10) == 0:
                    print(f"Progresso: {tempo_sobrevivido:.2f}M | L2(H): {erro_H_final:.2e} | L2(M_r): {erro_M_final:.2e}")
                    
                    # Salva os dados de evolução no txt e força a gravação no disco
                    arquivo_dados.write(f"{tempo_sobrevivido:.4f}, {erro_H_final:.8e}, {erro_M_final:.8e}\n")
                    arquivo_dados.flush()

            # ===========================================================
            # DETETIVE DE CRASH E AUTÓPSIA
            # ===========================================================
            if np.isnan(s).any() or np.max(np.abs(s)) > 1e11:
                print(f"\n[ALERTA VERMELHO] Crash detectado em t={i*h:.4f}M!")
                
                # Reconstrói estado anterior para autópsia
                c_a, c_b, c_chi = s_anterior[0], s_anterior[1], s_anterior[2]
                c_K, c_Aa, c_Theta = s_anterior[3], s_anterior[4], s_anterior[5]
                c_Lambda, c_alpha = s_anterior[6], s_anterior[7]
                c_beta, c_B = s_anterior[8], s_anterior[9]

                a_at = 1.0 + np.dot(c_a, psi_mat)
                b_at = 1.0 + np.dot(c_b, psi_mat)
                chi_at = 1.0 + np.dot(c_chi, psi_mat)
                K_at = np.dot(c_K, psi_mat)
                Aa_at = np.dot(c_Aa, psi_mat)
                Theta_at = np.dot(c_Theta, psi_mat)
                alpha_at = 1.0 + np.dot(c_alpha, psi_mat)
                B_at = np.dot(c_B, psi_mat)

                Lambda_at = np.dot(c_Lambda, psi2_mat)
                beta_at = np.dot(c_beta, psi2_mat)

                da_at = np.dot(c_a, rpsi_mat); db_at = np.dot(c_b, rpsi_mat)
                a_reg = np.sqrt(a_at**2 + 1e-12); b_reg = np.sqrt(b_at**2 + 1e-12)
                r_reg = np.sqrt(r_grid**2 + 1e-8)

                bar_Lambda = (1.0 / a_reg) * (da_at / (2.0 * a_reg) - db_at / b_reg - (2.0 / r_reg) * (1.0 - a_reg / b_reg))
                Zr_at = (a_reg / 2.0) * (Lambda_at - bar_Lambda)

                idx_max_erro = np.argmax(np.abs(Zr_at) + np.abs(Theta_at))
                r_critico = r_grid[idx_max_erro]

                print(f"\n--- RELATÓRIO DE AUTÓPSIA (T={i*h - h:.4f}M) ---")
                print(f"Ponto crítico inspecionado: r = {r_critico:.4f}M")
                print(f"1. Métrica: a = {a_at[idx_max_erro]:.4e} | b = {b_at[idx_max_erro]:.4e} | chi = {chi_at[idx_max_erro]:.4e}")
                print(f"2. Gauge: alpha = {alpha_at[idx_max_erro]:.4e} | beta = {beta_at[idx_max_erro]:.4e} | B = {B_at[idx_max_erro]:.4e}")
                print(f"3. Curvatura: K = {K_at[idx_max_erro]:.4e} | Aa = {Aa_at[idx_max_erro]:.4e} | Theta = {Theta_at[idx_max_erro]:.4e}")
                print(f"4. Vínculos: Lambda = {Lambda_at[idx_max_erro]:.4e} | bar_Lambda = {bar_Lambda[idx_max_erro]:.4e} | Zr = {Zr_at[idx_max_erro]:.4e}")
                break

    return tempo_sobrevivido, erro_H_final, erro_M_final

# =========================================================================
# 5. GERENCIADOR DO DIAGNÓSTICO
# =========================================================================
tempo_alvo = 20.0
kappa1_testes = [0.05, 0.1, 0.5 , 1.0, 2.0, 5.0]
eta_testes = [0.0, 0.5, 2.0, 4.0, 8.0]

# Calculando o número total de testes
total_combinacoes = len(kappa1_testes) * len(eta_testes)
contador = 1

print("Iniciando fCCZ4 ...")
print(f"Testando {total_combinacoes} combinações possíveis...\n")
print("=" * 60)

for k1 in kappa1_testes:
    for eta in eta_testes:
        print(f"\n[{contador}/{total_combinacoes}] Testando: kappa1 = {k1} | eta = {eta}")
        
        t_inicio = time.time()
        t_crash, err_H, err_M = simular_parametros_com_sonda(L0=5.0, N=300, tf=tempo_alvo, h=0.0001, k1=k1, eta_param=eta)
        t_fim = time.time()
        
        duracao_minutos = (t_fim - t_inicio) / 60.0
        status = "SOBREVIVEU!" if t_crash >= tempo_alvo - 0.01 else "CRASH"
        
        print(f"Resultado da Execução: {status} em t = {t_crash:.2f}M (Demorou {duracao_minutos:.1f} min)")
        print(f"Erro L2(H) final: {err_H:.2e} | L2(M_r) final: {err_M:.2e}")
        print("-" * 60)
        
        contador += 1
