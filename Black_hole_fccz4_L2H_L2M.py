import numpy as np
import time
import math

# =========================================================================
# 1. ARQUITETURA DE DOMÍNIO ÚNICO (Sem a origem r=0) E FILTRO ERFC
# =========================================================================
def configurar_base_unica(L0, N):
    j = np.arange(1, N + 2)
    xi = np.cos(j * np.pi / (N + 2)) 
    
    x_min, x_max = -1.0, 1.0
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
            dT_i, d2T_i = np.zeros_like(xi_flip), np.zeros_like(xi_flip)
        else:
            sin_t = np.sin(theta)
            dT_i = i * np.sin(i * theta) / sin_t
            d2T_i = -i**2 * np.cos(i * theta) / (sin_t**2) + i * np.sin(i * theta) * xi_flip / (sin_t**3)
        psi[i, :] = T_i
        rpsi[i, :] = dT_i * dxi_dr
        rrpsi[i, :] = d2T_i * (dxi_dr**2) + dT_i * d2xi_dr2

    inv_psi = np.linalg.inv(psi)      

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

    return {'r': r, 'psi': psi, 'rpsi': rpsi, 'rrpsi': rrpsi, 'inv_psi': inv_psi, 'N': N, 'filtro': filtro_erfc}

# =========================================================================
# 2. CONDIÇÕES INICIAIS (Fator Conforme chi)
# =========================================================================
def criar_condicoes_iniciais_fccz4(b, M=1.0):
    r = b['r']
    psi_bh = 1.0 + M / (2.0 * r)
    chi_0 = psi_bh**(-2)  
    alpha_0 = psi_bh**(-2)  
    zeros = np.zeros_like(r)
    inv_psi = b['inv_psi']
    return np.array([np.dot(zeros, inv_psi), np.dot(zeros, inv_psi), 
                     np.dot(chi_0 - 1.0, inv_psi), np.dot(zeros, inv_psi), np.dot(zeros, inv_psi), 
                     np.dot(zeros, inv_psi), np.dot(zeros, inv_psi), np.dot(alpha_0 - 1.0, inv_psi), 
                     np.dot(zeros, inv_psi), np.dot(zeros, inv_psi)])


# =========================================================================
# 3. EVOLUÇÃO fCCZ4 (FÍSICA CORRIGIDA E PURA - SEM QUINAS)
# =========================================================================
def calcular_taxas_fccz4(state, kappa1, kappa2, eta_param, b):
    c_a = state[0]; c_b = state[1]; c_chi = state[2]
    c_K = state[3]; c_Aa = state[4]; c_Theta = state[5]
    c_Lambda = state[6]; c_alpha = state[7]; c_beta = state[8]; c_B = state[9]
    
    psi, rpsi, rrpsi, inv_psi, r = b['psi'], b['rpsi'], b['rrpsi'], b['inv_psi'], b['r']
    
    a = 1.0 + np.dot(c_a, psi); da = np.dot(c_a, rpsi); dda = np.dot(c_a, rrpsi)
    b_met = 1.0 + np.dot(c_b, psi); db = np.dot(c_b, rpsi); ddb = np.dot(c_b, rrpsi)
    alpha = 1.0 + np.dot(c_alpha, psi); dalpha = np.dot(c_alpha, rpsi); ddalpha = np.dot(c_alpha, rrpsi)
    chi = 1.0 + np.dot(c_chi, psi); dchi = np.dot(c_chi, rpsi); ddchi = np.dot(c_chi, rrpsi)
    beta = np.dot(c_beta, psi); dbeta = np.dot(c_beta, rpsi); ddbeta = np.dot(c_beta, rrpsi)
    B_shift = np.dot(c_B, psi); Lambda = np.dot(c_Lambda, psi); dLambda = np.dot(c_Lambda, rpsi)
    K = np.dot(c_K, psi); dK = np.dot(c_K, rpsi); Aa = np.dot(c_Aa, psi); dAa = np.dot(c_Aa, rpsi)
    Theta = np.dot(c_Theta, psi); dTheta = np.dot(c_Theta, rpsi)
    
    # Regularização Suave Original (Sem pisos violentos)
    eps_sq = 1e-24
    chi_reg = np.sqrt(chi**2 + eps_sq)
    a_reg = np.sqrt(a**2 + eps_sq)
    b_reg = np.sqrt(b_met**2 + eps_sq)
    alpha_reg = np.sqrt(alpha**2 + eps_sq)
    
    chi_sq = chi_reg**2
    dchi_chi = dchi / chi_reg
    ddchi_chi = ddchi / chi_reg
    Ab = - (b_reg / (2.0 * a_reg)) * Aa 
    
    div_beta = dbeta + beta * (db / b_reg + da / (2.0 * a_reg) + 2.0 / r)
    d_div_beta = ddbeta + dbeta * (db / b_reg + da / (2.0 * a_reg) + 2.0 / r) + beta * ((ddb * b_reg - db**2) / (b_reg**2) + (dda * a_reg - da**2) / (2.0 * a_reg**2) - 2.0 / (r**2))
    
    bar_Lambda = (1.0 / a_reg) * (da / (2.0 * a_reg) - db / b_reg - (2.0 / r) * (1.0 - a_reg / b_reg))
    Zr = (a_reg / 2.0) * (Lambda - bar_Lambda)
    Zr_up = Zr / a_reg
    
    # =====================================================================
    # DERIVADA ANALÍTICA DE Zr (A cura do Aliasing Espectral - Mantida)
    # =====================================================================
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
    
    # FÍSICA CORRIGIDA (Ricci com Sinais Exatos)
    R_rr = bar_R_rr + 2.0 * ddchi_chi + (2.0 / r + db / b_reg - da / a_reg) * dchi_chi - 3.0 * dchi_chi**2
    R_tt = bar_R_tt + (r**2 * b_reg / a_reg) * (ddchi_chi + (3.0 / r + 1.5 * db / b_reg - 0.5 * da / a_reg) * dchi_chi - 2.0 * dchi_chi**2)
    Ricci = (chi_sq / a_reg) * R_rr + 2.0 * (chi_sq / (r**2 * b_reg)) * R_tt
    
    D2_alpha = (chi_sq / a_reg) * (ddalpha + dalpha * (2.0 / r + db / b_reg - da / (2.0 * a_reg) - dchi_chi))
    DrDr_alpha = (chi_sq / a_reg) * (ddalpha - dalpha * (da / (2.0 * a_reg) - dchi_chi)) 
    
    dt_a = beta * da + 2.0 * a_reg * dbeta - (2.0 / 3.0) * a_reg * div_beta - 2.0 * alpha_reg * a_reg * Aa
    dt_b = beta * db + 2.0 * b_reg * beta / r - (2.0 / 3.0) * b_reg * div_beta - 2.0 * alpha_reg * b_reg * Ab
    dt_chi = beta * dchi - (1.0 / 6.0) * chi_reg * div_beta + (1.0 / 6.0) * chi_reg * alpha_reg * K
    dt_K = - D2_alpha + alpha_reg * (Ricci + 2.0 * Dm_Zm + K**2 - 2.0 * Theta * K) + beta * dK - 3.0 * alpha_reg * kappa1 * (1.0 + kappa2) * Theta
    dt_Theta = beta * dTheta + 0.5 * alpha_reg * (Ricci + 2.0 * Dm_Zm - (Aa**2 + 2.0 * Ab**2) + (2.0 / 3.0) * K**2 - 2.0 * Theta * K) - Zr_up * dalpha - alpha_reg * kappa1 * (2.0 + kappa2) * Theta
    dt_Aa = beta * dAa - (DrDr_alpha - (1.0 / 3.0) * D2_alpha) + alpha_reg * ((chi_sq / a_reg) * R_rr - (1.0 / 3.0) * Ricci) + alpha_reg * (2.0 * Dr_Zr - (2.0 / 3.0) * Dm_Zm) + alpha_reg * Aa * (K - 2.0 * Theta)
    
    t1 = beta * dLambda - Lambda * dbeta + (1.0 / a_reg) * ddbeta + (2.0 / b_reg) * (dbeta / r - beta / (r**2))
    t2 = (1.0 / 3.0) * ((1.0 / a_reg) * d_div_beta + 2.0 * Lambda * div_beta)
    t3 = - (2.0 / a_reg) * (Aa * dalpha + alpha_reg * dAa)
    t4 = 2.0 * alpha_reg * (Aa * Lambda - (2.0 / (r * b_reg)) * (Aa - Ab))
    t5 = (2.0 * alpha_reg / a_reg) * (dAa - (2.0 / 3.0) * dK - 3.0 * Aa * dchi_chi + (Aa - Ab) * (2.0 / r + db / b_reg))
    t6 = (2.0 / a_reg) * (alpha_reg * dTheta - Theta * dalpha - (2.0 / 3.0) * alpha_reg * K * Zr)
    t7 = (2.0 / a_reg) * ((2.0 / 3.0) * Zr * div_beta - Zr * dbeta) - (2.0 / a_reg) * kappa1 * Zr
    dt_Lambda = t1 + t2 + t3 + t4 + t5 + t6 + t7
    
    # === A FÍSICA DO LAPSO (fCCZ4 Original - Ancorou o código de 24M) ===
    dt_alpha = - 2.0 * alpha * (K - 2.0 * Theta)
    
    # === A FÍSICA DO SHIFT ===
    dt_beta = B_shift
    eta_local = eta_param * (5.0 / (r + 5.0))
    dt_B = 0.75 * dt_Lambda - eta_local * B_shift

    return np.array([np.dot(dt_a, inv_psi), np.dot(dt_b, inv_psi), np.dot(dt_chi, inv_psi), np.dot(dt_K, inv_psi), np.dot(dt_Aa, inv_psi), np.dot(dt_Theta, inv_psi), np.dot(dt_Lambda, inv_psi), np.dot(dt_alpha, inv_psi), np.dot(dt_beta, inv_psi), np.dot(dt_B, inv_psi)])


# =========================================================================
# 4. INTEGRADOR RK4, SONDA E FILTRAGEM TOTAL
# =========================================================================
def passo_rk4(s, h, k1, k2, eta, b):
    k_1 = calcular_taxas_fccz4(s, k1, k2, eta, b)
    k_2 = calcular_taxas_fccz4(s + 0.5*h*k_1, k1, k2, eta, b)
    k_3 = calcular_taxas_fccz4(s + 0.5*h*k_2, k1, k2, eta, b)
    k_4 = calcular_taxas_fccz4(s + h*k_3, k1, k2, eta, b)
    return s + (h/6.0) * (k_1 + 2*k_2 + 2*k_3 + k_4)

def simular_parametros_com_sonda(L0, N, tf, h, k1, eta_param):
    b = configurar_base_unica(L0, N)
    s = criar_condicoes_iniciais_fccz4(b)
    
    k2 = 0.0 
    
    r_grid = b['r']
    psi_mat, rpsi_mat, rrpsi_mat = b['psi'], b['rpsi'], b['rrpsi']
    inv_psi = b['inv_psi']
    
    passos_totais = int(tf/h)
    passos_salvar = 500 
    
    tempo_sobrevivido = 0.0
    erro_H_final = 0.0
    erro_M_final = 0.0
    
    for i in range(passos_totais):
        s_anterior = np.copy(s)
        
        s = passo_rk4(s, h, k1, k2, eta_param, b)
        
        # FILTRAGEM TOTAL (A "cola" necessária para o Chebyshev não rasgar a origem)
        for j in range(10):
            s[j] *= b['filtro']
            
        if i % passos_salvar == 0 or i == passos_totais - 1:
            a_at = 1.0 + np.dot(s[0], psi_mat); b_at = 1.0 + np.dot(s[1], psi_mat); chi_at = 1.0 + np.dot(s[2], psi_mat)
            K_at = np.dot(s[3], psi_mat); Aa_at = np.dot(s[4], psi_mat); Ab_at = -0.5 * Aa_at 
            da_at = np.dot(s[0], rpsi_mat); db_at = np.dot(s[1], rpsi_mat); dchi_at = np.dot(s[2], rpsi_mat)
            dK_at = np.dot(s[3], rpsi_mat); dAa_at = np.dot(s[4], rpsi_mat)
            ddb_at = np.dot(s[1], rrpsi_mat); ddchi_at = np.dot(s[2], rrpsi_mat)
            
            eps_sq = 1e-12
            a_reg = np.sqrt(a_at**2 + eps_sq); b_reg = np.sqrt(b_at**2 + eps_sq); chi_reg = np.sqrt(chi_at**2 + eps_sq)
            chi_sq = chi_reg**2; dchi_chi = dchi_at / chi_reg; ddchi_chi = ddchi_at / chi_reg; db_b = db_at / b_reg
            
            r_reg = np.sqrt(r_grid**2 + 1e-8)
            r_reg_sq = r_reg**2
            
            bar_R_rr = - ddb_at / b_reg + (db_at**2) / (2.0 * b_reg**2) + (da_at * db_at) / (2.0 * a_reg * b_reg) + 2.0 * da_at / (r_reg * a_reg)
            bar_R_tt = - (r_reg_sq * ddb_at) / (2.0 * a_reg) - (3.0 * r_reg * db_at) / (2.0 * a_reg) + (r_reg_sq * da_at * db_at) / (4.0 * a_reg**2) + (r_reg * da_at) / (2.0 * a_reg) + 1.0 - a_reg / b_reg
            
            R_rr = bar_R_rr + 2.0 * ddchi_chi + (2.0 / r_reg + db_at / b_reg - da_at / a_reg) * dchi_chi - 3.0 * dchi_chi**2
            R_tt = bar_R_tt + (r_reg_sq * b_reg / a_reg) * (ddchi_chi + (3.0 / r_reg + 1.5 * db_at / b_reg - 0.5 * da_at / a_reg) * dchi_chi - 2.0 * dchi_chi**2)
            R_fisico = (chi_sq / a_reg) * R_rr + 2.0 * (chi_sq / (r_reg_sq * b_reg)) * R_tt
            
            M_r = dAa_at - (2.0/3.0)*dK_at - 3.0*Aa_at*dchi_chi + (Aa_at - Ab_at)*(2.0/r_reg + db_b)
            H_const = R_fisico - (Aa_at**2 + 2.0*Ab_at**2) + (2.0/3.0)*K_at**2
            
            mascara = r_grid > 1.0
            erro_H_final = np.sqrt(np.mean(H_const[mascara]**2))
            erro_M_final = np.sqrt(np.mean(M_r[mascara]**2))
            tempo_sobrevivido = i * h
            
            # =======================================================
            # O MONITOR DE BATIMENTOS (Imprime a cada 0.5M)
            # =======================================================
            if i % (passos_salvar * 10) == 0:
                print(f"Progresso: {tempo_sobrevivido:.2f}M | L2(H): {erro_H_final:.2e} | L2(M_r): {erro_M_final:.2e}")
            
        # =========================================================================
        # O DETETIVE DE CRASH
        # =========================================================================
        if np.isnan(s).any() or np.max(np.abs(s)) > 1e11:
            print(f"\n[ALERTA VERMELHO] Crash detectado em t={i*h:.4f}M!")
            print("Executando autópsia nas variáveis no passo imediatamente ANTERIOR ao colapso...")
            
            c_a = s_anterior[0]; c_b = s_anterior[1]; c_chi = s_anterior[2]
            c_K = s_anterior[3]; c_Aa = s_anterior[4]; c_Theta = s_anterior[5]
            c_Lambda = s_anterior[6]; c_alpha = s_anterior[7]; c_beta = s_anterior[8]; c_B = s_anterior[9]
            
            a_at = 1.0 + np.dot(c_a, psi_mat); b_at = 1.0 + np.dot(c_b, psi_mat); chi_at = 1.0 + np.dot(c_chi, psi_mat)
            K_at = np.dot(c_K, psi_mat); Aa_at = np.dot(c_Aa, psi_mat); Theta_at = np.dot(c_Theta, psi_mat)
            Lambda_at = np.dot(c_Lambda, psi_mat); alpha_at = 1.0 + np.dot(c_alpha, psi_mat)
            beta_at = np.dot(c_beta, psi_mat); B_at = np.dot(c_B, psi_mat)
            
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
            
            print(f"\nMax(abs) global na malha:")
            print(f"Max(alpha): {np.max(np.abs(alpha_at)):.2e} | Min(alpha): {np.min(alpha_at):.2e}")
            print(f"Max(a): {np.max(np.abs(a_at)):.2e} | Max(b): {np.max(np.abs(b_at)):.2e}")
            print(f"Max(Zr): {np.max(np.abs(Zr_at)):.2e} | Max(Theta): {np.max(np.abs(Theta_at)):.2e}")
            print("-" * 50)
            break
            
    return tempo_sobrevivido, erro_H_final, erro_M_final

# =========================================================================
# 5. GERENCIADOR DO DIAGNÓSTICO (AUTÓPSIA DO CRASH)
# =========================================================================
tempo_alvo = 40.0

# Apenas a combinação teórica ideal para ter o terminal limpo
kappa1_testes = [0.2]
eta_testes = [2.0]

print("Iniciando simulação com blindagem completa. Fique de olho no terminal...\n")

with open("Relatorio_Autopsia.txt", "w") as f:
    f.write("RELATORIO DE DIAGNOSTICO - BURACO NEGRO\n")
    f.write(f"{'Kappa1':<10} | {'Eta':<10} | {'Tempo de Crash (M)':<20} | {'Ultimo Erro L2(H)':<20}\n")
    f.write("-" * 68 + "\n")

for k1 in kappa1_testes:
    for eta in eta_testes:
        print(f"Testando: kappa1 = {k1}, eta = {eta}")
        
        t_inicio = time.time()
        
        simular_parametros_com_sonda(L0=5.0, N=300, tf=tempo_alvo, h=0.00002, k1=0.2, eta_param=2.0)
        
        t_fim = time.time()
        duracao_minutos = (t_fim - t_inicio) / 60.0
        
        status = "SOBREVIVEU!" if t_crash >= tempo_alvo - 0.01 else "CRASH"
        print(f"\nResultado da Execução: {status} em t = {t_crash:.2f}M (Demorou {duracao_minutos:.1f} min)")
        print(f"Erro L2(H) registrado um passo antes do crash: {err_H:.2e}")
        
        with open("Relatorio_Autopsia.txt", "a") as f:
            f.write(f"{k1:<10} | {eta:<10} | {t_crash:<20.2f} | {err_H:<20.2e}\n")

print("\nDiagnóstico finalizado.")
