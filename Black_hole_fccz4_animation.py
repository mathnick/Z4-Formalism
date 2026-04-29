import numpy as np
import matplotlib.pyplot as plt
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
# 3. EVOLUÇÃO fCCZ4 (EQUAÇÕES DO ARTIGO)
# =========================================================================
def calcular_taxas_fccz4(state, kappa1, kappa2, eta_param, b):
    # ATENÇÃO: Criação de cópias (não usar *= para não quebrar a memória do RK4)
    c_a = state[0] * b['filtro']; c_b = state[1] * b['filtro']; c_chi = state[2] * b['filtro']
    c_K = state[3] * b['filtro']; c_Aa = state[4] * b['filtro']; c_Theta = state[5] * b['filtro']
    c_Lambda = state[6] * b['filtro']; c_alpha = state[7] * b['filtro']; c_beta = state[8] * b['filtro']; c_B = state[9] * b['filtro']

    psi, rpsi, rrpsi, inv_psi, r = b['psi'], b['rpsi'], b['rrpsi'], b['inv_psi'], b['r']
    
    a = 1.0 + np.dot(c_a, psi); da = np.dot(c_a, rpsi); dda = np.dot(c_a, rrpsi)
    b_met = 1.0 + np.dot(c_b, psi); db = np.dot(c_b, rpsi); ddb = np.dot(c_b, rrpsi)
    alpha = 1.0 + np.dot(c_alpha, psi); dalpha = np.dot(c_alpha, rpsi); ddalpha = np.dot(c_alpha, rrpsi)
    chi = 1.0 + np.dot(c_chi, psi); dchi = np.dot(c_chi, rpsi); ddchi = np.dot(c_chi, rrpsi)
    beta = np.dot(c_beta, psi); dbeta = np.dot(c_beta, rpsi); ddbeta = np.dot(c_beta, rrpsi)
    B_shift = np.dot(c_B, psi); Lambda = np.dot(c_Lambda, psi); dLambda = np.dot(c_Lambda, rpsi)
    K = np.dot(c_K, psi); dK = np.dot(c_K, rpsi); Aa = np.dot(c_Aa, psi); dAa = np.dot(c_Aa, rpsi)
    Theta = np.dot(c_Theta, psi); dTheta = np.dot(c_Theta, rpsi)
    
    # Regularização Suave (Blindagem matemática analítica)
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
    d_div_beta = ddbeta + dbeta * (db / b_reg + da / (2.0 * a_reg) + 2.0 / r) + beta * ((ddb * b_reg - db**2) / (b_reg**2) + (dda * a_reg - da**2) / (2.0 * a_reg**2) - 2.0 / r**2)
    
    bar_Lambda = (1.0 / a_reg) * (da / (2.0 * a_reg) - db / b_reg - (2.0 / r) * (1.0 - a_reg / b_reg))
    Zr = (a_reg / 2.0) * (Lambda - bar_Lambda); Zr_up = Zr / a_reg
    dZr = np.dot(np.dot(Zr, inv_psi), rpsi); dZr_up = (1.0 / a_reg) * dZr - (Zr / a_reg**2) * da
    
    Dm_Zm = dZr_up + Zr_up * (da / (2.0 * a_reg) + db / b_reg + 2.0 / r - 3.0 * dchi_chi)
    Dr_Zr = dZr_up + Zr_up * (da / (2.0 * a_reg) - 1.0 * dchi_chi)

    bar_R_rr = - ddb / b_reg + (db**2) / (2.0 * b_reg**2) + (da * db) / (2.0 * a_reg * b_reg) + 2.0 * da / (r * a_reg)
    bar_R_tt = - (r**2 * ddb) / (2.0 * a_reg) - (3.0 * r * db) / (2.0 * a_reg) + (r**2 * da * db) / (4.0 * a_reg**2) + (r * da) / (2.0 * a_reg) + 1.0 - a_reg / b_reg
    
    R_rr = bar_R_rr + ddchi_chi - (da / (2.0 * a_reg)) * dchi_chi + 2.0 * dchi_chi * (1.0 / r + db / (2.0 * b_reg))
    R_tt = bar_R_tt + (r**2 * b_reg / a_reg) * ddchi_chi + (r**2 * b_reg / a_reg) * 2.0 * dchi_chi * (1.0 / r + db / (2.0 * b_reg) - da / (4.0 * a_reg))
    Ricci = (chi_sq / a_reg) * R_rr + 2.0 * (chi_sq / (r**2 * b_reg)) * R_tt

    D2_alpha = (chi_sq / a_reg) * ddalpha + (chi_sq / a_reg) * dalpha * (2.0 / r + db / b_reg - da / (2.0 * a_reg) + dchi_chi)
    DrDr_alpha = (chi_sq / a_reg) * (ddalpha - dalpha * (da / (2.0 * a_reg) + dchi_chi)) 
    
    dt_a = beta * da + 2.0 * a_reg * dbeta - (2.0 / 3.0) * a_reg * div_beta - 2.0 * alpha_reg * a_reg * Aa
    dt_b = beta * db + 2.0 * b_reg * beta / r - (2.0 / 3.0) * b_reg * div_beta - 2.0 * alpha_reg * b_reg * Ab
    dt_chi = beta * dchi - (1.0 / 6.0) * chi_reg * div_beta + (1.0 / 6.0) * chi_reg * alpha_reg * K
    dt_K = - D2_alpha + alpha_reg * (Ricci + 2.0 * Dm_Zm + K**2 - 2.0 * Theta * K) + beta * dK - 3.0 * alpha_reg * kappa1 * (1.0 + kappa2) * Theta
    dt_Theta = beta * dTheta + 0.5 * alpha_reg * (Ricci + 2.0 * Dm_Zm - (Aa**2 + 2.0 * Ab**2) + (2.0 / 3.0) * K**2 - 2.0 * Theta * K) - Zr_up * dalpha - alpha_reg * kappa1 * (2.0 + kappa2) * Theta
    dt_Aa = beta * dAa - (DrDr_alpha - (1.0 / 3.0) * D2_alpha) + alpha_reg * ((chi_sq / a_reg) * R_rr - (1.0 / 3.0) * Ricci) + alpha_reg * (2.0 * Dr_Zr - (2.0 / 3.0) * Dm_Zm) + alpha_reg * Aa * (K - 2.0 * Theta)
    
    t1 = beta * dLambda - Lambda * dbeta + (1.0 / a_reg) * ddbeta + (2.0 / b_reg) * (dbeta / r - beta / r**2)
    # CORREÇÃO 1: Usar Lambda (evoluído) em vez de bar_Lambda (analítico)
    t2 = (1.0 / 3.0) * ((1.0 / a_reg) * d_div_beta + 2.0 * Lambda * div_beta)
    t3 = - (2.0 / a_reg) * (Aa * dalpha + alpha_reg * dAa)
    # CORREÇÃO 2: Usar Lambda (evoluído) em vez de bar_Lambda (analítico)
    t4 = 2.0 * alpha_reg * (Aa * Lambda - (2.0 / (r * b_reg)) * (Aa - Ab))
    t5 = (2.0 * alpha_reg / a_reg) * (dAa - (2.0 / 3.0) * dK - 3.0 * Aa * dchi_chi + (Aa - Ab) * (2.0 / r + db / b_reg))
    t6 = (2.0 / a_reg) * (alpha_reg * dTheta - Theta * dalpha - (2.0 / 3.0) * alpha_reg * K * Zr)
    t7 = (2.0 / a_reg) * ((2.0 / 3.0) * Zr * div_beta - Zr * dbeta) - (2.0 / a_reg) * kappa1 * Zr
    dt_Lambda = t1 + t2 + t3 + t4 + t5 + t6 + t7
    
    # === A FÍSICA DO LAPSO (Eq. 2.25 do Artigo fCCZ4) ===
    # 1+log não-advectiva acoplada à restrição Theta
    dt_alpha = - 2.0 * alpha * (K - 2.0 * Theta)
    
    # === A FÍSICA DO SHIFT (Eq. 2.26 e 2.27 do Artigo) ===
    dt_beta = B_shift
    dt_B = 0.75 * dt_Lambda - eta_param * B_shift

    return np.array([np.dot(dt_a, inv_psi), np.dot(dt_b, inv_psi), np.dot(dt_chi, inv_psi), np.dot(dt_K, inv_psi), np.dot(dt_Aa, inv_psi), np.dot(dt_Theta, inv_psi), np.dot(dt_Lambda, inv_psi), np.dot(dt_alpha, inv_psi), np.dot(dt_beta, inv_psi), np.dot(dt_B, inv_psi)])

import matplotlib.animation as animation

# =========================================================================
# 4. INTEGRADOR E CAPTURA DE FRAMES PARA ANIMAÇÃO
# =========================================================================
def passo_rk4(s, h, k1, k2, eta, b):
    k_1 = calcular_taxas_fccz4(s, k1, k2, eta, b)
    k_2 = calcular_taxas_fccz4(s + 0.5*h*k_1, k1, k2, eta, b)
    k_3 = calcular_taxas_fccz4(s + 0.5*h*k_2, k1, k2, eta, b)
    k_4 = calcular_taxas_fccz4(s + h*k_3, k1, k2, eta, b)
    return s + (h/6.0) * (k_1 + 2*k_2 + 2*k_3 + k_4)

def simular_e_filmar(L0, N, tf, h, frames_qtd=100):
    b = configurar_base_unica(L0, N)
    s = criar_condicoes_iniciais_fccz4(b)
    k1, k2, eta = 2.0, 0.0, 6.5
    
    # Listas para guardar os frames (fotos do universo)
    r_grid = b['r']
    psi_mat = b['psi']
    frames_alpha = []
    frames_X = []
    frames_tempo = []
    
    passos_totais = int(tf/h)
    passo_por_frame = passos_totais // frames_qtd
    
    print(f"Rodando e Filmando fCCZ4 (N={N}, dt={h})...")
    
    for i in range(passos_totais):
        s = passo_rk4(s, h, k1, k2, eta, b)
        
        for j in range(10):
            s[j] *= b['filtro']
            
        # Tira uma "foto" a cada X passos
        if i % passo_por_frame == 0 or i == passos_totais - 1:
            alpha_atual = 1.0 + np.dot(s[7], psi_mat)
            chi_atual = 1.0 + np.dot(s[2], psi_mat)
            X_atual = chi_atual**2
            
            frames_alpha.append(np.copy(alpha_atual))
            frames_X.append(np.copy(X_atual))
            frames_tempo.append(i * h)
            print(f"Gravando frame em t={i*h:.2f}M")
            
        if np.isnan(s).any() or np.max(np.abs(s)) > 1e11:
            print(f"\nCrash t={i*h:.4f}M.")
            break
            
    return r_grid, frames_tempo, frames_alpha, frames_X

# =========================================================================
# 5. GERADOR DO CINEMA (Animação)
# =========================================================================

tf_filme = 10.0 
r, tempos, f_alpha, f_X = simular_e_filmar(L0=5.0, N=150, tf=tf_filme, h=0.00001, frames_qtd=100)

print("\nGerando animação... (isso pode levar alguns segundos)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Configuração da Câmera (Limites dos gráficos)
ax1.set_xlim(0.0, 15.0); ax1.set_ylim(-0.05, 1.05)
ax1.set_xlabel("Raio isotrópico (r/M)"); ax1.set_ylabel(r"Fator Lapso ($\alpha$)")
ax1.grid(True)

ax2.set_xlim(0.0, 15.0); ax2.set_ylim(-0.05, 1.05)
ax2.set_xlabel("Raio isotrópico (r/M)"); ax2.set_ylabel(r"Fator Conforme ($X$)")
ax2.grid(True)

linha_alpha, = ax1.plot([], [], 'b-', linewidth=2)
linha_X, = ax2.plot([], [], 'k-', linewidth=2)
titulo = fig.suptitle('', fontsize=16)

def animar(i):
    linha_alpha.set_data(r, f_alpha[i])
    linha_X.set_data(r, f_X[i])
    titulo.set_text(f'Evolução Wormhole $\\rightarrow$ Trombeta | Tempo = {tempos[i]:.2f}M')
    return linha_alpha, linha_X, titulo

ani = animation.FuncAnimation(fig, animar, frames=len(tempos), interval=100, blit=False)

# Exibe o player na tela
plt.tight_layout()
plt.show()
