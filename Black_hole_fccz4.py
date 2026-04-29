import numpy as np
import matplotlib.pyplot as plt
import math

# =========================================================================
# 1. ARQUITETURA DE DOMÍNIO ÚNICO (Filtro ERFC)
# =========================================================================
def configurar_base_unica(L0, N):
    j = np.arange(1, N + 2)
    xi = np.cos(j * np.pi / (N + 2)) 
    x_min, x_max = -1.0, 1.0
    x_global = x_min + (x_max - x_min) * (xi + 1.0) / 2.0
    r = L0 * (1.0 + x_global) / (1.0 - x_global)
    r = np.flip(r); xi_flip = np.flip(xi)
    
    psi = np.zeros([N+1, N+1]); rpsi = np.zeros([N+1, N+1]); rrpsi = np.zeros([N+1, N+1])
    dr_dx = (r + L0)**2 / (2.0 * L0); dx_dxi = (x_max - x_min) / 2.0
    dxi_dr = 1.0 / (dr_dx * dx_dxi); d2xi_dr2 = - (2.0 / (r + L0)) * dxi_dr

    for i in range(N+1):
        theta = np.arccos(xi_flip); T_i = np.cos(i * theta)
        if i == 0: dT_i, d2T_i = np.zeros_like(xi_flip), np.zeros_like(xi_flip)
        else:
            sin_t = np.sin(theta); dT_i = i * np.sin(i * theta) / sin_t
            d2T_i = -i**2 * np.cos(i * theta) / (sin_t**2) + i * np.sin(i * theta) * xi_flip / (sin_t**3)
        psi[i, :] = T_i; rpsi[i, :] = dT_i * dxi_dr; rrpsi[i, :] = d2T_i * (dxi_dr**2) + dT_i * d2xi_dr2

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
# 2. CONDIÇÕES INICIAIS (Perturbação Suave)
# =========================================================================
def criar_condicoes_iniciais_fccz4(b, M=1.0):
    r = b['r']; inv_psi = b['inv_psi']
    psi_bh = 1.0 + M / (2.0 * r)
    chi_0 = psi_bh**(-2); alpha_0 = psi_bh**(-2)
    zeros = np.zeros_like(r)
    return np.array([np.dot(zeros, inv_psi), np.dot(zeros, inv_psi), np.dot(chi_0 - 1.0, inv_psi), 
                     np.dot(zeros, inv_psi), np.dot(zeros, inv_psi), np.dot(zeros, inv_psi), 
                     np.dot(zeros, inv_psi), np.dot(alpha_0 - 1.0, inv_psi), np.dot(zeros, inv_psi), np.dot(zeros, inv_psi)])

# =========================================================================
# 3. EVOLUÇÃO fCCZ4 (A FÍSICA PURA)
# =========================================================================
def calcular_taxas_fccz4(state, kappa1, kappa2, eta_param, b):
    c_a, c_b, c_chi, c_K, c_Aa, c_Theta, c_Lambda, c_alpha, c_beta, c_B = state
    psi, rpsi, rrpsi, inv_psi, r = b['psi'], b['rpsi'], b['rrpsi'], b['inv_psi'], b['r']
    
    a = 1.0 + np.dot(c_a, psi); da = np.dot(c_a, rpsi); dda = np.dot(c_a, rrpsi)
    b_met = 1.0 + np.dot(c_b, psi); db = np.dot(c_b, rpsi); ddb = np.dot(c_b, rrpsi)
    alpha = 1.0 + np.dot(c_alpha, psi); dalpha = np.dot(c_alpha, rpsi); ddalpha = np.dot(c_alpha, rrpsi)
    chi = 1.0 + np.dot(c_chi, psi); dchi = np.dot(c_chi, rpsi); ddchi = np.dot(c_chi, rrpsi)
    beta = np.dot(c_beta, psi); dbeta = np.dot(c_beta, rpsi); ddbeta = np.dot(c_beta, rrpsi)
    B_shift = np.dot(c_B, psi); Lambda = np.dot(c_Lambda, psi); dLambda = np.dot(c_Lambda, rpsi)
    K = np.dot(c_K, psi); dK = np.dot(c_K, rpsi); Aa = np.dot(c_Aa, psi); dAa = np.dot(c_Aa, rpsi)
    Theta = np.dot(c_Theta, psi); dTheta = np.dot(c_Theta, rpsi)
    
    # REGULARIZAÇÃO SUAVE (C-infinito) DO CÓDIGO ORIGINAL
    eps_sq = 1e-24
    chi_reg = np.sqrt(chi**2 + eps_sq)
    a_reg = np.sqrt(a**2 + eps_sq)
    b_reg = np.sqrt(b_met**2 + eps_sq)
    
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

    dt_a = beta * da + 2.0 * a_reg * dbeta - (2.0 / 3.0) * a_reg * div_beta - 2.0 * alpha * a_reg * Aa
    dt_b = beta * db + 2.0 * b_reg * beta / r - (2.0 / 3.0) * b_reg * div_beta - 2.0 * alpha * b_reg * Ab
    dt_chi = beta * dchi - (1.0 / 6.0) * chi_reg * div_beta + (1.0 / 6.0) * chi_reg * alpha * K
    dt_K = - D2_alpha + alpha * (Ricci + 2.0 * Dm_Zm + K**2 - 2.0 * Theta * K) + beta * dK - 3.0 * alpha * kappa1 * (1.0 + kappa2) * Theta
    dt_Theta = beta * dTheta + 0.5 * alpha * (Ricci + 2.0 * Dm_Zm - (Aa**2 + 2.0 * Ab**2) + (2.0 / 3.0) * K**2 - 2.0 * Theta * K) - Zr_up * dalpha - alpha * kappa1 * (2.0 + kappa2) * Theta
    dt_Aa = beta * dAa - (DrDr_alpha - (1.0 / 3.0) * D2_alpha) + alpha * ((chi_sq / a_reg) * R_rr - (1.0 / 3.0) * Ricci) + alpha * (2.0 * Dr_Zr - (2.0 / 3.0) * Dm_Zm) + alpha * Aa * (K - 2.0 * Theta)
    t1 = beta * dLambda - Lambda * dbeta + (1.0 / a_reg) * ddbeta + (2.0 / b_reg) * (dbeta / r - beta / r**2)
    t2 = (1.0 / 3.0) * ((1.0 / a_reg) * d_div_beta + 2.0 * bar_Lambda * div_beta)
    t3 = - (2.0 / a_reg) * (Aa * dalpha + alpha * dAa)
    t4 = 2.0 * alpha * (Aa * bar_Lambda - (2.0 / (r * b_reg)) * (Aa - Ab))
    t5 = (2.0 * alpha / a_reg) * (dAa - (2.0 / 3.0) * dK - 3.0 * Aa * dchi_chi + (Aa - Ab) * (2.0 / r + db / b_reg))
    t6 = (2.0 / a_reg) * (alpha * dTheta - Theta * dalpha - (2.0 / 3.0) * alpha * K * Zr)
    t7 = (2.0 / a_reg) * ((2.0 / 3.0) * Zr * div_beta - Zr * dbeta) - (2.0 / a_reg) * kappa1 * Zr
    
    dt_Lambda = t1 + t2 + t3 + t4 + t5 + t6 + t7
    
    # A união perfeita: Congela em alpha=0 (evita o negativo) e mantém o Theta (fCCZ4)
    dt_alpha = - 2.0 * alpha * (K - 2.0 * Theta)
    
    dt_beta = B_shift
    dt_B = 0.75 * dt_Lambda - eta_param * B_shift

    return np.array([np.dot(dt_a, inv_psi), np.dot(dt_b, inv_psi), np.dot(dt_chi, inv_psi), np.dot(dt_K, inv_psi), np.dot(dt_Aa, inv_psi), np.dot(dt_Theta, inv_psi), np.dot(dt_Lambda, inv_psi), np.dot(dt_alpha, inv_psi), np.dot(dt_beta, inv_psi), np.dot(dt_B, inv_psi)])

# =========================================================================
# 4. INTEGRADOR RK4 E LOOP PRINCIPAL
# =========================================================================
def passo_rk4(s, h, k1, k2, eta, b):
    k_1 = calcular_taxas_fccz4(s, k1, k2, eta, b)
    k_2 = calcular_taxas_fccz4(s + 0.5*h*k_1, k1, k2, eta, b)
    k_3 = calcular_taxas_fccz4(s + 0.5*h*k_2, k1, k2, eta, b)
    k_4 = calcular_taxas_fccz4(s + h*k_3, k1, k2, eta, b)
    return s + (h/6.0) * (k_1 + 2*k_2 + 2*k_3 + k_4)

def executar_simulacao_com_perfis(L0, N, tf, h):
    b = configurar_base_unica(L0, N)
    s = criar_condicoes_iniciais_fccz4(b)
    k1, k2, eta = 2.0, 0.0, 6.5
    
    print(f"Rodando fCCZ4 - Reta Final (N={N}, dt={h})...")
    
    for i in range(int(tf/h)):
        s = passo_rk4(s, h, k1, k2, eta, b)
        
        # A FAXINA SILENCIOSA: Filtro ERFC aplicado apenas no final do RK4
        for j in range(10):
            s[j] *= b['filtro']
            
        if i % 20000 == 0:
            al_min = 1.0 + np.dot(s[7], b['psi'])[0]
            print(f"Tempo: {i*h:.2f}M | Alpha: {al_min:.5f}")
            
        if np.isnan(s).any() or np.max(np.abs(s)) > 1e11:
            print(f"\nCrash t={i*h:.4f}M.")
            break
            
    r_grid = b['r']
    psi_mat = b['psi']
    alpha_final = 1.0 + np.dot(s[7], psi_mat)
    chi_final = 1.0 + np.dot(s[2], psi_mat)
    X_final = chi_final**2 
    
    return r_grid, alpha_final, X_final

# =========================================================================
# 5. DISPARO E PLOTAGEM DOS PERFIS FINAIS
# =========================================================================
tf_estavel = 5.0
r, alpha, X = executar_simulacao_com_perfis(L0=5.0, N=150, tf=tf_estavel, h=0.00001)

plt.figure(figsize=(12, 5))

# Plot 1: Perfil do Lapso (Alpha)
plt.subplot(1, 2, 1)
plt.plot(r, alpha, 'b-', linewidth=2, label=r'$\alpha(r)$ final')
plt.title(f"Perfil Espacial do Lapso (t={tf_estavel}M)", fontsize=14)
plt.xlabel("Raio isotrópico (r/M)", fontsize=12)
plt.ylabel(r"Fator Lapso ($\alpha$)", fontsize=12)
plt.grid(True)
plt.ylim([-0.05, 1.05])
plt.xlim([0.0, 15.0]) # ZOOM NA FÍSICA REAL
plt.legend()

# Plot 2: Perfil do Fator Conforme Real (X)
plt.subplot(1, 2, 2)
plt.plot(r, X, 'k-', linewidth=2, label=r'$X(r)$ final')
plt.title(f"Perfil Espacial do Fator Conforme (t={tf_estavel}M)", fontsize=14)
plt.xlabel("Raio isotrópico (r/M)", fontsize=12)
plt.ylabel(r"Fator Conforme ($X = \chi^2$)", fontsize=12)
plt.grid(True)
plt.ylim([-0.05, 1.05])
plt.xlim([0.0, 15.0]) # ZOOM NA FÍSICA REAL
plt.legend()

plt.tight_layout()
plt.show()
