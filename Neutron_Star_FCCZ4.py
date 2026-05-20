import numpy as np
import time
import os
import math
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from numpy.polynomial.chebyshev import chebval
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings

warnings.filterwarnings("ignore")

# =========================================================================
# 1. LEITURA DE MATRIZES ESPECTRAIS (O MOTOR DO BURACO NEGRO)
# =========================================================================
pasta_atual = os.path.dirname(os.path.abspath(__file__))

def carregar_arquivo(nome):
    caminho = os.path.join(pasta_atual, nome)
    if not os.path.exists(caminho):
        caminho = caminho + '.txt'
    return np.loadtxt(caminho)

print("A carregar matrizes de alta precisão do Maple...")
L0 = 15.0  # Usaremos o mesmo fator de escala L0=15
p = 200

rcol = L0 * carregar_arquivo('rcol').flatten()
Alalpha = carregar_arquivo('Alalpha') 
DralphaAl = (1.0 / L0) * carregar_arquivo('DralphaAl')
DrralphaAl = (1.0 / L0**2) * carregar_arquivo('DrralphaAl')
Faa = np.copy(Alalpha)
DraFa = DralphaAl
DrraFa = DrralphaAl

eta1_vec = np.arange(1, p + 2) / (p + 1)
filter1 = np.exp(-36.0 * (eta1_vec**20))

# =========================================================================
# 2. SOLVER TOV (CONDIÇÕES INICIAIS DA ESTRELA)
# =========================================================================
def resolver_tov(rho_central, K_poly=100.0, Gamma_poly=2.0, r_max=30.0):
    P_central = K_poly * (rho_central ** Gamma_poly)
    
    def equacoes_tov(r_bar, y):
        m, P, Phi, r_iso = y
        if P <= 1e-14:
            P, rho_0, e = 1e-14, 0.0, 0.0
        else:
            rho_0 = (P / K_poly) ** (1.0 / Gamma_poly)
            e = rho_0 + P / (Gamma_poly - 1.0)
            
        dm_dr = 4.0 * np.pi * (r_bar**2) * e
        if r_bar < 1e-10:
            dP_dr, dPhi_dr, dln_r_iso_dr = 0.0, 0.0, 1.0 / 1e-10
        else:
            numerador_P = (e + P) * (m + 4.0 * np.pi * (r_bar**3) * P)
            denominador_P = r_bar * (r_bar - 2.0 * m)
            dP_dr = - numerador_P / denominador_P
            dPhi_dr = (m + 4.0 * np.pi * (r_bar**3) * P) / (r_bar * (r_bar - 2.0 * m))
            dln_r_iso_dr = 1.0 / (r_bar * np.sqrt(1.0 - 2.0 * m / r_bar))
            
        return [dm_dr, dP_dr, dPhi_dr, r_iso * dln_r_iso_dr]

    y0 = [0.0, P_central, 0.0, 1e-10] 
    sol = solve_ivp(equacoes_tov, [1e-10, r_max], y0, method='RK45', rtol=1e-8, atol=1e-10)
    
    r_bar_sol = sol.t
    m_sol, P_sol, Phi_sol, r_iso_sol = sol.y
    
    superficie_idx = np.argmin(np.abs(P_sol - 1e-14))
    M_total = m_sol[superficie_idx]
    R_estrela_bar = r_bar_sol[superficie_idx]
    
    Phi_superficie_exato = 0.5 * np.log(1.0 - 2.0 * M_total / R_estrela_bar)
    Phi_sol += (Phi_superficie_exato - Phi_sol[superficie_idx])
    alpha_sol = np.exp(Phi_sol)
    
    psi_sol = np.sqrt(r_bar_sol / r_iso_sol)
    psi_superficie_exato = 1.0 + M_total / (2.0 * r_iso_sol[superficie_idx])
    fator_escala_r = (psi_sol[superficie_idx] / psi_superficie_exato)**2
    r_iso_sol *= fator_escala_r
    psi_sol /= np.sqrt(fator_escala_r)
    
    rho_0_sol = np.zeros_like(P_sol)
    rho_0_sol[P_sol > 1e-13] = (P_sol[P_sol > 1e-13] / K_poly) ** (1.0 / Gamma_poly)
    
    print(f"[TOV] Estrela Gerada! Massa M = {M_total:.4f} | Raio R_iso = {r_iso_sol[superficie_idx]:.2f}")
    
    return interp1d(r_iso_sol, alpha_sol, fill_value=(alpha_sol[0], 1.0), bounds_error=False), \
           interp1d(r_iso_sol, psi_sol, fill_value=(psi_sol[0], 1.0), bounds_error=False), \
           interp1d(r_iso_sol, rho_0_sol, fill_value=(rho_0_sol[0], 0.0), bounds_error=False), \
           r_iso_sol[superficie_idx]

# =========================================================================
# 3. KERNEL SPH 1D E INTERPOLAÇÃO ESPECIAL (Mapeamento x = (r-L0)/(r+L0))
# =========================================================================
def kernel_1d(r_ij, h):
    q = np.abs(r_ij) / h
    sigma = 2.0 / (3.0 * h)
    W = np.zeros_like(q)
    m1, m2 = (q <= 1.0), ((q > 1.0) & (q <= 2.0))
    W[m1] = sigma * (1.0 - 1.5 * q[m1]**2 + 0.75 * q[m1]**3)
    W[m2] = sigma * 0.25 * (2.0 - q[m2])**3
    return W

def grad_kernel_1d(r_ij, h):
    q = np.abs(r_ij) / h
    sigma = 2.0 / (3.0 * h**2)
    sinal = np.sign(r_ij)
    grad_W = np.zeros_like(q)
    m1, m2 = (q <= 1.0), ((q > 1.0) & (q <= 2.0))
    grad_W[m1] = sinal[m1] * sigma * (-3.0 * q[m1] + 2.25 * q[m1]**2)
    grad_W[m2] = sinal[m2] * sigma * (-0.75 * (2.0 - q[m2])**2)
    return grad_W

def matriz_interpolacao_particulas(r_part, p, L0):
    """Constrói a matriz Psi nas coordenadas exatas das partículas SPH"""
    x_p = (r_part - L0) / (r_part + L0)
    Psi_p = np.zeros((len(r_part), p + 1))
    for l in range(p + 1):
        c_l = np.zeros(l + 1); c_l[l] = 1.0
        c_lp1 = np.zeros(l + 2); c_lp1[l + 1] = 1.0
        T_l = chebval(x_p, c_l)
        T_lp1 = chebval(x_p, c_lp1)
        Psi_p[:, l] = 0.5 * (T_lp1 - T_l)
    return Psi_p

# =========================================================================
# 4. TERMODINÂMICA E TRANSFERÊNCIA DE MATÉRIA
# =========================================================================
def processar_materia_sph(alpha, a, b, chi, beta, r_part, v_part, massa_part, Psi_p, h_sph, K_p, Gam_p):
    # Projeção da malha para as partículas
    al_coef = Alalpha @ (alpha - 1.0)
    a_coef = Faa @ (a - 1.0)
    b_coef = Faa @ (b - 1.0)
    chi_coef = Alalpha @ (chi - 1.0)
    beta_coef = Alalpha @ beta

    alpha_p = 1.0 + Psi_p @ al_coef
    a_p = 1.0 + Psi_p @ a_coef
    b_p = 1.0 + Psi_p @ b_coef
    chi_p = 1.0 + Psi_p @ chi_coef
    beta_p = Psi_p @ beta_coef
    
    sqrt_gamma_p = np.sqrt(a_p * (b_p**2) / (chi_p**3))
    area_sph = 4.0 * np.pi * (r_part**2 + 1e-4) * sqrt_gamma_p
    
    n_part = len(r_part)
    rho_p = np.zeros(n_part)
    for i in range(n_part):
        r_ij = r_part[i] - r_part
        r_ij_ghost = r_part[i] + r_part # Parede Fantasma em r=0
        soma = np.sum(massa_part * (kernel_1d(r_ij, h_sph) + kernel_1d(r_ij_ghost, h_sph)))
        rho_p[i] = soma / area_sph[i]
        
    P_p = K_p * (rho_p ** Gam_p)
    eps_p = np.zeros_like(P_p)
    mask_rho = rho_p > 1e-12
    eps_p[mask_rho] = P_p[mask_rho] / (rho_p[mask_rho] * (Gam_p - 1.0))
    h_ent_p = 1.0 + eps_p + (P_p / np.maximum(rho_p, 1e-12))
    
    gamma_rr_p = a_p / chi_p
    v2_p = np.clip(gamma_rr_p * (v_part ** 2), 0.0, 0.99)
    W_L_p = 1.0 / np.sqrt(1.0 - v2_p)
    
    E_p = rho_p * h_ent_p * (W_L_p ** 2) - P_p
    Sr_p = rho_p * h_ent_p * (W_L_p ** 2) * gamma_rr_p * v_part
    Srr_p = rho_p * h_ent_p * (W_L_p ** 2) * (gamma_rr_p * v_part)**2 + P_p * gamma_rr_p
    S_tr_p = rho_p * h_ent_p * (W_L_p ** 2) * v2_p + 3.0 * P_p
    
    E_g, Sr_g, Srr_g, S_tr_g = np.zeros(p+1), np.zeros(p+1), np.zeros(p+1), np.zeros(p+1)
    for i in range(p+1):
        distancias = rcol[i] - r_part
        dist_ghost = rcol[i] + r_part
        W_k = kernel_1d(distancias, h_sph) + kernel_1d(dist_ghost, h_sph)
        vol_fator = massa_part / np.maximum(rho_p, 1e-12)
        
        E_g[i] = np.sum(vol_fator * E_p * W_k)
        Sr_g[i] = np.sum(vol_fator * Sr_p * W_k)
        Srr_g[i] = np.sum(vol_fator * Srr_p * W_k)
        S_tr_g[i] = np.sum(vol_fator * S_tr_p * W_k)
        
    return {
        'E_g': E_g, 'Sr_g': Sr_g, 'Srr_g': Srr_g, 'S_tr_g': S_tr_g,
        'alpha_p': alpha_p, 'beta_p': beta_p, 'a_p': a_p, 'chi_p': chi_p,
        'rho_p': rho_p, 'P_p': P_p, 'h_ent_p': h_ent_p, 'W_L_p': W_L_p, 
        'gamma_rr_p': gamma_rr_p, 'area_sph': area_sph
    }
    
def calcular_aceleracao_particulas(r_p, v_p, rho_p, P_p, h_ent_p, W_L_p, gamma_rr_p, area_sph, m_p, h_sph, dalpha_p, da_p, dchi_p, metrica, Gam_p):
    n = len(r_p)
    a_pressao = np.zeros(n)
    a_gravidade = np.zeros(n)
    a_viscosidade = np.zeros(n)
    
    inercia_fisica = rho_p * h_ent_p * (W_L_p**2)
    inercia_1d = inercia_fisica * area_sph 
    c_s = np.sqrt(Gam_p * P_p / (inercia_fisica + 1e-12))
    
    alpha, a_met, chi = metrica['alpha_p'], metrica['a_p'], metrica['chi_p']

    for i in range(n):
        r_ij = r_p[i] - r_p
        r_ij_ghost = r_p[i] + r_p 
        grad_W = grad_kernel_1d(r_ij, h_sph) + grad_kernel_1d(r_ij_ghost, h_sph)
        
        # 1. FORÇA DE PRESSÃO (SPH 1D)
        t_i = P_p[i] / (inercia_1d[i]**2 + 1e-20)
        t_j = P_p / (inercia_1d**2 + 1e-20)
        a_pressao[i] = - area_sph[i] * np.sum(m_p * (t_i + t_j) * grad_W)
        
        # 2. VISCOSIDADE ARTIFICIAL 
        v_ij = v_p[i] - v_p
        dot_vr = v_ij * r_ij
        Pi_ij = np.zeros(n)
        mask_app = dot_vr < 0 
        if np.any(mask_app):
            mu_ij = (h_sph * dot_vr[mask_app]) / (r_ij[mask_app]**2 + 0.01 * h_sph**2)
            rho_1d_ij = 0.5 * (rho_p[i]*area_sph[i] + rho_p[mask_app]*area_sph[mask_app])
            cs_ij = 0.5 * (c_s[i] + c_s[mask_app])
            Pi_ij[mask_app] = (-1.0 * cs_ij * mu_ij + 2.0 * mu_ij**2) / rho_1d_ij
            
        a_viscosidade[i] = - area_sph[i] * np.sum(m_p * Pi_ij * grad_W)

        # 3. GRAVIDADE EXATA (Cinética)
        d_gamma_rr_i = (da_p[i] / chi[i]) - (a_met[i] * dchi_p[i] / chi[i]**2)
        E_kin = inercia_fisica[i]
        S_rr_kin = inercia_fisica[i] * (gamma_rr_p[i] * v_p[i])**2
        S_geom = - E_kin * dalpha_p[i] + 0.5 * alpha[i] * (S_rr_kin * d_gamma_rr_i)
        a_gravidade[i] = (S_geom / (inercia_fisica[i] + 1e-14))

    return a_pressao + a_gravidade + a_viscosidade - 0.01 * v_p

# =========================================================================
# 5. RHS Z4c HÍBRIDO (ESTRELA NO ESPAÇO Z4c DO BURACO NEGRO)
# =========================================================================
def rhs_hibrido(alpha, K, chi, a, b, Delta, Aa, beta, B, Z, Theta, r_part, v_part, m_part, Psi_p, k1, k2, xi, eta0, f0, nc, h_sph, K_p, Gam_p, aplicar_filtro=False):
    
    al_coef = Alalpha @ (alpha - 1.0)
    dralpha = DralphaAl @ al_coef
    drralpha = DrralphaAl @ al_coef

    c_coef = Alalpha @ (chi - 1.0)
    if aplicar_filtro: c_coef = filter1 * c_coef
    drchi = DralphaAl @ c_coef
    drrchi = DrralphaAl @ c_coef

    be_coef = Alalpha @ beta
    drbeta = DralphaAl @ be_coef
    drrbeta = DrralphaAl @ be_coef

    fa_coef = Faa @ (a - 1.0)
    dra = DraFa @ fa_coef
    drra = DrraFa @ fa_coef

    fb_coef = Faa @ (b - 1.0)
    drb = DraFa @ fb_coef
    drrb = DrraFa @ fb_coef

    cK_coef = Alalpha @ K
    if aplicar_filtro: cK_coef = filter1 * cK_coef
    drK = DralphaAl @ cK_coef

    f_coef = Alalpha @ Aa
    drAa = DralphaAl @ f_coef
    Del_coef = Alalpha @ Delta
    drDelta = DralphaAl @ Del_coef
    ze_coef = Alalpha @ Z
    drZ = DralphaAl @ ze_coef
    th_coef = Alalpha @ Theta
    drTheta = DralphaAl @ th_coef

    # 1. Acoplamento de Matéria (SPH)
    mat = processar_materia_sph(alpha, a, b, chi, beta, r_part, v_part, m_part, Psi_p, h_sph, K_p, Gam_p)
    E_g = mat['E_g']
    Sr_g = mat['Sr_g']
    Srr_g = mat['Srr_g']
    S_tr_g = mat['S_tr_g']

    # 2. Equações Geométricas Z4c (do Professor)
    dalpha_dt = -2.0 * alpha * K
    Divbeta = drbeta + beta * (0.5 * dra / a + drb / b) + 2.0 * beta / rcol
    dchi_dt = beta * drchi - (nc / 6.0) * chi * Divbeta + (nc / 6.0) * alpha * K * chi
    da_dt = beta * dra + 2.0 * a * drbeta - 2.0 * a * Divbeta / 3.0 - 2.0 * alpha * a * Aa
    db_dt = beta * drb + 2.0 * b * beta / rcol - 2.0 * b * Divbeta / 3.0 + alpha * b * Aa

    Zc = chi**(4.0 / nc) * Z / a
    divZ = chi**(4.0 / nc) * drZ / a + (-0.5 * dra / a + drb / b + 2.0 / rcol - (2.0 / nc) * drchi / chi) * Zc
    Lapalpha = chi**(4.0 / nc) / a * (drralpha - dralpha * (0.5 * dra / a + (2.0 / nc) * drchi / chi - drb / b - 2.0 / rcol))

    # Injeção de Matéria: O Trace de S é S_tr_g
    dK_dt = beta * drK - Lapalpha + (1.0 / 3.0) * alpha * K**2 + 1.5 * alpha * Aa**2 - 2.0 * alpha * K * Theta \
            - 3.0 * k1 * (1.0 + k2) * alpha * Theta + 2.0 * alpha * divZ + 4.0 * np.pi * alpha * (E_g + S_tr_g)

    TrLapalpha = (1.0 / 3.0) * chi**(4.0 / nc) / a * (2.0 * drralpha - dralpha * (-(8.0 / nc) * drchi / chi + dra / a + drb / b) - 2.0 * dralpha / rcol)

    TrRicci = chi**(4.0 / nc) / a * (
        (2.0 / 3.0) * a * drDelta - (2.0 / 3.0 / nc) * (2.0 / rcol + drb / b + dra / a) * drchi / chi
        - (1.0 / 3.0) * drra / a + (4.0 / 3.0 / nc) * drrchi / chi + (4.0 / 3.0 / nc**2) * (2.0 - nc) * drchi**2 / chi**2
        + (2.0 / rcol**2) * (1.0 - a / b) * (-1.0 / 3.0 - rcol * drb / b) + 0.5 * dra * Delta + (5.0 / 12.0) * dra**2 / a**2
        - dra / b / rcol - (1.0 / 3.0) * drb**2 / b**2 + (1.0 / 3.0) * drrb / b
        + (2.0 / 3.0) * drb / b / rcol * (3.0 - a / b)
    )

    S_rr_TF = Srr_g - (1.0 / 3.0) * a * S_tr_g
    dAa_dt = beta * drAa - TrLapalpha + alpha * TrRicci + alpha * K * Aa - 2.0 * alpha * Theta * Aa \
             + 2.0 * alpha * (chi**(4.0 / nc) * drZ / a - (0.5 * dra / a - (2.0 / nc) * drchi / chi) * Zc - (1.0 / 3.0) * divZ) \
             - 8.0 * np.pi * alpha * chi * S_rr_TF

    drDivbeta = drrbeta + drbeta * (0.5 * dra / a + drb / b) + beta * (0.5 * drra / a - 0.5 * dra**2 / a**2 + drrb / b - drb**2 / b**2) + 2.0 * (drbeta - beta / rcol) / rcol

    dDelta_dt = beta * drDelta - Delta * drbeta + drrbeta / a + 2.0 * (drbeta - beta / rcol) / (b * rcol) + (2.0 / 3.0) * Delta * Divbeta \
                + (1.0 / 3.0) * drDivbeta / a - (2.0 / a) * (Aa * dralpha + alpha * drAa) + 2.0 * alpha * (Aa * Delta - 3.0 * Aa / b / rcol) \
                + xi * alpha / a * (-(6.0 / nc) * Aa * drchi / chi - (2.0 / 3.0) * drK + drAa + 1.5 * (2.0 / rcol + drb / b) * Aa) \
                - 16.0 * np.pi * alpha * (Sr_g / a)

    R_curv = -(1.0 / a) * (chi**(4.0 / nc)) * (
        0.5 * drra / a + drrb / b + 0.5 * drb**2 / b**2 - a * drDelta - dra**2 / a**2
        - (8.0 / nc) * drrchi / chi + 8.0 * (1.0 + nc) / nc**2 * drchi**2 / chi**2
        + 4.0 / rcol**2 * (1.0 - a / b) + 2.0 / rcol * (3.0 - a / b) * drb / b
        + (8.0 / nc) * drchi / chi * (-2.0 / rcol - drb / b + 0.5 * dra / a)
    )

    Hc = R_curv + (2.0 / 3.0) * K**2 - 1.5 * Aa**2
    dTheta_dt = beta * drTheta + 0.5 * alpha * Hc - alpha * K * Theta - Zc * dralpha + alpha * divZ - k1 * (k2 + 2.0) * alpha * Theta - 8.0 * np.pi * alpha * E_g
    
    Mc = -(2.0 / 3.0) * drK + drAa - (6.0 / nc) * Aa * drchi / chi + 1.5 * Aa * (2.0 / rcol + drb / b)
    dZ_dt = beta * drZ + Z * drbeta + alpha * Mc - 2.0 * alpha * Z * Aa - (2.0 / 3.0) * alpha * K * Z + alpha * drTheta - Theta * dralpha - k1 * alpha * Z
    
    dbeta_dt = f0 * B
    dB_dt = dDelta_dt - eta0 * B

    # 3. Movimento SPH
    dt_r_p = mat['alpha_p'] * v_part - mat['beta_p']
    dalpha_p = Psi_p @ dralpha
    da_p = Psi_p @ dra
    dchi_p = Psi_p @ drchi
    
    dt_v_p = calcular_aceleracao_particulas(
        r_part, v_part, mat['rho_p'], mat['P_p'], mat['h_ent_p'], 
        mat['W_L_p'], mat['gamma_rr_p'], mat['area_sph'], 
        m_part, h_sph, dalpha_p, da_p, dchi_p, mat, Gam_p
    )

    return (dalpha_dt, dK_dt, dchi_dt, da_dt, db_dt, dDelta_dt, dAa_dt, dbeta_dt, dB_dt, dTheta_dt, dZ_dt, dt_r_p, dt_v_p)

# =========================================================================
# 6. INICIALIZAÇÃO DA ESTRELA
# =========================================================================
def iniciar_estrela_hibrida(N_part, rho_c, K_p, Gam_p):
    interp_alp, interp_psi, interp_rho, R_star = resolver_tov(rho_c, K_p, Gam_p)
    
    # Mapear variáveis TOV para o grid rcol do Z4c (p=200)
    chi = interp_psi(rcol)**(-4.0)
    a, b = np.ones_like(rcol), np.ones_like(rcol)
    alpha = interp_alp(rcol)
    zeros = np.zeros_like(rcol)
    K, Delta, Aa, beta, B, Z, Theta = zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()
    
    r_p = np.linspace(1e-3, R_star * 0.95, N_part)
    dr_p = r_p[1] - r_p[0]
    h_sph = 2.5 * dr_p
    v_p = np.zeros(N_part)
    
    rho_p = interp_rho(r_p)
    psi_p = interp_psi(r_p)
    m_p = rho_p * (4.0 * np.pi * (r_p**2) * (psi_p**6.0) * dr_p)
    
    Psi_p = matriz_interpolacao_particulas(r_p, p, L0)
    
    return alpha, K, chi, a, b, Delta, Aa, beta, B, Z, Theta, r_p, v_p, m_p, Psi_p, h_sph

# =========================================================================
# 7. MOTOR PRINCIPAL (RK4 COM VÍDEO)
# =========================================================================
if __name__ == "__main__":
    print("---------------------------------------------------------")
    print(" RELATIVIDADE NUMÉRICA: ESTRELA DE NEUTRÕES Z4c + SPH    ")
    print("---------------------------------------------------------")
    
    N_part = 200
    h_tempo = 0.0005
    passos = 1000
    passos_salvar = 10 
    
    k1, k2, xi, eta0, f0, nc = 0.1, 0.0, 2.0, 2.0, 3.0/4.0, 2.0
    K_poly, Gamma_poly = 100.0, 2.0
    rho_central = 0.00128
    
    alpha, K, chi, a, b, Delta, Aa, beta, B, Z, Theta, r_p, v_p, m_p, Psi_p, h_sph = iniciar_estrela_hibrida(N_part, rho_central, K_poly, Gamma_poly)
    
    frames_rho = []
    frames_tempo = []
    frames_r_p = []

    print("\nA evoluir RK4 Z4c...")
    t = 0.0
    for i in range(passos + 1):
        
        # O integrador agora recebe todas as variaveis separadas da malha BSSN
        da1, dK1, dc1, daa1, db1, dD1, dAa1, dbe1, dB1, dTh1, dZ1, dr1, dv1 = rhs_hibrido(
            alpha, K, chi, a, b, Delta, Aa, beta, B, Z, Theta, r_p, v_p, m_p, Psi_p, k1, k2, xi, eta0, f0, nc, h_sph, K_poly, Gamma_poly)

        da2, dK2, dc2, daa2, db2, dD2, dAa2, dbe2, dB2, dTh2, dZ2, dr2, dv2 = rhs_hibrido(
            alpha + 0.5*h_tempo*da1, K + 0.5*h_tempo*dK1, chi + 0.5*h_tempo*dc1, a + 0.5*h_tempo*daa1, b + 0.5*h_tempo*db1, 
            Delta + 0.5*h_tempo*dD1, Aa + 0.5*h_tempo*dAa1, beta + 0.5*h_tempo*dbe1, B + 0.5*h_tempo*dB1, 
            Z + 0.5*h_tempo*dZ1, Theta + 0.5*h_tempo*dTh1, r_p + 0.5*h_tempo*dr1, v_p + 0.5*h_tempo*dv1, m_p, Psi_p, k1, k2, xi, eta0, f0, nc, h_sph, K_poly, Gamma_poly)

        da3, dK3, dc3, daa3, db3, dD3, dAa3, dbe3, dB3, dTh3, dZ3, dr3, dv3 = rhs_hibrido(
            alpha + 0.5*h_tempo*da2, K + 0.5*h_tempo*dK2, chi + 0.5*h_tempo*dc2, a + 0.5*h_tempo*daa2, b + 0.5*h_tempo*db2, 
            Delta + 0.5*h_tempo*dD2, Aa + 0.5*h_tempo*dAa2, beta + 0.5*h_tempo*dbe2, B + 0.5*h_tempo*dB2, 
            Z + 0.5*h_tempo*dZ2, Theta + 0.5*h_tempo*dTh2, r_p + 0.5*h_tempo*dr2, v_p + 0.5*h_tempo*dv2, m_p, Psi_p, k1, k2, xi, eta0, f0, nc, h_sph, K_poly, Gamma_poly)

        da4, dK4, dc4, daa4, db4, dD4, dAa4, dbe4, dB4, dTh4, dZ4, dr4, dv4 = rhs_hibrido(
            alpha + h_tempo*da3, K + h_tempo*dK3, chi + h_tempo*dc3, a + h_tempo*daa3, b + h_tempo*db3, 
            Delta + h_tempo*dD3, Aa + h_tempo*dAa3, beta + h_tempo*dbe3, B + h_tempo*dB3, 
            Z + h_tempo*dZ3, Theta + h_tempo*dTh3, r_p + h_tempo*dr3, v_p + h_tempo*dv3, m_p, Psi_p, k1, k2, xi, eta0, f0, nc, h_sph, K_poly, Gamma_poly, aplicar_filtro=True)

        alpha += (h_tempo / 6.0) * (da1 + 2.0*da2 + 2.0*da3 + da4)
        K += (h_tempo / 6.0) * (dK1 + 2.0*dK2 + 2.0*dK3 + dK4)
        chi += (h_tempo / 6.0) * (dc1 + 2.0*dc2 + 2.0*dc3 + dc4)
        a += (h_tempo / 6.0) * (daa1 + 2.0*daa2 + 2.0*daa3 + daa4)
        b += (h_tempo / 6.0) * (db1 + 2.0*db2 + 2.0*db3 + db4)
        Delta += (h_tempo / 6.0) * (dD1 + 2.0*dD2 + 2.0*dD3 + dD4)
        Aa += (h_tempo / 6.0) * (dAa1 + 2.0*dAa2 + 2.0*dAa3 + dAa4)
        beta += (h_tempo / 6.0) * (dbe1 + 2.0*dbe2 + 2.0*dbe3 + dbe4)
        B += (h_tempo / 6.0) * (dB1 + 2.0*dB2 + 2.0*dB3 + dB4)
        Theta += (h_tempo / 6.0) * (dTh1 + 2.0*dTh2 + 2.0*dTh3 + dTh4)
        Z += (h_tempo / 6.0) * (dZ1 + 2.0*dZ2 + 2.0*dZ3 + dZ4)
        
        r_p += (h_tempo / 6.0) * (dr1 + 2.0*dr2 + 2.0*dr3 + dr4)
        v_p += (h_tempo / 6.0) * (dv1 + 2.0*dv2 + 2.0*dv3 + dv4)
        
        # Bater na parede r=0
        mask_centro = r_p < 0
        if np.any(mask_centro):
            r_p[mask_centro] = -r_p[mask_centro]
            v_p[mask_centro] = -v_p[mask_centro]
            
        # Atualizar malha de interpolacao com a nova posicao das particulas
        Psi_p = matriz_interpolacao_particulas(r_p, p, L0)

        t += h_tempo
        
        if i % passos_salvar == 0 or i == passos:
            rho_calc = np.zeros_like(r_p)
            for j in range(len(r_p)):
                rho_calc[j] = np.sum(m_p * kernel_1d(r_p[j] - r_p, h_sph)) / (4.0 * np.pi * (r_p[j]**2 + 1e-12))
            
            frames_rho.append(rho_calc)
            frames_tempo.append(t)
            frames_r_p.append(np.copy(r_p))
            print(f"Progresso: {t:.4f}M")

    print("\nSimulação Concluída! A compilar o vídeo...")

    fig, ax = plt.subplots(figsize=(8, 5))
    linha_rho, = ax.plot([], [], 'ro', ms=4, alpha=0.8, label='Partículas SPH')
    
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.0001, rho_central * 1.2)
    ax.set_xlabel("Raio Isotrópico (r/M)", fontsize=12)
    ax.set_ylabel(r"Densidade de Repouso ($\rho_0$)", fontsize=12)
    titulo = ax.set_title("Estrela SPH no Z4c (Base Racional)", fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    def init():
        linha_rho.set_data([], [])
        return linha_rho, titulo

    def animar(i):
        linha_rho.set_data(frames_r_p[i], frames_rho[i])
        titulo.set_text(f"Estrela Z4c+SPH | Tempo = {frames_tempo[i]:.4f} M")
        return linha_rho, titulo

    ani = animation.FuncAnimation(fig, animar, frames=len(frames_tempo), init_func=init, blit=False, interval=50)

    try:
        ani.save("Estrela_Z4c.mp4", writer='ffmpeg', fps=20, dpi=200)
        print("Sucesso! 'Estrela_Z4c.mp4' gerado.")
    except Exception as e:
        ani.save("Estrela_Z4c.gif", writer='pillow', fps=20)
        print("Sucesso! 'Estrela_Z4c.gif' gerado.")

    plt.show()
