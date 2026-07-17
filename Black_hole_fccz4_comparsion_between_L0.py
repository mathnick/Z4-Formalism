# =========================================================================
# SIMULATION ENGINE: fCCZ4 (Mode 1) | Maximal Slicing 
# Scanning Map Parameter (L0)
# =========================================================================
import numpy as np
import time
import os
import matplotlib.pyplot as plt

pasta_atual = os.path.dirname(os.path.abspath(__file__))
pasta_resultados = os.path.join(pasta_atual, "Resultados_Comparativos")
if not os.path.exists(pasta_resultados):
    os.makedirs(pasta_resultados)

p = 200
k1_fixo = 25.0  # Mantemos o kappa_1 que funcionou melhor
k2 = 0.0
tf = 40.0
h = 0.0002
idump = 500

xi, eta0, f0, nc = 2.0, 2.0, 3.0/4.0, 2.0

def carregar_matriz(nome):
    caminhos = [
        os.path.join(pasta_atual, f"{nome}_{p}.txt"),
        os.path.join(pasta_atual, f"{nome}_{p}"),
        os.path.join(pasta_atual, f"{nome}.txt"),
        os.path.join(pasta_atual, f"{nome}")
    ]
    for caminho in caminhos:
        if os.path.exists(caminho):
            return np.loadtxt(caminho)
    raise FileNotFoundError(f"\nERRO: Matriz {nome}_{p}.txt nao encontrada! Rode o gerador para N={p}.")

print(f"Loading base spectral matrices for N = {p}...")
base_rcol = carregar_matriz('rcol').flatten()
base_rqcol = carregar_matriz('rqcol').flatten()
wcolq = carregar_matriz('Wcolq').flatten()

AlphaAl = carregar_matriz('AlphaAl')
Alalpha = carregar_matriz('Alalpha') 
Faa = np.copy(Alalpha)
base_DralphaAl = carregar_matriz('DralphaAl')
base_DrralphaAl = carregar_matriz('DrralphaAl')
ChiqC = carregar_matriz('PhiqA')
base_DrchiqC = carregar_matriz('DrphiqA')
base_DrrchiqC = carregar_matriz('DrrphiqA')

Nq = int((3.0 / 2.0) * p)
eta1_vec = np.arange(1, p + 2) / (p + 1)
filter1 = np.exp(-36.0 * (eta1_vec**20))

def executar_simulacao_L0(L0_val):
    print(f"\n---> Running Simulation | fCCZ4 Mode 1 | L0 = {L0_val} | K1 = {k1_fixo} <---")
    
    # ---------------------------------------------------------
    # ESCALAMENTO DINÂMICO DA MALHA PELO L0
    # ---------------------------------------------------------
    rcol = L0_val * base_rcol
    rqcol = L0_val * base_rqcol
    DralphaAl = (1.0 / L0_val) * base_DralphaAl
    DrralphaAl = (1.0 / L0_val**2) * base_DrralphaAl
    DraFa = DralphaAl
    DrraFa = DrralphaAl
    DrchiqC = (1.0 / L0_val) * base_DrchiqC
    DrrchiqC = (1.0 / L0_val**2) * base_DrrchiqC

    # Initial Conditions (Buraco Negro no Vácuo / Brill-Lindquist)
    alpha = (1.0 + 1.0 / (2.0 * rcol))**(-2)
    K = np.zeros(p + 1)
    chi = (1.0 + 1.0 / (2.0 * rcol))**(-nc)
    a = np.ones(p + 1)
    b = np.ones(p + 1)
    Delta = np.zeros(p + 1)
    Lam = np.zeros(p + 1) 
    Aa = np.zeros(p + 1)
    beta = np.zeros(p + 1)
    B = np.zeros(p + 1)
    Z = np.zeros(p + 1)
    Theta = np.zeros(p + 1)

    def evaluate_rhs(alpha_val, K_val, chi_val, a_val, b_val, Delta_val, Aa_val, beta_val, B_val, Z_val, Theta_val, Lam_val, aplicar_filtro=False):
        c_coef = Alalpha @ (chi_val - 1.0)
        if aplicar_filtro: c_coef = filter1 * c_coef
        drchi = DralphaAl @ c_coef
        drrchi = DrralphaAl @ c_coef

        be_coef = Alalpha @ beta_val
        drbeta = DralphaAl @ be_coef
        drrbeta = DrralphaAl @ be_coef

        fa_coef = Faa @ (a_val - 1.0)
        dra = DraFa @ fa_coef
        drra = DrraFa @ fa_coef

        fb_coef = Faa @ (b_val - 1.0)
        drb = DraFa @ fb_coef
        drrb = DrraFa @ fb_coef

        f_coef = Alalpha @ Aa_val
        drAa = DralphaAl @ f_coef
        Del_coef = Alalpha @ Delta_val
        drDelta = DralphaAl @ Del_coef
        th_coef = Alalpha @ Theta_val
        drTheta = DralphaAl @ th_coef
        lam_coef = Alalpha @ Lam_val
        drLam = DralphaAl @ lam_coef

        # Variáveis do fCCZ4 Mode 1
        ze_coef = Alalpha @ Z_val
        drZ = DralphaAl @ ze_coef
        conn_for_R = Lam_val
        dr_conn_for_R = drLam

        Zc = chi_val**(4.0 / nc) * Z_val / a_val
        divZ = chi_val**(4.0 / nc) * drZ / a_val + (-0.5 * dra / a_val + drb / b_val + 2.0 / rcol - (2.0 / nc) * drchi / chi_val) * Zc

        K_val = np.zeros_like(K_val)
        drK = np.zeros_like(K_val)
        cK_coef = np.zeros_like(K_val)

        Gamma_term = 0.5 * dra / a_val + (2.0 / nc) * drchi / chi_val - drb / b_val - 2.0 / rcol
        S_term = 1.5 * Aa_val**2 - 3.0 * k1_fixo * (1.0 + k2) * Theta_val + 2.0 * divZ

        Q_vec = - (a_val / chi_val**(4.0 / nc)) * S_term
        Matrix_Alpha = DrralphaAl - Gamma_term[:, np.newaxis] * DralphaAl + Q_vec[:, np.newaxis] * AlphaAl
        RHS_Alpha = -Q_vec

        al_coef = np.linalg.solve(Matrix_Alpha, RHS_Alpha)
        alpha_val = 1.0 + AlphaAl @ al_coef
        dralpha = DralphaAl @ al_coef
        drralpha = DrralphaAl @ al_coef

        dalpha_dt = np.zeros_like(alpha_val)
        dK_dt = np.zeros_like(K_val)

        Divbeta = drbeta + beta_val * (0.5 * dra / a_val + drb / b_val) + 2.0 * beta_val / rcol
        dchi_dt = beta_val * drchi - (nc / 6.0) * chi_val * Divbeta + (nc / 6.0) * alpha_val * K_val * chi_val
        da_dt = beta_val * dra + 2.0 * a_val * drbeta - 2.0 * a_val * Divbeta / 3.0 - 2.0 * alpha_val * a_val * Aa_val
        db_dt = beta_val * drb + 2.0 * b_val * beta_val / rcol - 2.0 * b_val * Divbeta / 3.0 + alpha_val * b_val * Aa_val

        TrLapalpha = (1.0 / 3.0) * chi_val**(4.0 / nc) / a_val * (2.0 * drralpha - dralpha * (-(8.0 / nc) * drchi / chi_val + dra / a_val + drb / b_val) - 2.0 * dralpha / rcol)
        
        TrRicci = chi_val**(4.0 / nc) / a_val * (
            (2.0 / 3.0) * a_val * dr_conn_for_R - (2.0 / 3.0 / nc) * (2.0 / rcol + drb / b_val + dra / a_val) * drchi / chi_val
            - (1.0 / 3.0) * drra / a_val + (4.0 / 3.0 / nc) * drrchi / chi_val + (4.0 / 3.0 / nc**2) * (2.0 - nc) * drchi**2 / chi_val**2
            + (2.0 / rcol**2) * (1.0 - a_val / b_val) * (-1.0 / 3.0 - rcol * drb / b_val) + 0.5 * dra * conn_for_R + (5.0 / 12.0) * dra**2 / a_val**2
            - dra / b_val / rcol - (1.0 / 3.0) * drb**2 / b_val**2 + (1.0 / 3.0) * drrb / b_val
            + (2.0 / 3.0) * drb / b_val / rcol * (3.0 - a_val / b_val)
        )

        drDivbeta = drrbeta + drbeta * (0.5 * dra / a_val + drb / b_val) + beta_val * (0.5 * drra / a_val - 0.5 * dra**2 / a_val**2 + drrb / b_val - drb**2 / b_val**2) + 2.0 * (drbeta - beta_val / rcol) / rcol
        
        def adveccao_conexao(C_val, drC):
            return beta_val * drC - C_val * drbeta + drrbeta / a_val + 2.0 * (drbeta - beta_val / rcol) / (b_val * rcol) + (2.0 / 3.0) * C_val * Divbeta \
                   + (1.0 / 3.0) * drDivbeta / a_val - (2.0 / a_val) * (Aa_val * dralpha + alpha_val * drAa)

        fonte_conexao = xi * alpha_val / a_val * (-(6.0 / nc) * Aa_val * drchi / chi_val - (2.0 / 3.0) * drK + drAa + 1.5 * (2.0 / rcol + drb / b_val) * Aa_val)

        dDelta_dt = np.zeros_like(Delta_val)
        dLam_dt = adveccao_conexao(Lam_val, drLam) + 2.0 * alpha_val * (Aa_val * Lam_val - 3.0 * Aa_val / b_val / rcol) + fonte_conexao
        dB_dt = dLam_dt - eta0 * B_val
        dAa_dt = beta_val * drAa - TrLapalpha + alpha_val * TrRicci + alpha_val * K_val * Aa_val - 2.0 * alpha_val * Theta_val * Aa_val \
                 + 2.0 * alpha_val * (chi_val**(4.0 / nc) * drZ / a_val - (0.5 * dra / a_val - (2.0 / nc) * drchi / chi_val) * Zc - (1.0 / 3.0) * divZ)
        
        R_val = -(1.0 / a_val) * (chi_val**(4.0 / nc)) * (
            0.5 * drra / a_val + drrb / b_val + 0.5 * drb**2 / b_val**2 - a_val * drLam - dra**2 / a_val**2
            - (8.0 / nc) * drrchi / chi_val + 8.0 * (1.0 + nc) / nc**2 * drchi**2 / chi_val**2
            + 4.0 / rcol**2 * (1.0 - a_val / b_val) + 2.0 / rcol * (3.0 - a_val / b_val) * drb / b_val
            + (8.0 / nc) * drchi / chi_val * (-2.0 / rcol - drb / b_val + 0.5 * dra / a_val)
        )
        Hc = R_val + (2.0 / 3.0) * K_val**2 - 1.5 * Aa_val**2
        dTheta_dt = beta_val * drTheta + 0.5 * alpha_val * Hc - alpha_val * K_val * Theta_val - Zc * dralpha + alpha_val * divZ - k1_fixo * (k2 + 2.0) * alpha_val * Theta_val
        Mc = -(2.0 / 3.0) * drK + drAa - (6.0 / nc) * Aa_val * drchi / chi_val + 1.5 * Aa_val * (2.0 / rcol + drb / b_val)
        dZ_dt = beta_val * drZ + Z_val * drbeta + alpha_val * Mc - 2.0 * alpha_val * Z_val * Aa_val - (2.0 / 3.0) * alpha_val * K_val * Z_val + alpha_val * drTheta - Theta_val * dralpha - k1_fixo * alpha_val * Z_val

        dbeta_dt = f0 * B_val

        return (dalpha_dt, dK_dt, dchi_dt, da_dt, db_dt, dDelta_dt, dAa_dt, dbeta_dt, dB_dt, dTheta_dt, dZ_dt, dLam_dt, 
                al_coef, c_coef, be_coef, cK_coef, f_coef, fa_coef, fb_coef, Del_coef)

    t, niter = 0.0, 0
    Time_data = []
    L2HC_data = []

    while t <= tf:
        da1, dK1, dc1, daa1, db1, dD1, dAa1, dbe1, dB1, dTh1, dZ1, dL1, al_c1, c_c1, be_c1, cK_c1, f_c1, fa_c1, fb_c1, Del_c1 = evaluate_rhs(
            alpha, K, chi, a, b, Delta, Aa, beta, B, Z, Theta, Lam, aplicar_filtro=False)
        da2, dK2, dc2, daa2, db2, dD2, dAa2, dbe2, dB2, dTh2, dZ2, dL2, *_ = evaluate_rhs(
            alpha + 0.5*h*da1, K + 0.5*h*dK1, chi + 0.5*h*dc1, a + 0.5*h*daa1, b + 0.5*h*db1, 
            Delta + 0.5*h*dD1, Aa + 0.5*h*dAa1, beta + 0.5*h*dbe1, B + 0.5*h*dB1, 
            Z + 0.5*h*dZ1, Theta + 0.5*h*dTh1, Lam + 0.5*h*dL1, aplicar_filtro=False)
        da3, dK3, dc3, daa3, db3, dD3, dAa3, dbe3, dB3, dTh3, dZ3, dL3, *_ = evaluate_rhs(
            alpha + 0.5*h*da2, K + 0.5*h*dK2, chi + 0.5*h*dc2, a + 0.5*h*daa2, b + 0.5*h*db2, 
            Delta + 0.5*h*dD2, Aa + 0.5*h*dAa2, beta + 0.5*h*dbe2, B + 0.5*h*dB2, 
            Z + 0.5*h*dZ2, Theta + 0.5*h*dTh2, Lam + 0.5*h*dL2, aplicar_filtro=False)
        da4, dK4, dc4, daa4, db4, dD4, dAa4, dbe4, dB4, dTh4, dZ4, dL4, *_ = evaluate_rhs(
            alpha + h*da3, K + h*dK3, chi + h*dc3, a + h*daa3, b + h*db3, 
            Delta + h*dD3, Aa + h*dAa3, beta + h*dbe3, B + h*dB3, 
            Z + h*dZ3, Theta + h*dTh3, Lam + h*dL3, aplicar_filtro=True)

        alpha += (h / 6.0) * (da1 + 2.0*da2 + 2.0*da3 + da4)
        K += (h / 6.0) * (dK1 + 2.0*dK2 + 2.0*dK3 + dK4)
        chi += (h / 6.0) * (dc1 + 2.0*dc2 + 2.0*dc3 + dc4)
        a += (h / 6.0) * (daa1 + 2.0*daa2 + 2.0*daa3 + daa4)
        b += (h / 6.0) * (db1 + 2.0*db2 + 2.0*db3 + db4)
        Delta += (h / 6.0) * (dD1 + 2.0*dD2 + 2.0*dD3 + dD4)
        Aa += (h / 6.0) * (dAa1 + 2.0*dAa2 + 2.0*dAa3 + dAa4)
        beta += (h / 6.0) * (dbe1 + 2.0*dbe2 + 2.0*dbe3 + dbe4)
        B += (h / 6.0) * (dB1 + 2.0*dB2 + 2.0*dB3 + dB4)
        Theta += (h / 6.0) * (dTh1 + 2.0*dTh2 + 2.0*dTh3 + dTh4)
        Z += (h / 6.0) * (dZ1 + 2.0*dZ2 + 2.0*dZ3 + dZ4)
        Lam += (h / 6.0) * (dL1 + 2.0*dL2 + 2.0*dL3 + dL4)

        if niter % idump == 0:
            Kq = ChiqC @ cK_c1
            Aaq = ChiqC @ f_c1
            aq = ChiqC @ fa_c1 + np.ones(Nq + 1)
            draq = DrchiqC @ fa_c1
            drraq = DrrchiqC @ fa_c1
            bq = ChiqC @ fb_c1 + np.ones(Nq + 1)
            drbq = DrchiqC @ fb_c1
            drrbq = DrrchiqC @ fb_c1
            chiq = ChiqC @ c_c1 + np.ones(Nq + 1)
            drchiq = DrchiqC @ c_c1
            drrchiq = DrrchiqC @ c_c1
            
            dr_conn_q = DrchiqC @ (Alalpha @ Lam)

            Rq = - (1.0 / aq) * (chiq**(4.0 / nc)) * (
                0.5 * drraq / aq + drrbq / bq + 0.5 * drbq**2 / bq**2 - aq * dr_conn_q - draq**2 / aq**2
                - (8.0 / nc) * drrchiq / chiq + 8.0 * (1.0 + nc) / nc**2 * drchiq**2 / chiq**2
                + 4.0 / rqcol**2 * (1.0 - aq / bq) + 2.0 / rqcol * (3.0 - aq / bq) * drbq / bq
                + (8.0 / nc) * drchiq / chiq * (-2.0 / rqcol - drbq / bq + 0.5 * draq / aq)
            )
            HCq = Rq - 1.5 * Aaq**2 + (2.0 / 3.0) * Kq**2
            L2HC = np.sqrt(0.5 * np.dot(HCq**2, wcolq))

            Time_data.append(t)
            L2HC_data.append(L2HC)
            
            if np.isnan(L2HC) or L2HC > 1.0:
                print(f"     [ALERTA] Violação de restrição explodiu (L2HC = {L2HC:.2e}) em t={t:.2f}M! Interrompendo simulação.")
                break
                
        t += h
        niter += 1

    print(f"     -> Concluído! Último L2HC = {L2HC_data[-1]:.2e}")
    return Time_data, L2HC_data

# =========================================================================
# 4. VARREDURA DOS PARÂMETROS E PLOTAGEM ÚNICA
# =========================================================================

L0_values = [5.0, 10.0, 15.0, 20.0]
resultados_L0 = {}

for L0_test in L0_values:
    t_dados, hc_dados = executar_simulacao_L0(L0_test)
    resultados_L0[L0_test] = {'t': t_dados, 'hc': hc_dados}

print("\nGerando Gráfico Comparativo para Varredura de L0...")

plt.figure(figsize=(10, 7))

cores_estilos = {
    5.0: ('r-', 1.5),
    10.0: ('g-', 1.5),
    15.0: ('b-', 1.5),
    20.0: ('m--', 2.0)
}

for L0_test in L0_values:
    estilo, espessura = cores_estilos[L0_test]
    plt.plot(resultados_L0[L0_test]['t'], np.log10(resultados_L0[L0_test]['hc']), 
             estilo, lw=espessura, label=f'fCCZ4 Mode 1 ($L_0={L0_test}$)')

plt.title(f'Evolution of Hamiltonian Constraint Violation ($L_2$)\nMaximal Slicing | N = 200 | $\\kappa_1 = {k1_fixo}$', fontweight='bold', fontsize=14)
plt.xlabel('Time ($M$)', fontsize=13)
plt.ylabel(r'$\log_{10}(L2_{HC})$', fontsize=13)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right', fontsize=12, framealpha=1.0)

nome_grafico = os.path.join(pasta_resultados, 'Comparison_Mode1_L0_Scan.png')
plt.tight_layout()
plt.savefig(nome_grafico, dpi=300)
plt.show()

print(f" -> Gráfico gravado com sucesso em: {nome_grafico}")
print("\nConcluído! Pode visualizar o gráfico da varredura.")
