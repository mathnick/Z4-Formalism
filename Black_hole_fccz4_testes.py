import numpy as np
import time
import os

# =========================================================================
# PREPARAÇÃO DE DIRETÓRIOS E LEITURA DE MATRIZES BASE
# =========================================================================
pasta_atual = os.path.dirname(os.path.abspath(__file__))
pasta_resultados = os.path.join(pasta_atual, "Resultados_Erros")

# Cria a pasta de resultados se não existir
if not os.path.exists(pasta_resultados):
    os.makedirs(pasta_resultados)

def carregar_arquivo(nome):
    caminho = os.path.join(pasta_atual, nome)
    if not os.path.exists(caminho):
        caminho = caminho + '.txt'
    return np.loadtxt(caminho)

print("A carregar matrizes adimensionais de alta precisão...")
# Estas matrizes são invariantes à escala. O L0 será aplicado dentro do loop!
base_rcol = carregar_arquivo('rcol').flatten()
base_rqcol = carregar_arquivo('rqcol').flatten()
wcolq = carregar_arquivo('Wcolq').flatten()

AlphaAl = carregar_arquivo('AlphaAl')
Alalpha = carregar_arquivo('Alalpha') 
Faa = np.copy(Alalpha)

base_DralphaAl = carregar_arquivo('DralphaAl')
base_DrralphaAl = carregar_arquivo('DrralphaAl')

ChiqC = carregar_arquivo('PhiqA')
base_DrchiqC = carregar_arquivo('DrphiqA')
base_DrrchiqC = carregar_arquivo('DrrphiqA')

p = 200
Nq = int((3.0 / 2.0) * p)
eta1_vec = np.arange(1, p + 2) / (p + 1)
filter1 = np.exp(-36.0 * (eta1_vec**20))

# =========================================================================
# O MOTOR DE SIMULAÇÃO (ENCAPSULADO PARA O LOOP)
# =========================================================================
def executar_simulacao(L0, k1, tf=40.0, h=0.0002, idump=500):
    print(f"\n---> Iniciando Simulação | L0 = {L0} | kappa_1 = {k1} <---")
    t_inicio_sim = time.time()
    
    # 1. Aplicação da escala L0 nas matrizes locais
    rcol = L0 * base_rcol
    rqcol = L0 * base_rqcol
    DralphaAl = (1.0 / L0) * base_DralphaAl
    DrralphaAl = (1.0 / L0**2) * base_DrralphaAl
    DraFa = DralphaAl
    DrraFa = DrralphaAl
    DrchiqC = (1.0 / L0) * base_DrchiqC
    DrrchiqC = (1.0 / L0**2) * base_DrrchiqC

    # Constantes Físicas do BSSN/Z4c
    xi, eta0, f0, nc, k2 = 2.0, 2.0, 3.0/4.0, 2.0, 0.0

    # Condições Iniciais (dependentes do novo rcol)
    alpha = (1.0 + 1.0 / (2.0 * rcol))**(-2)
    K = np.zeros(p + 1)
    chi = (1.0 + 1.0 / (2.0 * rcol))**(-nc)
    a = np.ones(p + 1)
    b = np.ones(p + 1)
    Delta = np.zeros(p + 1)
    Aa = np.zeros(p + 1)
    beta = np.zeros(p + 1)
    B = np.zeros(p + 1)
    Z = np.zeros(p + 1)
    Theta = np.zeros(p + 1)

    # 2. Definição do RHS
    def evaluate_rhs(alpha_val, K_val, chi_val, a_val, b_val, Delta_val, Aa_val, beta_val, B_val, Z_val, Theta_val, aplicar_filtro=False):
        al_coef = Alalpha @ (alpha_val - 1.0)
        dralpha = DralphaAl @ al_coef
        drralpha = DrralphaAl @ al_coef

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

        cK_coef = Alalpha @ K_val
        if aplicar_filtro: cK_coef = filter1 * cK_coef
        drK = DralphaAl @ cK_coef

        f_coef = Alalpha @ Aa_val
        drAa = DralphaAl @ f_coef
        Del_coef = Alalpha @ Delta_val
        drDelta = DralphaAl @ Del_coef
        ze_coef = Alalpha @ Z_val
        drZ = DralphaAl @ ze_coef
        th_coef = Alalpha @ Theta_val
        drTheta = DralphaAl @ th_coef

        dalpha_dt = -2.0 * alpha_val * K_val
        Divbeta = drbeta + beta_val * (0.5 * dra / a_val + drb / b_val) + 2.0 * beta_val / rcol
        dchi_dt = beta_val * drchi - (nc / 6.0) * chi_val * Divbeta + (nc / 6.0) * alpha_val * K_val * chi_val
        da_dt = beta_val * dra + 2.0 * a_val * drbeta - 2.0 * a_val * Divbeta / 3.0 - 2.0 * alpha_val * a_val * Aa_val
        db_dt = beta_val * drb + 2.0 * b_val * beta_val / rcol - 2.0 * b_val * Divbeta / 3.0 + alpha_val * b_val * Aa_val

        Zc = chi_val**(4.0 / nc) * Z_val / a_val
        divZ = chi_val**(4.0 / nc) * drZ / a_val + (-0.5 * dra / a_val + drb / b_val + 2.0 / rcol - (2.0 / nc) * drchi / chi_val) * Zc
        Lapalpha = chi_val**(4.0 / nc) / a_val * (drralpha - dralpha * (0.5 * dra / a_val + (2.0 / nc) * drchi / chi_val - drb / b_val - 2.0 / rcol))

        dK_dt = beta_val * drK - Lapalpha + (1.0 / 3.0) * alpha_val * K_val**2 + 1.5 * alpha_val * Aa_val**2 - 2.0 * alpha_val * K_val * Theta_val \
                - 3.0 * k1 * (1.0 + k2) * alpha_val * Theta_val + 2.0 * alpha_val * divZ

        TrLapalpha = (1.0 / 3.0) * chi_val**(4.0 / nc) / a_val * (2.0 * drralpha - dralpha * (-(8.0 / nc) * drchi / chi_val + dra / a_val + drb / b_val) - 2.0 * dralpha / rcol)

        TrRicci = chi_val**(4.0 / nc) / a_val * (
            (2.0 / 3.0) * a_val * drDelta - (2.0 / 3.0 / nc) * (2.0 / rcol + drb / b_val + dra / a_val) * drchi / chi_val
            - (1.0 / 3.0) * drra / a_val + (4.0 / 3.0 / nc) * drrchi / chi_val + (4.0 / 3.0 / nc**2) * (2.0 - nc) * drchi**2 / chi_val**2
            + (2.0 / rcol**2) * (1.0 - a_val / b_val) * (-1.0 / 3.0 - rcol * drb / b_val) + 0.5 * dra * Delta_val + (5.0 / 12.0) * dra**2 / a_val**2
            - dra / b_val / rcol - (1.0 / 3.0) * drb**2 / b_val**2 + (1.0 / 3.0) * drrb / b_val
            + (2.0 / 3.0) * drb / b_val / rcol * (3.0 - a_val / b_val)
        )

        dAa_dt = beta_val * drAa - TrLapalpha + alpha_val * TrRicci + alpha_val * K_val * Aa_val - 2.0 * alpha_val * Theta_val * Aa_val \
                 + 2.0 * alpha_val * (chi_val**(4.0 / nc) * drZ / a_val - (0.5 * dra / a_val - (2.0 / nc) * drchi / chi_val) * Zc - (1.0 / 3.0) * divZ)

        drDivbeta = drrbeta + drbeta * (0.5 * dra / a_val + drb / b_val) + beta_val * (0.5 * drra / a_val - 0.5 * dra**2 / a_val**2 + drrb / b_val - drb**2 / b_val**2) + 2.0 * (drbeta - beta_val / rcol) / rcol

        dDelta_dt = beta_val * drDelta - Delta_val * drbeta + drrbeta / a_val + 2.0 * (drbeta - beta_val / rcol) / (b_val * rcol) + (2.0 / 3.0) * Delta_val * Divbeta \
                    + (1.0 / 3.0) * drDivbeta / a_val - (2.0 / a_val) * (Aa_val * dralpha + alpha_val * drAa) + 2.0 * alpha_val * (Aa_val * Delta_val - 3.0 * Aa_val / b_val / rcol) \
                    + xi * alpha_val / a_val * (-(6.0 / nc) * Aa_val * drchi / chi_val - (2.0 / 3.0) * drK + drAa + 1.5 * (2.0 / rcol + drb / b_val) * Aa_val)

        R = -(1.0 / a_val) * (chi_val**(4.0 / nc)) * (
            0.5 * drra / a_val + drrb / b_val + 0.5 * drb**2 / b_val**2 - a_val * drDelta - dra**2 / a_val**2
            - (8.0 / nc) * drrchi / chi_val + 8.0 * (1.0 + nc) / nc**2 * drchi**2 / chi_val**2
            + 4.0 / rcol**2 * (1.0 - a_val / b_val) + 2.0 / rcol * (3.0 - a_val / b_val) * drb / b_val
            + (8.0 / nc) * drchi / chi_val * (-2.0 / rcol - drb / b_val + 0.5 * dra / a_val)
        )

        Hc = R + (2.0 / 3.0) * K_val**2 - 1.5 * Aa_val**2
        dTheta_dt = beta_val * drTheta + 0.5 * alpha_val * Hc - alpha_val * K_val * Theta_val - Zc * dralpha + alpha_val * divZ - k1 * (k2 + 2.0) * alpha_val * Theta_val
        Mc = -(2.0 / 3.0) * drK + drAa - (6.0 / nc) * Aa_val * drchi / chi_val + 1.5 * Aa_val * (2.0 / rcol + drb / b_val)
        dZ_dt = beta_val * drZ + Z_val * drbeta + alpha_val * Mc - 2.0 * alpha_val * Z_val * Aa_val - (2.0 / 3.0) * alpha_val * K_val * Z_val + alpha_val * drTheta - Theta_val * dralpha - k1 * alpha_val * Z_val
        dbeta_dt = f0 * B_val
        dB_dt = dDelta_dt - eta0 * B_val

        # AQUI FOI CORRIGIDO O TYPO (dalpha_dt no lugar certo!)
        return (dalpha_dt, dK_dt, dchi_dt, da_dt, db_dt, dDelta_dt, dAa_dt, dbeta_dt, dB_dt, dTheta_dt, dZ_dt, 
                al_coef, c_coef, be_coef, cK_coef, f_coef, fa_coef, fb_coef, Del_coef)

    # 3. Arquivo de Output para esta iteração
    nome_arquivo_erros = os.path.join(pasta_resultados, f"Erros_L0_{L0}_k1_{k1}.txt")
    with open(nome_arquivo_erros, 'w') as f_out:
        f_out.write("Tempo_M\tL2HC\tL2MC\n")

    t = 0.0
    niter = 0

    # 4. Loop RK4
    while t <= tf:
        da1, dK1, dc1, daa1, db1, dD1, dAa1, dbe1, dB1, dTh1, dZ1, \
        al_c1, c_c1, be_c1, cK_c1, f_c1, fa_c1, fb_c1, _ = evaluate_rhs(
            alpha, K, chi, a, b, Delta, Aa, beta, B, Z, Theta, aplicar_filtro=False)

        da2, dK2, dc2, daa2, db2, dD2, dAa2, dbe2, dB2, dTh2, dZ2, *_ = evaluate_rhs(
            alpha + 0.5*h*da1, K + 0.5*h*dK1, chi + 0.5*h*dc1, a + 0.5*h*daa1, b + 0.5*h*db1, 
            Delta + 0.5*h*dD1, Aa + 0.5*h*dAa1, beta + 0.5*h*dbe1, B + 0.5*h*dB1, 
            Z + 0.5*h*dZ1, Theta + 0.5*h*dTh1, aplicar_filtro=False)

        da3, dK3, dc3, daa3, db3, dD3, dAa3, dbe3, dB3, dTh3, dZ3, *_ = evaluate_rhs(
            alpha + 0.5*h*da2, K + 0.5*h*dK2, chi + 0.5*h*dc2, a + 0.5*h*daa2, b + 0.5*h*db2, 
            Delta + 0.5*h*dD2, Aa + 0.5*h*dAa2, beta + 0.5*h*dbe2, B + 0.5*h*dB2, 
            Z + 0.5*h*dZ2, Theta + 0.5*h*dTh2, aplicar_filtro=False)

        da4, dK4, dc4, daa4, db4, dD4, dAa4, dbe4, dB4, dTh4, dZ4, *_ = evaluate_rhs(
            alpha + h*da3, K + h*dK3, chi + h*dc3, a + h*daa3, b + h*db3, 
            Delta + h*dD3, Aa + h*dAa3, beta + h*dbe3, B + h*dB3, 
            Z + h*dZ3, Theta + h*dTh3, aplicar_filtro=True)

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

        tdid = t
        t += h

        # 5. Cálculo Dinâmico e Gravação dos Erros L2 (A cada idump)
        if niter % idump == 0:
            Kq = ChiqC @ cK_c1
            drKq = DrchiqC @ cK_c1
            Aaq = ChiqC @ f_c1
            drAaq = DrchiqC @ f_c1
            aq = ChiqC @ fa_c1 + np.ones(Nq + 1)
            draq = DrchiqC @ fa_c1
            drraq = DrrchiqC @ fa_c1
            bq = ChiqC @ fb_c1 + np.ones(Nq + 1)
            drbq = DrchiqC @ fb_c1
            drrbq = DrrchiqC @ fb_c1
            chiq = ChiqC @ c_c1 + np.ones(Nq + 1)
            drchiq = DrchiqC @ c_c1
            drrchiq = DrrchiqC @ c_c1
            Del_c1 = Alalpha @ Delta
            drDeltaq = DrchiqC @ Del_c1

            MCq = - (2.0 / 3.0) * drKq + drAaq - (6.0 / nc) * Aaq * drchiq / chiq + 1.5 * Aaq * (2.0 / rqcol + drbq / bq)
            L2MC = np.sqrt(0.5 * np.dot(MCq**2, wcolq))

            Rq = - (1.0 / aq) * (chiq**(4.0 / nc)) * (
                0.5 * drraq / aq + drrbq / bq + 0.5 * drbq**2 / bq**2 - aq * drDeltaq - draq**2 / aq**2
                - (8.0 / nc) * drrchiq / chiq + 8.0 * (1.0 + nc) / nc**2 * drchiq**2 / chiq**2
                + 4.0 / rqcol**2 * (1.0 - aq / bq) + 2.0 / rqcol * (3.0 - aq / bq) * drbq / bq
                + (8.0 / nc) * drchiq / chiq * (-2.0 / rqcol - drbq / bq + 0.5 * draq / aq)
            )

            HCq = Rq - 1.5 * Aaq**2 + (2.0 / 3.0) * Kq**2
            L2HC = np.sqrt(0.5 * np.dot(HCq**2, wcolq))

            with open(nome_arquivo_erros, 'a') as f_out:
                f_out.write(f"{tdid:18.16f}\t{L2HC:17.16f}\t{L2MC:17.16f}\n")
            
            print(f"   Progresso: {tdid:.1f} M / {tf} M | L2HC: {L2HC:.2e}")

            # =================================================================
            # NOVO: GATILHO DE PARAGEM DE EMERGÊNCIA (EARLY STOPPING)
            # =================================================================
            # Se o erro for NaN (Not a Number) ou maior que 1.0 (explosão certa)
            if np.isnan(L2HC) or np.isnan(L2MC) or L2HC > 1.0 or L2MC > 1.0:
                print(f"   [ALERTA] Erro divergiu em t = {tdid:.2f} M! Abortando esta combinação de parâmetros para poupar tempo.")
                with open(nome_arquivo_erros, 'a') as f_out:
                    f_out.write(f"# SIMULACAO ABORTADA EM t={tdid:.2f}M DEVIDO A DIVERGENCIA NUMERICA\n")
                break # O 'break' quebra o 'while' e o código avança imediatamente para o próximo L0 e k1

        niter += 1

    tempo_gasto = (time.time() - t_inicio_sim) / 60.0
    print(f"---> Concluído! Erros gravados em: {nome_arquivo_erros} ({tempo_gasto:.2f} min) <---")

# =========================================================================
# CONFIGURAÇÃO DA VARREDURA (PARAMETER SWEEP)
# =========================================================================
if __name__ == "__main__":
    # Coloque aqui as listas de parâmetros que quer testar!
    lista_L0 = [5.0, 7.5, 10.0, 12.5, 15.0, 20.0]
    lista_kappa1 = [0.05, 0.2, 0.5, 1.5, 5.0, 10.0, 15.0, 25.0]
    
    # Parâmetros de tempo
    tempo_final = 40.0
    passo_rk4 = 0.0002
    
    print("=========================================================")
    print(" INICIANDO VARREDURA DE PARÂMETROS Z4c (MOVING PUNCTURE) ")
    print(f" Total de combinações a testar: {len(lista_L0) * len(lista_kappa1)}")
    print("=========================================================")
    
    tempo_global = time.time()
    
    for L0_teste in lista_L0:
        for k1_teste in lista_kappa1:
            executar_simulacao(L0_teste, k1_teste, tf=tempo_final, h=passo_rk4, idump=500)
            
    minutos_totais = (time.time() - tempo_global) / 60.0
    print("\n=========================================================")
    print(f" VARREDURA COMPLETA! Tempo total: {minutos_totais:.2f} minutos.")
    print(f" Pode encontrar todos os ficheiros .txt na pasta '{pasta_resultados}'")
    print("=========================================================")
