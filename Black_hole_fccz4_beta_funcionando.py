import numpy as np
import time
import math
import os
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# =========================================================================
# 1. ARQUITETURA DE DOMÍNIO (MALHA ASSIMÉTRICA - EVITA R=0)
# =========================================================================
def configurar_bases_fccz4(L0, N, n1_param=10.0):
    col = np.cos(np.arange(2*N + 4) * math.pi / (2*N + 3))
    colr = col[1:N+2]
    r1 = L0 * colr / (np.sqrt(1 - colr**2))
    r = np.flip(r1)

    SB = np.zeros([N+1, N+1])
    rSB = np.zeros([N+1, N+1])
    rrSB = np.zeros([N+1, N+1])

    SB2 = np.zeros([N+1, N+1])
    rSB2 = np.zeros([N+1, N+1])
    rrSB2 = np.zeros([N+1, N+1])

    theta = np.arctan(L0/r)
    
    for i in range(N+1):
        # BASE PAR (a, b, chi, K, Aa, Theta, alpha)
        k_odd = 2*i + 1
        SB[i, :] = np.sin(k_odd * theta)
        rSB[i, :] = -np.cos(k_odd * theta) * k_odd * L0 / (r**2 * (1 + L0**2 / r**2))
        rrSB[i, :] = -np.sin(k_odd * theta) * k_odd**2 * L0**2 / (r**4 * (1 + L0**2 / r**2)**2) \
                     + 2 * np.cos(k_odd * theta) * k_odd * L0 / (r**3 * (1 + L0**2 / r**2)) \
                     - 2 * np.cos(k_odd * theta) * k_odd * L0**3 / (r**5 * (1 + L0**2 / r**2)**2)
        
        # BASE ÍMPAR (Lambda, beta, B)
        k_even = 2*i + 2
        SB2[i, :] = np.sin(k_even * theta)
        rSB2[i, :] = -np.cos(k_even * theta) * k_even * L0 / (r**2 * (1 + L0**2 / r**2))
        rrSB2[i, :] = -np.sin(k_even * theta) * k_even**2 * L0**2 / (r**4 * (1 + L0**2 / r**2)**2) \
                      + 2 * np.cos(k_even * theta) * k_even * L0 / (r**3 * (1 + L0**2 / r**2)) \
                      - 2 * np.cos(k_even * theta) * k_even * L0**3 / (r**5 * (1 + L0**2 / r**2)**2)

    inv_psi = np.linalg.pinv(SB)
    inv_psi2 = np.linalg.pinv(SB2)

    erfc_vec = np.vectorize(math.erfc)
    eta1 = np.arange(1, N + 2) / (N + 1)
    u = eta1 - 0.5
    u_sq = u**2
    arg = np.clip(1.0 - 4.0 * u_sq, 1e-14, 1.0)
    denom = np.clip(4.0 * u_sq, 1e-14, 1.0)
    sqrt_term = np.sqrt(-np.log(arg) / denom)
    sqrt_term[np.abs(u) < 1e-8] = 1.0
    filtro_erfc = 0.5 * erfc_vec(2.0 * np.sqrt(n1_param) * u * sqrt_term)

    return {
        'r': r, 'N': N, 'filtro': filtro_erfc,
        'psi': SB, 'rpsi': rSB, 'rrpsi': rrSB, 'inv_psi': inv_psi,
        'psi2': SB2, 'rpsi2': rSB2, 'rrpsi2': rrSB2, 'inv_psi2': inv_psi2
    }

# =========================================================================
# 2. CONDIÇÕES INICIAIS FÍSICAS
# =========================================================================
def criar_condicoes_iniciais_fys(b, M=1.0):
    r = b['r']
    psi_bh = 1.0 + M / (2.0 * r)
    chi_0 = psi_bh**(-2)
    alpha_0 = psi_bh**(-2)
    
    zeros = np.zeros_like(r)
    ones = np.ones_like(r)

    # Vetor de Estado 100% no Domínio Físico
    return np.array([
        ones.copy(),      # 0: a
        ones.copy(),      # 1: b
        chi_0.copy(),     # 2: chi
        zeros.copy(),     # 3: K
        zeros.copy(),     # 4: Aa
        zeros.copy(),     # 5: Theta
        zeros.copy(),     # 6: Lambda (ÍMPAR)
        alpha_0.copy(),   # 7: alpha
        zeros.copy(),     # 8: beta (ÍMPAR)
        zeros.copy()      # 9: B (ÍMPAR)
    ])

# =========================================================================
# 3. EVOLUÇÃO fCCZ4 (RHS FÍSICO SEM REGULARIZAÇÃO)
# =========================================================================
def calcular_rhs_fys(state, kappa1, eta_param, b):
    a, b_met, chi, K, Aa, Theta, Lambda, alpha, beta, B_shift = state

    psi, rpsi, rrpsi, inv_psi = b['psi'], b['rpsi'], b['rrpsi'], b['inv_psi']
    psi2, rpsi2, rrpsi2, inv_psi2 = b['psi2'], b['rpsi2'], b['rrpsi2'], b['inv_psi2']
    r = b['r']

    # 1. Espectralização Exata (Ordem Vetor x Matriz)
    c_a = np.dot(a - 1.0, inv_psi)
    c_b = np.dot(b_met - 1.0, inv_psi)
    c_chi = np.dot(chi - 1.0, inv_psi)
    c_K = np.dot(K, inv_psi)
    c_Aa = np.dot(Aa, inv_psi)
    c_Theta = np.dot(Theta, inv_psi)
    c_alpha = np.dot(alpha - 1.0, inv_psi)
    
    c_Lambda = np.dot(Lambda, inv_psi2)
    c_beta = np.dot(beta, inv_psi2)
    c_B = np.dot(B_shift, inv_psi2)

    # 2. Derivadas no Domínio Físico
    da = np.dot(c_a, rpsi); dda = np.dot(c_a, rrpsi)
    db = np.dot(c_b, rpsi); ddb = np.dot(c_b, rrpsi)
    dchi = np.dot(c_chi, rpsi); ddchi = np.dot(c_chi, rrpsi)
    dK = np.dot(c_K, rpsi)
    dAa = np.dot(c_Aa, rpsi)
    dTheta = np.dot(c_Theta, rpsi)
    dalpha = np.dot(c_alpha, rpsi); ddalpha = np.dot(c_alpha, rrpsi)
    
    dLambda = np.dot(c_Lambda, rpsi2)
    dbeta = np.dot(c_beta, rpsi2); ddbeta = np.dot(c_beta, rrpsi2)
    dB_shift = np.dot(c_B, rpsi2)

    # 3. Componentes Geométricas (Divisões Cruas, sem eps_sq)
    chi_sq = chi**2
    dchi_chi = dchi / chi
    ddchi_chi = ddchi / chi
    Ab = - (b_met / (2.0 * a)) * Aa

    div_beta = dbeta + beta * (db / b_met + da / (2.0 * a) + 2.0 / r)
    d_div_beta = ddbeta + dbeta * (db / b_met + da / (2.0 * a) + 2.0 / r) + beta * ((ddb * b_met - db**2) / (b_met**2) + (dda * a - da**2) / (2.0 * a**2) - 2.0 / (r**2))

    bar_Lambda = (1.0 / a) * (da / (2.0 * a) - db / b_met - (2.0 / r) * (1.0 - a / b_met))
    Zr = (a / 2.0) * (Lambda - bar_Lambda)
    Zr_up = Zr / a

    term1 = 0.5 * (da * Lambda + a * dLambda)
    term2 = -0.25 * (dda / a - (da**2) / (a**2))
    term3 = 0.5 * (ddb / b_met - (db**2) / (b_met**2))
    term4 = -1.0 / (r**2)
    term5 = - (da / (r * b_met) - a / (r**2 * b_met) - (a * db) / (r * b_met**2))
    dZr = term1 + term2 + term3 + term4 + term5
    dZr_up = (1.0 / a) * dZr - (Zr / a**2) * da

    Dm_Zm = dZr_up + Zr_up * (da / (2.0 * a) + db / b_met + 2.0 / r - 3.0 * dchi_chi)
    Dr_Zr = dZr_up + Zr_up * (da / (2.0 * a) - 1.0 * dchi_chi)

    bar_R_rr = - ddb / b_met + (db**2) / (2.0 * b_met**2) + (da * db) / (2.0 * a * b_met) + 2.0 * da / (r * a)
    bar_R_tt = - (r**2 * ddb) / (2.0 * a) - (3.0 * r * db) / (2.0 * a) + (r**2 * da * db) / (4.0 * a**2) + (r * da) / (2.0 * a) + 1.0 - a / b_met

    R_rr = bar_R_rr + 2.0 * ddchi_chi + (2.0 / r + db / b_met - da / a) * dchi_chi - 3.0 * dchi_chi**2
    R_tt = bar_R_tt + (r**2 * b_met / a) * (ddchi_chi + (3.0 / r + 1.5 * db / b_met - 0.5 * da / a) * dchi_chi - 2.0 * dchi_chi**2)
    Ricci = (chi_sq / a) * R_rr + 2.0 * (chi_sq / (r**2 * b_met)) * R_tt

    D2_alpha = (chi_sq / a) * (ddalpha + dalpha * (2.0 / r + db / b_met - da / (2.0 * a) - dchi_chi))
    DrDr_alpha = (chi_sq / a) * (ddalpha - dalpha * (da / (2.0 * a) - dchi_chi))

    # 4. Taxas Físicas (Com Advecção)
    dt_a = beta * da + 2.0 * a * dbeta - (2.0 / 3.0) * a * div_beta - 2.0 * alpha * a * Aa
    dt_b = beta * db + 2.0 * b_met * beta / r - (2.0 / 3.0) * b_met * div_beta - 2.0 * alpha * b_met * Ab
    dt_chi = beta * dchi - (1.0 / 3.0) * chi * div_beta + (1.0 / 6.0) * chi * alpha * K
    
    dt_K = - D2_alpha + alpha * (Ricci + 2.0 * Dm_Zm + K**2 - 2.0 * Theta * K) + beta * dK - 3.0 * alpha * kappa1 * Theta
    dt_Theta = beta * dTheta + 0.5 * alpha * (Ricci + 2.0 * Dm_Zm - (Aa**2 + 2.0 * Ab**2) + (2.0 / 3.0) * K**2 - 2.0 * Theta * K) - Zr_up * dalpha - alpha * kappa1 * 2.0 * Theta
    dt_Aa = beta * dAa - (DrDr_alpha - (1.0 / 3.0) * D2_alpha) + alpha * ((chi_sq / a) * R_rr - (1.0 / 3.0) * Ricci) + alpha * (2.0 * Dr_Zr - (2.0 / 3.0) * Dm_Zm) + alpha * Aa * (K - 2.0 * Theta)

    t1 = beta * dLambda - Lambda * dbeta + (1.0 / a) * ddbeta + (2.0 / b_met) * (dbeta / r - beta / (r**2))
    t2 = (1.0 / 3.0) * ((1.0 / a) * d_div_beta + 2.0 * Lambda * div_beta)
    t3 = - (2.0 / a) * (Aa * dalpha + alpha * dAa)
    t4 = 2.0 * alpha * (Aa * Lambda - (2.0 / (r * b_met)) * (Aa - Ab))
    t5 = (2.0 * alpha / a) * (dAa - (2.0 / 3.0) * dK - 3.0 * Aa * dchi_chi + (Aa - Ab) * (2.0 / r + db / b_met))
    t6 = (2.0 / a) * (alpha * dTheta - Theta * dalpha - (2.0 / 3.0) * alpha * K * Zr)
    t7 = (2.0 / a) * ((2.0 / 3.0) * Zr * div_beta - Zr * dbeta) - (2.0 / a) * kappa1 * Zr
    dt_Lambda = t1 + t2 + t3 + t4 + t5 + t6 + t7

    # -------------------------------------------------------------
    # GAUGE (ZONA DE TESTE DE ADVECÇÃO)
    # -------------------------------------------------------------
    # Opção 1: Sem Advecção 
    dt_alpha = - 2.0 * alpha * (K - 2.0 * Theta)

    # Opção 2: Com Advecção
    # dt_alpha = beta * dalpha - 2.0 * alpha * (K - 2.0 * Theta)
    # -------------------------------------------------------------
    
    dt_beta = 0.75 * B_shift
    dt_B = dt_Lambda - eta_param * B_shift

    return np.array([dt_a, dt_b, dt_chi, dt_K, dt_Aa, dt_Theta, dt_Lambda, dt_alpha, dt_beta, dt_B])


# =========================================================================
# 4. INTEGRADOR E APLICAÇÃO DE FILTRO SELETIVO
# =========================================================================
def aplicar_filtro_seletivo_fys(state, b, vars_filtro):
    state_filtrado = state.copy()
    
    for idx in vars_filtro:
        # Verifica a paridade para usar as matrizes certas
        if idx in [6, 8, 9]:  # Lambda, beta, B
            inv_mat = b['inv_psi2']
            mat = b['psi2']
        else:
            inv_mat = b['inv_psi']
            mat = b['psi']

        # Extrai coeficientes (descontando 1.0 para as variáveis que não vão a zero)
        if idx in [0, 1, 2, 7]:  # a, b, chi, alpha
            c_var = np.dot(state[idx] - 1.0, inv_mat)
            c_var_filtrado = c_var * b['filtro']
            state_filtrado[idx] = 1.0 + np.dot(c_var_filtrado, mat)
        else:
            c_var = np.dot(state[idx], inv_mat)
            c_var_filtrado = c_var * b['filtro']
            state_filtrado[idx] = np.dot(c_var_filtrado, mat)

    return state_filtrado

def passo_rk4_fys(s, h, k1, eta, b, vars_filtro):
    k_1 = calcular_rhs_fys(s, k1, eta, b)
    k_2 = calcular_rhs_fys(s + 0.5*h*k_1, k1, eta, b)
    k_3 = calcular_rhs_fys(s + 0.5*h*k_2, k1, eta, b)
    k_4 = calcular_rhs_fys(s + h*k_3, k1, eta, b)
    s_new = s + (h/6.0) * (k_1 + 2*k_2 + 2*k_3 + k_4)
    
    return aplicar_filtro_seletivo_fys(s_new, b, vars_filtro)


# =========================================================================
# 5. GERENCIADOR DE SIMULAÇÃO E LIVE PREVIEW
# =========================================================================
def simular_e_filmar_gauge(L0, N, tf, h, k1, eta_param, n1_param, vars_filtro, frames_qtd=300):
    b = configurar_bases_fccz4(L0, N, n1_param)
    s = criar_condicoes_iniciais_fys(b)

    r_grid = b['r']
    passos_totais = int(tf/h)
    passos_salvar = max(1, int(0.5/h))
    passos_animacao = max(1, passos_totais // frames_qtd)

    tempo_sobrevivido = 0.0
    tempos_anim, f_alpha, f_beta = [], [], []

    plt.ion() 
    fig_live, (ax1_live, ax2_live) = plt.subplots(1, 2, figsize=(12, 5))
    limite_raio = 15.0 
    
    ax1_live.set_xlim(0.0, limite_raio); ax1_live.set_ylim(-0.05, 1.2)
    ax1_live.set_xlabel("Raio isotrópico (r/M)"); ax1_live.set_ylabel(r"Fator Lapso ($\alpha$)")
    ax1_live.grid(True)

    ax2_live.set_xlim(0.0, limite_raio); ax2_live.set_ylim(-0.10, 0.5)
    ax2_live.set_xlabel("Raio isotrópico (r/M)"); ax2_live.set_ylabel(r"Vetor Shift ($\beta^r$)")
    ax2_live.grid(True)

    linha_a_live, = ax1_live.plot([], [], 'b-', linewidth=2.5)
    linha_b_live, = ax2_live.plot([], [], 'r-', linewidth=2.5)
    titulo_live = fig_live.suptitle('Inicializando...', fontsize=16, fontweight='bold')
    plt.show() 

    nome_arquivo = f"evolucao_L2_{tf}M_k{k1:.2f}_eta{eta_param:.2f}.txt"
    with open(nome_arquivo, "w") as arquivo_dados:
        arquivo_dados.write("Tempo_M, L2_H\n")

        for i in range(passos_totais):
            s = passo_rk4_fys(s, h, k1, eta_param, b, vars_filtro)
            tempo_atual = (i + 1) * h

            if i % passos_salvar == 0 or i == passos_totais - 1:
                a, b_met, chi, K, Aa, Theta, Lambda, alpha, beta, B_shift = s
                
                c_a, c_b, c_chi = np.dot(a-1.0, b['inv_psi']), np.dot(b_met-1.0, b['inv_psi']), np.dot(chi-1.0, b['inv_psi'])
                da, ddb = np.dot(c_a, b['rpsi']), np.dot(c_b, b['rrpsi'])
                db, dchi, ddchi = np.dot(c_b, b['rpsi']), np.dot(c_chi, b['rpsi']), np.dot(c_chi, b['rrpsi'])
                
                chi_sq = chi**2
                dchi_chi = dchi / chi
                ddchi_chi = ddchi / chi
                
                bar_R_rr = - ddb / b_met + (db**2) / (2.0 * b_met**2) + (da * db) / (2.0 * a * b_met) + 2.0 * da / (r_grid * a)
                bar_R_tt = - (r_grid**2 * ddb) / (2.0 * a) - (3.0 * r_grid * db) / (2.0 * a) + (r_grid**2 * da * db) / (4.0 * a**2) + (r_grid * da) / (2.0 * a) + 1.0 - a / b_met
                R_rr = bar_R_rr + 2.0 * ddchi_chi + (2.0 / r_grid + db / b_met - da / a) * dchi_chi - 3.0 * dchi_chi**2
                R_tt = bar_R_tt + (r_grid**2 * b_met / a) * (ddchi_chi + (3.0 / r_grid + 1.5 * db / b_met - 0.5 * da / a) * dchi_chi - 2.0 * dchi_chi**2)
                R_fisico = (chi_sq / a) * R_rr + 2.0 * (chi_sq / (r_grid**2 * b_met)) * R_tt

                Ab = - (b_met / (2.0 * a)) * Aa
                H_const = R_fisico - (Aa**2 + 2.0*Ab**2) + (2.0/3.0)*K**2

                mascara = r_grid > 1.0
                r_ext = r_grid[mascara]
                H_ext = H_const[mascara]
                
                integral_H2 = np.trapezoid(H_ext**2, x=r_ext)
                erro_H_final = np.sqrt(integral_H2 / (r_ext[-1] - r_ext[0]))
                
                tempo_sobrevivido = tempo_atual
                if i % (passos_salvar * 10) == 0:
                    print(f"Progresso: {tempo_sobrevivido:.2f}M | L2(H): {erro_H_final:.2e}")
                arquivo_dados.write(f"{tempo_sobrevivido:.4f}, {erro_H_final:.8e}\n")
                arquivo_dados.flush()

            if i % passos_animacao == 0 or i == passos_totais - 1:
                tempos_anim.append(tempo_atual)
                f_alpha.append(s[7].copy())
                f_beta.append(s[8].copy())

                linha_a_live.set_data(r_grid, s[7])
                linha_b_live.set_data(r_grid, s[8])
                titulo_live.set_text(f'Evolução ao Vivo | Tempo = {tempo_atual:.2f}M')
                fig_live.canvas.draw()
                fig_live.canvas.flush_events()

            if np.isnan(s).any() or np.max(np.abs(s)) > 1e11:
                print(f"\n[ALERTA VERMELHO] Crash detectado em t={tempo_atual:.4f}M!")
                break

    plt.ioff()
    plt.close(fig_live)
    return r_grid, tempos_anim, f_alpha, f_beta

# =========================================================================
# 6. EXECUÇÃO DA SIMULAÇÃO
# =========================================================================
if __name__ == "__main__":
    tf_filme = 30.0
    k1_alvo = 2.0
    eta_alvo = 2.0
    n1_alvo = 10.0

    # -------------------------------------------------------------
    # Índices -> 0:a, 1:b, 2:chi, 3:K, 4:Aa, 5:Theta, 6:Lambda, 7:alpha, 8:beta, 9:B
    # -------------------------------------------------------------
    lista_de_filtros = [0,1,2,3,4,5,6,7,8,9] 

    print(f"Iniciando simulação fCCZ4  (k1={k1_alvo}, eta={eta_alvo}, n1={n1_alvo})...")
    r, tempos, f_alpha, f_beta = simular_e_filmar_gauge(
        L0=5.0, N=150, tf=tf_filme, h=0.0001, 
        k1=k1_alvo, eta_param=eta_alvo, n1_param=n1_alvo, 
        vars_filtro=lista_de_filtros, frames_qtd=300
    )
    
    print("\nSimulação concluída! Renderizando o vídeo final em alta qualidade...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    limite_raio = 15.0 

    ax1.set_xlim(0.0, limite_raio); ax1.set_ylim(-0.05, 1.2)
    ax1.set_xlabel("Raio isotrópico (r/M)"); ax1.set_ylabel(r"Fator Lapso ($\alpha$)")
    ax1.grid(True)

    ax2.set_xlim(0.0, limite_raio); ax2.set_ylim(-0.10, 0.4)
    ax2.set_xlabel("Raio isotrópico (r/M)"); ax2.set_ylabel(r"Vetor Shift ($\beta^r$)")
    ax2.grid(True)

    linha_alpha, = ax1.plot([], [], 'b-', linewidth=2.5)
    linha_beta, = ax2.plot([], [], 'r-', linewidth=2.5)
    titulo = fig.suptitle('', fontsize=16, fontweight='bold')

    def animar(i):
        linha_alpha.set_data(r, f_alpha[i])
        linha_beta.set_data(r, f_beta[i])
        titulo.set_text(f'Evolução da Trombeta | Tempo = {tempos[i]:.2f}M')
        return linha_alpha, linha_beta, titulo

    ani = animation.FuncAnimation(fig, animar, frames=len(tempos), interval=80, blit=False)

    nome_video = f"Evolucao_Gauge_Trombeta_{tf_filme}M.mp4"
    try:
        ani.save(nome_video, writer='ffmpeg', fps=15, dpi=200)
        print(f"Sucesso! Vídeo MP4 salvo como '{nome_video}'.")
    except Exception as e:
        nome_gif = f"Evolucao_Gauge_Trombeta_{tf_filme}M.gif"
        print(f"Erro ao salvar MP4 ({e}). Tentando fallback para GIF animado...")
        ani.save(nome_gif, writer='pillow', fps=15)
        print(f"GIF salvo com sucesso como '{nome_gif}'!")
