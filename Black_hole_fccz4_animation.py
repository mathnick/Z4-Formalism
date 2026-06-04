import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# =========================================================================
# PARÂMETROS GERAIS
# =========================================================================
tf = 40.0
h = 0.0002
L0 = 15.0

xi = 2.0
eta0 = 2.0
f0 = 3.0 / 4.0
nc = 2.0
k1 = 15.0
k2 = 0.0
p = 200
Nq = int((3.0 / 2.0) * p)

io = 1       # Modos: escreve os modos a, b, c, etc.
idump = 200
t2 = 0.0     # Instante a partir do qual salvo os modos

# =========================================================================
# LIMPEZA DE ARQUIVOS E INICIALIZAÇÃO
# =========================================================================
open('ErrorL2HC.txt', 'w').close()
open('ErrorL2MC.txt', 'w').close()

if io == 1:
    arquivos_modos = ['al.txt', 'c.txt', 'be.txt', 'cK.txt', 'f.txt', 'fa.txt', 'fb.txt']
    for arquivo in arquivos_modos:
        open(arquivo, 'w').close()

niter = 0
t = 0.0

print(f'Evolução do Puncture iniciada às {time.strftime("%Hh %Mm")}')
t_start = time.time()

# =========================================================================
# LEITURA DE MATRIZES EXTERNAS (ALTA PRECISÃO)
# =========================================================================
pasta_atual = os.path.dirname(os.path.abspath(__file__))

def carregar_arquivo(nome):
    caminho = os.path.join(pasta_atual, nome)
    if not os.path.exists(caminho):
        caminho_com_txt = caminho + '.txt'
        if os.path.exists(caminho_com_txt):
            caminho = caminho_com_txt
        else:
            raise FileNotFoundError(f"Não encontrei o ficheiro {nome} nem {nome}.txt na pasta {pasta_atual}")
    return np.loadtxt(caminho)

print("A carregar as matrizes de alta precisão do Maple...")

rcol = L0 * carregar_arquivo('rcol').flatten()
rqcol = L0 * carregar_arquivo('rqcol').flatten()
wcolq = carregar_arquivo('Wcolq').flatten()

AlphaAl = carregar_arquivo('AlphaAl')
# Usamos a inversa exata calculada com 220 casas pelo Maple!
Alalpha = carregar_arquivo('Alalpha') 
Faa = np.copy(Alalpha)

DralphaAl = (1.0 / L0) * carregar_arquivo('DralphaAl')
DrralphaAl = (1.0 / L0**2) * carregar_arquivo('DrralphaAl')
DraFa = DralphaAl
DrraFa = DrralphaAl

ChiqC = carregar_arquivo('PhiqA')
DrchiqC = (1.0 / L0) * carregar_arquivo('DrphiqA')
DrrchiqC = (1.0 / L0**2) * carregar_arquivo('DrrphiqA')

# Filtro Exponencial
eta1_vec = np.arange(1, p + 2) / (p + 1)
filter1 = np.exp(-36.0 * (eta1_vec**20))

# =========================================================================
# CONDIÇÕES INICIAIS
# =========================================================================
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

# Listas para guardar a animação do vídeo
frames_alpha = []
frames_beta = []
frames_tempo = []

# =========================================================================
# FUNÇÃO RHS (AVALIAÇÃO DAS EQUAÇÕES DE EVOLUÇÃO E DERIVADAS)
# =========================================================================
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

    return (dalpha_dt, dK_dt, dchi_dt, da_dt, db_dt, dDelta_dt, dAa_dt, dbeta_dt, dB_dt, dTheta_dt, dZ_dt, 
            al_coef, c_coef, be_coef, cK_coef, f_coef, fa_coef, fb_coef, Del_coef)

# =========================================================================
# LOOP PRINCIPAL DO RK4
# =========================================================================
print("Calculando o avanço no tempo...")
while t < tf:
    
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

    with open('ErrorL2HC.txt', 'a') as f_out:
        f_out.write(f"{t:18.16f} {L2HC:17.16f}\n")

    with open('ErrorL2MC.txt', 'a') as f_out:
        f_out.write(f"{t:18.16f} {L2MC:17.16f}\n")

    # Captura de frames para a Animação
    if niter % idump == 0:
        frames_alpha.append(alpha.copy())
        frames_beta.append(beta.copy())
        frames_tempo.append(tdid)
        print(f"Evolução: {tdid:.2f} M / {tf} M concluídos...")

    if io == 1:
        if (t > t2) and (niter % idump == 0):
            def write_mode(filename, array_data):
                with open(filename, 'a') as f_out:
                    f_out.write(f"{tdid:18.16f}\t")
                    f_out.write("\t".join([f"{val:18.16f}" for val in array_data]))
                    f_out.write("\n")

            write_mode('al.txt', al_c1)
            write_mode('c.txt', c_c1)
            write_mode('be.txt', be_c1)
            write_mode('cK.txt', cK_c1)
            write_mode('f.txt', f_c1)
            write_mode('fa.txt', fa_c1)
            write_mode('fb.txt', fb_c1)

    niter += 1

print(f"Tempo total de cálculo: {time.time() - t_start:.1f} segundos.")

# =========================================================================
# GRÁFICO FINAL DO ERRO L2 (COMO NO MATLAB)
# =========================================================================
try:
    dados_L2HC = np.loadtxt('ErrorL2HC.txt')
    x = dados_L2HC[:, 0]
    y = np.log10(dados_L2HC[:, 1])

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=2, c='blue', alpha=0.5)
    plt.grid(True, linestyle='--')
    plt.title('Erro L2 Hamiltoniano (Vínculo Z4c)', fontweight='bold')
    plt.xlabel('Tempo ($M$)')
    plt.ylabel(r'$\log_{10}(L2_{HC})$')
    plt.show() # Feche esta janela para iniciar a gravação do vídeo!
except Exception as e:
    print("Não foi possível gerar o gráfico L2 final.", e)

# =========================================================================
# GERAÇÃO DA ANIMAÇÃO 3D DO LAPSO (O "POÇO DO TEMPO")
# =========================================================================
import mpl_toolkits.mplot3d.axes3d as p3

print("\nA compilar o vídeo 3D da geometria...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

limite_raio = 10.0 
mask = rcol <= limite_raio

# Criar a malha polar e converter para cartesiana para o gráfico 3D
r_plot = rcol[mask]
theta = np.linspace(0, 2 * np.pi, 60)
R, Theta = np.meshgrid(r_plot, theta)
X = R * np.cos(Theta)
Y = R * np.sin(Theta)

# Configuração visual do gráfico 3D
ax.set_xlim(-limite_raio, limite_raio)
ax.set_ylim(-limite_raio, limite_raio)
ax.set_zlim(0, 1.1)
ax.set_xlabel('X ($M$)')
ax.set_ylabel('Y ($M$)')
ax.set_zlabel(r'Fator Lapso ($\alpha$)')

# Ajustar o ângulo de visão da câmera (Elevação, Azimute)
ax.view_init(elev=25, azim=45)

titulo = ax.set_title('', fontsize=16, fontweight='bold')

# Variável global para armazenar a superfície a cada frame
surf = None

def animate_3d(i):
    global surf
    if surf:
        surf.remove() # Remove a malha antiga para não sobrecarregar a memória
    
    # O Lapso é simétrico, então repetimos o array 1D para todos os ângulos da malha
    Z = np.tile(frames_alpha[i][mask], (len(theta), 1))
    
    # Desenhar a nova superfície com um mapa de cores elegante (magma, plasma ou viridis)
    surf = ax.plot_surface(X, Y, Z, cmap='magma', edgecolor='none', alpha=0.9)
    titulo.set_text(f'Evolução 3D do Lapso | Tempo = {frames_tempo[i]:.2f} M')
    
    # Gira a câmera lentamente durante a animação
    ax.view_init(elev=25, azim=45 + (i * 0.5)) 
    
    return surf, titulo

# blit=False é obrigatório em animações 3D no Matplotlib
ani_3d = animation.FuncAnimation(fig, animate_3d, frames=len(frames_tempo), blit=False, interval=50)

try:
    nome_arquivo_3d = "Evolucao_Lapso_3D.mp4"
    print(f"Gravando {nome_arquivo_3d}... (Isto pode levar uns segundos)")
    ani_3d.save(nome_arquivo_3d, writer='ffmpeg', fps=20, dpi=200)
    print("Sucesso! Vídeo 3D guardado.")
except Exception as e:
    nome_arquivo_3d = "Evolucao_Lapso_3D.gif"
    print("Gravando GIF animado 3D...")
    ani_3d.save(nome_arquivo_3d, writer='pillow', fps=20)
    print("Sucesso! GIF 3D guardado.")

plt.show()
