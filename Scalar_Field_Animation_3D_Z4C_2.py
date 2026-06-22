import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

# =========================================================================
# 1. FUNÇÃO PARA CONFIGURAR AS BASES ESPECTRAIS
# =========================================================================
def configurar_bases_espectrais(L0_valor, N_valor, M_plot=300):
    """
    Configura as bases espectrais para evolução e para a plotagem 3D.
    M_plot define a resolução da malha para o vídeo 3D.
    """
    col = np.cos(np.arange(2*N_valor + 4)*math.pi /(2*N_valor + 3))
    colr = col[1:N_valor+2]

    r1 = L0_valor * colr/(np.sqrt(1-colr**2))
    r = np.flip(r1)

    # Matrizes Base (Tchebyshev Polinomials)
    SB = np.zeros([N_valor+3, N_valor+1])
    rSB = np.zeros([N_valor+3, N_valor+1])
    rrSB = np.zeros([N_valor+3, N_valor+1])

    for i in range(N_valor+1+1+1):
        SB[i,] = np.sin((2*i+1)*np.arctan(L0_valor/r))
        rSB[i,] = -np.cos((2*i+1)*np.arctan(L0_valor/r))*(2*i+1)*L0_valor/(r**2*(1+L0_valor**2/r**2))
        rrSB[i,] = -np.sin((2*i+1)*np.arctan(L0_valor/r))*(2*i+1)**2*L0_valor**2/(r**4*(1+L0_valor**2/r**2)**2)+2*np.cos((2*i+1)*np.arctan(L0_valor/r))*(2*i+1)*L0_valor/(r**3*(1+L0_valor**2/r**2))-2*np.cos((2*i+1)*np.arctan(L0_valor/r))*(2*i+1)*L0_valor**3/(r**5*(1+L0_valor**2/r**2)**2)

    psi = SB[0:N_valor+1,:]
    rpsi = rSB[0:N_valor+1,:]
    rrpsi = rrSB[0:N_valor+1,:]
    inv_psi = np.linalg.inv(psi)

    # Bases para Krr
    SB1 = 1/2*(SB[1:(N_valor+2),:] + SB[0:(N_valor+1),:])
    rSB1 = 1/2*(rSB[1:(N_valor+2),:] + rSB[0:(N_valor+1),:])

    # Bases para Beta e Z
    SB2 = np.zeros([N_valor+1, N_valor+1])
    rSB2 = np.zeros([N_valor+1, N_valor+1])
    for i in range(N_valor+1):
        SB2[i,] = np.sin((2*(i+1/2)+1)*np.arctan(L0_valor/r))
        rSB2[i,] = -np.cos((2*i+2)*np.arctan(L0_valor/r))*(2*i+2)*L0_valor/(r**2*(1+L0_valor**2/r**2))
    inv_SB2 = np.linalg.pinv(SB2)

    # Alpha na origem
    psi_0 = np.zeros(N_valor+1)
    for i in range(N_valor+1):
        psi_0[i,] = np.sin((2*i+1)*math.pi/2)

    # --- Bases de Plotagem (Para o Visual 3D) ---
    rplot = np.linspace(0.000001, 10, M_plot)
    SBplot = np.zeros([N_valor+1, M_plot])
    for i in range(N_valor+1):
        SBplot[i,] = np.sin((2*i+1)*np.arctan(L0_valor/rplot))
    psiplot = SBplot

    return {
        'r': r, 'psi': psi, 'rpsi': rpsi, 'rrpsi': rrpsi, 'inv_psi': inv_psi,
        'SB1': SB1, 'rSB1': rSB1, 'SB2': SB2, 'rSB2': rSB2, 'inv_SB2': inv_SB2,
        'psi_0': psi_0, 'rplot': rplot, 'psiplot': psiplot,
        'L0': L0_valor, 'N': N_valor
    }

# =========================================================================
# 2. CONDIÇÕES INICIAIS (NEWTON-RAPHSON)
# =========================================================================
def criar_condicoes_iniciais(A0_valor, r0_valor, sigma_valor, bases):
    r = bases['r']
    psi = bases['psi']
    rpsi = bases['rpsi']
    rrpsi = bases['rrpsi']
    inv_psi = bases['inv_psi']
    N_valor = bases['N']
    
    # Campo Escalar Inicial (Phi e Pi)
    Phi_0 = A0_valor*r**2*(np.exp(-(r-r0_valor)**2/sigma_valor**2)+np.exp(-(r+r0_valor)**2/sigma_valor**2))
    a0 = np.dot(Phi_0, inv_psi)
    Pi_0 = np.zeros(N_valor+1)
    b0 = np.dot(Pi_0, inv_psi) 

    # Solução da Restrição Hamiltoniana (Newton Raphson para Chi)
    c0 = 0.001 * np.ones([N_valor+1])
    rPhi= np.dot(a0, rpsi)

    tol = 1e-18
    for n in range(51):
        Chi=np.dot(c0,psi)
        rChi=np.dot(c0,rpsi)
        rrChi=np.dot(c0,rrpsi)
        H0 = 4*rChi**2 + 4*rrChi + 8/r*rChi + 1/2*(rPhi)**2
        JH = 8*np.dot(c0,rpsi)*rpsi + 4*rrpsi + 8/r*rpsi
        cnew = c0 - np.dot(H0, np.linalg.inv(JH))
        if min(abs(cnew-c0)) < tol:
            break
        c0 = cnew

    # Inicialização Z4
    theta0 = np.zeros(N_valor+1)
    z0 = np.zeros(N_valor+1)

    return c0, a0, b0, theta0, z0

# =========================================================================
# 3. MÓDULO RK4 (FUNÇÃO RHS)
# =========================================================================
def calcular_taxas_z4(c_coefs, a_coefs, b_coefs, theta_coefs, z_coefs, kappa1, kappa2, bases):
    psi = bases['psi']; rpsi = bases['rpsi']; rrpsi = bases['rrpsi']; inv_psi = bases['inv_psi']
    SB1 = bases['SB1']; rSB1 = bases['rSB1']; SB2 = bases['SB2']; rSB2 = bases['rSB2']; inv_SB2 = bases['inv_SB2']
    psi_0 = bases['psi_0']; r = bases['r']
    V = 0.0 # Potencial do campo escalar
    
    # 1. Recuperar Campos Físicos
    Chi = np.dot(c_coefs, psi)
    rChi = np.dot(c_coefs, rpsi)
    rrChi = np.dot(c_coefs, rrpsi)
    Phi = np.dot(a_coefs, psi)
    rPhi = np.dot(a_coefs, rpsi)
    rrPhi = np.dot(a_coefs, rrpsi)
    Pi = np.dot(b_coefs, psi)
    rPi = np.dot(b_coefs, rpsi) 
    Theta = np.dot(theta_coefs, psi)
    rTheta = np.dot(theta_coefs, rpsi)
    Z = np.dot(z_coefs, SB2)
    rZ = np.dot(z_coefs, rSB2)

    # 2. Krr (Restrição de Momento)
    Matrix_Krr = 2 * rChi * SB1 + rSB1 + 3 / r * SB1
    ck0 = np.dot(- Pi * rPhi * np.exp(4 * Chi), np.linalg.inv(Matrix_Krr))
    Krr = np.dot(ck0, SB1)
    rKrr = np.dot(ck0, rSB1)

    # 3. Lapso (Alpha)
    Matrix_Alpha = (rrpsi + 2*(1/r + rChi)*rpsi -
                    (3/2)*np.exp(-4*Chi)*Krr**2*psi -
                    np.exp(4*Chi)*(Pi**2 - V)*psi -
                    2*np.exp(4*Chi)*rZ*psi -
                    2*np.exp(4*Chi)*(6*rChi + 2/r)*Z*psi)
    rhsal = ((3/2)*np.exp(-4*Chi)*Krr**2 + np.exp(4*Chi)*(Pi**2 - V) +
             2*np.exp(4*Chi)*rZ + 2*np.exp(4*Chi)*(6*rChi + 2/r)*Z)
             
    al0 = np.dot(rhsal, np.linalg.inv(Matrix_Alpha))
    Alpha = 1 + np.dot(al0, psi)
    rAlpha = np.dot(al0, rpsi)
    Alpha_central = 1 + np.dot(al0, psi_0)

    # 4. Shift (Beta)
    Matrix_Beta = rSB2/r - SB2/r**2
    rhsbe = 3/2 * Alpha * np.exp(-4*Chi) * Krr / r
    be0 = np.dot(rhsbe, np.linalg.inv(Matrix_Beta))
    Beta = np.dot(be0, SB2)
    rBeta = np.dot(be0, rSB2)
    
    # 5. Evolução (Taxas de variação)
    dChi_dt = np.dot(Beta * rChi + Beta / (2*r) + Alpha / 4 * np.exp(-4*Chi) * Krr, inv_psi)
    dPhi_dt = np.dot(Alpha * Pi + Beta * rPhi, inv_psi)
    dPi_dt = np.dot((Beta*rPi + np.exp(-4*Chi)*(2*Alpha/r + rAlpha + 2*rChi*Alpha)*rPhi + np.exp(-4*Chi)*Alpha*rrPhi - Alpha*V), inv_psi)
    
    R = -8*np.exp(-4*Chi)*(rrChi + rChi**2 + 2*rChi/r)
    dtheta_dt = np.dot((Beta*rTheta + 0.5*Alpha*(R - 1.5*np.exp(-8*Chi)*Krr**2 - Pi**2 - np.exp(-4*Chi)*rPhi**2)
                       - Z*rAlpha + Alpha*(rZ + (6*rChi + 2/r)*Z) - 0.5*Alpha*kappa1*(kappa2 + 2)*Theta), inv_psi)

    dz_dt = np.dot((Beta*rZ - Z*rBeta +
                    Alpha*np.exp(-4*Chi)*(rKrr + 3*Krr/r + 2*rChi*Krr + np.exp(4*Chi)*Pi*rPhi) +
                    np.exp(-4*Chi)*(Alpha*rTheta - Theta*rAlpha) - kappa1*Alpha*Z), inv_SB2)
    
    return dChi_dt, dPhi_dt, dPi_dt, dtheta_dt, dz_dt, Alpha_central

def passo_rk4_z4(c0, a0, b0, theta0, z0, h, kappa1, kappa2, bases):
    K1_rc, K1_ra, K1_rb, K1_rtheta, K1_rz, _ = calcular_taxas_z4(c0, a0, b0, theta0, z0, kappa1, kappa2, bases)
    K1, L1, N1, dtheta1, dz1 = h*K1_rc, h*K1_ra, h*K1_rb, h*K1_rtheta, h*K1_rz

    K2_rc, K2_ra, K2_rb, K2_rtheta, K2_rz, _ = calcular_taxas_z4(c0+K1/2, a0+L1/2, b0+N1/2, theta0+dtheta1/2, z0+dz1/2, kappa1, kappa2, bases)
    K2, L2, N2, dtheta2, dz2 = h*K2_rc, h*K2_ra, h*K2_rb, h*K2_rtheta, h*K2_rz

    K3_rc, K3_ra, K3_rb, K3_rtheta, K3_rz, _ = calcular_taxas_z4(c0+K2/2, a0+L2/2, b0+N2/2, theta0+dtheta2/2, z0+dz2/2, kappa1, kappa2, bases)
    K3, L3, N3, dtheta3, dz3 = h*K3_rc, h*K3_ra, h*K3_rb, h*K3_rtheta, h*K3_rz

    K4_rc, K4_ra, K4_rb, K4_rtheta, K4_rz, _ = calcular_taxas_z4(c0+K3, a0+L3, b0+N3, theta0+dtheta3, z0+dz3, kappa1, kappa2, bases)
    K4, L4, N4, dtheta4, dz4 = h*K4_rc, h*K4_ra, h*K4_rb, h*K4_rtheta, h*K4_rz

    return (c0 + (K1 + 2*K2 + 2*K3 + K4)/6,
            a0 + (L1 + 2*L2 + 2*L3 + L4)/6,
            b0 + (N1 + 2*N2 + 2*N3 + N4)/6,
            theta0 + (dtheta1 + 2*dtheta2 + 2*dtheta3 + dtheta4)/6,
            z0 + (dz1 + 2*dz2 + 2*dz3 + dz4)/6)

# =========================================================================
# 4. PARÂMETROS E EXECUÇÃO DA SIMULAÇÃO
# =========================================================================
N = 80
L0 = 5
A0 = 0.05
r0 = 2
sigma = 1
kappa1 = 1000
kappa2 = 0

h = 0.0002
tf = 7
It = int(tf/h)
SAVE_STEP = 50

print(f"-> Configurando simulação espectral (N={N}, A0={A0})...")
bases = configurar_bases_espectrais(L0, N, M_plot=300)
c_atual, a_atual, b_atual, theta_atual, z_atual = criar_condicoes_iniciais(A0, r0, sigma, bases)

phi_set = []
print("-> Iniciando integração RK4 no tempo...")

t_atual = 0.0
for i in range(It + 1):
    # Salvar frame para a animação
    if i % SAVE_STEP == 0:
        phi_set.append(np.dot(a_atual, bases['psiplot']))
        
    # Verificar colapso para prevenir explosões matemáticas no horizonte
    _, _, _, _, _, Alpha_central = calcular_taxas_z4(c_atual, a_atual, b_atual, theta_atual, z_atual, kappa1, kappa2, bases)
    if Alpha_central < 1e-6:
        print(f"\n   [!] Formação de Buraco Negro detetada em t = {t_atual:.4f}M (Lapso colapsou).")
        break
        
    # Avanço no tempo
    c_atual, a_atual, b_atual, theta_atual, z_atual = passo_rk4_z4(c_atual, a_atual, b_atual, theta_atual, z_atual, h, kappa1, kappa2, bases)
    t_atual += h
    
    if i % 5000 == 0:
        print(f"   Simulado: t = {t_atual:.2f} M / {tf} M")

print("-> Simulação concluída com sucesso. Preparando renderização 3D...")

# =========================================================================
# 5. GERAÇÃO DA ANIMAÇÃO EXCLUSIVAMENTE 3D
# =========================================================================
fig_anim = plt.figure(figsize=(10, 8))
ax = fig_anim.add_subplot(111, projection='3d')
fig_anim.patch.set_facecolor('black') # Fundo preto para visual cinemático
ax.set_facecolor('black')

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10) 
ax.set_zlim(-1.5, 1.0)

# Estética dos eixos para fundo escuro
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('white')
ax.yaxis.pane.set_edgecolor('white')
ax.zaxis.pane.set_edgecolor('white')
ax.tick_params(colors='white')
ax.set_xlabel('x', color='white', fontsize=12)
ax.set_ylabel('y', color='white', fontsize=12)
ax.set_zlabel(r'$\phi(t, r)$', color='white', fontsize=14)

# Malha 3D
rplot = bases['rplot']
theta = np.linspace(0, 2*np.pi, len(rplot))
xn = np.outer(rplot, np.cos(theta))
yn = np.outer(rplot, np.sin(theta))
zn = np.zeros_like(xn)

# Superfície inicial
surface = [ax.plot_surface(xn, yn, zn, rstride=3, cstride=3, cmap='magma', edgecolor='none')]

time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, color='white', fontsize=14, fontweight='bold')
a0_text = ax.text2D(0.05, 0.90, f'$A_0$ = {A0}', transform=ax.transAxes, color='white', fontsize=12)

def init_anim():
    for surf in surface:
        surf.remove()
    surface[0] = ax.plot_surface(xn, yn, zn, rstride=3, cstride=3, cmap='magma', edgecolor='none', alpha=0.9)
    time_text.set_text("Time = 0.00 M")
    return surface[0], time_text

def animate_func(i):
    y = phi_set[i]
    for surf in surface:
        surf.remove()
        
    for j in range(len(rplot)):
        zn[j, :] = np.full_like(zn[0, :], y[j])
        
    # Renderização da superfície com alta performance
    surface[0] = ax.plot_surface(xn, yn, zn, rstride=4, cstride=4, cmap='magma', edgecolor='none', alpha=0.9)
    
    current_time = i * SAVE_STEP * h
    time_text.set_text(f"Time = {current_time:.2f} M")
    
    return surface[0], time_text

TOTAL_FRAMES = len(phi_set)
FPS = 30

anim = FuncAnimation(fig_anim, animate_func, init_func=init_anim, frames=TOTAL_FRAMES, interval=(1000/FPS), blit=False)

print(f"-> Renderizando {TOTAL_FRAMES} frames da superfície 3D (pode demorar alguns segundos)...")
writer = mpl.animation.FFMpegWriter(fps=FPS, metadata=dict(artist='Z4c Spectral Solver'), bitrate=2500)
anim.save("Colapso_Campo_Escalar_3D.mp4", writer=writer)

print("\n[+] SUCESSO! Animação 3D renderizada e guardada como 'Colapso_Campo_Escalar_3D.mp4'.")
plt.show()
