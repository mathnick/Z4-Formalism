import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from matplotlib.animation import FuncAnimation
# Import necessário para salvar em MP4 com ffmpeg
import matplotlib as mpl

# --- Parâmetros de Simulação ---
N = 80 # Truncation ordem
L0 = 5 # Map parameter
col = np.cos(np.arange(2*N + 4)*math.pi /(2*N + 3)) 
colr = col[1:N+2]
r1 = L0 * colr/(np.sqrt(1-colr**2))
r = np.flip(r1)

# Base Matrix (Tchebyshev Polinomials):
SB = np.zeros([N+3,N+1])
rSB = np.zeros([N+3,N+1])
rrSB = np.zeros([N+3,N+1])

for i in range(N+1+1+1):
  SB[i,] = np.sin((2*i+1)*np.arctan(L0/r))
for i in range(N+1+1+1):
  rSB[i,] = -np.cos((2*i+1)*np.arctan(L0/r))*(2*i+1)*L0/(r**2*(1+L0**2/r**2))
for i in range(N+1+1+1):
  rrSB[i,] = -np.sin((2*i+1)*np.arctan(L0/r))*(2*i+1)**2*L0**2/(r**4*(1+L0**2/r**2)**2)+2*np.cos((2*i+1)*np.arctan(L0/r))*(2*i+1)*L0/(r**3*(1+L0**2/r**2))-2*np.cos((2*i+1)*np.arctan(L0/r))*(2*i+1)*L0**3/(r**5*(1+L0**2/r**2)**2)

psi = SB[0:N+1,:] # Base function
rpsi = rSB[0:N+1,:]
rrpsi = rrSB[0:N+1,:]

# Initial conditions of Phi (Scalar field)
r0 = 2
sigma = 1
A0 = 0.09
Phi_0 = A0*r**2*(np.exp(-(r-r0)**2/sigma**2)+np.exp(-(r+r0)**2/sigma**2))
inv_psi = np.linalg.inv(psi)
a0 = np.dot(Phi_0, inv_psi)
Phi = np.dot(a0, psi)
rPhi= np.dot(a0, rpsi)

# --- Plot Setup ---
M = 3000 # plot truncation (alta resolução para gráficos estáticos)
rplot = np.linspace(0.000001,10,M)
colplot = rplot/np.sqrt(L0**2 + rplot**2)
SBplot = np.zeros([N+1,M])
rSBplot = np.zeros([N+1,M])
rrSBplot = np.zeros([N+1,M])

for i in range(N+1):
  SBplot[i,] = np.sin((2*i+1)*np.arctan(L0/rplot))
for i in range(N+1):
  rSBplot[i,] = -np.cos((2*i+1)*np.arctan(L0/rplot))*(2*i+1)*L0/(rplot**2*(1+L0**2/rplot**2))
for i in range(N+1):
  rrSBplot[i,] = -np.sin((2*i+1)*np.arctan(L0/rplot))*(2*i+1)**2*L0**2/(rplot**4*(1+L0**2/rplot**2)**2)+2*np.cos((2*i+1)*np.arctan(L0/rplot))*(2*i+1)*L0/(rplot**3*(1+L0**2/rplot**2))-2*np.cos((2*i+1)*np.arctan(L0/rplot))*(2*i+1)*L0**3/(rplot**5*(1+L0**2/rplot**2)**2)

psiplot = SBplot[0:(N+1),:]
rpsiplot = rSBplot[0:(N+1),:]
rrpsiplot = rrSBplot[0:(N+1),:]

Phi_plot0 = A0*rplot**2*(np.exp(-(rplot-r0)**2/sigma**2)+np.exp(-(rplot+r0)**2/sigma**2))
Phiplot = np.dot(a0, psiplot)

# --- Plot: Initial Conditions of Phi ---
plt.figure()
plt.plot(rplot, Phiplot, rplot, Phi_plot0, "--r")
plt.xlabel('r')
plt.xlim(0,8)
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.title("Condição Inicial Phi")
plt.show()

# --- Restante da Inicialização ---
Pi_0 = np.zeros(N+1)
b0 = np.dot(Pi_0, psi)
Pi = np.dot(b0, psi)
c0 = np.zeros([N+1]) 
for i in range(N+1):
  c0[i]  =  0.001
Chi=np.dot(c0,psi)
rChi=np.dot(c0,rpsi)
rrChi=np.dot(c0,rrpsi)

# Newton Raphson Loop... (omitted for brevity, assume c0 is solved)
# ...

# --- Inicialização para RK4 ---
SB1 = 1/2*(SB[1:(N+2),:] + SB[0:(N+1),:])
rSB1 = 1/2*(rSB[1:(N+2),:] + rSB[0:(N+1),:])
rrSB1 = 1/2*(rrSB[1:(N+2),:] + rrSB[0:(N+1),:])

SB2 = np.zeros([N+1,N+1])
rSB2 = np.zeros([N+1,N+1])
rrSB2 = np.zeros([N+1,N+1])

for i in range(N+1):
  SB2[i,] = np.sin((2*(i+1/2)+1)*np.arctan(L0/r))

for i in range(N+1):
  rSB2[i,] = -np.cos((2*i+2)*np.arctan(L0/r))*(2*i+2)*L0/(r**2*(1+L0**2/r**2))

for i in range(N+1):
  rrSB2[i,] = -np.sin((2*i+2)*np.arctan(L0/r))*(2*i+2)**2*L0**2/(r**4*(1+L0**2/r**2)**2)+2*np.cos((2*i+2)*np.arctan(L0/r))*(2*i+2)*L0/(r**3*(1+L0**2/r**2))-2*np.cos((2*i+2)*np.arctan(L0/r))*(2*i+2)*L0**3/(r**5*(1+L0**2/r**2)**2)

inv_SB2 = np.linalg.pinv(SB2)

z0 = np.zeros(N+1)
Z = np.dot(z0, SB2)
rZ = np.dot(z0, rSB2)
theta0 = np.zeros(N+1)
Theta = np.dot(theta0, psi)
rTheta = np.dot(theta0, rpsi)

Nq = int(3/2*N)
gauss_quadrature = np.polynomial.legendre.leggauss(Nq + 1)
new_col = gauss_quadrature[0]
P = np.zeros([Nq+3,Nq+1])
colP = np.zeros([Nq+3,Nq+1])

P[0,] = 1
P[1,] = new_col
for i in range(2,Nq+3):
  P[i,] = ((2*i-1)*new_col*P[i-1,] - (i-1)*P[i-2,])/(i)
for i in range(2,Nq+3):
  colP[i,] = i*P[i-1] + new_col*colP[i-1]

P_max = P[Nq+1]
colP_max = colP[Nq+1]
wq_col = 2/((1-new_col**2)*colP_max**2)
rq = L0*(1+new_col)/(1-new_col)

# ... (outras inicializações de bases)

# --- Loop Principal RK4 OTIMIZADO ---
h = 0.0002 # step size
tf = 7
It = int(tf/h) # 35,000 passos

# --- Otimização para Animação ---
SAVE_STEP = 50 # Salvar apenas a cada 50 passos
t = 0
current_step = 0
# -----------------------------------

kappa1 = 1000
kappa2 = 0
V = 0
M0 = 2*np.dot(np.arange(1, 2*N + 2, 2), c0)

Madm = []
Alpha_origin = []
phi_origin = []
L2HC = []
L2MC = []
phi_set = [] # Agora conterá apenas It / SAVE_STEP elementos
Madm_error = []


while t <= tf:
    # ----------------------------------------------------
    # ATENÇÃO: Somente salva o estado a cada SAVE_STEP
    # ----------------------------------------------------
    if current_step % SAVE_STEP == 0:
        # SALVA O CAMPO ESCALAR PARA ANIMAÇÃO
        phi_set.append(np.dot(a0, psiplot))
    
    current_step += 1
    # ----------------------------------------------------


    # =============================================
    # PRIMEIRO ESTÁGIO RK4 (e MONITORAMENTO)
    # [O restante das 4 etapas do RK4 foi mantido idêntico]
    # =============================================

    # Recuperar variáveis atuais
    Phi_1 = np.dot(a0, psi)
    rPhi_1 = np.dot(a0, rpsi)
    rrPhi_1 = np.dot(a0, rrpsi)
    Pi_1 = np.dot(b0, psi)
    rPi_1 = np.dot(b0, rpsi)
    Chi_1 = np.dot(c0, psi)
    rChi_1 = np.dot(c0, rpsi)
    rrChi_1 = np.dot(c0, rrpsi)

    #VARIÁVEIS Z4 (COMENTADAS/OPCIONAIS)
    Theta_1 = np.dot(theta0, psi)
    rTheta_1 = np.dot(theta0, rpsi)
    Z_1 = np.dot(z0, SB2)
    rZ_1 = np.dot(z0, rSB2)


    # Equação para Krr (MOMENTUM CONSTRAINT) - MESMA PARA ADM E Z4
    Matrix_Krr_1 = 2*rChi_1*SB1 + rSB1 + 3/r*SB1
    inv_matrix_krr_1 = np.linalg.inv(Matrix_Krr_1)
    rhsk_1 = - Pi_1*rPhi_1*np.exp(4*Chi_1)
    ck0_1 = np.dot(rhsk_1, inv_matrix_krr_1)
    Krr_1 = np.dot(ck0_1, SB1)
    rKrr_1 = np.dot(ck0_1, rSB1)

    # Lapse - VERSÃO Z4 CORRIGIDA (sem κ₁, κ₂)
    Matrix_Alpha_1 = (rrpsi + 2*(1/r + rChi_1)*rpsi -
            (3/2)*np.exp(-4*Chi_1)*Krr_1**2*psi -
            np.exp(4*Chi_1)*(Pi_1**2 - V)*psi -
            2*np.exp(4*Chi_1)*rZ_1*psi -
            2*np.exp(4*Chi_1)*(6*rChi_1 + 2/r)*Z_1*psi)

    inv_matrix_alpha_1 = np.linalg.inv(Matrix_Alpha_1)

    rhsal_1 = ((3/2)*np.exp(-4*Chi_1)*Krr_1**2 +
            np.exp(4*Chi_1)*(Pi_1**2 - V) +
            2*np.exp(4*Chi_1)*rZ_1 +
            2*np.exp(4*Chi_1)*(6*rChi_1 + 2/r)*Z_1)

    al0_1 = np.dot(rhsal_1, inv_matrix_alpha_1)
    Alpha_1 = 1 + np.dot(al0_1, psi)
    rAlpha_1 = np.dot(al0_1, rpsi)

    # Equação para Beta - MESMA PARA ADM E Z4
    Matrix_Beta_1 = rSB2/r - SB2/r**2
    inv_matrix_beta_1 = np.linalg.inv(Matrix_Beta_1)
    rhsbe_1 = 3/2*Alpha_1*np.exp(-4*Chi_1)*Krr_1/r
    be0_1 = np.dot(rhsbe_1, inv_matrix_beta_1)
    Beta_1 = np.dot(be0_1, SB2)
    rBeta_1 = np.dot(be0_1, rSB2)

    #Equação para a densidade - Mesma para ADM e Z4
    rho_1 = Pi_1**2/2 + (rPhi_1**2)*np.exp(-4*Chi_1)/2 + V

    # EQUAÇÕES DE EVOLUÇÃO - VERSÃO ADM (ORIGINAL)
    db_1 = np.dot(Beta_1*rPi_1 + np.exp(-4*Chi_1)*(2*Alpha_1/r + rAlpha_1 + 2*rChi_1*Alpha_1)*rPhi_1 + np.exp(-4*Chi_1)*Alpha_1*rrPhi_1 - Alpha_1*V, inv_psi)
    dc_1 = np.dot(Beta_1*rChi_1 + Beta_1/2/r + Alpha_1/4*np.exp(-4*Chi_1)*Krr_1, inv_psi)
    da_1 = np.dot(Alpha_1*Pi_1 + Beta_1*rPhi_1, inv_psi)


    # EVOLUÇÃO THETA - Versão MATLAB
    R_1 = -8*np.exp(-4*Chi_1)*(rrChi_1 + rChi_1**2 + 2*rChi_1/r)

    # EVOLUÇÃO THETA - K1 e K2
    dTheta_dt_1 = (Beta_1*rTheta_1 + 0.5*Alpha_1*(R_1 - 1.5*np.exp(-8*Chi_1)*Krr_1**2 - Pi_1**2 - np.exp(-4*Chi_1)*rPhi_1**2)
                - Z_1*rAlpha_1 + Alpha_1*(rZ_1 + (6*rChi_1 + 2/r)*Z_1) - 0.5*Alpha_1*kappa1*(kappa2 + 2)*Theta_1)
    dtheta_dt_1 = np.dot(dTheta_dt_1, inv_psi)


    # EVOLUÇÃO Z - COM K1 E K2
    dZ_dt_1 = (Beta_1*rZ_1 - Z_1*rBeta_1 +
            Alpha_1*np.exp(-4*Chi_1)*(rKrr_1 + 3*Krr_1/r + 2*rChi_1*Krr_1 + np.exp(4*Chi_1)*Pi_1*rPhi_1) +
            np.exp(-4*Chi_1)*(Alpha_1*rTheta_1 - Theta_1*rAlpha_1) - kappa1*Alpha_1*Z_1)
    dz_dt_1 = np.dot(dZ_dt_1, inv_SB2)


    # Primeiros incrementos RK4
    K1 = h * dc_1
    L1 = h * da_1
    N1 = h * db_1

    # INCREMENTOS Z4 (COMENTADOS)
    dtheta1 = h * dtheta_dt_1
    dz1 = h * dz_dt_1

    # MONITORAMENTO
    qPhi = np.dot(a0, qpsi)
    rqPhi = np.dot(a0, rqpsi)
    qPi = np.dot(b0, qpsi)
    qChi = np.dot(c0, qpsi)
    rqChi = np.dot(c0, rqpsi)
    rrqChi = np.dot(c0, rrqpsi)
    qKrr = np.dot(ck0_1, qSB1)
    H = 4*rqChi**2 + 4*rrqChi + 8*rqChi/rq + 3/4*np.exp(-4*qChi)*qKrr**2 + np.exp(4*qChi)*(1/2*qPi**2 + np.exp(-4*qChi)/2*rqPhi**2)
    
    if current_step % SAVE_STEP == 0:
      L2HC.append((1/2*np.dot(H**2,wq_col))**(1/2))
      rqKrr = np.dot(ck0_1, rqSB1)
      M = 2*rqChi*qKrr + rqKrr + 3/rq*qKrr + qPi*rqPhi*np.exp(4*qChi)
      L2MC.append((1/2*np.dot(M**2,wq_col))**1/2)

      Alpha_0 = 1 + np.dot(al0_1, psi_0)
      Alpha_origin.append(Alpha_0)

      phi_0 = np.dot(a0, psi_0)
      phi_origin.append(phi_0)
      
      Madm = 2*np.dot(np.arange(1, 2*N + 2, 2), c0)
      Madm_pc = abs(Madm - M0)/M0 * 100
      Madm_error.append(Madm_pc)

    # --- Estágios 2, 3 e 4 do RK4 (Código completo omitido para não duplicar, mas é o mesmo que você forneceu) ---
    # ... (Cálculo K2, L2, N2)
    # ... (Cálculo K3, L3, N3)
    # ... (Cálculo K4, L4, N4)
    # --- FIM ESTÁGIOS RK4

    # =============================================
    # ATUALIZAÇÃO FINAL
    # =============================================

    t = t + h

    # Atualizar todos os coeficientes ADM
    a0 = filter1 * (a0 + (L1 + 2*L2 + 2*L3 + L4)/6)
    b0 = filter1 * (b0 + (N1 + 2*N2 + 2*N3 + N4)/6)
    c0 = filter1 * (c0 + (K1 + 2*K2 + 2*K3 + K4)/6)

    # ATUALIZAÇÃO Z4 (COMENTADA)
    theta0 = filter1 * (theta0 + (dtheta1 + 2*dtheta2 + 2*dtheta3 + dtheta4)/6)
    z0 = filter1 * (z0 + (dz1 + 2*dz2 + 2*dz3 + dz4)/6)

# --- FIM DO LOOP WHILE

# --- Geração da Animação OTIMIZADA ---

# Seu código original de setup da figura 3D (mantido)
# ...

# Configuração da Animação
fig_anim = plt.figure(figsize=(10, 5))
ax = plt.axes(xlim=(0, 10), ylim = (-2, 1.5))
line, = ax.plot([], [], lw=2, color='b')
ax.set_title(r"Evolução do Campo Escalar $\phi(t, r)$")
ax.set_xlabel("r")
ax.set_ylabel(r"$\phi$")

# Usamos len(phi_set) para o número real de frames salvos
TOTAL_FRAMES_SALVOS = len(phi_set)
FPS = 30 # Velocidade do vídeo

# Textos para exibir o tempo
initA0_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)
x = rplot

def init_anim():
    line.set_data([], [])
    initA0_text.set_text(f"$A_0 = {A0}$")
    time_text.set_text("Time = 0.00")
    return line, initA0_text, time_text

def animate_func(i):
    # i é o índice do frame, que vai de 0 a TOTAL_FRAMES_SALVOS - 1
    y = phi_set[i]
    line.set_data(x, y)
    
    # Calculamos o tempo real: tempo_real = i * SAVE_STEP * h
    current_time = i * SAVE_STEP * h
    time_text.set_text(f"Time = {current_time:.2f}")
    
    return line, initA0_text, time_text

# Crio a animação
anim = FuncAnimation(fig_anim, animate_func, init_func=init_anim,
                     frames=TOTAL_FRAMES_SALVOS, interval=(1000/FPS), blit=True)

# ----------------------------------------------------
# OTIMIZAÇÃO CRÍTICA: Salvar como MP4 usando 'ffmpeg'
# ----------------------------------------------------
print(f"Salvando {TOTAL_FRAMES_SALVOS} frames em MP4. Isso levará alguns minutos...")

# Configure a velocidade de quadros e o nome do arquivo
writer = mpl.animation.FFMpegWriter(fps=FPS, metadata=dict(artist='Z4c Solver'), bitrate=1800)
anim.save("animation_MS_collapse_optimized.mp4", writer=writer)

print("Animação salva com sucesso em 'animation_MS_collapse_optimized.mp4'")

plt.show()
