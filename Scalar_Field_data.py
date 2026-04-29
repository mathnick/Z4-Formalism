import numpy as np
import math

# =========================================================================
# PARÂMETROS GLOBAIS DA SIMULAÇÃO
# =========================================================================
V = 0  # Potencial

# Parâmetros Físicos e Numéricos (Ajustados para a sua requisição)
A0_FIXO = 0.09     # Amplitude inicial alterada conforme solicitado
L0_FIXO = 10       # Parâmetro de mapa
N_FIXO = 100       # Ordem de truncamento
KAPPA1_FIXO = 80   # Amortecimento Z4
KAPPA2_FIXO = 0.5  # Amortecimento Z4

# Parâmetros de Tempo
h = 0.0002
tf = 15
It = int(tf/h)
t = 0.0 

# Parâmetros de Exportação (Blender)
SAVE_STEP = 50     # Salva 1 frame a cada 50 iterações RK4
M_PLOT = 500       # Resolução da malha no Blender (Quantidade de anéis concêntricos)
R_MAX = 15.0       # Raio físico máximo que será exportado para a malha 3D

# =========================================================================
# 1. FUNÇÕES DE PREPARAÇÃO (BASES E CONDIÇÕES INICIAIS)
# =========================================================================

def configurar_bases_espectrais(L0_valor, N_valor):
    col = np.cos(np.arange(2*N_valor + 4)*math.pi /(2*N_valor + 3))  
    colr = col[1:N_valor+2]
    r1 = L0_valor * colr/(np.sqrt(1-colr**2))       
    r = np.flip(r1)

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

    SB1 = 1/2*(SB[1:(N_valor+2),:] + SB[0:(N_valor+1),:])
    rSB1 = 1/2*(rSB[1:(N_valor+2),:] + rSB[0:(N_valor+1),:])                              
    
    SB2 = np.zeros([N_valor+1, N_valor+1])
    rSB2 = np.zeros([N_valor+1, N_valor+1])

    for i in range(N_valor+1):
        SB2[i,] = np.sin((2*(i+1/2)+1)*np.arctan(L0_valor/r))
        rSB2[i,] = -np.cos((2*i+2)*np.arctan(L0_valor/r))*(2*i+2)*L0_valor/(r**2*(1+L0_valor**2/r**2))
    inv_SB2 = np.linalg.pinv(SB2)

    psi_0 = np.zeros(N_valor+1)
    for i in range(N_valor+1):
        psi_0[i,] = np.sin((2*i+1)*math.pi/2)

    return {
        'r': r, 'psi': psi, 'rpsi': rpsi, 'rrpsi': rrpsi, 'inv_psi': inv_psi,
        'SB1': SB1, 'rSB1': rSB1, 'SB2': SB2, 'rSB2': rSB2, 'inv_SB2': inv_SB2,
        'psi_0': psi_0, 'L0': L0_valor, 'N': N_valor
    }                                                

def criar_condicoes_iniciais(A0_valor, bases):
    r = bases['r']; psi = bases['psi']; rpsi = bases['rpsi']; rrpsi = bases['rrpsi']
    inv_psi = bases['inv_psi']; N_valor = bases['N']
    
    r0 = 2
    sigma = 1

    Phi_0 = A0_valor*r**2*(np.exp(-(r-r0)**2/sigma**2)+np.exp(-(r+r0)**2/sigma**2))
    a0 = np.dot(Phi_0, inv_psi)
    Pi_0 = np.zeros(N_valor+1)
    b0 = np.dot(Pi_0, inv_psi) 

    c0 = 0.001 * np.ones([N_valor+1])
    rPhi= np.dot(a0, rpsi)

    tol = 1e-18
    n = 0
    nf = 50            
    while n <= nf:
        rChi=np.dot(c0,rpsi)
        rrChi=np.dot(c0,rrpsi)
        H0 = 4*rChi**2 + 4*rrChi + 8/r*rChi + 1/2*(rPhi)**2
        JH = 8*np.dot(c0,rpsi)*rpsi + 4*rrpsi + 8/r*rpsi
        inv_JH = np.linalg.inv(JH)
        cnew = c0 - np.dot(H0, inv_JH)
        if min(abs(cnew-c0)) < tol:
            break
        c0 = cnew
        n = n + 1

    theta0 = np.zeros(N_valor+1)
    z0 = np.zeros(N_valor+1)
    return c0, a0, b0, theta0, z0

# =========================================================================
# 2. FUNÇÕES DE EVOLUÇÃO E INTEGRAÇÃO RK4
# =========================================================================

def calcular_taxas_z4(c_coefs, a_coefs, b_coefs, theta_coefs, z_coefs, t, kappa1, kappa2, bases):
    psi = bases['psi']; rpsi = bases['rpsi']; rrpsi = bases['rrpsi']; inv_psi = bases['inv_psi']
    SB1 = bases['SB1']; rSB1 = bases['rSB1']; SB2 = bases['SB2']; rSB2 = bases['rSB2']; inv_SB2 = bases['inv_SB2']
    psi_0 = bases['psi_0']; r = bases['r']
    
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
    
    Matrix_Krr = 2 * rChi * SB1 + rSB1 + 3 / r * SB1
    inv_matrix_krr = np.linalg.inv(Matrix_Krr)
    rhsk = - Pi * rPhi * np.exp(4 * Chi)
    ck0 = np.dot(rhsk, inv_matrix_krr)
    Krr = np.dot(ck0, SB1)
    rKrr = np.dot(ck0, rSB1)

    Matrix_Alpha = (rrpsi + 2*(1/r + rChi)*rpsi -
                    (3/2)*np.exp(-4*Chi)*Krr**2*psi -
                    np.exp(4*Chi)*(Pi**2 - V)*psi -
                    2*np.exp(4*Chi)*rZ*psi -
                    2*np.exp(4*Chi)*(6*rChi + 2/r)*Z*psi)
    
    inv_matrix_alpha = np.linalg.inv(Matrix_Alpha)
    rhsal = ((3/2)*np.exp(-4*Chi)*Krr**2 +
             np.exp(4*Chi)*(Pi**2 - V) +
             2*np.exp(4*Chi)*rZ +
             2*np.exp(4*Chi)*(6*rChi + 2/r)*Z)
             
    al0 = np.dot(rhsal, inv_matrix_alpha)
    Alpha = 1 + np.dot(al0, psi)
    rAlpha = np.dot(al0, rpsi)
    Alpha_central = 1 + np.dot(al0, psi_0)                                  
    
    Matrix_Beta = rSB2/r - SB2/r**2
    inv_matrix_beta = np.linalg.inv(Matrix_Beta)
    rhsbe = 3/2 * Alpha * np.exp(-4*Chi) * Krr / r
    be0 = np.dot(rhsbe, inv_matrix_beta)
    Beta = np.dot(be0, SB2)
    rBeta = np.dot(be0, rSB2)
    
    dChi_dt = np.dot(Beta * rChi + Beta / (2*r) + Alpha / 4 * np.exp(-4*Chi) * Krr, inv_psi)
    dPhi_dt = np.dot(Alpha * Pi + Beta * rPhi, inv_psi)
    
    dPi_dt_expr = (Beta*rPi + np.exp(-4*Chi)*(2*Alpha/r + rAlpha + 2*rChi*Alpha)*rPhi + np.exp(-4*Chi)*Alpha*rrPhi - Alpha*V)
    dPi_dt = np.dot(dPi_dt_expr, inv_psi)            
    
    R = -8*np.exp(-4*Chi)*(rrChi + rChi**2 + 2*rChi/r)

    dTheta_dt_expr = (Beta*rTheta + 0.5*Alpha*(R - 1.5*np.exp(-8*Chi)*Krr**2 - Pi**2 - np.exp(-4*Chi)*rPhi**2)
                       - Z*rAlpha + Alpha*(rZ + (6*rChi + 2/r)*Z) - 0.5*Alpha*kappa1*(kappa2 + 2)*Theta)
    dtheta_dt = np.dot(dTheta_dt_expr, inv_psi)

    dZ_dt_expr = (Beta*rZ - Z*rBeta +
                  Alpha*np.exp(-4*Chi)*(rKrr + 3*Krr/r + 2*rChi*Krr + np.exp(4*Chi)*Pi*rPhi) +
                  np.exp(-4*Chi)*(Alpha*rTheta - Theta*rAlpha) - kappa1*Alpha*Z)
    dz_dt = np.dot(dZ_dt_expr, inv_SB2)
    
    return dChi_dt, dPhi_dt, dPi_dt, dtheta_dt, dz_dt, Alpha_central, ck0

def passo_rk4_z4(c0, a0, b0, theta0, z0, t, h, kappa1, kappa2, bases):
    K1_rc, K1_ra, K1_rb, K1_rtheta, K1_rz, _, _ = calcular_taxas_z4(c0, a0, b0, theta0, z0, t, kappa1, kappa2, bases)
    K1 = h * K1_rc; L1 = h * K1_ra; N1 = h * K1_rb; dtheta1 = h * K1_rtheta; dz1 = h * K1_rz

    c_temp2 = c0 + K1/2; a_temp2 = a0 + L1/2; b_temp2 = b0 + N1/2
    theta_temp2 = theta0 + dtheta1/2; z_temp2 = z0 + dz1/2
    K2_rc, K2_ra, K2_rb, K2_rtheta, K2_rz, _, _ = calcular_taxas_z4(c_temp2, a_temp2, b_temp2, theta_temp2, z_temp2, t + h/2, kappa1, kappa2, bases)
    K2 = h * K2_rc; L2 = h * K2_ra; N2 = h * K2_rb; dtheta2 = h * K2_rtheta; dz2 = h * K2_rz          
    
    c_temp3 = c0 + K2/2; a_temp3 = a0 + L2/2; b_temp3 = b0 + N2/2
    theta_temp3 = theta0 + dtheta2/2; z_temp3 = z0 + dz2/2 
    K3_rc, K3_ra, K3_rb, K3_rtheta, K3_rz, _, _ = calcular_taxas_z4(c_temp3, a_temp3, b_temp3, theta_temp3, z_temp3, t + h/2, kappa1, kappa2, bases)
    K3 = h * K3_rc; L3 = h * K3_ra; N3 = h * K3_rb; dtheta3 = h * K3_rtheta; dz3 = h * K3_rz

    c_temp4 = c0 + K3; a_temp4 = a0 + L3; b_temp4 = b0 + N3
    theta_temp4 = theta0 + dtheta3; z_temp4 = z0 + dz3
    K4_rc, K4_ra, K4_rb, K4_rtheta, K4_rz, _, _ = calcular_taxas_z4(c_temp4, a_temp4, b_temp4, theta_temp4, z_temp4, t + h, kappa1, kappa2, bases)
    K4 = h * K4_rc; L4 = h * K4_ra; N4 = h * K4_rb; dtheta4 = h * K4_rtheta; dz4 = h * K4_rz

    c_novo = c0 + (K1 + 2*K2 + 2*K3 + K4)/6
    a_novo = a0 + (L1 + 2*L2 + 2*L3 + L4)/6
    b_novo = b0 + (N1 + 2*N2 + 2*N3 + N4)/6
    theta_novo = theta0 + (dtheta1 + 2*dtheta2 + 2*dtheta3 + dtheta4)/6
    z_novo = z0 + (dz1 + 2*dz2 + 2*dz3 + dz4)/6
    
    return c_novo, a_novo, b_novo, theta_novo, z_novo


# =========================================================================
# 3. EXECUÇÃO E EXPORTAÇÃO DE DADOS
# =========================================================================

print(f"Configurando bases para N={N_FIXO}, L0={L0_FIXO}...")
bases = configurar_bases_espectrais(L0_FIXO, N_FIXO)
c_atual, a_atual, b_atual, theta_atual, z_atual = criar_condicoes_iniciais(A0_FIXO, bases)

# Geração do Grid Uniforme para o Blender
print(f"Gerando grid de plotagem para o Blender (Raio Max = {R_MAX}, Resolução = {M_PLOT})...")
rplot = np.linspace(0.000001, R_MAX, M_PLOT)
psiplot = np.zeros([N_FIXO+1, M_PLOT])
for i in range(N_FIXO+1):
    psiplot[i,] = np.sin((2*i+1)*np.arctan(L0_FIXO/rplot))

phi_set = []
t_atual = t

print(f"Iniciando simulação principal (A0 = {A0_FIXO}). Total de iterações: {It}")
for i in range(It):
    # Salvar frame atual para a animação
    if i % SAVE_STEP == 0:
        Phi_plot = np.dot(a_atual, psiplot)
        phi_set.append(Phi_plot)

    # Executar o passo RK4
    c_atual, a_atual, b_atual, theta_atual, z_atual = passo_rk4_z4(
        c_atual, a_atual, b_atual, theta_atual, z_atual, t_atual, h, KAPPA1_FIXO, KAPPA2_FIXO, bases
    )
    t_atual += h

    # Check de Colapso (Para evitar erros numéricos)
    if i % 500 == 0: # Checa o Lapse apenas de vez em quando para não atrasar o código
        _, _, _, _, _, Alpha_central, _ = calcular_taxas_z4(c_atual, a_atual, b_atual, theta_atual, z_atual, t_atual, KAPPA1_FIXO, KAPPA2_FIXO, bases)
        if Alpha_central < 1e-6:
             print(f"-> Colapso Crítico detectado em t = {t_atual:.4f}. Parando simulação e salvando até aqui.")
             break

print("\nSimulação concluída! Preparando arquivo de exportação...")

# =========================================================================
# 4. SALVANDO O ARQUIVO PARA O BLENDER
# =========================================================================
data_to_save = {
    'r_coords': rplot,
    'phi_values': np.array(phi_set)
}

file_name = 'scalar_field_A009.npy'
np.save(file_name, data_to_save)

print(f"Sucesso! O arquivo '{file_name}' foi salvo.")
print("Agora basta abrir o Blender, mudar o DATA_FILE_NAME no script para este nome e renderizar!")
