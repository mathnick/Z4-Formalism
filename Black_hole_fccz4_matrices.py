import mpmath
import numpy as np
import os

def fabricar_matrizes_fccz4(p=200, L0_val=15.0):
    # 220 Casas Decimais de Precisão Absoluta
    mpmath.mp.dps = 220
    L0 = mpmath.mpf(str(L0_val))
    print(f"[{p}x{p}] Iniciando compilação matemática com 220 casas decimais...")
    
    # 1. Pontos de Colocação (rcol) - Raízes Interiores de Chebyshev
    x_col = [-mpmath.cos(mpmath.pi * k / (p + 2)) for k in range(1, p + 2)]
    r_col = [L0 * (mpmath.mpf('1') + x) / (mpmath.mpf('1') - x) for x in x_col]
    
    # 2. Malha de Quadratura (rqcol e Wcolq) - Gauss-Legendre
    Nq = (3 * p) // 2
    n_leg = 2 * Nq + 2
    print(f"Calculando raízes de Legendre para integração de erro (Nq = {Nq})...")
    
    # Busca hiperprecisa de raízes (Newton-Raphson sobre semente Numpy)
    approx_roots, _ = np.polynomial.legendre.leggauss(n_leg)
    roots, weights = [], []
    for x0 in approx_roots:
        x = mpmath.mpf(str(x0))
        for _ in range(15): # 15 Passos de Newton garantem convergência extrema
            Pn = mpmath.legendre(n_leg, x)
            Pn_1 = mpmath.legendre(n_leg - 1, x)
            dPn = n_leg / (x**2 - mpmath.mpf('1')) * (x * Pn - Pn_1)
            x_new = x - Pn / dPn
            if mpmath.almosteq(x, x_new):
                x = x_new
                break
            x = x_new
        roots.append(x)
        # Cálculo do peso da quadratura de Gauss-Legendre
        Pn_1 = mpmath.legendre(n_leg - 1, x)
        dPn = n_leg / (x**2 - mpmath.mpf('1')) * (x * mpmath.legendre(n_leg, x) - Pn_1)
        w = mpmath.mpf('2') / ((mpmath.mpf('1') - x**2) * dPn**2)
        weights.append(w)
        
    # Extração das raízes positivas e mapeamento geométrico
    pos_roots = roots[Nq+1:]
    Wcolq = weights[Nq+1:]
    rq_col = [L0 * y / mpmath.sqrt(mpmath.mpf('1') - y**2) for y in pos_roots]
    
    # 3. Funções de Base Recombinadas (O segredo da estabilidade)
    def TL(k, r):
        x = (r - L0) / (r + L0)
        return mpmath.cos(k * mpmath.acos(x))

    def dTL_dr(k, r):
        if k == 0: return mpmath.mpf('0')
        x = (r - L0) / (r + L0)
        dx_dr = mpmath.mpf('2') * L0 / ((r + L0)**2)
        dT_dx = k * mpmath.sin(k * mpmath.acos(x)) / mpmath.sin(mpmath.acos(x))
        return dT_dx * dx_dr

    def d2TL_dr2(k, r):
        if k == 0: return mpmath.mpf('0')
        x = (r - L0) / (r + L0)
        dx_dr = mpmath.mpf('2') * L0 / ((r + L0)**2)
        d2x_dr2 = -mpmath.mpf('4') * L0 / ((r + L0)**3)
        dT_dx = k * mpmath.sin(k * mpmath.acos(x)) / mpmath.sin(mpmath.acos(x))
        T_k = mpmath.cos(k * mpmath.acos(x))
        d2T_dx2 = (-k**2 * T_k + x * dT_dx) / (mpmath.mpf('1') - x**2)
        return d2T_dx2 * (dx_dr**2) + dT_dx * d2x_dr2

    # A base psi_l que força 0 no infinito
    def psi(k, r):
        return mpmath.mpf('0.5') * (TL(k + 1, r) - TL(k, r))
    def dpsi_dr(k, r):
        return mpmath.mpf('0.5') * (dTL_dr(k + 1, r) - dTL_dr(k, r))
    def d2psi_dr2(k, r):
        return mpmath.mpf('0.5') * (d2TL_dr2(k + 1, r) - d2TL_dr2(k, r))

    # 4. Construção das Matrizes de Colocação
    print("Construindo Matrizes de Colocação (AlphaAl, DralphaAl...)...")
    AlphaAl = mpmath.matrix(p + 1, p + 1)
    DralphaAl = mpmath.matrix(p + 1, p + 1)
    DrralphaAl = mpmath.matrix(p + 1, p + 1)
    
    for l in range(p + 1):
        for k in range(p + 1):
            r = r_col[k]
            AlphaAl[k, l] = psi(l, r)
            DralphaAl[k, l] = L0 * dpsi_dr(l, r)
            DrralphaAl[k, l] = (L0**2) * d2psi_dr2(l, r)
            
    print("Invertendo AlphaAl (Aproveitando as 220 casas de precisão)...")
    Alalpha = AlphaAl**-1
    
    # 5. Construção das Matrizes de Quadratura (Para avaliação L2)
    print("Construindo Matrizes de Quadratura (PhiqA, DrphiqA...)...")
    PhiqA = mpmath.matrix(Nq + 1, p + 1)
    DrphiqA = mpmath.matrix(Nq + 1, p + 1)
    DrrphiqA = mpmath.matrix(Nq + 1, p + 1)
    
    for l in range(p + 1):
        for k in range(Nq + 1):
            r = rq_col[k]
            PhiqA[k, l] = psi(l, r)
            DrphiqA[k, l] = L0 * dpsi_dr(l, r)
            DrrphiqA[k, l] = (L0**2) * d2psi_dr2(l, r)

    # 6. Exportação Absoluta Garantida
    pasta_destino = os.path.dirname(os.path.abspath(__file__))
    print(f"\nSalvando ficheiros formatados na pasta: {pasta_destino}")
    
    def salvar(nome, dados):
        caminho = os.path.join(pasta_destino, nome)
        np.savetxt(caminho, dados, delimiter=' ')
        print(f" -> {nome} gravado com sucesso.")
        
    salvar('rcol.txt', np.array([float(r/L0) for r in r_col]))
    salvar('rqcol.txt', np.array([float(r/L0) for r in rq_col]))
    salvar('Wcolq.txt', np.array([float(w) for w in Wcolq]))
    
    salvar('AlphaAl.txt', np.array(AlphaAl.tolist(), dtype=float))
    salvar('Alalpha.txt', np.array(Alalpha.tolist(), dtype=float))
    salvar('DralphaAl.txt', np.array(DralphaAl.tolist(), dtype=float))
    salvar('DrralphaAl.txt', np.array(DrralphaAl.tolist(), dtype=float))
    
    salvar('PhiqA.txt', np.array(PhiqA.tolist(), dtype=float))
    salvar('DrphiqA.txt', np.array(DrphiqA.tolist(), dtype=float))
    salvar('DrrphiqA.txt', np.array(DrrphiqA.tolist(), dtype=float))
    
    print("\nSUCESSO TOTAL! Pode rodar a simulação RK4 sem erros.")

if __name__ == "__main__":
    fabricar_matrizes_fccz4(p=200, L0_val=15.0)
