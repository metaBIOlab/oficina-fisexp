"""
Espectro de potencia de Laplace-Beltrami: calculo com cache e graficos
======================================================================

Bruno Mota -- metaBIO lab, Instituto de Fisica / UFRJ

Terceiro da familia, depois de spectralmesh_dipolar_modes e de
Spectralmesh_power_spectrum. Faz a mesma conta que o segundo, mas preparado
para as duas coisas que aparecem quando se quer muitos modos: o calculo nao
cabe de uma vez na memoria, e a gente sempre quer mais modos do que rodou da
ultima vez.


O QUE ESTE PROGRAMA FAZ
-----------------------
Dada uma superficie fechada (.stl, .ply ou .obj), ele:

1. Procura na pasta de saida um espectro ja calculado. O que ja estiver la e
   reaproveitado; so os modos que faltam sao calculados. Rodar de novo pedindo
   mais modos continua de onde parou, em vez de recomecar. O cache so e aceito
   se a area gravada nele bater com a da malha carregada, para que trocar a
   malha sem trocar o nome da pasta nao reuse o espectro errado.

2. Calcula os autopares que faltam em janelas: em vez de pedir N modos de uma
   vez, pede blocos de --window-size modos em torno de deslocamentos espectrais
   sucessivos (shift-invert), projeta o campo na hora e descarta os autovetores.
   Guardar 3000 autovetores de uma malha de 150 mil vertices sao 3,5 GB; guardar
   os coeficientes sao alguns kilobytes. As janelas se sobrepoem em 15% para que
   nenhum autovalor caia entre duas.

3. Faz o mesmo para o fecho convexo da superficie, remalhado. O fecho e liso por
   construcao, e serve de controle: qualquer coisa que apareca nas duas curvas
   nao vem do dobramento. Como o fecho tem menos area, precisa de menos modos
   para chegar ao mesmo comprimento de onda, e o programa ajusta isso sozinho
   pela lei de Weyl. Use --no-hull para pular.

4. Desenha tres paineis e grava tudo na pasta de saida.


AS TRES QUANTIDADES DOS PAINEIS
-------------------------------
A. Espectro de potencia. P_n = |c_n|^2 com c_n = phi_n^T M f, contra o
   comprimento de onda ell = 2 sqrt(2/mu). Pontos claros sao modos individuais;
   os circulos sao medias por quantil, com o mesmo numero de modos em cada
   ponto. As retas guia tem inclinacao dada por --ref-beta, no sentido
   P ~ mu^(-beta).

B. Area efetiva -- so no campo posicao. Para ele vale a identidade exata

       soma_n mu_n P_n = 2 A

   (porque |grad r|^2 = 2 em qualquer superficie). A soma parcial ate um dado
   mu define entao uma area associada a escala correspondente,

       A_ef(ell) = (1/2) soma_{mu_n < 8/ell^2} mu_n P_n

   que vai de zero ate a area total conforme se incluem modos mais finos. As
   retas guia tem inclinacao 2 - d_f para cada d_f de --ref-df.

   Para qualquer outro campo a mesma soma parcial e a energia de Dirichlet
   acumulada, integral de |grad f|^2 ate aquela escala. Continua sendo uma
   quantidade legitima e informativa, mas nao e uma area: o painel muda de
   rotulo, as guias em d_f nao sao desenhadas, e o eixo passa a ser E(ell).

C. Expoente local, d log A_ef / d log ell, ajustado numa janela deslizante de
   --slope-window decadas. As horizontais estao em 2 - d_f para cada valor de
   --ref-df. Fora do campo posicao o painel mostra a inclinacao de E(ell) e as
   horizontais somem, porque a leitura em d_f nao se aplica.

O programa desenha as guias e nao conclui nada a partir delas. A leitura fica
com voce.


COMO USAR
---------
    python Spectralmesh_Power_Spectrum_Plots.py malha.stl
    python Spectralmesh_Power_Spectrum_Plots.py malha.stl --n-modes 3000
    python Spectralmesh_Power_Spectrum_Plots.py malha.ply --no-visualize
    python Spectralmesh_Power_Spectrum_Plots.py --ref-df 2 2.4 2.5 2.6
    python Spectralmesh_Power_Spectrum_Plots.py               # lote

Sem argumento, processa todos os .stl, .ply e .obj da pasta _Input_Meshes.

    --n-modes N        quantos modos ter ao final, contando os que ja estao em
                       cache (padrao 800).
    --window-size K    modos por janela (padrao 200). Diminua se faltar memoria;
                       aumente se sobrar, que fica mais rapido.
    --field CAMPO      'position' (padrao) ou 'curvature'. O segundo e o
                       primeiro multiplicado por mu^2/4, exatamente.
    --ref-df D [D...]  valores de d_f das guias dos paineis B e C (padrao 2 2.5).
    --ref-beta B [B..] inclinacoes das guias do painel A (padrao 1.5 2 2.5).
    --slope-window W   meia-largura, em decadas, da janela do painel C
                       (padrao 0.175).
    --modes-per-point  modos por ponto nas medias do painel A (padrao 40).
    --no-hull          nao calcula o fecho convexo.
    --no-visualize     nao abre a janela; o png e o csv saem do mesmo jeito.
    --recompute        ignora o cache e recalcula do zero.


O QUE SAI
---------
Uma pasta com o nome da malha, contendo:

    <malha>.<ext>                copia da malha de entrada.
    <malha>_spectrum.csv         espectro da superficie, uma linha por modo.
    <malha>_hull_spectrum.csv    espectro do fecho convexo, se calculado.
    <malha>_spectrum.png         os tres paineis.

Colunas dos csv:

    mode_index          n, a partir de 1 (o modo constante fica de fora).
    eigenvalue          mu_n.
    power               P_n do campo posicao, sempre -- e esta coluna que serve
                        de cache. Fora do campo posicao aparece tambem uma
                        coluna power_<campo> com a potencia transformada.
    wavelength          ell_n = 2 sqrt(2/mu_n), nas unidades da malha.
    area_eff            A_ef acumulada ate este modo. So no campo posicao;
                        nos demais a coluna se chama dirichlet_acc, e e a
                        energia de Dirichlet acumulada.
    area_eff_over_A     a mesma coisa dividida pela area total (ou, fora do
                        campo posicao, dirichlet_acc_over_total, dividida pelo
                        total acumulado).
    area_total          area da superficie. Repete em todas as linhas, e serve
                        para validar o cache: se a area gravada nao bater com a
                        da malha carregada, o cache e descartado em vez de
                        reaproveitado por engano.

Sao esses csv que o programa le no inicio da execucao seguinte.
"""

import csv
import gc
import shutil
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import trimesh
import pymeshlab
import robust_laplacian
import scipy.sparse.linalg as sla

n_modes_default = 800
window_size_default = 200
WINDOW_OVERLAP = 0.85          # janelas avancam 85% da propria largura

MESH_EXTENSIONS = (".stl", ".ply", ".obj")
FIELDS = ("position", "curvature")

# -------------------- Mesh preparation --------------------

def _meshlab_absolute(x: float):
    """AbsoluteValue virou PureValue entre versoes do pymeshlab; tenta as duas."""
    for name in ("PureValue", "AbsoluteValue"):
        cls = getattr(pymeshlab, name, None)
        if cls is not None:
            return cls(float(x))
    return float(x)

def remesh_regular(mesh, target_edge: Optional[float] = None, iterations: int = 10):
    """
    Remalhagem isotropica pelo MeshLab. O fecho convexo devolvido pelo qhull tem
    triangulos muito alongados, que estragam o espectro; aqui ele e refeito com
    o mesmo comprimento de aresta da superficie original, para que os dois
    carreguem erro de discretizacao comparavel.
    """
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(np.asarray(mesh.vertices, dtype=float),
                               np.asarray(mesh.faces, dtype=np.int32)))
    kwargs = {} if target_edge is None else dict(targetlen=_meshlab_absolute(target_edge))
    ms.meshing_isotropic_explicit_remeshing(iterations=iterations, **kwargs)
    out = ms.current_mesh()
    return trimesh.Trimesh(out.vertex_matrix(), out.face_matrix(), process=False)

def load_largest_component(path):
    mesh = trimesh.load_mesh(path)
    return max(mesh.split(only_watertight=False), key=lambda m: len(m.vertices))

# -------------------- Spectrum, windowed and cached --------------------

def _project(evec, M_mat, Mf, chunk: int = 25) -> np.ndarray:
    """
    Normaliza (phi^T M phi = 1) e projeta, em blocos de colunas. Em bloco porque
    M @ evec inteiro duplicaria a base em memoria, que e justamente o que se
    esta tentando evitar.
    """
    k = evec.shape[1]
    P = np.empty(k)
    for a in range(0, k, chunk):
        b = min(a + chunk, k)
        Mb = M_mat @ evec[:, a:b]
        evec[:, a:b] /= np.sqrt(np.einsum('ij,ij->j', evec[:, a:b], Mb))
        c = evec[:, a:b].T @ Mf
        P[a:b] = np.einsum('nc,nc->n', c, c) if c.ndim == 2 else c ** 2
        del Mb
    return P

def _merge_new(have: Optional[np.ndarray], new: np.ndarray) -> np.ndarray:
    """
    Junta os modos de uma janela nova aos que ja existem.

    Deliberadamente NAO deduplica por valor. Autovalores repetidos costumam ser
    degenerescencias de verdade -- numa esfera o autovalor l(l+1) tem
    multiplicidade 2l+1 -- e jogar fora os repetidos apagaria modos legitimos e
    a potencia que eles carregam. Como o que ja existe e sempre um bloco
    contiguo a partir de mu = 0, o criterio pode ser posicional: aproveita-se da
    janela tudo que esta acima do topo, e os empates exatamente no topo sao
    resolvidos por contagem.
    """
    new = new[np.argsort(new[:, 0])]
    if have is None or len(have) == 0:
        return new
    have = have[np.argsort(have[:, 0])]
    mu_top = float(have[-1, 0])
    tol = 1e-9 * max(1.0, abs(mu_top))
    n_tie_have = int(np.sum(np.abs(have[:, 0] - mu_top) <= tol))
    tie_mask = np.abs(new[:, 0] - mu_top) <= tol
    extra = new[tie_mask][:max(0, int(tie_mask.sum()) - n_tie_have)]
    above = new[new[:, 0] > mu_top + tol]
    out = np.vstack([have, extra, above])
    return out[np.argsort(out[:, 0])]

def compute_spectrum_windowed(verts, faces, area, n_modes,
                              window_size=window_size_default,
                              cached: Optional[np.ndarray] = None,
                              label: str = ""):
    """
    Devolve um array (n, 2) de pares (mu, P), calculando so o que falta em
    relacao a `cached`. As janelas sao centradas em deslocamentos sucessivos e
    dimensionadas pela densidade de modos de Weyl, rho = A / 4pi, que e quantos
    autovalores existem por unidade de mu.
    """
    L, M_mat = robust_laplacian.mesh_laplacian(verts, faces)
    Mf = M_mat @ (verts - verts.mean(axis=0))
    rho = area / (4.0 * np.pi)
    dmu = WINDOW_OVERLAP * window_size / rho

    have = (cached[np.argsort(cached[:, 0])]
            if cached is not None and len(cached) else np.empty((0, 2)))
    if len(have) >= n_modes:
        print(f"{label}cache ja tem {len(have)} modos; nada a calcular.")
        return have

    if len(have):
        print(f"{label}cache com {len(have)} modos (mu ate {have[:, 0].max():.5g}); "
              f"continuando.")

    while len(have) < n_modes:
        # Comeca a proxima janela um pouco abaixo do topo do que ja existe, para
        # que as duas se sobreponham e nenhum autovalor caia no vao.
        mu_top = float(have[:, 0].max()) if len(have) else 0.0
        sigma = max(1e-8, mu_top + 0.35 * window_size / rho)
        ev, evec = sla.eigsh(L, k=window_size, M=M_mat, sigma=sigma,
                             which='LM', ncv=2 * window_size + 30)
        P = _project(evec, M_mat, Mf)
        del evec
        gc.collect()
        before = len(have)
        have = _merge_new(have, np.column_stack([ev, P]))
        gained = len(have) - before
        print(f"{label}janela em sigma = {sigma:.5g}: mu ate {have[:, 0].max():.5g}, "
              f"+{gained} modos, {len(have)} acumulados", flush=True)
        if gained == 0:
            print(f"{label}janela nao acrescentou nada; parando em {len(have)} modos.")
            break

    return have

# -------------------- Derived quantities --------------------

def is_position_field(field: str) -> bool:
    """
    A identidade soma_n mu_n P_n = 2A vale porque |grad r|^2 = 2 em qualquer
    superficie, e portanto SO para o campo posicao. Para qualquer outro campo a
    mesma soma da a energia de Dirichlet, integral de |grad f|^2, que nao e uma
    area. O acumulado continua sendo uma quantidade legitima e util, mas muda de
    nome e perde a leitura em d_f.
    """
    return field == "position"

def spectrum_table(D: np.ndarray, area: float, field: str = "position"):
    """
    (mu, P, ell, cum) com o modo constante removido e o campo aplicado.

    `cum` e a soma parcial de mu_n P_n: dividida por 2 no campo posicao, onde
    ela e a area efetiva A_ef; crua nos demais, onde e a energia de Dirichlet
    acumulada.
    """
    mu, P = D[:, 0], D[:, 1]
    ok = mu > 1e-9
    mu, P = mu[ok], P[ok]
    if field == "curvature":
        P = P * mu ** 2 / 4.0
    ell = 2.0 * np.sqrt(2.0 / mu)
    cum = np.cumsum(mu * P)
    if is_position_field(field):
        cum = cum / 2.0
    return mu, P, ell, cum

def quantile_means(x, y, per_point=40):
    """Medias com o mesmo numero de modos por ponto, e nao por faixa em log."""
    groups = np.array_split(np.arange(len(x)), max(3, len(x) // per_point))
    return (np.array([np.exp(np.log(x[g]).mean()) for g in groups]),
            np.array([y[g].mean() for g in groups]))

def local_slope(x, y, half_decade=0.175):
    """d log y / d log x por ajuste numa janela deslizante."""
    lx, ly = np.log10(x), np.log10(y)
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        s = np.abs(lx - lx[i]) <= half_decade
        if s.sum() >= 5:
            out[i] = np.polyfit(lx[s], ly[s], 1)[0]
    return out

# -------------------- Output --------------------

def prepare_output_folder(mesh_path: Path) -> Path:
    out_dir = Path(__file__).resolve().parent / mesh_path.stem
    out_dir.mkdir(exist_ok=True)
    return out_dir

def save_spectrum_csv(csv_path: Path, D: np.ndarray, area: float, field: str):
    mu, P_field, ell, cum = spectrum_table(D, area, field)
    _, P_raw, _, _ = spectrum_table(D, area, "position")
    pos = is_position_field(field)
    # A coluna `power` guarda SEMPRE o campo posicao, porque este arquivo e o
    # cache: se gravasse o campo transformado, a leitura seguinte reaplicaria a
    # transformacao por cima. A potencia da curvatura e mu^2/4 vezes esta, e sai
    # das duas primeiras colunas com uma multiplicacao.
    names = (["area_eff", "area_eff_over_A"] if pos
             else ["dirichlet_acc", "dirichlet_acc_over_total"])
    denom = area if pos else max(cum[-1], 1e-300)
    header = ["mode_index", "eigenvalue", "power", "wavelength",
              names[0], names[1], "area_total"]
    if not pos:
        header.insert(3, f"power_{field}")
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for n in range(len(mu)):
            row = [n + 1, f"{mu[n]:.10g}", f"{P_raw[n]:.10g}"]
            if not pos:
                row.append(f"{P_field[n]:.10g}")
            row += [f"{ell[n]:.10g}", f"{cum[n]:.10g}",
                    f"{cum[n] / denom:.10g}", f"{area:.10g}"]
            w.writerow(row)

def load_spectrum_csv(csv_path: Path, area: Optional[float] = None,
                      rtol: float = 1e-3) -> Optional[np.ndarray]:
    """
    Le o cache e devolve pares (mu, P) crus, sem o campo aplicado.

    Se `area` for passada, confere contra a coluna area_total gravada no
    arquivo e descarta o cache se as duas nao baterem dentro de `rtol`. Sem essa
    checagem, trocar a malha mantendo o nome da pasta faria o programa reusar
    silenciosamente o espectro da malha antiga. A area e um resumo barato e
    sensivel da geometria: qualquer mudanca de malha que importe para o espectro
    a move muito mais que 0,1%.
    """
    if not csv_path.exists():
        return None
    try:
        raw = np.genfromtxt(csv_path, delimiter=",", names=True)
        if raw.size == 0:
            return None
        if area is not None and raw.dtype.names and "area_total" in raw.dtype.names:
            cached_area = float(np.atleast_1d(raw["area_total"])[0])
            if abs(cached_area - area) > rtol * max(abs(area), 1e-30):
                print(f"  (cache em {csv_path.name} foi feito numa malha de area "
                      f"{cached_area:.6g}, mas esta tem {area:.6g}; ignorando o cache)")
                return None
        return np.column_stack([np.atleast_1d(raw["eigenvalue"]),
                                np.atleast_1d(raw["power"])])
    except Exception as e:
        print(f"  (cache em {csv_path.name} ilegivel: {type(e).__name__}; recalculando)")
        return None

# -------------------- Plot --------------------

def make_figure(png_path: Path, name: str, surf, hull,
                ref_df=(2.0, 2.5), ref_beta=(1.5, 2.0, 2.5),
                slope_window=0.175, per_point=40, field="position", show=True):
    """
    surf e hull sao dicionarios com mu, P, ell, cum, area; hull pode ser None.

    Os paineis B e C mudam de rotulo conforme o campo. So no campo posicao o
    acumulado e uma area e a inclinacao local se le como 2 - d_f; nos demais e a
    energia de Dirichlet acumulada, que continua sendo uma quantidade util mas
    nao admite essa leitura, e por isso as referencias em d_f nao sao desenhadas.
    """
    pos = is_position_field(field)
    import matplotlib.pyplot as plt

    series = [(surf, 'tab:red', name)]
    if hull is not None:
        series.append((hull, 'tab:blue', 'fecho convexo'))

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.0))

    # --- A: espectro de potencia ---
    a = ax[0]
    for d, col, lab in series:
        a.loglog(d['ell'], d['P'], '.', color=col, ms=1.5, alpha=0.16)
        x, y = quantile_means(d['ell'], d['P'], per_point)
        a.loglog(x, y, 'o-', color=col, ms=5, lw=1.7, label=lab)
    x0, y0 = quantile_means(surf['ell'], surf['P'], per_point)
    anchor = len(x0) // 3
    xg = np.array([x0[anchor], x0.min()])
    for beta in ref_beta:
        yg = y0[anchor] * (xg / xg[0]) ** (2 * beta)
        a.loglog(xg, yg, '--', color='0.35', lw=1.1)
        a.annotate(rf'$\beta={beta:g}$', (xg[1], yg[1]), fontsize=9,
                   color='0.35', ha='right', va='top')
    a.set_xlabel(r'comprimento de onda  $\ell = 2\sqrt{2/\mu}$')
    a.set_ylabel(r'potência por modo  $P$')
    a.set_title('A. espectro de potência', fontsize=11)
    a.invert_xaxis()
    a.grid(alpha=.25, which='both')
    a.legend(fontsize=8.5, loc='lower left')

    # --- B: area efetiva ---
    b = ax[1]
    for d, col, lab in series:
        y = d['cum'] / d['area'] if pos else d['cum']
        b.loglog(d['ell'], y, color=col, lw=2.2, label=lab)
    if pos:
        i0 = np.argmin(np.abs(surf['ell'] - np.exp(np.log(surf['ell']).mean())))
        xg = np.array([surf['ell'][i0], surf['ell'].min()])
        y0b = surf['cum'][i0] / surf['area']
        for df in ref_df:
            b.loglog(xg, y0b * (xg / xg[0]) ** (2.0 - df), '--', color='0.35', lw=1.1)
            b.annotate(rf'$d_f={df:g}$', (xg[1], y0b * (xg[1] / xg[0]) ** (2.0 - df)),
                       fontsize=9, color='0.35', ha='right', va='bottom')
    b.set_xlabel(r'$\ell$')
    if pos:
        b.set_ylabel(r'$A_{ef}(\ell)\,/\,A$')
        b.set_title(r'B. área efetiva:  $A_{ef}=\frac{1}{2}\sum_{\mu_n<8/\ell^2}\mu_n P_n$',
                    fontsize=11)
    else:
        b.set_ylabel(r'$E(\ell) = \sum_{\mu_n<8/\ell^2}\mu_n P_n$')
        b.set_title('B. energia de Dirichlet acumulada\n'
                    f'(campo {field}: não é área)', fontsize=11)
    b.invert_xaxis()
    b.grid(alpha=.25, which='both')
    b.legend(fontsize=8.5, loc='lower right')

    # --- C: expoente local ---
    c = ax[2]
    for d, col, lab in series:
        m = d['cum'] > 0
        c.semilogx(d['ell'][m], local_slope(d['ell'][m], d['cum'][m], slope_window),
                   color=col, lw=2.2, label=lab)
    if pos:
        for df in ref_df:
            c.axhline(2.0 - df, color='k', ls='--', lw=1.2)
            c.annotate(rf'$d_f = {df:g}$', xy=(0.03, 2.0 - df),
                       xycoords=('axes fraction', 'data'),
                       fontsize=9.5, ha='left', va='center',
                       bbox=dict(boxstyle='round,pad=0.18', fc='white',
                                 ec='0.7', lw=0.6, alpha=0.92))
    c.set_xlabel(r'$\ell$')
    c.set_ylabel((r'$d\log A_{ef}\,/\,d\log \ell$' if pos
                  else r'$d\log E\,/\,d\log \ell$'))
    c.set_title((f'C. expoente local (janela de {2*slope_window:g} década)' if pos
                 else f'C. inclinação local de $E$ (janela de {2*slope_window:g} década)\n'
                      'sem leitura em $d_f$'), fontsize=11)
    c.invert_xaxis()
    c.grid(alpha=.25, which='both')
    c.legend(fontsize=8.5, loc='lower left')

    fig.suptitle(f'{name} — espectro de Laplace–Beltrami', fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(png_path, dpi=125)
    print(f"Wrote {png_path.name}")
    if show:
        plt.show()
    plt.close(fig)

# -------------------- Processing --------------------

def process_mesh(mesh_file: str,
                 n_modes: int = n_modes_default,
                 window_size: int = window_size_default,
                 field: str = "position",
                 ref_df=(2.0, 2.5),
                 ref_beta=(1.5, 2.0, 2.5),
                 slope_window: float = 0.175,
                 per_point: int = 40,
                 with_hull: bool = True,
                 recompute: bool = False,
                 visualize: bool = True):
    mesh_path = Path(mesh_file).resolve()
    out_dir   = prepare_output_folder(mesh_path)
    dest_mesh = out_dir / mesh_path.name
    if not dest_mesh.exists():
        shutil.copy2(mesh_path, dest_mesh)

    name = mesh_path.stem
    print(f"Processing: {dest_mesh.name}")

    mesh  = load_largest_component(dest_mesh)
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    print(f"Surface: {len(verts)} vertices, {len(faces)} faces, area {mesh.area:.6g}")

    csv_surf = out_dir / f"{name}_spectrum.csv"
    cached   = None if recompute else load_spectrum_csv(csv_surf, mesh.area)
    D_surf   = compute_spectrum_windowed(verts, faces, mesh.area, n_modes,
                                         window_size, cached, label="[surface] ")
    save_spectrum_csv(csv_surf, D_surf, mesh.area, field)
    print(f"Wrote {csv_surf.name}: {len(D_surf)} modes")

    mu, P, ell, cum = spectrum_table(D_surf, mesh.area, field)
    surf = dict(mu=mu, P=P, ell=ell, cum=cum, area=mesh.area)
    tail = (f"sum mu_n P_n / 2A = {cum[-1] / mesh.area:.4f}"
            if is_position_field(field)
            else f"energia de Dirichlet acumulada = {cum[-1]:.6g}")
    print(f"  ell from {ell.max():.4g} down to {ell.min():.4g}; " + tail)

    hull = None
    if with_hull:
        hull_mesh = remesh_regular(mesh.convex_hull,
                                   target_edge=float(mesh.edges_unique_length.mean()))
        hv, hf = np.asarray(hull_mesh.vertices), np.asarray(hull_mesh.faces)
        # Mesmo ell minimo com menos modos: por Weyl, N escala com a area.
        n_hull = max(window_size, int(round(n_modes * hull_mesh.area / mesh.area)))
        print(f"Convex hull (remeshed): {len(hv)} vertices, area {hull_mesh.area:.6g}; "
              f"target {n_hull} modes")
        csv_hull = out_dir / f"{name}_hull_spectrum.csv"
        cached_h = None if recompute else load_spectrum_csv(csv_hull, hull_mesh.area)
        D_hull   = compute_spectrum_windowed(hv, hf, hull_mesh.area, n_hull,
                                             window_size, cached_h, label="[hull]    ")
        save_spectrum_csv(csv_hull, D_hull, hull_mesh.area, field)
        print(f"Wrote {csv_hull.name}: {len(D_hull)} modes")
        hmu, hP, hell, hcum = spectrum_table(D_hull, hull_mesh.area, field)
        hull = dict(mu=hmu, P=hP, ell=hell, cum=hcum, area=hull_mesh.area)

    make_figure(out_dir / f"{name}_spectrum.png", name, surf, hull,
                ref_df=ref_df, ref_beta=ref_beta, slope_window=slope_window,
                per_point=per_point, field=field, show=visualize)

def batch_process_meshes(folder, **kw):
    mesh_dir = Path(folder)
    if not mesh_dir.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = sorted(f for ext in MESH_EXTENSIONS for f in mesh_dir.glob(f"*{ext}"))
    for mesh_file in files:
        print(f"\n--- Processing {mesh_file.name} ---")
        process_mesh(str(mesh_file), **kw)

def main():
    p = argparse.ArgumentParser(
        description="Laplace-Beltrami power spectrum: cached computation and plots"
    )
    p.add_argument("mesh", nargs="?", help="Path to mesh file (.stl, .ply, .obj)")
    p.add_argument("--n-modes", type=int, default=n_modes_default,
                   help=f"Total modes to end up with, cache included (default {n_modes_default}).")
    p.add_argument("--window-size", type=int, default=window_size_default,
                   help=f"Modes per shift-invert window (default {window_size_default}).")
    p.add_argument("--field", choices=FIELDS, default="position",
                   help="Field expanded in the eigenbasis (default: position).")
    p.add_argument("--ref-df", type=float, nargs="+", default=[2.0, 2.5],
                   metavar="D", help="d_f guide values for panels B and C (default: 2 2.5).")
    p.add_argument("--ref-beta", type=float, nargs="+", default=[1.5, 2.0, 2.5],
                   metavar="B", help="Slope guides for panel A (default: 1.5 2 2.5).")
    p.add_argument("--slope-window", type=float, default=0.175,
                   help="Half-width in decades of the sliding fit in panel C (default 0.175).")
    p.add_argument("--modes-per-point", type=int, default=40,
                   help="Modes averaged per plotted point in panel A (default 40).")
    p.add_argument("--no-hull", action="store_true", help="Skip the convex-hull control.")
    p.add_argument("--recompute", action="store_true", help="Ignore any cached spectrum.")
    p.add_argument("--no-visualize", action="store_true",
                   help="Do not open the window; the png and csv are written anyway.")

    a = p.parse_args()
    kw = dict(n_modes=a.n_modes, window_size=a.window_size, field=a.field,
              ref_df=tuple(a.ref_df), ref_beta=tuple(a.ref_beta),
              slope_window=a.slope_window, per_point=a.modes_per_point,
              with_hull=not a.no_hull, recompute=a.recompute,
              visualize=not a.no_visualize)

    if a.mesh:
        process_mesh(a.mesh, **kw)
    else:
        import os
        cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(cwd)
        batch_process_meshes("brains", **kw)

if __name__ == "__main__":
    main()