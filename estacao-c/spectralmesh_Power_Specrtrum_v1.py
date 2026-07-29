"""
Espectro de potencia do operador de Laplace-Beltrami
====================================================

Bruno Mota -- metaBIO lab, Instituto de Fisica / UFRJ

Companheiro do spectralmesh_dipolar_modes. La usamos os primeiros modos, os
dipolares, para medir direcoes. Aqui usamos muitos modos de uma vez, e olhamos
para como a superficie distribui estrutura entre as escalas espaciais.


O QUE ESTE PROGRAMA FAZ
-----------------------
Dada uma superficie fechada (.stl ou .ply), ele:

1. Monta o operador de Laplace-Beltrami da malha (robust_laplacian, o mesmo do
   programa dos modos dipolares) e resolve

       L phi_n = mu_n M phi_n,      n = 0, 1, 2, ...

   para os primeiros n_modes autopares. As autofuncoes sao normalizadas de modo
   que phi_i^T M phi_j = delta_ij, ou seja, ortonormais no produto interno de
   L2 da propria superficie -- e por isso que a matriz de massa M aparece em
   todas as projecoes daqui para a frente.

2. Projeta um campo definido sobre a superficie nessa base:

       c_n = phi_n^T M f

   O campo padrao e o proprio vetor posicao r, centrado no centroide. Nesse
   caso f tem tres componentes e c_n tambem; o programa trata as tres juntas.

3. Chama de potencia do modo n a soma dos quadrados dos coeficientes,

       P_n = |c_n|^2 = soma sobre as componentes de f,

   e chama de espectro de potencia o conjunto dos pares (mu_n, P_n). Modos de
   autovalor pequeno tem comprimento de onda grande: a parte esquerda do
   espectro descreve a forma geral do objeto, e a parte direita, os detalhes
   finos. Se a superficie for auto-similar, os pares (mu_n, P_n) caem sobre uma
   reta em escala log-log, e a inclinacao dessa reta e a grandeza de interesse.

4. Agrupa os modos em faixas de mu igualmente espacadas em log (o histograma),
   toma a media de P_n dentro de cada faixa, ajusta uma reta a esses pontos por
   minimos quadrados, desenha o grafico e exporta tudo para um csv.

O programa nao interpreta a inclinacao nem calcula nada a partir dos
autovalores alem disso. O csv traz todos os mu_n, um por linha, justamente para
que voce possa fazer isso por conta propria.


COMO USAR
---------
    python Spectralmesh_power_spectrum_v1.py malha.stl
    python Spectralmesh_power_spectrum_v1.py malha.ply --n-modes 500
    python Spectralmesh_power_spectrum_v1.py malha.stl --no-visualize
    python Spectralmesh_power_spectrum_v1.py                  # lote

Sem argumento, processa todos os .stl, .ply e .obj da pasta _Input_Meshes.

    --n-modes N      quantos autopares calcular (padrao 300). O custo cresce
                     rapido com N; comece pequeno para ver se o pipeline roda,
                     e so depois aumente.
    --field CAMPO    'position' (padrao) ou 'curvature'. Veja abaixo.
    --n-bins N       numero de faixas do histograma (padrao 24).
    --no-visualize   nao abre a janela do grafico. Use em lote, ou em maquina
                     sem tela. O csv sai do mesmo jeito.


OS DOIS CAMPOS
--------------
'position'   expande o vetor posicao r. E a decomposicao da forma da superficie
             em si: quanto de r e explicado por cada escala.

'curvature'  expande o vetor curvatura media H*n. Nao e uma segunda conta: como
             L r = 2 M (H n) na convencao usada aqui, vale exatamente

                 c_n(H n) = (mu_n / 2) c_n(r),   logo   P_curv = (mu^2/4) P_pos

             Ou seja, o espectro da curvatura e o da posicao multiplicado por
             mu^2/4. Modo a modo isso desloca a inclinacao em log-log de
             exatamente 2 unidades; depois do agrupamento em faixas o
             deslocamento fica muito perto disso, mas nao exatamente, porque a
             media de mu^2 P dentro de uma faixa nao e mu_centro^2 vezes a media
             de P. Vale rodar os dois e conferir de quanto e a diferenca.

O modo n = 0 e constante sobre a superficie (mu_0 = 0) e nao carrega estrutura;
ele e descartado do espectro.


O QUE SAI
---------
Uma pasta com o nome da malha, contendo uma copia da malha e o arquivo

    <malha>_power_spectrum.csv

com uma linha por modo e as colunas:

    mode_index        n, a partir de 1 (o modo constante fica de fora).
    eigenvalue        mu_n.
    power             P_n, a potencia do modo.
    bin_index         a qual faixa do histograma este modo pertence.
    bin_center        centro da faixa (media geometrica das bordas).
    bin_mean_power    media de P_n dentro da faixa. E o ponto que aparece no
                      grafico e que entra na regressao.
    bin_count         quantos modos caem na faixa.
    fit_slope         inclinacao da reta ajustada a log10(bin_mean_power)
                      contra log10(bin_center).
    fit_intercept     coeficiente linear da mesma reta.
    fit_r2            R^2 do ajuste.

As colunas bin_* se repetem para todos os modos de uma mesma faixa, e as fit_*
se repetem em todas as linhas. Para recuperar so os pontos do grafico, basta
eliminar as linhas duplicadas em bin_index.
"""

import csv
import argparse
from pathlib import Path
from typing import Tuple

import sys

import numpy as np
import trimesh
import robust_laplacian
import scipy.sparse.linalg as sla

n_modes_default = 300
n_bins_default = 24

# Extensions picked up in batch mode. trimesh.load_mesh reads all of these.
MESH_EXTENSIONS = (".stl", ".ply", ".obj")

FIELDS = ("position", "curvature")

# -------------------- Spectrum --------------------

def compute_spectrum_LB(verts: np.ndarray,
                        faces: np.ndarray,
                        n_modes: int,
                        sigma: float = 1e-8) -> tuple:
    """
    Lowest n_modes eigenpairs of L phi = mu M phi, with L and M from
    robust_laplacian (same operator the dipolar-modes program uses).

    Returns (eigenvalues, modes, M), including the trivial mode n = 0. The
    eigenvectors are explicitly renormalised so that phi^T M phi = 1: ARPACK's
    normalisation convention for the generalised problem is not something to
    rely on, and every projection below assumes M-orthonormality.
    """
    L, M_mat = robust_laplacian.mesh_laplacian(verts, faces)
    evals, evecs = sla.eigsh(L, k=n_modes, M=M_mat, sigma=sigma, which='LM')
    idx = np.argsort(evals)
    evals, evecs = evals[idx], evecs[:, idx]
    norms = np.sqrt(np.einsum('ij,ij->j', evecs, M_mat @ evecs))
    evecs = evecs / norms
    return evals, evecs, M_mat

def power_spectrum(verts: np.ndarray,
                   eigenvalues: np.ndarray,
                   modes: np.ndarray,
                   M_mat,
                   field: str = "position") -> Tuple[np.ndarray, np.ndarray]:
    """
    Project the chosen field onto the eigenbasis and return (mu, P), with the
    constant mode n = 0 dropped.

    'position'  expands the centred position vector r. P_n = |phi_n^T M r|^2,
                summed over the three coordinates.
    'curvature' expands the mean-curvature vector H n. Since L r = 2 M (H n),
                c_n(H n) = (mu_n / 2) c_n(r) exactly, so the curvature spectrum
                is the position one times mu^2 / 4. No second projection needed.
    """
    if field not in FIELDS:
        raise ValueError(f"Unknown field '{field}'. Available: {list(FIELDS)}")

    r = np.asarray(verts, dtype=float)
    r = r - r.mean(axis=0)
    coeffs = modes.T @ (M_mat @ r)              # (n_modes, 3)
    P = np.einsum('nc,nc->n', coeffs, coeffs)

    mu = eigenvalues[1:]
    P = P[1:]
    if field == "curvature":
        P = P * mu ** 2 / 4.0
    return mu, P

# -------------------- Histogram & regression --------------------

def bin_spectrum(mu: np.ndarray, P: np.ndarray, n_bins: int = n_bins_default):
    """
    Group the modes into n_bins bands equally spaced in log(mu) and average the
    power inside each. Returns (bin_index_per_mode, centres, mean_power,
    counts); empty bands are dropped and bin_index is -1 for modes that fall in
    one (which can only happen if mu <= 0, i.e. numerical noise at the bottom).
    """
    good = mu > 0
    edges = np.logspace(np.log10(mu[good].min()), np.log10(mu[good].max()),
                        n_bins + 1)
    edges[-1] *= 1.000001                       # keep the largest mu inside
    idx = np.digitize(mu, edges) - 1
    idx[~good] = -1
    idx[idx >= n_bins] = n_bins - 1

    centres, means, counts, keep = [], [], [], []
    for b in range(n_bins):
        sel = idx == b
        if not np.any(sel):
            continue
        centres.append(np.sqrt(edges[b] * edges[b + 1]))
        means.append(float(P[sel].mean()))
        counts.append(int(sel.sum()))
        keep.append(b)

    # Renumber the surviving bins consecutively from 1.
    remap = {b: i + 1 for i, b in enumerate(keep)}
    idx_out = np.array([remap.get(int(b), 0) for b in idx])
    return idx_out, np.array(centres), np.array(means), np.array(counts)

def fit_loglog(x: np.ndarray, y: np.ndarray):
    """Least squares on log10(y) against log10(x). Returns (slope, intercept, r2)."""
    ok = (x > 0) & (y > 0)
    if ok.sum() < 2:
        return float('nan'), float('nan'), float('nan')
    lx, ly = np.log10(x[ok]), np.log10(y[ok])
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(slope), float(intercept), r2

# -------------------- Output --------------------

def prepare_output_folder(mesh_path: Path) -> Path:
    project_root = Path(__file__).resolve().parent
    out_dir = project_root / mesh_path.stem
    out_dir.mkdir(exist_ok=True)
    return out_dir

def save_power_spectrum_csv(csv_path: Path, mu, P, bin_idx, centres, means,
                            counts, slope, intercept, r2):
    """One row per mode; bin_* columns repeat within a band, fit_* everywhere."""
    header = ["mode_index", "eigenvalue", "power",
              "bin_index", "bin_center", "bin_mean_power", "bin_count",
              "fit_slope", "fit_intercept", "fit_r2"]

    def fmt(x):
        return "" if x is None else f"{x:.10g}"

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for n in range(len(mu)):
            b = int(bin_idx[n])
            if b >= 1:
                c, mp, ct = centres[b - 1], means[b - 1], counts[b - 1]
            else:
                c = mp = ct = None
            writer.writerow([
                n + 1, fmt(mu[n]), fmt(P[n]),
                b if b >= 1 else "", fmt(c), fmt(mp), fmt(ct),
                fmt(slope), fmt(intercept), fmt(r2),
            ])

def show_plot(mu, P, centres, means, slope, intercept, r2, title, field):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.loglog(mu, P, '.', color='lightgray', markersize=3,
              label='modos individuais')
    ax.loglog(centres, means, 'o', color='tab:blue', markersize=7,
              label='media por faixa')
    xs = np.array([centres.min(), centres.max()])
    ax.loglog(xs, 10 ** (intercept + slope * np.log10(xs)), '-',
              color='tab:red', linewidth=2,
              label=f'ajuste: inclinacao {slope:.3f}, $R^2$ = {r2:.4f}')
    ax.set_xlabel(r'autovalor $\mu$')
    ax.set_ylabel(r'potencia $P(\mu)$')
    ax.set_title(f'{title}  --  campo: {field}')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plt.show()

# -------------------- Processing --------------------

def process_mesh(mesh_file: str,
                 n_modes: int = n_modes_default,
                 n_bins: int = n_bins_default,
                 field: str = "position",
                 visualize: bool = True):
    import shutil

    mesh_path = Path(mesh_file).resolve()
    out_dir   = prepare_output_folder(mesh_path)
    dest_mesh = out_dir / mesh_path.name
    if not dest_mesh.exists():
        shutil.copy2(mesh_path, dest_mesh)

    print(f"Processing: {dest_mesh.name}")

    mesh       = trimesh.load_mesh(dest_mesh)
    components = mesh.split(only_watertight=False)
    mesh       = max(components, key=lambda m: len(m.vertices))
    verts      = np.asarray(mesh.vertices)
    faces      = np.asarray(mesh.faces)
    print(f"Surface: {len(verts)} vertices, {len(faces)} faces, area {mesh.area:.6g}")

    print(f"Computing {n_modes} eigenpairs…")
    eigenvalues, modes, M_mat = compute_spectrum_LB(verts, faces, n_modes)
    print(f"  mu_0 = {eigenvalues[0]:.3e} (should be ~0), "
          f"mu_max = {eigenvalues[-1]:.6g}")

    mu, P = power_spectrum(verts, eigenvalues, modes, M_mat, field=field)
    bin_idx, centres, means, counts = bin_spectrum(mu, P, n_bins=n_bins)
    slope, intercept, r2 = fit_loglog(centres, means)
    print(f"Power spectrum ({field}): {len(mu)} modes in {len(centres)} bands")
    print(f"  fit: slope = {slope:.4f}, intercept = {intercept:.4f}, R^2 = {r2:.5f}")

    csv_path = out_dir / f"{mesh_path.stem}_power_spectrum.csv"
    save_power_spectrum_csv(csv_path, mu, P, bin_idx, centres, means, counts,
                            slope, intercept, r2)
    print(f"Wrote {csv_path.name}")

    if visualize:
        show_plot(mu, P, centres, means, slope, intercept, r2,
                  mesh_path.stem, field)

def batch_process_meshes(folder, n_modes=n_modes_default,
                         n_bins=n_bins_default, field="position",
                         visualize=True):
    mesh_dir = Path(folder)
    if not mesh_dir.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    mesh_files = sorted(f for ext in MESH_EXTENSIONS
                        for f in mesh_dir.glob(f"*{ext}"))
    for mesh_file in mesh_files:
        print(f"\n--- Processing {mesh_file.name} ---")
        process_mesh(str(mesh_file), n_modes=n_modes, n_bins=n_bins,
                     field=field, visualize=visualize)

def main():
    parser = argparse.ArgumentParser(
        description="Compute & plot the Laplace-Beltrami power spectrum"
    )
    parser.add_argument(
        "mesh",
        nargs="?",
        help="Path to mesh file (.stl, .ply, .obj, etc.)"
    )
    parser.add_argument(
        "--n-modes",
        type=int,
        default=n_modes_default,
        help=f"Number of eigenpairs to compute (default {n_modes_default})."
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=n_bins_default,
        help=f"Number of log-spaced bands in the histogram (default {n_bins_default})."
    )
    parser.add_argument(
        "--field",
        choices=FIELDS,
        default="position",
        help="Field expanded in the eigenbasis (default: position)."
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Suppress the plot window; the CSV is written either way."
    )

    args = parser.parse_args()
    visualize = not args.no_visualize

    if args.mesh:
        process_mesh(args.mesh, n_modes=args.n_modes, n_bins=args.n_bins,
                     field=args.field, visualize=visualize)
    else:
        import os
        cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(cwd)
        batch_process_meshes(
            "brains",
            n_modes=args.n_modes,
            n_bins=args.n_bins,
            field=args.field,
            visualize=visualize,
        )

if __name__ == "__main__":
    main()