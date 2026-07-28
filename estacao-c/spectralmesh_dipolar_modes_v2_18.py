"""
Modos dipolares e indices de girificacao direcionais
===================================================

Bruno Mota -- metaBIO lab, Instituto de Fisica / UFRJ


O QUE ESTE PROGRAMA FAZ
-----------------------
Dada uma superficie fechada (tipicamente um cortex, em .stl), ele:

1. Calcula os autovalores e autofuncoes mais baixos do operador de
   Laplace-Beltrami da malha.

2. Identifica entre eles tres modos *dipolares* -- campos com um unico maximo
   e um unico minimo sobre a superficie, orientados segundo os tres eixos
   anatomicos. Os dois primeiros modos nao triviais em geral ja sao dipolares.
   O terceiro normalmente nao e, e por isso e construido como uma combinacao

       phi(theta) = cos(theta) * e3 + sin(theta) * e4,

   varrendo theta em [0, pi) e ficando com o intervalo angular ("arco") em que
   a combinacao passa no teste de dipolaridade. Dentro desse arco, o angulo
   escolhido e o ponto medio (padrao) ou o que maximiza o gap espectral da
   covariancia do gradiente (opcao --spectral-gap).

3. Repete os passos 1 e 2 para o fecho convexo (convex hull) da mesma
   superficie. O fecho devolvido pelo qhull herda so os vertices da casca
   externa, e por isso vem com triangulos muito alongados; antes de calcular o
   espectro ele passa por uma remalhagem isotropica do MeshLab, com comprimento
   de aresta alvo igual ao da superficie original. A area usada como Ae continua
   sendo a do fecho exato, e nao a da versao remalhada.

4. Compara os dois. O indice de girificacao direcional na direcao i e

       g_i = comprimento de onda do modo i na superficie
             ---------------------------------------------
             comprimento de onda do modo i no fecho convexo

   Aqui o comprimento de onda de um modo de autovalor lambda e tomado como
   2*sqrt(2/lambda) -- numa esfera de raio R o modo dipolar tem lambda = 2/R^2,
   e essa expressao devolve 2R, o diametro. Como g_i e uma razao, a constante
   se cancela e a escolha de convencao nao afeta o resultado.


COMO USAR
---------
    python spectralmesh_dipolar_modes_v2_18.py malha.stl
    python spectralmesh_dipolar_modes_v2_18.py malha.stl --no-visualize
    python spectralmesh_dipolar_modes_v2_18.py                 # lote

Sem argumento, processa todos os .stl da pasta _Input_Meshes.

    --no-visualize   nao abre a janela interativa do PyVista. Use em lote, ou
                     em maquina sem tela.
    --spectral-gap   escolhe o terceiro modo pelo criterio do gap espectral
                     (Criterio 3) em vez do ponto medio do arco (Criterio 1,
                     que e o padrao).
    --verbose        mostra o progresso das varreduras angulares, que sao a
                     parte lenta em malhas grandes.


O QUE SAI
---------
Uma pasta com o nome da malha, contendo:

    <malha>.stl                 copia da malha de entrada.

    <malha>_dipolar.csv         os modos dipolares da *superficie*. A primeira
                                linha traz os autovalores; cada linha seguinte
                                e um vertice, com o valor do campo em cada modo.

    <malha>_gyrification.csv    uma linha por direcao, comparando superficie e
                                fecho convexo.


COMO LER O ARQUIVO DE GIRIFICACAO
---------------------------------
As colunas sao:

    direction                indice do modo dipolar (1, 2, 3).
    lambda_surface           autovalor do modo na superficie.
    lambda_hull              autovalor do modo correspondente no fecho convexo.
    wavelength_surface       2*sqrt(2/lambda) para a superficie.
    wavelength_hull          idem para o fecho convexo.
    g_dir                    o indice direcional, wavelength_surface dividido
                             por wavelength_hull.
    axis_alignment           |cos| do angulo entre o eixo do dipolo do modo na
                             superficie e o do modo no fecho convexo. Os modos
                             sao pareados pela ordem em que aparecem, e nao
                             pelo eixo; este numero e a verificacao de que o
                             pareamento faz sentido. Perto de 1 esta certo;
                             longe de 1 quer dizer que as duas superficies
                             ordenaram os eixos de maneira diferente e as
                             linhas precisam ser trocadas na mao.
    axis_surface_x/y/z       eixo do dipolo do modo na superficie, normalizado.
    axis_hull_x/y/z          idem no fecho convexo.
    area_surface             area total da superficie, At.
    area_hull                area do fecho convexo, Ae.
    g_global                 At / Ae, o indice de girificacao usual.
    g_dir_product            g_1 * g_2 * g_3.
    g_global_pow_1p5         g_global elevado a 3/2, para comparar com o
                             produto dos tres indices direcionais.

As quatro ultimas colunas sao resumos da malha inteira e por isso se repetem
identicas nas tres linhas.

"""

import csv
import math
import argparse
import shutil
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

import sys
import warnings

import numpy as np
import trimesh
import pymeshlab
import pyvista as pv
import robust_laplacian
import scipy.sparse.linalg as sla
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

n_modes_default = 4

PLOT_PARAMS = {
    'edge_color': 'lightgray',
    'font_size': {'title': 14, 'mode': 9},
    'cmap': 'coolwarm',
    'line_width': 4,
}

# -------------------- Utility: simple console progress --------------------

def _print_progress(prefix: str, current: int, total: int, width: int = 40, force_newline: bool = False):
    if total <= 0:
        return
    frac = current / total
    filled = int(round(width * frac))
    bar = '█' * filled + '-' * (width - filled)
    pct = frac * 100
    if sys.stdout.isatty():
        sys.stdout.write(f"\r{prefix} |{bar}| {pct:6.2f}% ({current}/{total})")
        sys.stdout.flush()
        if current == total or force_newline:
            sys.stdout.write("\n")
            sys.stdout.flush()
    else:
        if current == total or (total >= 10 and current % max(1, total // 10) == 0):
            print(f"{prefix}: {current}/{total} ({pct:.1f}%)")

# -------------------- Utility: connectivity counting --------------------

def count_components_masked(adj_csr: sp.csr_matrix, nodes: np.ndarray) -> int:
    """
    Number of connected components in the subgraph of `adj_csr` induced by
    `nodes`. Isolated nodes count as their own component.

    Uses the induced submatrix adj_csr[nodes][:, nodes] and scipy's compiled
    csgraph routine. (An internal-edge variant — select internal edges, relabel,
    rebuild a small k×k graph per call — was tried as a speedup but profiled
    ~5-10% *slower* than this slice from ~7k to ~290k vertices, because scipy's
    CSR slicing is already C-optimized and the per-call rebuild costs more. The
    slice is kept as the simpler and faster option.)
    """
    if nodes.size == 0:
        return 0
    sub = adj_csr[nodes][:, nodes]
    return connected_components(sub, directed=False, return_labels=False)

# -------------------- Geometry & Spectrum Helpers --------------------

def build_adjacency_csr_and_edge_lists(n_vertices: int, faces: np.ndarray) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """
    Build symmetric adjacency CSR and unique undirected edge lists (u < v).
    Vectorized construction for speed.
    """
    a = faces[:, 0].astype(np.int64)
    b = faces[:, 1].astype(np.int64)
    c = faces[:, 2].astype(np.int64)
    rows = np.concatenate([a, b, b, c, c, a])
    cols = np.concatenate([b, a, c, b, a, c])
    data = np.ones(rows.shape[0], dtype=np.int8)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))
    A.sum_duplicates()
    A.data[:] = 1
    A = A.tocsr()
    A_triu = sp.triu(A, k=1).tocoo()
    edges_u = A_triu.row.astype(np.int32)
    edges_v = A_triu.col.astype(np.int32)
    return A, edges_u, edges_v

def is_dipolar_mode(
    values: np.ndarray,
    adj_csr: sp.csr_matrix,
    tol: float = 1e-8,
    max_order: int = 4,
) -> bool:
    """
    Strict dipolarity test via connected-components on the mesh adjacency.
    For each dyadic threshold x, the super-level set {values >= x+tol} and the
    sub-level set {values <= x-tol} must each form a single connected
    component. Early exit on the first failure.

    Thresholds are generated coarse-first: level 0 is the absolute level 0.0,
    then levels 1..max_order place thresholds at successively finer dyadic
    fractions of [min, max]. Because multipolar modes typically fail at a
    coarse level, the per-threshold early exit means most rejected modes never
    pay for the finest levels. The full set of thresholds — and therefore the
    True/False verdict for any mode — is identical to testing them all at once.
    """
    vmin, vmax = float(values.min()), float(values.max())
    span = vmax - vmin

    def component_check(x: float) -> bool:
        pos = np.nonzero(values >= tol + x)[0]
        neg = np.nonzero(values <= -tol + x)[0]
        if pos.size == 0 or neg.size == 0:
            return False
        if count_components_masked(adj_csr, pos) != 1:
            return False
        if count_components_masked(adj_csr, neg) != 1:
            return False
        return True

    # Level 0: absolute threshold at 0.0 (preserves original behavior).
    if not component_check(0.0):
        return False
    # Levels 1..max_order: finer dyadic fractions of [vmin, vmax], coarse first.
    for lvl in range(1, max_order + 1):
        step = 1 << lvl
        for k in range(1, step, 2):
            x = vmin + span * (k / step)
            if not component_check(x):
                return False
    return True

def compute_spectrum_LB(verts: np.ndarray,
                        faces: np.ndarray,
                        n_modes: int,
                        sigma: float = 1e-8) -> tuple:
    L, M_mat = robust_laplacian.mesh_laplacian(verts, faces)
    evals, evecs = sla.eigsh(L, k=n_modes, M=M_mat, sigma=sigma, which='LM')
    idx = np.argsort(evals)
    eigenvalues = evals[idx][1:n_modes]
    modes       = evecs[:, idx][:, 1:n_modes]
    return eigenvalues, modes

def compute_color_limit(modes: np.ndarray) -> float:
    abs_max = np.max(np.abs(modes))
    exp     = np.floor(np.log10(abs_max))
    step    = 10 ** exp
    return np.ceil(abs_max / step) * step

# -------------------- Plotting Setup --------------------

def setup_plotter(surface, shape=(3, 3), size=(1200, 900)):
    plotter = pv.Plotter(shape=shape, window_size=size, border=False)
    center = surface.center
    plotter.camera_position = [
        (center[0] + 2 * surface.length, center[1], center[2]),
        center,
        (0, 0, 1),
    ]
    plotter.enable_parallel_projection()
    return plotter

def add_original_mesh(plotter, surface):
    plotter.subplot(0, 0)
    plotter.add_mesh(surface, color=PLOT_PARAMS['edge_color'], show_edges=False)
    plotter.add_text('Original Mesh',
                     font_size=PLOT_PARAMS['font_size']['title'],
                     position=(0.4, 0.01), viewport=True)

def add_mesh_scale_bar(plotter, surface):
    plotter.subplot(0, 1)
    xmin, _, ymin, ymax, zmin, _ = surface.bounds
    mesh_width = ymax - ymin

    if mesh_width >= 10:
        bar_length = np.ceil(mesh_width / 10) * 10
        label = f"{int(bar_length)}"
    else:
        bar_length = np.ceil(mesh_width * 10) / 10
        label = f"{int(round(bar_length))}"

    start = (xmin, ymin, zmin)
    end   = (xmin, ymin + bar_length, zmin)
    bar   = pv.Line(start, end)

    plotter.add_mesh(bar, color='black', line_width=PLOT_PARAMS['line_width'])
    plotter.add_point_labels(
        points=np.array([end]),
        labels=[label],
        font_size=12,
        text_color="black",
        shape_opacity=0.0,
        always_visible=True,
    )

def add_shared_colorbar(plotter, limit):
    plotter.subplot(0, 2)
    dummy = pv.PolyData(np.zeros((1, 3)))
    dummy["eigenmode"] = np.array([0.0])

    plotter.add_mesh(
        dummy,
        scalars="eigenmode",
        cmap=PLOT_PARAMS['cmap'],
        opacity=0.0,
        clim=[-limit, limit],
        show_edges=False,
        show_scalar_bar=True,
        scalar_bar_args=dict(
            vertical=False,
            position_x=0.2,
            position_y=0.25,
            width=0.60,
            height=0.2,
            title=None,
            label_font_size=15,
            fmt="%.2g",
        ),
    )
    plotter.hide_axes()

# -------------------- Per-mode green scale bar --------------------

def _mode_scale_bar_length(lam: float, tol: float = 1e-12) -> float:
    """
    Length of the per-mode green scale bar, from the eigenvalue `lam`.

    Returns 2√2/√λ, a Euclidean length (λ has units 1/length²). On a sphere of
    radius R the l=1 (dipole) eigenvalue is λ = 2/R², so 2√2/√λ = 2R — exactly
    the sphere's diameter.
    """
    safe = max(float(lam), tol)
    return 2.0 * math.sqrt(2.0 / safe)

def add_mode_dipole_scale_bar(plotter, surface, values, lam,
                              view_dir=(1.0, 0.0, 0.0), margin_frac=0.03):
    """
    Draw a green scale bar of length _mode_scale_bar_length(lam), oriented along
    the mode's φ-weighted dipole axis — the first moment of the field,

        d = Σ_i A_i φ_i (p_i − c),   c = Σ_i A_i p_i / Σ_i A_i  (area centroid),

    which points from the negative lobe toward the positive lobe and uses every
    vertex (unlike a max–min segment). The bar is offset perpendicular to d,
    within the view plane, so it sits beside the mesh rather than through it,
    and is labelled with its length.

    Assumes the caller has already selected the target subplot.
    """
    pts = np.asarray(surface.points)

    # Per-vertex (lumped) areas from the triangle faces: each face area is split
    # equally among its three vertices.
    tri = np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
    face_area = 0.5 * np.linalg.norm(
        np.cross(pts[tri[:, 1]] - pts[tri[:, 0]],
                 pts[tri[:, 2]] - pts[tri[:, 0]]), axis=1)
    vert_area = np.zeros(len(pts))
    np.add.at(vert_area, tri.ravel(), np.repeat(face_area, 3) / 3.0)

    # φ-weighted first moment about the area-weighted centroid → dipole axis.
    c = (vert_area[:, None] * pts).sum(0) / vert_area.sum()
    d = ((vert_area * values)[:, None] * (pts - c)).sum(0)
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-12:
        return
    axis = d / d_norm

    bar_len = _mode_scale_bar_length(lam)

    # Offset direction: perpendicular to the axis AND lying in the view plane
    # (axis × view_dir is ⟂ view_dir), so the shift shows on screen.
    vd = np.asarray(view_dir, dtype=float)
    vd /= np.linalg.norm(vd)
    perp = np.cross(axis, vd)
    if np.linalg.norm(perp) < 1e-8:          # axis ∥ view direction: fall back
        perp = np.cross(axis, np.array([0.0, 0.0, 1.0]))
    perp /= np.linalg.norm(perp)

    # Push out just past the silhouette edge on the +perp side, plus a margin.
    halfwidth = float(((pts - c) @ perp).max())
    offset = perp * (halfwidth + margin_frac * float(surface.length))

    half = 0.5 * bar_len * axis
    start = c + offset - half
    end   = c + offset + half

    bar = pv.Line(start, end)
    plotter.add_mesh(bar, color='green', line_width=PLOT_PARAMS['line_width'])
    plotter.add_point_labels(
        points=np.array([end]),
        labels=[f"{bar_len:.1f}"],
        font_size=12,
        text_color="green",
        shape_opacity=0.0,
        always_visible=True,
    )

def dipole_axis(verts: np.ndarray, faces: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Unit dipole axis of a mode: the phi-weighted first moment about the
    area-weighted centroid,

        d = sum_i A_i phi_i (p_i - c),   c = sum_i A_i p_i / sum_i A_i,

    normalised. Points from the negative lobe toward the positive one. Same
    definition used by the green scale bar in add_mode_dipole_scale_bar; kept
    standalone here so it can be called without a plotter.
    Returns a zero vector if the moment is degenerate.
    """
    pts = np.asarray(verts, dtype=float)
    tri = np.asarray(faces)
    face_area = 0.5 * np.linalg.norm(
        np.cross(pts[tri[:, 1]] - pts[tri[:, 0]],
                 pts[tri[:, 2]] - pts[tri[:, 0]]), axis=1)
    vert_area = np.zeros(len(pts))
    np.add.at(vert_area, tri.ravel(), np.repeat(face_area, 3) / 3.0)
    c = (vert_area[:, None] * pts).sum(0) / vert_area.sum()
    d = ((vert_area * values)[:, None] * (pts - c)).sum(0)
    n = np.linalg.norm(d)
    return d / n if n > 1e-12 else np.zeros(3)

def add_eigenmode_subplot(plotter, surface, modes, eigenvalues,
                          faces, index, row, col, text_position,
                          adj_csr, tol):
    mode_values = modes[:, index]
    dip = is_dipolar_mode(mode_values, adj_csr, tol)
    plotter.subplot(row, col)
    mcopy = surface.copy()
    mcopy["eigenmode"] = mode_values
    plotter.add_mesh(
        mcopy,
        scalars="eigenmode",
        cmap=PLOT_PARAMS['cmap'],
        show_edges=False,
        show_scalar_bar=False
    )

    eigen_label = f"k{index+1} = {eigenvalues[index]:.3g}"
    dip_label   = "dipolar" if dip else "multipolar"
    plotter.add_text(
        eigen_label + "\n" + dip_label,
        font_size=PLOT_PARAMS['font_size']['mode'],
        position=text_position,
        viewport=True
    )

    # Green reference bar: length 2√2/√λ (the sphere diameter for a dipole),
    # along this mode's φ-weighted dipole axis, offset to sit beside the mesh.
    add_mode_dipole_scale_bar(plotter, mcopy, mode_values, eigenvalues[index])

def plot_eigenmodes(plotter, surface, faces,
                    eigenvalues, modes, n_modes, n_columns,
                    adj_csr):
    for i in range(n_modes):
        row = (i // n_columns) + 1
        col = i % n_columns
        add_eigenmode_subplot(
            plotter, surface, modes, eigenvalues,
            faces, i, row, col, (0.4, -0.01),
            adj_csr, tol=1e-8
        )

def show_plot(surface, faces, eigenvalues, modes, n_modes, n_columns,
              adj_csr):
    n_rows_modes = math.ceil(n_modes / n_columns)
    total_rows   = 1 + n_rows_modes
    plotter = setup_plotter(
        surface,
        shape=(total_rows, n_columns),
        size=(min(300 * n_columns, 1800),
              min(275 * total_rows, 1100))
    )

    add_original_mesh(plotter, surface)
    add_mesh_scale_bar(plotter, surface)
    limit = compute_color_limit(modes)
    add_shared_colorbar(plotter, limit)
    plot_eigenmodes(plotter, surface, faces,
                    eigenvalues, modes, n_modes, n_columns,
                    adj_csr)

    plotter.link_views()
    plotter.show()

# -------------------- Spectrum Cache Logic --------------------

def save_cached_spectrum(csv_path: Path, eigenvalues, modes):
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index"] + [f"{lam:.6g}" for lam in eigenvalues])
        for idx, row in enumerate(modes):
            writer.writerow([idx] + row.tolist())

def save_gyrification_csv(csv_path: Path, surf: dict, hull: dict,
                          area_surface: float, area_hull: float):
    """
    Write one row per dipolar direction comparing the surface with its convex
    hull: eigenvalues, wavelengths, the directional gyrification index
    g_i = wavelength_surface / wavelength_hull, and the dipole axes of both
    modes plus |cos| between them (modes are paired by order, so that number is
    the check that the pairing is meaningful). The last four columns are
    whole-mesh summaries and repeat on every row.
    """
    header = [
        "direction",
        "lambda_surface", "lambda_hull",
        "wavelength_surface", "wavelength_hull",
        "g_dir", "axis_alignment",
        "axis_surface_x", "axis_surface_y", "axis_surface_z",
        "axis_hull_x", "axis_hull_y", "axis_hull_z",
        "area_surface", "area_hull",
        "g_global", "g_dir_product", "g_global_pow_1p5",
    ]

    g_global = area_surface / area_hull if area_hull > 0 else float("nan")

    per_dir = []
    for i in range(3):
        has_s = i < len(surf["eigs"])
        has_h = i < len(hull["eigs"])
        lam_s = float(surf["eigs"][i]) if has_s else None
        lam_h = float(hull["eigs"][i]) if has_h else None
        wl_s = _mode_scale_bar_length(lam_s) if has_s else None
        wl_h = _mode_scale_bar_length(lam_h) if has_h else None
        g_dir = (wl_s / wl_h) if (has_s and has_h and wl_h > 0) else None
        ax_s = (dipole_axis(surf["verts"], surf["faces"], surf["modes"][i])
                if has_s else np.zeros(3))
        ax_h = (dipole_axis(hull["verts"], hull["faces"], hull["modes"][i])
                if has_h else np.zeros(3))
        align = abs(float(np.dot(ax_s, ax_h))) if (has_s and has_h) else None
        per_dir.append((lam_s, lam_h, wl_s, wl_h, g_dir, align, ax_s, ax_h))

    g_list = [d[4] for d in per_dir]
    g_prod = (g_list[0] * g_list[1] * g_list[2]
              if all(g is not None for g in g_list) else None)

    def fmt(x):
        return "" if x is None else f"{x:.10g}"

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, (lam_s, lam_h, wl_s, wl_h, g_dir, align, ax_s, ax_h) in enumerate(per_dir):
            writer.writerow([
                i + 1,
                fmt(lam_s), fmt(lam_h), fmt(wl_s), fmt(wl_h),
                fmt(g_dir), fmt(align),
                fmt(ax_s[0]), fmt(ax_s[1]), fmt(ax_s[2]),
                fmt(ax_h[0]), fmt(ax_h[1]), fmt(ax_h[2]),
                fmt(area_surface), fmt(area_hull),
                fmt(g_global), fmt(g_prod), fmt(g_global ** 1.5),
            ])
    print(f"Wrote {csv_path.name}: g_global = {g_global:.4f}, "
          f"g_global^1.5 = {g_global ** 1.5:.4f}, "
          f"g_dir product = {'n/a' if g_prod is None else f'{g_prod:.4f}'}")


def prepare_output_folder(mesh_path: Path) -> Path:
    project_root = Path(__file__).resolve().parent
    out_dir = project_root / mesh_path.stem
    out_dir.mkdir(exist_ok=True)
    return out_dir

# -------------------- Dipolar Arc Search (half-range) --------------------

def _extract_circular_runs(mask: np.ndarray, min_steps: int = 1) -> List[np.ndarray]:
    """
    Return contiguous runs of True in a *circular* boolean array, each as an
    array of indices in natural circular order. Runs shorter than `min_steps`
    are discarded. The domain [0, π) is circular here because θ and θ+π give
    the same mode (up to sign), so a run may legitimately wrap the boundary.
    """
    n = mask.size
    if mask.all():
        return [np.arange(n)]
    if not mask.any():
        return []

    # Rotate so index 0 is False; no True run can then be split at the seam.
    first_false = int(np.argmax(~mask))
    rolled = np.roll(mask, -first_false)

    runs = []
    i = 0
    while i < n:
        if rolled[i]:
            j = i
            while j < n and rolled[j]:
                j += 1
            if (j - i) >= min_steps:
                runs.append((np.arange(i, j) + first_false) % n)
            i = j
        else:
            i += 1
    return runs

def per_face_gradients(verts: np.ndarray, faces: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v0 = verts[faces[:, 0]];  v1 = verts[faces[:, 1]];  v2 = verts[faces[:, 2]]
    p0 =  phi[faces[:, 0]];   p1 =  phi[faces[:, 1]];   p2 =  phi[faces[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0
    nv = np.cross(e1, e2)
    nn = np.einsum('fi,fi->f', nv, nv)

    mask = nn > 1e-28
    area = np.zeros_like(nn)
    grad = np.zeros((faces.shape[0], 3), dtype=float)

    if np.any(mask):
        e1m = e1[mask]; e2m = e2[mask]; nvm = nv[mask]; nnm = nn[mask]
        p0m = p0[mask]; p1m = p1[mask]; p2m = p2[mask]
        area_mask = 0.5 * np.sqrt(nnm)
        dp1  = (p1m - p0m)[:, None]
        dp2  = (p2m - p0m)[:, None]
        grad_mask = (dp1 * np.cross(e2m, nvm)
                   + dp2 * np.cross(nvm, e1m)) / nnm[:, None]
        grad[mask] = grad_mask
        area[mask] = area_mask
    return grad, area

def _gradient_covariance_spectral_gap_from_precomputed(
    grad_e3: np.ndarray,
    grad_e4: np.ndarray,
    area: np.ndarray,
    c: float,
    s: float
) -> float:
    grad_phi = c * grad_e3 + s * grad_e4
    G = np.einsum('f,fi,fj->ij', area, grad_phi, grad_phi)
    evals = np.sort(np.linalg.eigvalsh(G))[::-1]
    return float(evals[0] - evals[1])

def find_dipolar_arcs(
    e3: np.ndarray,
    e4: np.ndarray,
    faces: np.ndarray,
    adj_csr: sp.csr_matrix,
    n_scan: int = 360,
    tol: float = 1e-8,
    min_arc_frac: float = 0.05,
    coarse_factor: int = 10,
    verbose: bool = False,
) -> List[np.ndarray]:
    """
    Find every angular arc on [0, π) where φ(θ) = cos θ · e3 + sin θ · e4 is
    dipolar. Returns a list of index arrays (into the fine θ grid), sorted by
    decreasing length; arcs shorter than min_arc_frac · n_scan are discarded.

    Strategy — coarse→fine, no approximation for retained arcs:
      1. Test dipolarity on a coarse grid of n_scan/coarse_factor angles.
      2. Dilate the coarse hits by ±1 bin (circularly) and refine every fine
         angle that falls inside a dilated coarse bin.

    Guarantee: any arc we keep contains at least one coarse sample, so the
    coarse pass cannot miss it. This holds when the coarse spacing
    (n_scan / n_coarse fine steps) is no larger than the smallest kept arc
    (min_steps fine steps). If the requested coarse_factor would violate this,
    n_coarse is automatically tightened (with a warning) so the guarantee always
    holds; correctness is never traded for the coarse speedup.
    """
    thetas = np.linspace(0, math.pi, n_scan, endpoint=False)

    # ---- Coarse grid, sized to guarantee no arc is missed ----
    # The no-missed-arc property needs the coarse spacing (n_scan / n_coarse
    # fine steps) to be no larger than the smallest kept arc (min_steps fine
    # steps): then any retained arc spans at least one coarse sample. We honour
    # the requested coarse_factor but tighten n_coarse upward if it would
    # violate the guarantee, warning rather than failing or silently missing.
    min_steps = max(1, int(np.ceil(min_arc_frac * n_scan)))
    required_n_coarse = int(math.ceil(n_scan / min_steps))
    requested_n_coarse = max(8, n_scan // coarse_factor)
    n_coarse = max(requested_n_coarse, required_n_coarse)
    if requested_n_coarse < required_n_coarse:
        warnings.warn(
            f"coarse_factor={coarse_factor} gives n_coarse={requested_n_coarse}, "
            f"too sparse to guarantee detection of arcs >= {min_steps} steps; "
            f"tightened to n_coarse={required_n_coarse}.",
            stacklevel=2,
        )

    coarse_thetas = np.linspace(0, math.pi, n_coarse, endpoint=False)
    coarse_dipolar = np.zeros(n_coarse, dtype=bool)

    show_progress = verbose or sys.stdout.isatty()
    if show_progress:
        print(f"Coarse scan (half-range): {n_coarse} angles")

    for i, t in enumerate(coarse_thetas):
        if show_progress:
            _print_progress("Coarse scan", i + 1, n_coarse)
        phi = math.cos(t) * e3 + math.sin(t) * e4
        coarse_dipolar[i] = is_dipolar_mode(phi, adj_csr, tol)

    if show_progress:
        _print_progress("Coarse scan", n_coarse, n_coarse, force_newline=True)
        print(f"Coarse positives: {int(coarse_dipolar.sum())}/{n_coarse}")

    if not coarse_dipolar.any():
        if show_progress:
            print("No coarse positives found.")
        return []

    # ---- Circular ±1-bin dilation, then map dilated coarse bins to fine angles ----
    dilated = (coarse_dipolar
               | np.roll(coarse_dipolar, 1)
               | np.roll(coarse_dipolar, -1))

    refine_fine = np.zeros(n_scan, dtype=bool)
    for b in np.nonzero(dilated)[0]:
        a0 = math.pi * b / n_coarse
        a1 = math.pi * (b + 1) / n_coarse
        refine_fine |= (thetas >= a0) & (thetas < a1)
    fine_indices = np.nonzero(refine_fine)[0]

    if show_progress:
        print(f"Refining {fine_indices.size} fine angles inside dilated coarse arcs")

    # ---- Fine refinement: authoritative test on every candidate fine angle ----
    dipolar_mask = np.zeros(n_scan, dtype=bool)
    for pos, k in enumerate(fine_indices):
        if show_progress and (pos % max(1, fine_indices.size // 10) == 0):
            _print_progress("Fine refine", pos + 1, fine_indices.size)
        t = float(thetas[k])
        phi = math.cos(t) * e3 + math.sin(t) * e4
        dipolar_mask[k] = is_dipolar_mode(phi, adj_csr, tol)
    if show_progress:
        _print_progress("Fine refine", fine_indices.size, fine_indices.size, force_newline=True)
        print(f"Refined dipolar samples: {int(dipolar_mask.sum())}/{n_scan}")

    # ---- Extract arcs, discard short ones (min_steps computed above) ----
    arcs = _extract_circular_runs(dipolar_mask, min_steps=min_steps)
    arcs.sort(key=len, reverse=True)
    return arcs

# -------------------- Quasi-mode selection criteria --------------------
#
# A "criterion" chooses θ* within a single, already-identified dipolar arc.
# Every criterion shares the signature
#
#     criterion(arc, thetas, ctx) -> float        # θ* in radians
#
# so new ones can be added below and registered in CRITERIA without touching
# the caller. Planned future additions (each fits this same signature):
#   - proto-saddle gradient minimum  (Criterion 2: max over the arc of the
#     smallest |∇φ| away from the true extrema)
#   - nodal-area balance             (Criterion 4: |Area(φ>0) − Area(φ<0)| min)
#   - gradient-axis orthogonality    (Criterion 5: ∇φ most orthogonal to
#     ∇e1, ∇e2 as direction fields)
# A criterion reads only the ctx fields it needs.

@dataclass
class CriterionContext:
    """Everything any selection criterion might need; criteria use a subset."""
    e3: np.ndarray
    e4: np.ndarray
    verts: np.ndarray
    faces: np.ndarray
    adj_csr: sp.csr_matrix
    grad_e3: Optional[np.ndarray] = None
    grad_e4: Optional[np.ndarray] = None
    area_faces: Optional[np.ndarray] = None
    lam3: Optional[float] = None
    lam4: Optional[float] = None

def _arc_midpoint_theta(arc: np.ndarray, thetas: np.ndarray) -> float:
    return float(thetas[arc[len(arc) // 2]])

def criterion_arc_midpoint(arc: np.ndarray, thetas: np.ndarray,
                           ctx: CriterionContext) -> float:
    """
    Criterion 1 — arc midpoint.
    The centre of the arc maximises the angular margin to both bifurcation
    boundaries (where a new max/min pair is about to nucleate). Cheap, robust;
    the default.
    """
    return _arc_midpoint_theta(arc, thetas)

def criterion_spectral_gap(arc: np.ndarray, thetas: np.ndarray,
                           ctx: CriterionContext) -> float:
    """
    Criterion 3 — gradient-covariance spectral gap.
    Returns the angle in the arc maximising λ1 − λ2 of
    G = Σ_f area_f (∇φ_f ⊗ ∇φ_f): the representative whose gradient energy is
    most concentrated on a single surface axis, i.e. furthest (in function
    space) from acquiring a second gradient axis at a bifurcation.
    Uses ctx's precomputed per-face gradients/areas, recomputing if absent.
    """
    grad_e3, grad_e4, area = ctx.grad_e3, ctx.grad_e4, ctx.area_faces
    if grad_e3 is None or grad_e4 is None or area is None:
        grad_e3, area = per_face_gradients(ctx.verts, ctx.faces, ctx.e3)
        grad_e4, _ = per_face_gradients(ctx.verts, ctx.faces, ctx.e4)

    valid = area > 0
    if not np.any(valid):
        return _arc_midpoint_theta(arc, thetas)
    g3, g4, a = grad_e3[valid], grad_e4[valid], area[valid]

    best_theta = _arc_midpoint_theta(arc, thetas)
    best_gap = -np.inf
    for idx in arc:
        t = float(thetas[idx])
        c, s = math.cos(t), math.sin(t)
        gap = _gradient_covariance_spectral_gap_from_precomputed(g3, g4, a, c, s)
        if gap > best_gap:
            best_gap, best_theta = gap, t
    return best_theta

# Registry of within-arc criteria. Add new criteria here by name.
CRITERIA = {
    "arc_midpoint": criterion_arc_midpoint,   # Criterion 1 (default)
    "spectral_gap": criterion_spectral_gap,   # Criterion 3
}

def select_dipolar_angle(
    e3: np.ndarray,
    e4: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    arc_indices: List[np.ndarray],
    adj_csr: sp.csr_matrix,
    grad_e3: Optional[np.ndarray] = None,
    grad_e4: Optional[np.ndarray] = None,
    area_faces: Optional[np.ndarray] = None,
    lam3: Optional[float] = None,
    lam4: Optional[float] = None,
    use_spectral_gap: bool = False,
    criterion: Optional[str] = None,
    n_scan: int = 360,
    verbose: bool = False,
) -> Optional[float]:
    """
    Select θ* from the dipolar arcs (θ on [0, π)), in two stages.

    Stage 1 — choose the arc: take the longest arc; if several tie for longest
    and lam3, lam4 are supplied, break the tie by the largest wavelength
    λ(θ)^(-2) at the midpoint, with λ(θ) = cos²θ · lam3 + sin²θ · lam4
    (equivalently, the smallest-eigenvalue arc).

    Stage 2 — choose the angle within that arc: delegate to a named criterion
    from CRITERIA. Default "arc_midpoint" (Criterion 1); "spectral_gap" is
    Criterion 3. `use_spectral_gap=True` is shorthand for criterion=
    "spectral_gap", kept for backward compatibility; an explicit `criterion`
    name takes precedence.
    """
    if not arc_indices:
        return None

    thetas = np.linspace(0, math.pi, n_scan, endpoint=False)

    # Resolve the criterion (explicit name wins over the legacy boolean).
    if criterion is None:
        criterion = "spectral_gap" if use_spectral_gap else "arc_midpoint"
    if criterion not in CRITERIA:
        raise ValueError(f"Unknown criterion '{criterion}'. "
                         f"Available: {sorted(CRITERIA)}")

    # ---- Stage 1: choose the arc ----
    max_len = len(arc_indices[0])  # arc_indices is sorted by decreasing length
    longest = [a for a in arc_indices if len(a) == max_len]
    if len(longest) == 1 or lam3 is None or lam4 is None:
        best_arc = longest[0]
    else:
        def midpoint_wavelength(arc: np.ndarray) -> float:
            th = _arc_midpoint_theta(arc, thetas)
            c, s = math.cos(th), math.sin(th)
            lam = c * c * lam3 + s * s * lam4
            return lam ** (-2.0) if lam > 0 else math.inf
        best_arc = max(longest, key=midpoint_wavelength)
        if verbose:
            print(f"Tie among {len(longest)} equal-length arcs; "
                  f"chose largest-wavelength midpoint.")

    # ---- Stage 2: choose the angle within the arc ----
    if verbose:
        print(f"Selecting θ* in arc of {len(best_arc)} samples via '{criterion}'.")
    ctx = CriterionContext(
        e3=e3, e4=e4, verts=verts, faces=faces, adj_csr=adj_csr,
        grad_e3=grad_e3, grad_e4=grad_e4, area_faces=area_faces,
        lam3=lam3, lam4=lam4,
    )
    return CRITERIA[criterion](best_arc, thetas, ctx)

# -------------------- Processing --------------------

def _meshlab_absolute(x: float):
    """
    MeshLab's absolute-length wrapper. The class was renamed from
    AbsoluteValue to PureValue across pymeshlab versions; try both, and fall
    back to a bare float for very old ones.
    """
    for name in ("PureValue", "AbsoluteValue"):
        cls = getattr(pymeshlab, name, None)
        if cls is not None:
            return cls(float(x))
    return float(x)

def remesh_regular(mesh: "trimesh.Trimesh",
                   target_edge: Optional[float] = None,
                   iterations: int = 10) -> "trimesh.Trimesh":
    """
    Isotropic explicit remeshing through MeshLab (pymeshlab), used to replace
    the sliver-rich triangulation qhull returns for a convex hull with one of
    roughly uniform, roughly equilateral triangles.

    `target_edge` is the target edge length in mesh units; pass the mean edge
    length of the surface being compared against, so both meshes carry a
    comparable O(h^2) discretisation error and it largely cancels in the ratio
    g_i. If None, MeshLab's default (1% of the bounding-box diagonal) is used.

    Vertices are reprojected onto the input surface, so the remeshed mesh is
    inscribed in the original and its area comes out marginally smaller. Use
    the original hull for areas and the remeshed one for the spectrum.
    """
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(np.asarray(mesh.vertices, dtype=float),
                               np.asarray(mesh.faces, dtype=np.int32)))
    kwargs = {} if target_edge is None else dict(targetlen=_meshlab_absolute(target_edge))
    ms.meshing_isotropic_explicit_remeshing(iterations=iterations, **kwargs)
    out = ms.current_mesh()
    return trimesh.Trimesh(out.vertex_matrix(), out.face_matrix(), process=False)

def compute_dipolar_triplet(verts: np.ndarray,
                            faces: np.ndarray,
                            n_modes: int = n_modes_default,
                            use_spectral_gap: bool = False,
                            verbose: bool = False,
                            label: str = ""):
    """
    The spectral work that used to sit inline in process_mesh: spectrum,
    dipolarity test on e1 and e2, arc search in span{e3, e4}, and selection of
    theta*. Unchanged in substance; factored out so the identical procedure can
    be run on a surface and on its convex hull. `label` prefixes console output.
    """
    nv = verts.shape[0]

    total_required = n_modes + 2
    print(f"{label}Computing {n_modes} candidate modes (requesting {total_required} eigs)…")
    eigenvalues_full, modes_full = compute_spectrum_LB(verts, faces, total_required)

    # Extract first four nontrivial modes
    e1, e2, e3, e4 = (modes_full[:, i] for i in (0, 1, 2, 3))
    λ1, λ2, λ3, λ4 = (eigenvalues_full[i] for i in (0, 1, 2, 3))

    # Build adjacency CSR for the connectivity tests (masked component counts).
    adj_csr, _, _ = build_adjacency_csr_and_edge_lists(nv, faces)

    # Precompute per-face gradients for e3 and e4 (used for spectral-gap)
    grad_e3, area_faces = per_face_gradients(verts, faces, e3)
    grad_e4, _ = per_face_gradients(verts, faces, e4)

    # Test e1, e2 for dipolarity
    dipolar_modes = []
    dipolar_eigs  = []
    if is_dipolar_mode(e1, adj_csr, tol=1e-8):
        dipolar_modes.append(e1)
        dipolar_eigs.append(λ1)
    if is_dipolar_mode(e2, adj_csr, tol=1e-8):
        dipolar_modes.append(e2)
        dipolar_eigs.append(λ2)

    # Search for dipolar quasi-mode in span{e3, e4}
    N_SCAN = 360
    ed3    = None
    λ_ed3  = None

    print(f"{label}Scanning span{{e3, e4}} for dipolar arcs (coarse-to-fine)...")
    arcs = find_dipolar_arcs(
        e3, e4, faces, adj_csr,
        n_scan=N_SCAN, tol=1e-8, min_arc_frac=0.05,
        coarse_factor=10, verbose=verbose
    )

    if arcs:
        # Report each arc's angular width and its midpoint angle/wavelength.
        # Half-range: each fine step spans 180 / N_SCAN degrees.
        deg_per_step = 180.0 / N_SCAN
        print(f"{label}Found {len(arcs)} dipolar arc(s):")
        for ai, a in enumerate(arcs):
            width_deg = len(a) * deg_per_step
            th_mid = float(np.linspace(0, math.pi, N_SCAN, endpoint=False)[a[len(a) // 2]])
            c, s = math.cos(th_mid), math.sin(th_mid)
            lam_mid = c * c * λ3 + s * s * λ4
            wl_mid = lam_mid ** (-2.0) if lam_mid > 0 else math.inf
            print(f"{label}  arc {ai+1}: width = {width_deg:5.1f}°, "
                  f"midpoint θ = {math.degrees(th_mid):6.1f}°, "
                  f"λ_mid = {lam_mid:.4g}, wavelength(λ^-2) = {wl_mid:.4g}")

        theta_star = select_dipolar_angle(
            e3, e4, verts, faces, arcs, adj_csr,
            grad_e3=grad_e3, grad_e4=grad_e4, area_faces=area_faces,
            lam3=λ3, lam4=λ4,
            use_spectral_gap=use_spectral_gap,
            n_scan=N_SCAN, verbose=verbose
        )
        method_name = ("spectral-gap (Criterion 3)" if use_spectral_gap else "arc-midpoint (Criterion 1)")
        print(f"{label}Selected θ* = {np.degrees(theta_star):.1f}°  [{method_name}]")

        c, s = math.cos(theta_star), math.sin(theta_star)
        ed3 = c * e3 + s * e4
        λ_ed3 = c**2 * λ3 + s**2 * λ4
        dipolar_modes.append(ed3)
        dipolar_eigs.append(λ_ed3)
    else:
        print(f"{label}No dipolar arc found in span{{e3, e4}}.")

    print(f"{label}Found {len(dipolar_modes)} dipolar mode(s).")

    return dict(
        verts=verts, faces=faces,
        modes=dipolar_modes, eigs=dipolar_eigs,
        e1=e1, e2=e2, e3=e3, e4=e4,
        lam=(λ1, λ2, λ3, λ4),
        adj_csr=adj_csr, ed3=ed3, lam_ed3=λ_ed3,
    )

def process_mesh(mesh_file: str,
                 n_modes: int = n_modes_default,
                 visualize: bool = True,
                 use_spectral_gap: bool = False,
                 verbose: bool = False):
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

    # Convex hull of the loaded surface. It gets exactly the same treatment as
    # the surface itself, and supplies the denominator of every g_i. Its qhull
    # triangulation is remeshed first (see remesh_regular); areas are taken from
    # the exact hull, the spectrum from the remeshed one.
    hull_mesh   = mesh.convex_hull
    target_edge = float(mesh.edges_unique_length.mean())
    hull_remesh = remesh_regular(hull_mesh, target_edge=target_edge)
    hverts      = np.asarray(hull_remesh.vertices)
    hfaces      = np.asarray(hull_remesh.faces)

    print(f"Surface:     {len(verts):7d} vertices, {len(faces):7d} faces, "
          f"area {mesh.area:.6g}")
    print(f"Convex hull: {len(hull_mesh.vertices):7d} vertices, "
          f"{len(hull_mesh.faces):7d} faces, area {hull_mesh.area:.6g}")
    print(f"  remeshed:  {len(hverts):7d} vertices, {len(hfaces):7d} faces, "
          f"area {hull_remesh.area:.6g}  (target edge {target_edge:.4g})")

    csv_path = out_dir / f"{mesh_path.stem}_dipolar.csv"
    gyr_path = out_dir / f"{mesh_path.stem}_gyrification.csv"

    surf = compute_dipolar_triplet(verts, faces, n_modes=n_modes,
                                   use_spectral_gap=use_spectral_gap,
                                   verbose=verbose, label="[surface] ")
    hull = compute_dipolar_triplet(hverts, hfaces, n_modes=n_modes,
                                   use_spectral_gap=use_spectral_gap,
                                   verbose=verbose, label="[hull]    ")

    print(f"Saving {len(surf['modes'])} surface dipolar mode(s) to CSV.")
    if surf['modes']:
        save_cached_spectrum(csv_path, surf['eigs'], np.column_stack(surf['modes']))
    else:
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index"])

    save_gyrification_csv(gyr_path, surf, hull, mesh.area, hull_mesh.area)

    if visualize:
        λ1, λ2, λ3, λ4 = surf['lam']
        if surf['ed3'] is not None:
            edview = surf['ed3']
            lam_edview = surf['lam_ed3']
        else:
            # Fallback view when no synthetic dipole was found: show e3+e4. Its
            # Rayleigh quotient is (λ3+λ4)/2 (e3, e4 are M-orthonormal), which is
            # the correct eigenvalue to label — a 0.0 sentinel here would make
            # the 1/√λ scale bar blow up.
            edview = surf['e3'] + surf['e4']
            lam_edview = 0.5 * (λ3 + λ4)

        modes_to_show = [surf['e1'], surf['e2'], edview]
        eigs_to_show  = [λ1, λ2, lam_edview]

        n_show    = 3
        n_columns = max(n_show // min(math.isqrt(n_show - 1) + 1, 3), 3)
        faces_pv  = np.hstack([[3] + list(f) for f in faces]).astype(np.int32)
        surface   = pv.PolyData(verts, faces_pv)

        show_plot(
            surface,
            faces,
            eigs_to_show,
            np.column_stack(modes_to_show),
            n_show,
            n_columns,
            surf['adj_csr']
        )

def batch_process_meshes(folder, n_modes=4, visualize=True,
                         use_spectral_gap=False, verbose: bool = False):
    mesh_dir = Path(folder)
    if not mesh_dir.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    for mesh_file in sorted(mesh_dir.glob("*.stl")):
        print(f"\n--- Processing {mesh_file.name} ---")
        process_mesh(str(mesh_file), n_modes=n_modes, visualize=visualize,
                     use_spectral_gap=use_spectral_gap, verbose=verbose)

def main():
    parser = argparse.ArgumentParser(
        description="Compute & visualize dipolar LB modes"
    )
    parser.add_argument(
        "mesh",
        nargs="?",
        help="Path to mesh file (.stl, .obj, etc.)"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Suppress the interactive PyVista window"
    )
    parser.add_argument(
        "--spectral-gap",
        action="store_true",
        help=(
            "Select the third dipolar quasi-mode by maximising the gradient-"
            "covariance spectral gap (Criterion 3) instead of the arc "
            "midpoint (Criterion 1, default)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable progress reporting during arc scans and spectral-gap selection."
    )

    args = parser.parse_args()
    visualize        = not args.no_visualize
    use_spectral_gap = args.spectral_gap
    verbose          = args.verbose

    auto_verbose = verbose or sys.stdout.isatty()

    if args.mesh:
        process_mesh(args.mesh, visualize=visualize,
                     use_spectral_gap=use_spectral_gap, verbose=auto_verbose)
    else:
        import os
        cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(cwd)
        batch_process_meshes(
            "brains",
            visualize=visualize,
            use_spectral_gap=use_spectral_gap,
            verbose=auto_verbose,
        )

if __name__ == "__main__":
    main()