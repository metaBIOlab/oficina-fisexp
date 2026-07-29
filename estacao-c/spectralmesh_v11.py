# Mesh spectral decomposer v.11
# Bruno Mota – metaBIO lab

"""
Compute Laplace–Beltrami eigenmodes on a closed mesh, visualize them
with PyVista, detect dipolarity, and save main component mesh, L and M matrices, eigenvalues &
eigenvectors to directory.
"""

import argparse
import csv
import itertools
import math
from math import cos, sin, pi
import os
import shutil
import time
from pathlib import Path

import numpy as np
import networkx as nx
import scipy.sparse.linalg as sla
from scipy.sparse import spmatrix, csr_matrix, load_npz, save_npz
import trimesh
import pyvista as pv

import robust_laplacian

from typing import Optional, Tuple

from datetime import datetime, timezone

n_modes_default = 12
PLOT_PARAMS = {
    'edge_color': 'lightgray',
    'font_size': {'title': 14, 'mode': 9},
    'cmap': 'coolwarm',
    'line_width': 4,
}

# -------------------- Spectrum Cache Logic --------------------

def load_mesh_and_spectrum_with_version(
    raw_path,
    cache_dir,
    version_file,
    current_version,
    mode_count
):
    raw_path     = Path(raw_path)
    cache_dir    = Path(cache_dir)
    version_file = Path(version_file)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Mesh caching + version check
    mesh_cache = cache_dir / f"{raw_path.stem}.ply"
    if version_file.exists():
        saved_version = version_file.read_text().strip()
    else:
        saved_version = ""

    if not mesh_cache.exists() or saved_version != current_version:
        mesh = trimesh.load(raw_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        mesh.merge_vertices()
        submeshes = mesh.split(only_watertight=False)
        if len(submeshes) > 1:
            mesh = max(submeshes, key=lambda m: len(m.vertices))
            print(f"Kept largest component ({len(mesh.faces)} faces), "
                  f"dropped {len(submeshes)-1} parts")
        mesh.remove_unreferenced_vertices()
        mesh.export(str(mesh_cache), file_type="ply")
        version_file.write_text(current_version)
        print(f"Exported cleaned mesh to {mesh_cache}")
    else:
        mesh = trimesh.load(mesh_cache, process=False)
        print(f"Loaded cleaned mesh from {mesh_cache}")

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    n_verts = len(verts)

    # 2. L and M caching
    L_cache = cache_dir / "L.npz"
    M_cache = cache_dir / "M.npz"
    if L_cache.exists() and M_cache.exists():
        L     = load_npz(str(L_cache))
        M_mat = load_npz(str(M_cache))
        print(f"Loaded L and M matrices from {cache_dir}")
    else:
        L, M_mat = robust_laplacian.mesh_laplacian(verts, faces)
        save_npz(str(L_cache), L)
        save_npz(str(M_cache), M_mat)
        print(f"Saved new L and M matrices to {cache_dir}")
    M = M_mat  # alias for return

    # 3. Spectrum caching + logging + trivial‐mode drop
    modes_cache = cache_dir / f"{raw_path.stem}_modes.npz"
    if modes_cache.exists():
        data         = np.load(str(modes_cache))
        cached_evals = data['evals']
        cached_evecs = data['evecs']
    else:
        cached_evals = np.empty(0)
        cached_evecs = np.zeros((n_verts, 0))

    if len(cached_evals) >= mode_count:
        print(f"Loaded {mode_count} eigenmodes from cache.")
        return mesh, L, M, cached_evals[:mode_count], cached_evecs[:, :mode_count]

    num_cached = len(cached_evals)
    needed     = mode_count - num_cached
    total_k    = num_cached + needed + 1  # +1 for trivial

    # guard: ARPACK needs k < N
    if total_k >= n_verts:
        total_k = max(n_verts - 1, 1)
        print(f"Adjusted k to {total_k} (n_verts={n_verts}) to avoid k>=N")

    print(f"Cache has {num_cached} modes; computing {needed} more (k={total_k})…")

    start = time.time()
    # ask compute_spectrum_LB to return the trivial mode as well
    eigvals_all, eigvecs_all = compute_spectrum_LB(
        verts,
        faces,
        total_k,
        L=L,
        M_mat=M,
        discard_trivial=False
    )
    print(f"Computed {needed} additional modes in {time.time() - start:.2f} s")

    # sort, drop exactly one trivial zero‐mode, then take 'needed'
    idx         = np.argsort(eigvals_all)
    vals_sorted = eigvals_all[idx]
    vecs_sorted = eigvecs_all[:, idx]

    new_vals = vals_sorted[1: needed+1]
    new_vecs = vecs_sorted[:,    1: needed+1]

    # combine with cache
    evals = np.hstack([cached_evals, new_vals])
    evecs = np.hstack([cached_evecs, new_vecs])

    np.savez(str(modes_cache), evals=evals, evecs=evecs)
    return mesh, L, M, evals, evecs

def save_cached_spectrum(path: Path, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> None:
    """
    Save computed non-trivial eigenpairs into a compressed .npz.

    Parameters
    ----------
    path : Path
        Filepath to write (.npz will be appended if missing).
    eigenvalues : (k,) ndarray
    eigenvectors : (n, k) ndarray
    """
    # ensure suffix
    if path.suffix.lower() != ".npz":
        path = path.with_suffix(path.suffix + ".npz")
    np.savez_compressed(path, evals=eigenvalues, modes=eigenvectors)    

# -------------------- Geometry & Spectrum Helpers --------------------

def build_connectivity_graph(n_vertices, faces):
    G = nx.Graph()
    G.add_nodes_from(range(n_vertices))
    for face in faces:
        for i, j in itertools.combinations(face, 2):
            G.add_edge(i, j)
            G.add_edge(j, i)
    return G

def dyadic_search(f, Xmin, Xmax, order: int = 4):
    """
    A drop‐in replacement for golden_section_search that samples at dyadic
    fractions up to 2^order.  It evaluates f at the points:

      0,
      Xmax/2, Xmin/2,                    # order = 1
      Xmin*(1/4), Xmin*(3/4),           # order = 2 (negative side)
      Xmax*(1/4), Xmax*(3/4),           # order = 2 (positive side)
      Xmin*(1/8), Xmin*(3/8), …         # order = 3
      Xmax*(1/8), Xmax*(3/8), …
      …
      up to order `order`

    and returns the x that gives the largest f(x).
    """

    # build the sequence of test points
    xs = [0.0]

    for lvl in range(1, order + 1):
        step = 1 << lvl               # 2**lvl
        # negative‐side dyads: Xmin * (k/2^lvl), k = 1,3,5,…,2^lvl-1
        for k in range(1, step, 2):
            xs.append(Xmin * (k / step))

        # positive‐side dyads: Xmax * (k/2^lvl), k = 1,3,5,…,2^lvl-1
        for k in range(1, step, 2):
            xs.append(Xmax * (k / step))

    # scan all sample points and pick the best
    best_x = xs[0]
    best_y = f(best_x)
    for x in xs[1:]:
        y = f(x)
        if y > best_y:
            best_y, best_x = y, x

    return best_x, best_y

def is_dipolar_mode(values, faces, G, tol=1e-6):
    def component_sum(x: float) -> float:
        pos = [i for i, v in enumerate(values) if v >= tol + x]
        neg = [i for i, v in enumerate(values) if v <= -tol + x]

        if len(pos) == 0 or len(neg) == 0:
            raise ValueError("Insufficient regions at x = {}".format(x))

        p_c = nx.number_connected_components(G.subgraph(pos))
        n_c = nx.number_connected_components(G.subgraph(neg))

        if p_c != 1 or n_c != 1:
            raise ValueError("Mode not dipolar at x = {}".format(x))

        return 2.0  # You can replace this with any constant; it won't matter

    try:
        x_min, x_max = np.min(values), np.max(values)
        _ = dyadic_search(lambda x: -component_sum(x), x_min, x_max, order=3)
        return True, None, None
    except ValueError as e:
        return False, None, str(e)

def compute_spectrum_LB(verts: np.ndarray, faces: np.ndarray, n_modes: int,
    sigma: float = 1e-8,
    discard_trivial: bool = True,
    which: str = 'LM',
    tol: Optional[float] = None,
    maxiter: Optional[int] = None,
    L: Optional[spmatrix] = None,
    M_mat: Optional[spmatrix] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenpairs of the Laplace–Beltrami operator on a triangular mesh.
    All parameters and doc omitted for brevity…
    """
    # input checks omitted…

    if L is None or M_mat is None:
        L, M_mat = robust_laplacian.mesh_laplacian(verts, faces)

    # build ARPACK kwargs
    arpack_kwargs = {}
    if tol is not None:
        arpack_kwargs['tol'] = tol
    if maxiter is not None:
        arpack_kwargs['maxiter'] = maxiter

    # call eigsh without any None
    evals, evecs = sla.eigsh(
        A=L,
        k=n_modes,
        M=M_mat,
        sigma=sigma,
        which=which,
        **arpack_kwargs
    )

    idx = np.argsort(evals)
    if discard_trivial:
        evals = evals[idx][1:]
        evecs = evecs[:, idx][:, 1:]
    else:
        evals = evals[idx]
        evecs = evecs[:, idx]

    i_max = np.argmax(verts[:, 0])    # --- Align sign at the vertex with highest x 
    for j in range(evecs.shape[1]):
        if evecs[i_max, j] < 0:
            evecs[:, j] *= -1

    return evals, evecs


def compute_color_limit(modes):
    abs_max = np.max(np.abs(modes))
    exp = np.floor(np.log10(abs_max))
    step = 10 ** exp
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
    plotter.add_text('Original Mesh', font_size=PLOT_PARAMS['font_size']['title'],
                     position=(0.4, 0.01), viewport=True)

def add_mesh_scale_bar(plotter, surface):
    plotter.subplot(0, 1)
    xmin, _, ymin, ymax, zmin, _ = surface.bounds
    mesh_width = ymax - ymin

    if mesh_width >= 10:
        bar_length = np.ceil(mesh_width / 10) * 10
        label_text = f"{int(bar_length)}"
    else:
        bar_length = np.ceil(mesh_width * 10) / 10
        label_text = f"{int(round(bar_length))}"

    start = (xmin, ymin, zmin)
    end = (xmin, ymin + bar_length, zmin)
    scale_bar = pv.Line(start, end)

    plotter.add_mesh(scale_bar, color='black', line_width=PLOT_PARAMS['line_width'])
    plotter.add_point_labels(
        points=np.array([end]),
        labels=[label_text],
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

def add_eigenmode_subplot(plotter, surface, modes, eigenvalues, faces, index, row, col, text_position, graph, tol=1e-8):
    # copy mesh and attach this mode as a scalar field
    mcopy = surface.copy()
    mcopy["eigenmode"] = modes[:, index]

    # test for dipolarity
    dipolar, pos_comp, neg_comp = is_dipolar_mode(
        modes[:, index], faces, graph, tol
    )
    eigen_label = f"k{index+1} = {eigenvalues[index]:.4f}"
    dip_label   = "dipolar" if dipolar else "multipolar"
    label       = eigen_label + "\n" + dip_label

    # draw the mesh and label
    plotter.subplot(row, col)
    plotter.add_mesh(
        mcopy,
        scalars="eigenmode",
        cmap=PLOT_PARAMS['cmap'],
        show_edges=False,
        show_scalar_bar=False
    )
    plotter.add_text(
        label,
        font_size=PLOT_PARAMS['font_size']['mode'],
        position=text_position,
        viewport=True
    )

    # clamp tiny/negative eigenvalues to avoid NaN in sqrt
    eig      = eigenvalues[index]
    safe_eig = max(eig, tol)
    EM_typ_length = np.pi / np.sqrt(safe_eig)
    green_label   = f"{EM_typ_length:.1f}"

    # determine where to draw the green bar
    x_min, x_max, y_min, y_max, z_min, z_max = mcopy.bounds
    z_extent = z_max - z_min
    delta_z  = 0.05 * z_extent if z_extent > 0 else 1.0
    bar_z    = z_min - delta_z

    bar_start = (x_min, y_min,           bar_z)
    bar_end   = (x_min, y_min + EM_typ_length, bar_z)
    green_bar = pv.Line(bar_start, bar_end)

    # draw the bar and its label
    plotter.add_mesh(
        green_bar,
        color='green',
        line_width=PLOT_PARAMS['line_width']
    )
    plotter.add_point_labels(
        points=np.array([[x_min, y_min + EM_typ_length, bar_z]]),
        labels=[green_label],
        font_size=12,
        text_color="green",
        shape_opacity=0.0
    )
  

def plot_eigenmodes(plotter, surface, faces, eigenvalues, modes, n_modes=6, n_columns=3):
    graph = build_connectivity_graph(len(surface.points), faces)
    for i in range(n_modes):
        row = (i // n_columns) + 1
        col = i % n_columns
        text_position = (0.4, -0.01)
        add_eigenmode_subplot(
            plotter, surface, modes, eigenvalues, faces,
            index=i,
            row=row,
            col=col,
            text_position=text_position,
            graph=graph
        )

def show_plot(surface, faces, eigenvalues, modes, n_modes, n_columns, clim=None):
    n_rows_modes = math.ceil(n_modes / n_columns)
    total_rows = 1 + n_rows_modes
    plotter = setup_plotter(
        surface,
        shape=(total_rows, n_columns),
        size=(min(300 * n_columns, 1800), min(275 * total_rows, 1100))
    )

    add_original_mesh(plotter, surface)
    add_mesh_scale_bar(plotter, surface)
    limit = compute_color_limit(modes)
    add_shared_colorbar(plotter, limit)
    plot_eigenmodes(plotter, surface, faces, eigenvalues, modes, n_modes, n_columns)

    plotter.link_views()
    plotter.show()

def prepare_output_folder(mesh_path: Path) -> Path:
    project_root = Path(__file__).resolve().parent
    folder_name = mesh_path.stem
    out_dir = project_root / folder_name
    out_dir.mkdir(exist_ok=True)
    return out_dir

# ─── process_mesh ──────────────────────────────────────
def process_mesh(mesh_file: str, n_modes: int = n_modes_default):
    mesh_path = Path(mesh_file).resolve()
    out_dir   = prepare_output_folder(mesh_path)

    # copy + timestamp
    dest_mesh = out_dir / mesh_path.name
    if not dest_mesh.exists():
        shutil.copy2(mesh_path, dest_mesh)

    current_version = datetime.fromtimestamp(
        dest_mesh.stat().st_mtime, tz=timezone.utc
    ).isoformat()
    print(f"Processing: {dest_mesh.name} (version: {current_version})")

    # 1. Request exactly n_modes (no extra +1 here)
    mesh, L, M, evals_full, modes_full = \
        load_mesh_and_spectrum_with_version(
            raw_path        = dest_mesh,
            cache_dir       = out_dir,
            version_file    = out_dir / 'mesh.version',
            current_version = current_version,
            mode_count      = n_modes
        )

    # 2. These come back with the trivial already dropped,
    #    so evals_full/modes_full each have shape (n_modes,)
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    # 3. Cache & return
    save_cached_spectrum(
        out_dir / f"{mesh_path.stem}_spectrum.npz",
        evals_full, modes_full
    )

    return verts, faces, evals_full, modes_full

def batch_process_meshes(folder: str,
                         n_modes: int = n_modes_default,
                         visualize: bool = False):
    """
    Process all meshes in `folder`, optionally visualizing each one.
    After successful processing, remove the original mesh file.
    """
    mesh_dir = Path(folder)
    mesh_dir.mkdir(exist_ok=True)
    files = sorted(mesh_dir.glob("*.*"))

    if not files:
        print(f"No mesh files found in '{folder}'.")
        return

    for mesh_file in files:
        try:
            # Compute & cache eigenmodes
            verts, faces, eigenvalues, modes = process_mesh(
                str(mesh_file),
                n_modes=n_modes
            )

            # Visualize if requested
            if visualize:
                visualize_modes(verts, faces, eigenvalues, modes, n_modes)

            # Remove original mesh
            mesh_file.unlink()

        except Exception as e:
            print(f"Failed to process {mesh_file.name}: {e}")

def visualize_modes(verts: np.ndarray,
                    faces: np.ndarray,
                    eigenvalues: np.ndarray,
                    modes: np.ndarray,
                    n_modes: int):
    """
    Render the first n_modes eigenfunctions on the mesh using PyVista.
    """
    # build pyvista surface
    faces_pv = np.hstack([[3] + list(f) for f in faces]).astype(np.int32)
    surface  = pv.PolyData(verts, faces_pv)

    # arrange grid & colormap limits
    n_cols = max(n_modes // min(math.isqrt(n_modes - 1) + 1, 3), 3)
    limit  = compute_color_limit(modes)

    show_plot(surface,
              faces,
              eigenvalues,
              modes,
              n_modes,
              n_cols,
              clim=[-limit, limit])

def process_dipolar_modes(
    verts: np.ndarray,
    faces: np.ndarray,
    evals: np.ndarray,
    modes: np.ndarray,
    max_iter: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    From the first 4 modes extract all dipolar ones (d1,d2).
    If only two are dipolar, mix the two non-dipolar to find a third (d3)
    by enumerating x = ±(2n+1)/2^k for k=1..max_iter.
    Prints status messages along the way.
    Returns:
      - d_evs  : array of length 3
      - d_mods : ndarray of shape (n_vertices, 3)
    """
    # 1) build connectivity graph for your dipolar test
    n_verts = verts.shape[0]
    G        = build_connectivity_graph(n_verts, faces)

    # 2) identify true dipolar among the first four
    dip_idxs = []
    for i in range(min(4, evals.shape[0])):
        dip, _, _ = is_dipolar_mode(modes[:, i], faces, G)
        if dip:
            dip_idxs.append(i)

    print(f"Found {len(dip_idxs)} dipolar modes among the first 4")

    # collect d1, d2
    d_evs, d_mods = [], []
    for idx in dip_idxs:
        d_evs.append(evals[idx])
        d_mods.append(modes[:, idx])

    # if already have 3 or more, just truncate to three
    if len(d_evs) >= 3:
        d_evs  = np.array(d_evs[:3])
        d_mods = np.stack(d_mods[:3], axis=1)
        return d_evs, d_mods

    # only two dipolar → mix the two nondipolar to find a third
    nondip = [i for i in range(4) if i not in dip_idxs]
    if len(nondip) < 2:
        raise RuntimeError("Not enough modes to mix for a third dipolar")

    print("Searching for non-eigenmodal dipolar modes")
    i1, i2   = nondip[:2]
    ev1, ev2 = evals[i1], evals[i2]
    m1, m2   = modes[:, i1], modes[:, i2]

    found = False
    # enumerate x = ±(2n+1)/2^k
    for k in range(1, max_iter + 1):
        denom = 2 ** k
        for n in range(2 ** (k - 1)):
            x_base = (2*n + 1) / denom
            for x in ( x_base, -x_base ):
                print(f"Testing dip_candidate = {cos(pi*(x)/2):.3f}*m1 + {sin(pi*(x)/2):.3f}*m2")
                dc = cos(pi*(x)/2) * m1 + sin(pi*(x)/2) * m2
                dip, _, _ = is_dipolar_mode(dc, faces, G)
                if dip:
                    xd    = x
                    d3_ev = cos(pi*(xd)/2)**2 *ev1  + sin(pi*(xd)/2)**2 *ev2
                    d3_md = dc
                    print(f"Found dipolar mode d3 = {cos(pi*(xd)/2):.3f}*m1 + {sin(pi*(xd)/2):.3f}*m2")
                    found = True
                    break
            if found:
                break
        if found:
            break

    if found:
        d_evs.append(d3_ev)
        d_mods.append(d3_md)
    else:
        print("Warning: no mixed dipolar found; using fallback mode")
        d_evs.append(ev2)
        d_mods.append(m2)

    # now we have exactly three dipolar modes
    d_evs  = np.array(d_evs)
    d_mods = np.stack(d_mods, axis=1)
    return d_evs, d_mods

# main

def main():
    parser = argparse.ArgumentParser(
        description="Spectral Laplacian: compute & visualize LB eigenmodes"
    )
    parser.add_argument(
        "mesh",
        nargs="?",
        help="path to mesh file (.stl, .obj, ...)"
    )
    parser.add_argument(
        "--modes",
        type=int,
        default=n_modes_default,
        help="number of nontrivial eigenmodes to process"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="compute/cache modes but do not launch the visualization"
    )
    parser.add_argument(
        "--dipolar-modes",
        action="store_true",
        help="override --modes: extract exactly three dipolar modes"
    )
    args      = parser.parse_args()
    visualize = not args.no_visualize

    # decide how many modes to compute
    if args.dipolar_modes:
        n_req = 4
    else:
        n_req = args.modes

    # process_mesh as before, but with n_req
    if args.mesh:
        verts, faces, evals, modes = process_mesh(
            args.mesh, n_modes=n_req
        )
        out_dir  = prepare_output_folder(Path(args.mesh).resolve())
        stem     = Path(args.mesh).stem

        if args.dipolar_modes:
            # extract and optionally mix dipolar modes
            d_evs, d_mods = process_dipolar_modes(
                verts, faces, evals, modes
            )
            # save to <stem>_spectrum_dipolar.npz
            dip_path = out_dir / f"{stem}_spectrum_dipolar.npz"
            save_cached_spectrum(dip_path, d_evs, d_mods)
            if visualize:
                visualize_modes(verts, faces, d_evs, d_mods, n_modes = len(d_evs))
        else:
            if visualize:
                visualize_modes(verts, faces, evals, modes, args.modes)

    else:
        import os
        cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(cwd)
        mesh_dir = Path("brains")
        for mesh_file in mesh_dir.iterdir():
            if mesh_file.suffix.lower() not in (".stl", ".obj", ".ply"):
                continue
            verts, faces, evals, modes = process_mesh(
                mesh_file, n_modes=n_req
            )
            out_dir = prepare_output_folder(mesh_file.resolve())
            stem    = mesh_file.stem

            if args.dipolar_modes:
                d_evs, d_mods = process_dipolar_modes(
                    verts, faces, evals, modes
                )
                dip_path = out_dir / f"{stem}_spectrum_dipolar.npz"
                save_cached_spectrum(dip_path, d_evs, d_mods)
                if visualize:
                    visualize_modes(verts, faces, d_evs, d_mods, len(d_evs))
            else:
                if visualize:
                    visualize_modes(
                        verts, faces, evals, modes, args.modes
                    )


if __name__ == "__main__":
    main()

