"""
fem_solver.py

Assemble and solve a small 2D linear-elastic static problem using linear triangular (3-node) elements.

Assumptions:
- Plane stress formulation.
- Small strains, linear material (Hooke's law).
- Direct dense solver via numpy.linalg.solve (OK for moderate sized mesh).
- Boundary conditions: left edge fixed (ux=uy=0). Right edge loaded downward (distributed nodal force).
"""

import numpy as np
from geometry import load_mesh, boundary_nodes
import math


def plane_stress_D(E, nu):
    """
    Elastic constitutive matrix for plane stress (3x3)
    """
    factor = E / (1.0 - nu * nu)
    D = factor * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0]
    ])
    return D


def element_stiffness_tri(Xy, D, thickness=1.0):
    """
    Compute 3x3x2x2 element stiffness for a linear triangular element.

    Xy: 3x2 array with node coords [[x0,y0],[x1,y1],[x2,y2]]
    D: 3x3 constitutive matrix
    returns: kel (6x6) stiffness matrix for element (2 DOF per node)
    """
    x0, y0 = Xy[0]
    x1, y1 = Xy[1]
    x2, y2 = Xy[2]

    # area  = 0.5 * ((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0))
    detJ = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    area = 0.5 * detJ
    if area <= 1e-12:
        # degenerate element, return zeros
        return np.zeros((6, 6))

    # B matrix (3x6)
    b0 = y1 - y2
    b1 = y2 - y0
    b2 = y0 - y1
    c0 = x2 - x1
    c1 = x0 - x2
    c2 = x1 - x0
    B = np.zeros((3, 6))
    B[0, 0::2] = [b0, b1, b2]
    B[1, 1::2] = [c0, c1, c2]
    B[2, 0::2] = [c0, c1, c2]
    B[2, 1::2] = [b0, b1, b2]
    B = B / (2.0 * area)

    kel = thickness * area * (B.T @ D @ B)
    return kel


def assemble_global(nodes, elems, D, thickness=1.0):
    n_nodes = nodes.shape[0]
    ndof = 2 * n_nodes
    K = np.zeros((ndof, ndof))
    # loop elements
    for tri in elems:
        Xy = nodes[tri, :]
        ke = element_stiffness_tri(Xy, D, thickness)
        # assemble to K
        dof_indices = np.zeros(6, dtype=np.int64)
        for i in range(3):
            dof_indices[2 * i] = 2 * tri[i]
            dof_indices[2 * i + 1] = 2 * tri[i] + 1
        # add ke to K at dof_indices
        for i in range(6):
            ii = dof_indices[i]
            for j in range(6):
                jj = dof_indices[j]
                K[ii, jj] += ke[i, j]
    return K


def apply_boundary_conditions(K, f, fixed_dofs):
    """
    Apply essential (Dirichlet) boundary conditions by modifying stiffness and load.
    Simple approach: zero rows/cols and set diagonal to 1 and rhs to BC value (zero here).
    fixed_dofs: list/array of DOF indices to fix (zero displacement).
    """
    K_mod = K.copy()
    f_mod = f.copy()
    for d in fixed_dofs:
        K_mod[d, :] = 0.0
        K_mod[:, d] = 0.0
        K_mod[d, d] = 1.0
        f_mod[d] = 0.0
    return K_mod, f_mod


def distribute_nodal_forces(nodes, elems, right_edge_indices, total_force=-1000.0):
    """
    Distribute a total vertical force (negative = downward) among specified nodes.
    Returns force vector F of length 2*n_nodes.
    """
    n_nodes = nodes.shape[0]
    F = np.zeros(2 * n_nodes)
    if len(right_edge_indices) == 0:
        return F
    per_node = total_force / float(len(right_edge_indices))
    for idx in right_edge_indices:
        F[2 * idx + 1] += per_node
    return F


def compute_element_strain_stress(nodes, elems, u, D):
    """
    For each triangular element compute strain (eps) and stress (sigma) in Voigt form.
    Returns arrays of shape (n_elems, 3) for strain and stress.
    """
    n_elems = elems.shape[0]
    strains = np.zeros((n_elems, 3))
    stresses = np.zeros((n_elems, 3))
    for e_idx, tri in enumerate(elems):
        Xy = nodes[tri, :]
        x0, y0 = Xy[0]
        x1, y1 = Xy[1]
        x2, y2 = Xy[2]
        detJ = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        area = 0.5 * detJ if detJ != 0 else 0.0
        # compute B matrix components (same as in stiffness)
        b0 = y1 - y2
        b1 = y2 - y0
        b2 = y0 - y1
        c0 = x2 - x1
        c1 = x0 - x2
        c2 = x1 - x0
        B = np.zeros((3, 6))
        B[0, 0::2] = [b0, b1, b2]
        B[1, 1::2] = [c0, c1, c2]
        B[2, 0::2] = [c0, c1, c2]
        B[2, 1::2] = [b0, b1, b2]
        if area != 0.0:
            B = B / (2.0 * area)
            ue = np.zeros(6)
            for i in range(3):
                ue[2 * i] = u[2 * tri[i]]
                ue[2 * i + 1] = u[2 * tri[i] + 1]
            eps = B @ ue
            sig = D @ eps
            strains[e_idx, :] = eps
            stresses[e_idx, :] = sig
    return strains, stresses


def von_mises_from_stress(sig):
    """
    Compute von Mises stress for plane stress state for array sig shape (n,3) [sig_xx, sig_yy, sig_xy]
    """
    sxx = sig[:, 0]
    syy = sig[:, 1]
    sxy = sig[:, 2]
    vm = np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * sxy ** 2)
    return vm


def run_static_analysis(mesh_file="mesh.npz", E=7e10, nu=0.33, thickness=0.01, total_force=-500.0):
    """
    High-level function: load mesh, assemble, apply BCs, solve, postprocess.
    Saves results to 'results.npz' by convention.
    """
    nodes, elems = load_mesh(mesh_file)
    D = plane_stress_D(E, nu)

    print(f"Loaded mesh: nodes={nodes.shape[0]}, elems={elems.shape[0]}")

    # assemble global stiffness
    K = assemble_global(nodes, elems, D, thickness)
    n_nodes = nodes.shape[0]
    ndof = 2 * n_nodes

    # boundary detection
    bnds = boundary_nodes(nodes)
    left_nodes = bnds["left"]
    right_nodes = bnds["right"]

    # set fixed dofs (both ux,uy) for left edge
    fixed = []
    for idx in left_nodes:
        fixed.append(2 * idx)
        fixed.append(2 * idx + 1)

    # forcing: distribute on right edge nodes downward
    F = distribute_nodal_forces(nodes, elems, right_nodes, total_force=total_force)

    # Apply BCs
    Kmod, Fmod = apply_boundary_conditions(K, F, fixed)

    # Solve linear system
    print("Solving linear system...")
    u = np.linalg.solve(Kmod, Fmod)

    # compute strains and stresses per element
    strains, stresses = compute_element_strain_stress(nodes, elems, u, D)
    vonm = von_mises_from_stress(stresses)

    # Save results
    np.savez_compressed("results.npz", nodes=nodes, elems=elems, u=u, strains=strains, stresses=stresses, vonm=vonm)
    print("Saved results to results.npz")
    return nodes, elems, u, strains, stresses, vonm


if __name__ == "__main__":
    run_static_analysis()
