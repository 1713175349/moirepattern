import numpy as np
from ase.build import make_supercell

def _norm(v): 
    return np.linalg.norm(v)

def _embed_2x2_to_3x3(L2, axes=(0, 1)):
    """把 2x2 行变换矩阵 L2 嵌入 3x3 左乘矩阵，仅作用于 axes 这两条晶格矢量。"""
    P = np.eye(3, dtype=int)
    for i, row_idx in enumerate(axes):
        P[row_idx, :] = 0
        P[row_idx, axes[0]] = int(L2[i, 0])
        P[row_idx, axes[1]] = int(L2[i, 1])
    return P

def _right_handed_sign(cell, axes=(0,1)):
    """相对给定顺序 (axes[0], axes[1], out) 的有向体积符号（>0 为右手系）。"""
    a = cell[axes[0]]
    b = cell[axes[1]]
    out = ({0,1,2} - set(axes)).pop()
    c = cell[out]
    return np.sign(np.dot(np.cross(a, b), c))

def reduce_surface_mn(
    cell0,mn, tol=1e-10, max_iter=10_000, trace=False
):
    """
    2D Lagrange/Gauss 规约 + 右手系保障。
    返回: (reduced_atoms, P_final(3x3 int), L2(2x2 int), flipped_out_axis: bool)
    """
    cell = np.dot(mn,cell0)
    a = cell[0].copy()
    b = cell[1].copy()

    # 累计“对行”的 2x2 整数变换：new_rows = L2 @ old_rows
    L2 = np.eye(2, dtype=int)
    it = 0
    while True:
        it += 1
        if it > max_iter:
            raise RuntimeError("Reduction did not converge. Try loosening tol or check the input cell.")

        changed = False

        # 1) If a·b < 0 replace b by -b
        if np.dot(a, b) < -tol:
            b = -b
            L2 = np.array([[1, 0], [0, -1]], dtype=int) @ L2
            changed = True
            if trace: print("b -> -b")

        # 2) |a| > |b| ? swap
        if _norm(a) > _norm(b) + tol:
            a, b = b, a
            L2 = np.array([[0, 1], [1, 0]], dtype=int) @ L2
            changed = True
            if trace: print("swap a,b")

        # 3) |b| > |b+a| ? b -> b + a
        if _norm(b) > _norm(b + a) + tol:
            b = b + a
            L2 = np.array([[1, 0], [1, 1]], dtype=int) @ L2
            changed = True
            if trace: print("b -> b + a")

        # 4) |b| > |b-a| ? b -> b - a
        elif _norm(b) > _norm(b - a) + tol:
            b = b - a
            L2 = np.array([[1, 0], [-1, 1]], dtype=int) @ L2
            changed = True
            if trace: print("b -> b - a")

        if not changed:
            break

    # 组装 3x3 左乘矩阵（只变换 in-plane 行）
    new_mn=np.dot(L2,mn)
    P=[[new_mn[0,0],new_mn[0,1],0],[new_mn[1,0],new_mn[1,1],0],[0,0,1]]
    if np.linalg.det(P)<0:
        new_mn=np.array([[new_mn[1,0],new_mn[1,1]],[new_mn[0,0],new_mn[0,1]]])


    return new_mn