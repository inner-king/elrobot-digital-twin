"""복원 메쉬를 시뮬레이션에 넣기 전 다듬기.

복원 겉면의 작은 돌기·주름(TSDF 3 mm 격자 + 깊이 잡음)이 입자 크기(2.5 mm)와 비슷해서, 입자를 채우면 몸통과 이어지지
않은 작은 입자 덩어리가 생긴다(자르기 전부터 흩어짐). Taubin 다듬기로 돌기를 지우고, 줄어든 부피는 가운데 기준 배율로 되돌린다.
"""
import numpy as np
import trimesh


def smooth_keep_volume(mesh, iters):
    m = mesh.copy()
    if iters <= 0:
        return m, 0.0
    v0, c = m.volume, m.center_mass
    trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=-0.53, iterations=iters)
    shrink = m.volume / v0 - 1
    m.vertices = c + (m.vertices - c) * (v0 / m.volume) ** (1 / 3)
    return m, shrink
