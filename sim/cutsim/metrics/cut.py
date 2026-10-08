"""절단 판정 지표: 조각 라벨, 틈 폭, 분리 유지(재결합), 연결 성분, 부피, 힘 곡선.

절단면 위치(cut_x)를 알고 있으므로 조각은 "초기 위치가 절단면의 어느 쪽인가"로 라벨링한다.
틈 폭은 입자 크기보다 작을 수 있어 복셀 연결로는 못 잰다. 대신 절단면에 맞닿은 입자층끼리의
최근접 거리가 자르기 전(같은 라벨로 잰 기준값)보다 얼마나 늘었는지로 잰다.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def piece_labels(pos0, cut_xs):
    """초기 x 가 절단면 몇 개보다 큰가 → 0..len(cut_xs)."""
    return np.searchsorted(np.sort(np.asarray(cut_xs)), pos0[:, 0])


def face_c2c(pos, pos0, labels, cut_x, a, b, layer):
    """절단면에 맞닿은 b 쪽 입자층(초기 x ∈ [cut_x, cut_x+layer))에서 a 조각까지의 최근접 거리 중앙값."""
    face_b = (labels == b) & (pos0[:, 0] >= cut_x) & (pos0[:, 0] < cut_x + layer)
    face_a = (labels == a) & (pos0[:, 0] < cut_x) & (pos0[:, 0] >= cut_x - layer)
    if face_a.sum() == 0 or face_b.sum() == 0:
        return np.nan
    d, _ = cKDTree(pos[face_a]).query(pos[face_b])
    return float(np.median(d))


def gap_width(pos, pos0, labels, cut_x, a, b, layer):
    """틈 폭 ≈ (지금 맞닿은 층 거리) - (자르기 전 같은 층 거리)."""
    return face_c2c(pos, pos0, labels, cut_x, a, b, layer) - face_c2c(pos0, pos0, labels, cut_x, a, b, layer)


def nn_spacing(pos0):
    d, _ = cKDTree(pos0).query(pos0, k=2)
    return float(np.median(d[:, 1]))


def n_components(pos, radius, min_frac=0.01):
    """반경 radius 안의 입자끼리 이은 그래프의 연결 성분 수(전체의 min_frac 미만 성분은 버림)."""
    pairs = cKDTree(pos).query_pairs(radius, output_type="ndarray")
    n = len(pos)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    k, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab)
    return int((sizes >= min_frac * n).sum()), sorted(sizes.tolist(), reverse=True)[:8]


def centroids(pos, labels):
    return {int(l): pos[labels == l].mean(0) for l in np.unique(labels)}


def follow_ratio(pos_before, pos_after, labels, pushed, other):
    """밀린 조각 대비 반대쪽 조각의 이동량 비율(재결합 지표). 0 이면 완전히 분리, 1 이면 한 덩어리.

    returns (크기 비율, 밀린 조각 이동 mm 아닌 m, 반대쪽 이동 m, 미는 방향 성분 비율)
    미는 방향 성분 비율: 반대쪽 이동을 밀린 조각의 이동 방향에 투영한 비율. +면 끌려옴(재결합), -면 밀려남.
    """
    c0, c1 = centroids(pos_before, labels), centroids(pos_after, labels)
    v_p, v_o = c1[pushed] - c0[pushed], c1[other] - c0[other]
    d_p = np.linalg.norm(v_p)
    d_o = np.linalg.norm(v_o)
    signed = float(v_o @ v_p / max(d_p, 1e-9) ** 2)
    return float(d_o / max(d_p, 1e-9)), float(d_p), float(d_o), signed


def volume_change(J):
    """입자 부피 비율 det(F) 평균 - 1 (같은 물체 안에서 초기 입자 부피가 같다고 근사)."""
    return float(np.mean(J) - 1.0)


def smooth(x, k):
    if k <= 1:
        return np.asarray(x)
    w = np.ones(k) / k
    return np.convolve(np.pad(x, (k // 2, k - 1 - k // 2), mode="edge"), w, mode="valid")


def force_curve_checks(depth, force, full_depth):
    """힘 곡선 모양 판정(경험 규칙): 접촉 후 상승, 절단 중 유지, 관통 직전(마지막 10%) 급락.

    depth: 날 끝이 재료 윗면에서 내려간 깊이(m), force: 위쪽(+) 방향 절단 저항력(N).
    """
    m = (depth > 0) & (depth <= full_depth * 1.02)
    if m.sum() < 10:
        return {"ok": False, "reason": "접촉 구간 없음"}
    d, f = depth[m], force[m]
    peak = float(f.max())
    d_peak = float(d[f.argmax()])
    early = f[d < 0.15 * full_depth]
    mid = f[(d > 0.3 * full_depth) & (d < 0.8 * full_depth)]
    late = f[d > 0.92 * full_depth]
    mid_mean = float(mid.mean()) if len(mid) else np.nan
    res = {
        "peak_N": peak,
        "depth_at_peak_frac": d_peak / full_depth,
        "mean_early_N": float(early.mean()) if len(early) else np.nan,
        "mean_mid_N": mid_mean,
        "mean_late_N": float(late.mean()) if len(late) else np.nan,
        # 상승: 접촉 직후(첫 2%)보다 봉우리가 뚜렷이 크고, 봉우리가 바닥 직전이 아니다
        "rise": bool(peak > 3 * max(float(f[d < 0.02 * full_depth].mean()) if (d < 0.02 * full_depth).any() else 0, 1e-3)
                     and d_peak < 0.9 * full_depth),
        # 유지: 절단 중간(30~80%)에 힘이 끊기지 않는다(봉우리 5% 넘는 표본이 90% 이상, 평균이 봉우리 10% 이상)
        "sustain": bool(len(mid) and (mid > 0.05 * peak).mean() >= 0.9 and mid_mean > 0.1 * peak),
        # 급락: 마지막 8% 의 평균이 중간 평균의 절반 미만
        "drop": bool(len(late) and len(mid) and late.mean() < 0.5 * mid_mean),
    }
    res["ok"] = res["rise"] and res["sustain"] and res["drop"]
    return res
