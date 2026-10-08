"""Genesis 1.4.3 MPM 패치 생성기: (1) 조각 짝/홀 두 속도장(two-field) (2) 칼 안쪽 격자점 분리
(3) 렌더 꼭짓점 지지 입자 찾기 KD-tree 화(메모리 폭주 수정) (4) 조각 사이 접촉을 꽉 찬 격자점에서만
(5) 입자별 감쇠(서브스텝마다 속도와 APIC C 함께).

문제: MPM 은 입자들이 한 격자를 공유하므로, 칼을 뺀 뒤 떨어진 두 조각이 격자 1~2칸 안에 있으면
격자 속도가 섞여 한 덩어리처럼 끌려간다(T3 재결합). CPIC 는 칼이 그 자리에 있을 때만 양쪽을 가른다.

해결: 입자마다 짝/홀 라벨(p_field 0|1)을 두고, 라벨별로 격자를 따로 쓴다(grid, grid1).
두 격자가 함께 질량을 가진 칸에서는 "서로 파고드는 상대속도만" 없애는 접촉을 건다(떨어지는 건 자유).
접촉 법선 = 그 칸에서 본 라벨1 입자 질량중심 - 라벨0 입자 질량중심. 마찰 계수 mf_mu(기본 0).
라벨을 한 번도 안 바꾸면(전부 0) 원래 Genesis 와 똑같이 동작한다.

사용: python patches/make_genesis_multifield.py <site-packages/genesis>  (원본 파일을 직접 고친다)
되돌리기: pip install --force-reinstall --no-deps genesis-world==1.4.3
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
MARK = "# [cutsim-multifield]"


def patch(path, pairs, mark=MARK):
    s = path.read_text()
    if mark in s:
        print(f"이미 패치됨({mark}): {path}")
        return
    for old, new in pairs:
        n = s.count(old)
        assert n == 1, f"{path.name}: 기준 문자열이 {n}번 나옴:\n{old[:200]}"
        s = s.replace(old, new)
    path.write_text(s)
    print(f"패치 완료({mark}): {path}")


# ------------------------------------------------------------------ mpm_solver.py
solver_pairs = [
    # 1) 두 번째 격자, 접촉 법선용 오프셋, 입자 라벨, 마찰 계수
    (
        """        self.grid = grid_cell_state.field(
            shape=(self._sim.substeps_local, *self._grid_res, self._B), needs_grad=True, layout=qd.Layout.SOA
        )
""",
        """        self.grid = grid_cell_state.field(
            shape=(self._sim.substeps_local, *self._grid_res, self._B), needs_grad=True, layout=qd.Layout.SOA
        )
        # [cutsim-multifield] 라벨 1 입자용 두 번째 격자, 라벨별 질량가중 (입자-격자점) 오프셋, 입자 라벨, 마찰
        self.grid1 = grid_cell_state.field(
            shape=(self._sim.substeps_local, *self._grid_res, self._B), needs_grad=False, layout=qd.Layout.SOA
        )
        self.grid_off0 = qd.Vector.field(3, gs.qd_float, shape=(self._sim.substeps_local, *self._grid_res, self._B))
        self.grid_off1 = qd.Vector.field(3, gs.qd_float, shape=(self._sim.substeps_local, *self._grid_res, self._B))
        self.p_field = qd.field(gs.qd_int, shape=(max(self._n_particles, 1), self._B))
        self.mf_mu = qd.field(gs.qd_float, shape=())
""",
    ),
    # 2) P2G: 라벨에 따라 다른 격자로
    (
        """                    if sep_geom_idx == -1:
                        cell_ijk = base - self._grid_offset + offset
                        self.grid[f, cell_ijk, i_b].vel_in += weight * (
                            self.particles_info[i_p].mass * self.particles[f, i_p, i_b].vel + affine @ dpos
                        )
                        mass_contrib = weight * self.particles_info[i_p].mass
                        prev_mass = qd.atomic_add(self.grid[f, cell_ijk, i_b].mass, mass_contrib)
""",
        """                    if sep_geom_idx == -1:
                        cell_ijk = base - self._grid_offset + offset
                        mom = weight * (self.particles_info[i_p].mass * self.particles[f, i_p, i_b].vel + affine @ dpos)
                        mass_contrib = weight * self.particles_info[i_p].mass
                        prev_mass = gs.qd_float(0.0)
                        # [cutsim-multifield] 라벨별 격자. dpos = 격자점 - 입자 이므로 오프셋에는 -dpos 를 쌓는다
                        if self.p_field[i_p, i_b] == 1:
                            self.grid1[f, cell_ijk, i_b].vel_in += mom
                            prev_mass = qd.atomic_add(self.grid1[f, cell_ijk, i_b].mass, mass_contrib)
                            self.grid_off1[f, cell_ijk, i_b] += -mass_contrib * dpos
                        else:
                            self.grid[f, cell_ijk, i_b].vel_in += mom
                            prev_mass = qd.atomic_add(self.grid[f, cell_ijk, i_b].mass, mass_contrib)
                            self.grid_off0[f, cell_ijk, i_b] += -mass_contrib * dpos
""",
    ),
    # 3) G2P: 자기 라벨의 격자에서 속도를 읽는다
    (
        """                    grid_vel = self.grid[f, base - self._grid_offset + offset, i_b].vel_out
""",
        """                    grid_vel = self.grid[f, base - self._grid_offset + offset, i_b].vel_out
                    if self.p_field[i_p, i_b] == 1:  # [cutsim-multifield]
                        grid_vel = self.grid1[f, base - self._grid_offset + offset, i_b].vel_out
""",
    ),
    # 4) 희소 초기화에 두 번째 격자·오프셋 포함
    (
        """                self.grid[f, i, j, k, i_b].vel_out = qd.Vector.zero(gs.qd_float, 3)
                if i_b == 0:
""",
        """                self.grid[f, i, j, k, i_b].vel_out = qd.Vector.zero(gs.qd_float, 3)
                # [cutsim-multifield]
                self.grid1[f, i, j, k, i_b].mass = gs.qd_float(0.0)
                self.grid1[f, i, j, k, i_b].vel_in = qd.Vector.zero(gs.qd_float, 3)
                self.grid1[f, i, j, k, i_b].vel_out = qd.Vector.zero(gs.qd_float, 3)
                self.grid_off0[f, i, j, k, i_b] = qd.Vector.zero(gs.qd_float, 3)
                self.grid_off1[f, i, j, k, i_b] = qd.Vector.zero(gs.qd_float, 3)
                if i_b == 0:
""",
    ),
    # 5) G2P 직전에 두 격자 사이 접촉
    (
        """    def substep_post_coupling(self, f):
        self.g2p(
""",
        """    def substep_post_coupling(self, f):
        if not self._sim.requires_grad:  # [cutsim-multifield]
            self.multifield_contact(f)
        self.g2p(
""",
    ),
    # 6) 접촉 커널과 라벨 설정 API
    (
        """    @qd.kernel
    def _is_state_valid(self, f: qd.i32) -> qd.i32:
""",
        """    # [cutsim-multifield] ------------------------------------------------------------
    @qd.kernel
    def multifield_contact(self, f: qd.i32):
        for ii, jj, kk, i_b in qd.ndrange(*self._grid_res, self._B):
            I = (ii, jj, kk)
            m0 = self.grid[f, I, i_b].mass
            m1 = self.grid1[f, I, i_b].mass
            if m0 > gs.EPS and m1 > gs.EPS:
                n = self.grid_off1[f, I, i_b] / m1 - self.grid_off0[f, I, i_b] / m0  # 라벨0 → 라벨1 방향
                n_norm = n.norm()
                if n_norm > gs.EPS:
                    n = n / n_norm
                    v0 = self.grid[f, I, i_b].vel_out
                    v1 = self.grid1[f, I, i_b].vel_out
                    if (v0 - v1).dot(n) > 0:  # 서로 다가가는 중일 때만 접촉
                        vcm = (m0 * v0 + m1 * v1) / (m0 + m1)
                        d0 = (v0 - vcm).dot(n)
                        d1 = (v1 - vcm).dot(n)
                        v0 = v0 - d0 * n
                        v1 = v1 - d1 * n
                        mu = self.mf_mu[None]
                        if mu > 0:
                            t0 = v0 - vcm
                            t0 = t0 - t0.dot(n) * n
                            t1 = v1 - vcm
                            t1 = t1 - t1.dot(n) * n
                            v0 = v0 - t0 * qd.min(1.0, mu * qd.abs(d0) / t0.norm(gs.EPS))
                            v1 = v1 - t1 * qd.min(1.0, mu * qd.abs(d1) / t1.norm(gs.EPS))
                        self.grid[f, I, i_b].vel_out = v0
                        self.grid1[f, I, i_b].vel_out = v1

    @qd.kernel
    def _kernel_set_particles_field(self, start: qd.i32, n: qd.i32, fld: qd.types.ndarray()):
        for i_p, i_b in qd.ndrange(n, self._B):
            self.p_field[start + i_p, i_b] = fld[i_p]

    def set_particles_field(self, entity, fld):
        \"\"\"entity 입자의 짝/홀 라벨(0|1) 설정. fld: int32 (entity.n_particles,)\"\"\"
        import numpy as _np

        fld = _np.ascontiguousarray(fld, dtype=_np.int32)
        assert fld.shape == (entity.n_particles,)
        self._kernel_set_particles_field(entity._particle_start, entity.n_particles, fld)

    def set_multifield_friction(self, mu):
        self.mf_mu[None] = float(mu)

    @qd.kernel
    def _is_state_valid(self, f: qd.i32) -> qd.i32:
""",
    ),
]

# ------------------------------------------------------------------ legacy_coupler.py
coupler_pairs = [
    (
        """    def preprocess(self, f):
""",
        """    # [cutsim-multifield] 라벨 1 격자의 격자 연산: 운동량→속도, 중력, 강체 결합(칼·도마), 경계
    @qd.kernel
    def mpm_grid_op_field1(
        self,
        f: qd.i32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_info: array_class.RigidInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
    ):
        for ii, jj, kk, i_b in qd.ndrange(*self.mpm_solver.grid_res, self.mpm_solver._B):
            I = (ii, jj, kk)
            if self.mpm_solver.grid1[f, I, i_b].mass > gs.EPS:
                vel_mpm = (1 / self.mpm_solver.grid1[f, I, i_b].mass) * self.mpm_solver.grid1[f, I, i_b].vel_in
                vel_mpm += self.mpm_solver.substep_dt * self.mpm_solver._gravity[i_b]
                pos = (I + self.mpm_solver.grid_offset) * self.mpm_solver.dx
                mass_mpm = self.mpm_solver.grid1[f, I, i_b].mass / self.mpm_solver._particle_volume_scale
                if qd.static(self._rigid_mpm):
                    vel_mpm = self._func_collide_with_rigid(
                        f,
                        pos,
                        vel_mpm,
                        mass_mpm,
                        i_b,
                        geoms_state=geoms_state,
                        geoms_info=geoms_info,
                        links_state=links_state,
                        rigid_info=rigid_info,
                        sdf_info=sdf_info,
                        collider_static_config=collider_static_config,
                    )
                _, self.mpm_solver.grid1[f, I, i_b].vel_out = self.mpm_solver.boundary.impose_pos_vel(pos, vel_mpm)

    def preprocess(self, f):
""",
    ),
    (
        """                collider_static_config=self.rigid_solver.collider.collider_config,
            )

        # SPH <-> Rigid
""",
        """                collider_static_config=self.rigid_solver.collider.collider_config,
            )
            # [cutsim-multifield]
            self.mpm_grid_op_field1(
                f,
                geoms_state=self.rigid_solver.dyn_state.geoms,
                geoms_info=self.rigid_solver.dyn_info.geoms,
                links_state=self.rigid_solver.dyn_state.links,
                rigid_info=self.rigid_solver.rigid_info,
                sdf_info=self.rigid_solver.collider._sdf._sdf_info,
                collider_static_config=self.rigid_solver.collider.collider_config,
            )

        # SPH <-> Rigid
""",
    ),
]

# ------------------------------------------------------------------ 2단계: 칼 안쪽 격자점 분리
# 칼날이 격자 칸보다 얇으면 칼 정중앙에 격자점이 놓일 때만 그 점의 법선이 애매해져, 한쪽 재료만 칼과 같은 편으로
# 처리된다(한쪽 조각만 튀어 나가고 절단력이 칼 위치에 따라 0.1N~10N 으로 바뀜). 얇은 강체(평면 제외) 안쪽에 있는
# 격자점은 어느 쪽 입자와도 떼어 놓아, 칼-재료 상호작용을 입자 단위 충돌(CPIC 의 분리 경로)로만 하게 한다.
# solver.set_cpic_inside_sep(1) 로 켠다(기본 0 = 원래 동작).
MARK2 = "# [cutsim-cpic-inside]"
solver_pairs2 = [
    (
        """        self.mf_mu = qd.field(gs.qd_float, shape=())
""",
        """        self.mf_mu = qd.field(gs.qd_float, shape=())
        self.cpic_inside_sep = qd.field(gs.qd_int, shape=())  # [cutsim-cpic-inside]
""",
    ),
    (
        """                                if sdf_normal_particle.dot(sdf_normal_cell) < 0:  # separated by geom i_g
                                    sep_geom_idx = i_g
                                    break
""",
        """                                if sdf_normal_particle.dot(sdf_normal_cell) < 0:  # separated by geom i_g
                                    sep_geom_idx = i_g
                                    break
                                # [cutsim-cpic-inside] 얇은 강체 안쪽 격자점은 양쪽 입자 모두와 분리
                                if self.cpic_inside_sep[None] == 1 and geoms_info.type[i_g] != gs.GEOM_TYPE.PLANE:
                                    if sdf.sdf_func_world(i_g, i_b, cell_pos, geoms_state, geoms_info, sdf_info) < 0:
                                        sep_geom_idx = i_g
                                        break
""",
    ),
    (
        """    def set_multifield_friction(self, mu):
        self.mf_mu[None] = float(mu)
""",
        """    def set_multifield_friction(self, mu):
        self.mf_mu[None] = float(mu)

    def set_cpic_inside_sep(self, flag):  # [cutsim-cpic-inside]
        self.cpic_inside_sep[None] = int(bool(flag))
""",
    ),
]

# ------------------------------------------------------------------ 3단계: 렌더 꼭짓점 지지 입자 찾기를 KD-tree 로
# 원래 코드는 (꼭짓점 수 × 입자 수 × 3) float64 거리 행렬을 통째로 만들어, 꼭짓점이 수만 개인 메쉬(복원·마칭큐브
# 메쉬)와 입자 수만 개를 쓰면 수십 GB 를 잡고 프로세스가 죽는다(사과 격자 384: 50GB). 같은 k-최근접을 KD-tree 로 구하고
# 정렬 규칙(거리, 같으면 번호)은 그대로 둔다.
MARK3 = "# [cutsim-kdtree]"
entity_pairs3 = [
    (
        """        dist2 = np.sum(np.square(self._vverts[:, None, :] - self._particles[None, :, :]), axis=2)
        support_idxs = np.argpartition(dist2, self.solver._n_vvert_supports - 1, axis=1)[
            :, : self.solver._n_vvert_supports
        ]
        row_indices = np.arange(dist2.shape[0])[:, None]
        sorted_order = np.lexsort((support_idxs, dist2[row_indices, support_idxs]))
        support_idxs = support_idxs[row_indices, sorted_order].astype(gs.np_int)
""",
        """        # [cutsim-kdtree] 꼭짓점×입자 전체 거리 행렬 대신 KD-tree k-최근접(메모리 O(꼭짓점 수))
        from scipy.spatial import cKDTree

        k = self.solver._n_vvert_supports
        n_vv = len(self._vverts)
        dist, support_idxs = cKDTree(self._particles).query(self._vverts, k=k)
        dist = np.asarray(dist, dtype=np.float64).reshape(n_vv, k)
        support_idxs = np.asarray(support_idxs).reshape(n_vv, k)
        row_indices = np.arange(n_vv)[:, None]
        sorted_order = np.lexsort((support_idxs, dist))
        support_idxs = support_idxs[row_indices, sorted_order].astype(gs.np_int)
""",
    ),
]

# ------------------------------------------------------------------ 4단계: 조각 사이 접촉은 꽉 찬 격자점에서만
# 입자는 격자점 1.5칸(256 격자에서 5.9mm)까지 질량을 나눠 주므로, 라벨이 다른 두 조각은 11.7mm 떨어져 있어도 같은
# 격자점에 질량을 갖고 "다가가면" 접촉이 걸린다. 칼에 밀려 다가오는 조각이 7mm 떨어진 이웃 조각을 밀어 넘어뜨렸다.
# 두 라벨 질량을 합친 채움 비율 (m0 + m1) / (rho dx^3) 이 mf_fill 보다 작으면(사이에 빈틈) 접촉하지 않는다.
# 빈틈이 없으면 비율이 1 에 가깝고, 가운데 격자점 기준 빈틈 1mm 면 0.8, 4mm 면 0.33 이다.
# solver.set_multifield_fill(0.75, rho) 로 켠다(기본 0 = 원래 동작).
MARK4 = "# [cutsim-mf-fill]"
solver_pairs4 = [
    (
        """        self.cpic_inside_sep = qd.field(gs.qd_int, shape=())  # [cutsim-cpic-inside]
""",
        """        self.cpic_inside_sep = qd.field(gs.qd_int, shape=())  # [cutsim-cpic-inside]
        self.mf_fill = qd.field(gs.qd_float, shape=())  # [cutsim-mf-fill] 접촉에 필요한 채움 비율(0 = 끔)
        self.mf_inv_mref = qd.field(gs.qd_float, shape=())  # 1 / 꽉 찬 격자점 질량
""",
    ),
    (
        """            if m0 > gs.EPS and m1 > gs.EPS:
                n = self.grid_off1[f, I, i_b] / m1 - self.grid_off0[f, I, i_b] / m0  # 라벨0 → 라벨1 방향
""",
        """            filled = self.mf_fill[None] <= 0 or (m0 + m1) * self.mf_inv_mref[None] >= self.mf_fill[None]
            if m0 > gs.EPS and m1 > gs.EPS and filled:  # [cutsim-mf-fill] 빈틈이 있는 격자점은 접촉 아님
                n = self.grid_off1[f, I, i_b] / m1 - self.grid_off0[f, I, i_b] / m0  # 라벨0 → 라벨1 방향
""",
    ),
    (
        """    def set_cpic_inside_sep(self, flag):  # [cutsim-cpic-inside]
        self.cpic_inside_sep[None] = int(bool(flag))
""",
        """    def set_cpic_inside_sep(self, flag):  # [cutsim-cpic-inside]
        self.cpic_inside_sep[None] = int(bool(flag))

    def set_multifield_fill(self, fill, rho):  # [cutsim-mf-fill]
        \"\"\"조각 사이 접촉에 필요한 격자점 채움 비율(0 = 원래 동작). rho: 재료 밀도(꽉 찬 격자점 질량 계산용)\"\"\"
        self.mf_fill[None] = float(fill)
        self.mf_inv_mref[None] = 1.0 / (float(rho) * self._dx**3 * self._particle_volume_scale)
""",
    ),
]

# ------------------------------------------------------------------ 5단계: 입자별 감쇠(서브스텝마다, 속도와 APIC C 함께)
# 파이썬에서 스텝(=124 서브스텝)마다 입자 속도만 줄이면 서브스텝 사이에는 감쇠가 없고, 회전은 C 에 남아 거의 안 줄어든다
# (조각 분리 감쇠가 끝나자 서 있던 조각이 동전처럼 굴러갔다). G2P 에서 입자마다 새 속도·C 에 1/(1 + c dt) 를 곱한다.
# solver.set_particles_damping(entity, c) 로 준다(기본 0 = 원래 동작).
MARK5 = "# [cutsim-pdamp]"
solver_pairs5 = [
    (
        """        self.mf_inv_mref = qd.field(gs.qd_float, shape=())  # 1 / 꽉 찬 격자점 질량
""",
        """        self.mf_inv_mref = qd.field(gs.qd_float, shape=())  # 1 / 꽉 찬 격자점 질량
        self.p_damp = qd.field(gs.qd_float, shape=(max(self._n_particles, 1), self._B))  # [cutsim-pdamp] 1/s
""",
    ),
    (
        """                # compute actual new_pos with new_vel
""",
        """                if self.p_damp[i_p, i_b] > 0:  # [cutsim-pdamp] 입자별 감쇠(속도와 회전 C 함께)
                    s_damp = 1.0 / (1.0 + self.p_damp[i_p, i_b] * self.substep_dt)
                    new_vel = new_vel * s_damp
                    new_C = new_C * s_damp

                # compute actual new_pos with new_vel
""",
    ),
    (
        """        self.mf_inv_mref[None] = 1.0 / (float(rho) * self._dx**3 * self._particle_volume_scale)
""",
        """        self.mf_inv_mref[None] = 1.0 / (float(rho) * self._dx**3 * self._particle_volume_scale)

    @qd.kernel
    def _kernel_set_particles_damping(self, start: qd.i32, n: qd.i32, c: qd.types.ndarray()):  # [cutsim-pdamp]
        for i_p, i_b in qd.ndrange(n, self._B):
            self.p_damp[start + i_p, i_b] = c[i_p]

    def set_particles_damping(self, entity, c):
        \"\"\"entity 입자별 감쇠율(1/s, float32 (entity.n_particles,)). 0 이면 원래 동작\"\"\"
        import numpy as _np

        c = _np.ascontiguousarray(c, dtype=_np.float32)
        assert c.shape == (entity.n_particles,)
        self._kernel_set_particles_damping(entity._particle_start, entity.n_particles, c)
""",
    ),
]

patch(root / "engine/solvers/mpm_solver.py", solver_pairs)
patch(root / "engine/couplers/legacy_coupler.py", coupler_pairs)
patch(root / "engine/solvers/mpm_solver.py", solver_pairs2, MARK2)
patch(root / "engine/entities/particle_entity.py", entity_pairs3, MARK3)
patch(root / "engine/solvers/mpm_solver.py", solver_pairs4, MARK4)
patch(root / "engine/solvers/mpm_solver.py", solver_pairs5, MARK5)
