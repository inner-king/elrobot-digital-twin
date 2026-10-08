#!/bin/bash
# 합성 시연을 물리 보정(손 그립·도마 접촉 높이·분리 감쇠)을 켜고 다시 재생한 뒤, 입자 대신 겉면 메쉬로 그린 영상을 만든다.
#   bash scripts/run_realistic.sh                 # 오이
#   bash scripts/run_realistic.sh apple_quarters  # 다른 시연(data/demos/<id>)
# 결과: reports/08_realistic/TR_<id>/{mosaic.mp4(입자), mosaic_render.mp4(겉면), render/, frames.npz, metrics.json}
#
# 보정 네 가지(README 의 01_cut_primitive.py 옵션 참고):
#   --grip                              손 상자 아래 재료를 입자 구속으로 붙잡는다(누르는 상자는 칼질마다 몸통이 밀림)
#   --set board.collision_offset_dx=1   도마 충돌 면을 격자 한 칸 내린다(떨어진 조각이 한 칸 위에 떠서 멈추던 것)
#   --mf_fill 0.75                      조각 사이 접촉을 꽉 찬 격자점에서만(7mm 떨어진 조각을 밀어 넘어뜨리던 것, 패치 4단계)
#   --sep_damp 300 --sep_damp_r 0.015   라벨 순간부터 칼이 빠져나갈 때까지, 칼 가까이 걸친 조각과 거기 맞닿은
#                                       조각만 통째로 감쇠(쌓인 탄성이 풀리며 조각이 튀던 것. 떨어져 넘어지는 조각은 그대로)
PY=${PY:-python}
cd "$(dirname "$0")/.."
mkdir -p reports/08_realistic
fix="--grip --set board.collision_offset_dx=1.0 --mf_fill 0.75 --sep_damp 300 --sep_damp_r 0.015 --save_frames 25"
common="--out_root reports/08_realistic --cut_force --no_baseline $fix"
declare -A EXTRA=(
  [potato_press_saw]="--z_force_budget 12"
  [apple_quarters]="--set knife.length=0.13 --domain_pad 0.015"
  [cucumber_slices]="--domain_pad 0.02"
)
ids=("$@")
[ ${#ids[@]} -eq 0 ] && ids=(cucumber_slices)
for id in "${ids[@]}"; do
  echo "== $id $(date +%T)"
  log=reports/08_realistic/$id.log
  $PY scripts/03_import_recon.py data/demos/$id --cut --cut_args "$common --tag $id ${EXTRA[$id]}" > $log 2>&1
  echo "   sim EXIT=$? $(date +%T)"
  title="$(sed -n 's/^task: //p' data/demos/$id/meta.yaml)"
  $PY scripts/make_mosaic.py reports/08_realistic/TR_$id --title "$title" >> $log 2>&1
  $PY scripts/render_surface.py reports/08_realistic/TR_$id >> $log 2>&1
  echo "   render EXIT=$? $(date +%T)"
  $PY scripts/make_mosaic.py reports/08_realistic/TR_$id --src render --title "$title" >> $log 2>&1
done
