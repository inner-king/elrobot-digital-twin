# ElRobot Digital Twin

[NormaCore ElRobot](https://github.com/norma-core/norma-core/tree/main/hardware/elrobot)(7+1 DoF, ST3215 서보) 팔을 위한 웹 콘솔과, iPhone LiDAR를 이용한 디지털 트윈 작업 공간입니다.

- **웹 콘솔:** URDF 3D 모델, 모터별 실시간 상태, 슬라이더 조작, 캘리브레이션 파일 관리, 비상정지
- **iPhone LiDAR 스트림:** USB로 RGB·깊이·카메라 위치(ARKit)를 받아 3D 화면에 실시간 점군으로 표시

## 구성 (입력 → 출력)

| 모듈 | 입력 | 출력 |
|---|---|---|
| `server.py` (서보 버스) | ST3215 버스, 1 Mbps (Waveshare 드라이버 보드) | 모터 8개 상태 ≈25 Hz (위치 0–4095, 속도, 부하, 전류 mA, 전압, 온도, 상태 비트), WebSocket `/ws` JSON |
| `server.py` (명령) | 브라우저 명령 (토크, 목표 위치, 설정, 캘리브레이션) | 서보 RAM/EEPROM 쓰기 |
| `camera/stream.py` | iPhone (Record3DStream 앱), USB ≈60 fps: RGB 1920×1440 JPEG, 깊이 256×192 float32 [m], 신뢰도 {0,1,2}, 내부 파라미터, 카메라→세계 4×4 | ≈10 Hz 점군 (N×3 float32 [m], ARKit 세계 좌표·y 위쪽) + 색, 카메라 위치, AprilTag 36h11 위치 `T_world_tag`, WebSocket `/ws_cam` 바이너리 |
| `static/index.html` | `/ws`, `/ws_cam`, ElRobot URDF | 3D 화면 (로봇 + 점군 + iPhone 위치), 패널 (모터·캘리브레이션·시스템·로그·조작·카메라) |

관절 각도 매핑은 norma-core station-viewer와 같습니다: `p = (pos − min) / (max − min)`, `angle = lower + p · (upper − lower)` (URDF 관절 한계).

## 준비

```
8_DoF_arm/
├── norma-core/        # git clone https://github.com/norma-core/norma-core  (URDF·메시를 여기서 읽음)
└── arm_viewer/        # 이 저장소
```

필요한 것: Python 3.12, [uv](https://docs.astral.sh/uv/), macOS (iPhone USB 포워딩은 `pymobiledevice3` 사용)

### iPhone 스트림 SDK (직접 받기)

카메라 기능은 [Record3DStream (PathOn)](https://apps.apple.com/us/app/record3dstream-pathon/id6761314229) 앱과 그 Python SDK를 씁니다. SDK는 **개인·비상업 용도만 허용되고 재배포가 금지된 라이선스**라 이 저장소에 포함하지 않습니다. 라이선스를 확인한 뒤 직접 받아 주세요.

```bash
git clone --depth 1 https://github.com/PathOn-AI/pathon_opensource /tmp/pathon
mkdir -p camera/r3ds_sdk && cp -R /tmp/pathon/software/iphone_sensor_suite/Record3DStream/sdk/sdk camera/r3ds_sdk/ && cp /tmp/pathon/LICENSE camera/r3ds_sdk/
```

SDK가 없으면 카메라 패널만 비활성화되고 로봇 콘솔은 그대로 동작합니다.

## 실행

```bash
uv run --with pyserial --with fastapi --with "uvicorn[standard]" --with numpy --with opencv-python --with pymobiledevice3 python server.py
```

브라우저에서 `http://localhost:8765`. 시리얼 포트는 `/dev/cu.usbmodem*`을 자동으로 찾고, `ARM_PORT`로 지정할 수 있습니다.

## 안전 장치

- 토크를 켜기 직전에 목표 위치를 현재 위치로 맞춥니다 (목표 레지스터에 남은 값으로 튀는 것 방지).
- 위치 명령은 캘리브레이션 범위 안으로 제한되고, 캘리브레이션 전에는 막혀 있습니다.
- 토크 상한: 1–7번 60%, 8번(그리퍼) 25%. 상단 E-STOP 또는 `Esc`로 전체 토크 OFF.

## 캘리브레이션

캘리브레이션 탭에서 새 파일을 만들고, 토크가 꺼진 상태에서 관절을 손으로 양 끝까지 움직입니다.

- 끝 위치는 **0.1초 동안 ±10 step(≈0.9°) 안에 정지**했을 때만 기록됩니다. 세게 눌러 순간적으로 튄 값이나 지나가는 값은 무시됩니다.
- 저장하면 범위 중앙이 2048이 되도록 offset을 EEPROM에 씁니다. 범위가 0/4095 경계를 넘어도 안전합니다.
- 파일은 `calibrations/<이름>.json`이고, 다른 파일을 적용하면 그 파일의 offset이 EEPROM에 다시 써집니다.

## 하드웨어 메모

- **서보 ID 설정:** 새 ST3215는 모두 ID 1입니다. 이미 조립된 상태라면 버스에 하나씩 추가하면서 바꾸고, 겹치지 않게 임시 ID를 씁니다.
- **다른 기어비/용도로 쓰던 서보:** 그리퍼에 쓴 7.4V 서보가 `phase=76`이라 구동 방향이 센서와 반대였습니다 (목표를 주면 반대쪽 끝으로 밀어붙임). 나머지와 같은 `phase=12`로 쓰고 **전원을 껐다 켜야** 적용됩니다.
- iPhone 앱이 NeRFCapture라면 CycloneDDS **0.10.x**로 맞춰야 합니다. 11.x의 탐색 메시지를 받으면 앱이 죽습니다 (`camera/nerfcapture_probe.py`). 단, App Store 버전은 Send 버튼을 누를 때 한 장씩만 보냅니다.

## 다음 단계

1. AprilTag로 로봇 베이스 ↔ ARKit 세계 좌표 정렬 → 점군을 로봇 좌표계로 표시
2. TSDF 메시 + 중력·맨해튼 정렬
3. 물체: 마스크 → TRELLIS 메시 → LiDAR 부분 점군으로 실제 크기·위치 보정 (Any6D 방식) → FoundationPose 6DoF 추적 (GPU 머신)
4. 가상 목표 자세 → IK → 실제 잡기
