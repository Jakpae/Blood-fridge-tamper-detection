# coldchain_lab: 냉동시설 센서 위변조 탐지·복원 모델링

서로 다른 적혈구 보관 냉장고 8대를 물리식으로 시뮬레이션하고, 온도·압력 보고값 위조를
**전력(제어망과 분리된 독립 채널) 기반 열 모델**과 **압력 기반 가상 온도 센서**로 탐지·복원한다.
모든 데이터는 합성 데이터이며 실제 설비에 연결하지 않는다.

## 빠른 시작 (Windows 11, PowerShell)

```powershell
cd simulation
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

python step1_generate.py   # 냉장고 8대 데이터 생성 (약 1분)
python step2_train.py      # 냉장고별 학습·임계값·신뢰 구간 (약 20분)
python step3_evaluate.py   # 공격 350회 + 누락 50회 평가 (약 15~20분)
python step4_plots.py      # 그림 저장 → results/
```

`run_all.bat`을 더블클릭해도 네 단계가 차례로 실행된다 (가상환경을 먼저 만들어 둘 것).

냉장고 수와 공격 횟수는 config.py의 N_FRIDGES, EPISODES_PER_ATTACK로 바꾼다.
GPU 비교 실험은 gpu_compare.py (PyTorch cu128 필요, 가이드 5장).

## 파일

| 파일 | 역할 |
| --- | --- |
| config.py | 가상 설비 파라미터, 경로, 실험 설정 |
| sim.py | 1차 열 모델 + 냉매 압력 + 전력 시뮬레이터 |
| attacks.py | 공격 7종 + 데이터 누락 주입 |
| models.py | 전력 상태(가우시안 혼합), 열 모델(Huber + 보정층), 가상 온도 센서(그래디언트 부스팅) |
| detect.py | 표준화 → CUSUM 탐지, 조작 시작 추정, 원인 판별 |
| integrity.py | HMAC-SHA256, 시퀀스·시각 검사 |
| step1\~4 | 실행 단계 |
| gpu_compare.py | (선택) GRU 딥러닝 모델과 물리 모델 비교, PyTorch 필요 |
| reference_results/ | 개발 환경에서 미리 돌린 결과 (내 결과와 비교용) |

## 결과 파일 (results/)

- detection_by_config.csv, false_alarms.csv: 구성별 탐지율·지연·오경보
- episode_detail.csv: 에피소드별 경보 시각, 시작 추정, 원인 판별, 남은 시간
- reconstruction.csv, missing_reconstruction.csv: 복원 오차와 95% 구간 포함률
- hmac_test.csv: 전송 변조·재전송 판정
- fig_*.png: 발표용 그림
