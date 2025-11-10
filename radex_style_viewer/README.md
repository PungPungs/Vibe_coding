# RadExPro-Style Professional SEG-Y Viewer

RadExPro Professional을 참고하여 만든 전문적인 SEG-Y 지구물리학 데이터 시각화 도구입니다.

**✨ 최신 업데이트**: segy_viz의 robust한 SEG-Y 로더 통합

## 주요 기능

### 강력한 SEG-Y 로더 (segy_viz 기반)
- **다양한 포맷 지원**:
  - Format 1: IBM floating point
  - Format 2: 4-byte signed integer
  - Format 3: 2-byte signed integer
  - Format 5: IEEE floating point
  - Format 8: 1-byte unsigned integer
- **Robust한 text header 디코딩**: cp500, cp1147, ASCII, Latin-1
- **완벽한 에러 처리**: 파일 손상 감지 및 명확한 에러 메시지
- **통계 정보**: Min, Max, Mean, Std 자동 계산

### 다양한 디스플레이 모드
- **Wiggle**: 전통적인 wiggle trace 표시
- **Variable Area (VA)**: 양수 부분을 채운 wiggle trace
- **Variable Density (VD)**: 컬러맵을 사용한 밀도 표시
- **Wiggle + VA**: Wiggle과 Variable Area 결합 (기본값)
- **Wiggle + VD**: Wiggle과 Variable Density 결합

### Gain Control
- **AGC (Automatic Gain Control)**: 자동 이득 조절
  - RMS 방식
  - Mean 방식
  - 윈도우 크기 조절 (10-500 samples)
- **Clipping**: 백분위수 기반 진폭 제한 (90-100%)
- **RMS 기반 스케일링**: 3-sigma 기반 자동 스케일링

### Colormap 시스템 (6가지)
- Seismic (Blue-White-Red) - 기본값
- Grayscale
- Jet
- Viridis
- Red-White-Blue
- Brown-White-Green

### Header 정보
- **Text Header**: EBCDIC/ASCII 자동 감지 및 표시
- **Binary Header**: 파싱된 메타데이터
  - Sample interval, format, measurement system 등
- **Trace Header**: 선택한 trace의 상세 정보
  - Source/Receiver 좌표
  - Trace sequence 정보

### 인터랙티브 Navigation
- **마우스 휠**: 줌 인/아웃
- **우클릭 드래그**: 화면 이동 (Pan)
- **좌클릭**: Trace 선택 및 헤더 정보 표시
- **화살표 키**: 상하좌우 Pan 이동
- **+/- 키**: 줌 인/아웃
- **Home**: 수평 위치 리셋
- **R 키**: 전체 뷰 리셋

### 전문적인 UI
- 3-패널 레이아웃 (Control Panel, Display, Header Viewer)
- Dockable 윈도우
- 직관적인 컨트롤
- 실시간 피드백
- **모든 trace 표시** (제한 없음)

## 설치

```bash
# 의존성 설치
pip install -r requirements.txt
```

## 실행

```bash
cd src
python main.py
```

## 사용 방법

### 1. 파일 열기
- 툴바에서 "Open SEG-Y" 클릭 또는 `Ctrl+O`
- SEG-Y 파일 선택 (.sgy, .segy)
- 콘솔에서 로딩 정보 확인:
  ```
  [SEG-Y Loader] Successfully loaded: file.sgy
    Traces: 9834
    Samples per trace: 1792
    Sample interval: 16 μs (0.02 ms)
    Sample format: 3
    Data range: [-3.28e+04, 3.28e+04]
    Mean: 7.38e+00, Std: 3.74e+03
  ```

### 2. Display Mode 선택
- Control Panel에서 원하는 display mode 선택
- **Wiggle + VA** (기본값): 가장 일반적인 지구물리학 표시
- **Variable Density**: 빠른 overview

### 3. Gain Control 조절
- **AGC 활성화**: 약한 신호를 강화
  - 윈도우 크기: 100-200 samples (권장)
  - 방식: RMS (권장) 또는 Mean
- **Clip Percentile**: 99% (기본값)
  - 95%로 낮추면 더 많은 디테일

### 4. Wiggle Amplitude 조정
- 슬라이더로 진폭 조절
- 범위: 0.1 - 2.0
- 기본값: 1.5

### 5. Navigation
- **마우스**: 휠로 줌, 우클릭으로 Pan
- **키보드**: 화살표로 이동, +/-로 줌
- **R**: 전체 뷰 리셋

### 6. Trace 분석
- 좌클릭으로 trace 선택
- Header Viewer에서 상세 정보 확인

### 7. 이미지 Export
- 툴바에서 "Export Image" 클릭 또는 `Ctrl+E`
- PNG/JPEG로 저장

## 프로젝트 구조

```
radex_style_viewer/
├── src/
│   ├── main.py              # 메인 윈도우 및 애플리케이션
│   ├── segy_loader.py       # SEG-Y 로더 (segy_viz 기반)
│   ├── gl_display.py        # OpenGL 디스플레이 위젯
│   ├── display_modes.py     # Display mode 정의
│   ├── colormap.py          # Colormap 시스템
│   ├── agc.py              # AGC 및 gain control
│   ├── header_viewer.py    # Header 정보 뷰어
│   └── control_panel.py    # 컨트롤 패널 위젯
├── create_test_segy.py     # 테스트 데이터 생성기
├── requirements.txt
├── README.md
└── QUICK_START.md
```

## 기술 스택

- **Python 3.7+**
- **PyQt5**: GUI 프레임워크
- **OpenGL**: 고성능 그래픽 렌더링
- **NumPy**: 수치 계산
- **SciPy**: AGC 필터링

## 특징

### segy_viz 기반 로더
- **5가지 포맷 지원**: IBM float, IEEE float, 정수형
- **다중 인코딩**: EBCDIC, ASCII 등 자동 감지
- **Robust한 IBM float 변환**: NumPy 벡터화 연산
- **완벽한 에러 처리**

### 실시간 처리
- AGC 실시간 적용
- 인터랙티브한 파라미터 조절
- RMS 기반 자동 스케일링

### OpenGL 렌더링
- 부드러운 줌/팬
- 빠른 렌더링 성능
- Texture 기반 Variable Density
- **무제한 trace 표시**

## 테스트

### 테스트 데이터 생성
```bash
python create_test_segy.py
# Creates: test_data.sgy (100 traces x 600 samples)
```

### 실제 데이터 테스트
실제 SEG-Y 파일로 테스트 완료:
- ✅ Format 3 (2-byte integer): 9,834 traces
- ✅ Format 5 (IEEE float): 100 traces
- ✅ 대용량 파일 (17MB+)

## 문제 해결

### 데이터가 안 보이는 경우
1. **콘솔 출력 확인**: 데이터 범위와 통계 확인
2. **AGC 활성화**: Enable AGC 체크
3. **Wiggle Amplitude 증가**: 슬라이더를 오른쪽으로
4. **Variable Density 모드**: 항상 뭔가 보임
5. **줌 인**: + 키 또는 마우스 휠

### 파일 로딩 실패
- 콘솔에서 에러 메시지 확인
- 지원되는 포맷: 1, 2, 3, 5, 8
- 파일이 손상되지 않았는지 확인

## 성능

### 대용량 파일 (>5000 traces)
- **Variable Density 모드** 사용 (더 빠름)
- **Zoom In** 해서 부분만 보기
- 모든 trace가 한 번에 표시됨 (제한 없음)

### 최적화
- NumPy 벡터화 연산
- OpenGL 하드웨어 가속
- 효율적인 메모리 관리

## RadExPro Professional과의 비교

RadExPro Professional의 주요 기능들을 구현:
- ✅ Multiple display modes
- ✅ AGC (Automatic Gain Control)
- ✅ Colormap system
- ✅ Header viewer
- ✅ Interactive navigation
- ✅ Professional UI layout
- ✅ Robust SEG-Y loader

## 라이선스

MIT License

## 참고

- SEG-Y 로더: segy_viz 프로젝트 기반
- UI 디자인: RadExPro Professional 참고
- 교육 및 연구용 도구
