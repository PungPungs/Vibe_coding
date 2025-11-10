# SEG-Y 2D Viewer with First Break Picking

SEG-Y 파일을 시각화하고 first break picking을 수행하는 OpenGL 기반 2D 뷰어입니다.

## 주요 기능

1. **확대/축소 기능**: 마우스 휠을 사용하여 데이터를 확대/축소할 수 있습니다.
2. **SEG-Y 뷰어**: OpenGL을 사용한 고성능 2D 지진파 데이터 시각화
   - mmap을 이용한 빠른 파일 로딩
   - 병렬 처리를 통한 대용량 파일 지원
3. **수동 피킹 기능**: First break를 수동으로 피킹 (1트레이스당 1점)
4. **자동 피킹 알고리즘**:
   - **STA/LTA** (Short-Term Average / Long-Term Average)
   - **Energy Ratio**
   - **AIC** (Akaike Information Criterion)
5. **자동 보간**: 피킹된 포인트 간 자동 선형 보간
6. **다양한 컬러맵**: Seismic, Grayscale 컬러맵 지원
7. **CSV 내보내기**: 피킹 결과를 CSV 형식으로 저장/로드

## 시스템 요구사항

- Python 3.7 이상
- OpenGL 2.1 이상 지원 GPU
- 최소 2GB RAM

## 설치

### 1. 저장소 클론 (또는 다운로드)

```bash
cd segy_2d_viewer
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

필요한 라이브러리:
- PyQt5: GUI 프레임워크
- PyOpenGL: OpenGL 바인딩
- numpy: 수치 연산 및 SEG-Y 파일 파싱

## 실행

```bash
cd segy_2d_viewer/src
python main.py
```

또는 루트 디렉토리에서:

```bash
cd segy_2d_viewer
python -m src.main
```

## 빠른 시작

### 1. 테스트 데이터 생성

```bash
python create_test_segy.py
```

이 명령은 `resources/test_data.sgy` 파일을 생성합니다.

### 2. 프로그램 실행

```bash
python src/main.py
```

### 3. 파일 열기

- 툴바에서 "Open SEG-Y" 클릭
- `resources/test_data.sgy` 선택

### 4. 피킹 시작

**수동 피킹:**
- 왼쪽 마우스 버튼으로 first break 지점 클릭 (1트레이스당 1점)
- 여러 트레이스에 피킹하면 자동으로 보간됩니다

**자동 피킹:**
- 알고리즘 선택: STA/LTA, Energy Ratio, AIC
- "Auto Pick" 버튼 클릭
- 결과 확인 및 수동 수정 가능

**저장:**
- "Save Picks"로 결과 저장

## 사용법

### 마우스 조작

- **왼쪽 클릭**: 피킹 추가/수정
- **마우스 휠**: 확대/축소
- **중간/오른쪽 버튼 드래그**: 패닝 (화면 이동)

### 메뉴 및 컨트롤

- **Open SEG-Y**: SEG-Y 파일 열기
- **Reset View**: 뷰 초기화
- **Save Picks**: 피킹 데이터를 CSV로 저장
- **Load Picks**: 저장된 피킹 데이터 로드
- **Clear Picks**: 모든 피킹 제거
- **Auto Pick**: 자동 피킹 실행
- **Enable Picking**: 수동 피킹 모드 on/off
- **Show Picks**: 피킹 표시 on/off
- **Auto Pick Algorithm**: 자동 피킹 알고리즘 선택 (STA/LTA, Energy Ratio, AIC)
- **Colormap**: 컬러맵 선택 (seismic/grayscale)

### 상태바

- 왼쪽: 현재 상태 메시지
- 오른쪽: 현재 마우스 위치 (Trace, Sample)

## 프로젝트 구조

```
segy_2d_viewer/
├── src/
│   ├── __init__.py          # 패키지 초기화
│   ├── main.py              # 메인 애플리케이션
│   ├── segy_reader.py       # SEG-Y 파일 파싱
│   ├── gl_widget.py         # OpenGL 뷰어 위젯
│   ├── picking_manager.py   # 피킹 관리
│   └── utils.py             # 유틸리티 함수
├── resources/               # 리소스 파일 (테스트 데이터)
├── docs/                    # 문서
│   └── USER_GUIDE.md        # 사용자 가이드
├── create_test_segy.py      # 테스트 데이터 생성 스크립트
├── requirements.txt         # 의존성
└── README.md                # 이 파일
```

## 기술 스택

- **GUI Framework**: PyQt5
- **Graphics**: OpenGL 2.1
- **SEG-Y Parsing**: 직접 구현 (mmap + numpy)
- **Numerical Computing**: NumPy
- **Parallel Processing**: multiprocessing
- **Auto Picking**: STA/LTA, Energy Ratio, AIC 알고리즘

## 개발자 정보

### 아키텍처

- **MVC 패턴**: Model (SegyReader), View (SegyGLWidget), Controller (MainWindow)
- **시그널/슬롯**: PyQt5 시그널/슬롯 메커니즘 활용
- **OpenGL 렌더링**: 텍스처 매핑을 통한 효율적인 렌더링

### 주요 클래스

- `SegyReader`: SEG-Y 파일 읽기 (mmap + 병렬 처리)
- `SegyGLWidget`: OpenGL 기반 렌더링 및 사용자 상호작용
- `PickingManager`: 피킹 데이터 관리 및 보간
- `AutoPicker`: 자동 피킹 알고리즘 (STA/LTA, Energy Ratio, AIC)
- `MainWindow`: 메인 애플리케이션 UI

### 성능 최적화

- **mmap**: 메모리 매핑을 통한 빠른 파일 I/O
- **병렬 처리**: multiprocessing을 사용한 트레이스 로딩 및 자동 피킹
- **OpenGL 텍스처**: 하드웨어 가속 렌더링

## 문제 해결

자세한 문제 해결 가이드는 [docs/USER_GUIDE.md](docs/USER_GUIDE.md)를 참조하세요.

## 라이선스

MIT License

## 기여

버그 리포트, 기능 제안, 풀 리퀘스트를 환영합니다!

## 자동 피킹 알고리즘

### 1. STA/LTA (Short-Term Average / Long-Term Average)
- 가장 널리 사용되는 first break picking 방법
- 단기 평균과 장기 평균의 비율을 계산하여 신호의 시작점 감지
- 파라미터: `sta_window=5`, `lta_window=50`, `threshold=3.0`

### 2. Energy Ratio
- 신호의 누적 에너지 비율을 사용
- 전체 에너지 대비 특정 임계값을 넘는 첫 지점 감지
- 파라미터: `window=20`, `threshold=0.1`

### 3. AIC (Akaike Information Criterion)
- 통계적 방법으로 신호의 특성 변화 지점을 찾음
- AIC가 최소가 되는 지점이 first break
- 파라미터: `window=100`

## 향후 계획

- [x] 자동 피킹 알고리즘 추가
- [ ] 자동 피킹 파라미터 조정 UI
- [ ] 다중 피킹 레이어 지원
- [ ] 3D 시각화 지원
- [ ] 추가 컬러맵 옵션
- [ ] 키보드 단축키 지원
- [ ] 피킹 편집 기능 (삭제, 이동)
