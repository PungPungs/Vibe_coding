# SEG-Y 2D Viewer with First Break Picking

SEG-Y 파일을 시각화하고 first break picking을 수행하는 OpenGL 기반 2D 뷰어입니다.

## 주요 기능

1. **확대/축소 기능**: 마우스 휠을 사용하여 데이터를 확대/축소할 수 있습니다.
2. **SEG-Y 뷰어**: OpenGL을 사용한 고성능 2D 지진파 데이터 시각화
3. **피킹 기능**: First break를 수동으로 피킹하고 저장할 수 있습니다.
4. **자동 보간**: 피킹된 포인트 간 자동 선형 보간
5. **다양한 컬러맵**: Seismic, Grayscale 컬러맵 지원
6. **CSV 내보내기**: 피킹 결과를 CSV 형식으로 저장/로드

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
- segyio: SEG-Y 파일 파싱
- numpy: 수치 연산
- matplotlib: 추가 시각화 (옵션)

## 실행

```bash
python src/main.py
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

- 왼쪽 마우스 버튼으로 first break 지점 클릭
- 여러 트레이스에 피킹하면 자동으로 보간됩니다
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
- **Enable Picking**: 피킹 모드 on/off
- **Show Picks**: 피킹 표시 on/off
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
- **SEG-Y Parsing**: segyio
- **Numerical Computing**: NumPy
- **Data Visualization**: Matplotlib

## 개발자 정보

### 아키텍처

- **MVC 패턴**: Model (SegyReader), View (SegyGLWidget), Controller (MainWindow)
- **시그널/슬롯**: PyQt5 시그널/슬롯 메커니즘 활용
- **OpenGL 렌더링**: 텍스처 매핑을 통한 효율적인 렌더링

### 주요 클래스

- `SegyReader`: SEG-Y 파일 읽기 및 데이터 정규화
- `SegyGLWidget`: OpenGL 기반 렌더링 및 사용자 상호작용
- `PickingManager`: 피킹 데이터 관리 및 보간
- `MainWindow`: 메인 애플리케이션 UI

## 문제 해결

자세한 문제 해결 가이드는 [docs/USER_GUIDE.md](docs/USER_GUIDE.md)를 참조하세요.

## 라이선스

MIT License

## 기여

버그 리포트, 기능 제안, 풀 리퀘스트를 환영합니다!

## 향후 계획

- [ ] 자동 피킹 알고리즘 추가
- [ ] 다중 피킹 레이어 지원
- [ ] 3D 시각화 지원
- [ ] 추가 컬러맵 옵션
- [ ] 키보드 단축키 지원
- [ ] 피킹 편집 기능 (삭제, 이동)
