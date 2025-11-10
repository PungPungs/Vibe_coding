# SEG-Y Trace Viewer

간단한 SEG-Y 파일 viewer로 OpenGL 기반의 wiggle trace 시각화를 제공합니다.

## 기능

- **Zoom In/Out**: 마우스 휠로 확대/축소
- **Pan**: 우클릭 드래그로 화면 이동
- **좌우 Navigation**: 키보드 화살표 키로 trace 범위 이동
- **Window 크기 조절**: +/- 키로 한 번에 보여줄 trace 개수 조절
- **mmap 기반**: 큰 파일도 빠르게 로드
- **실시간 렌더링**: OpenGL을 사용한 부드러운 시각화

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
cd src
python main.py
```

## 조작법

### 마우스
- **휠**: 줌 인/아웃
- **우클릭 드래그**: 화면 이동 (pan)

### 키보드
- **←/→**: 이전/다음 trace 그룹으로 이동
- **Home/End**: 처음/끝으로 이동
- **+/-**: Trace window 크기 조절 (한 번에 보여줄 trace 개수)
- **R**: 뷰 리셋

## 구조

```
segy_trace_viewer/
├── src/
│   ├── segy_loader.py   # mmap을 이용한 SEG-Y 로더
│   ├── gl_viewer.py     # OpenGL viewer 위젯
│   └── main.py          # 메인 윈도우
├── requirements.txt
└── README.md
```

## 특징

- **정규화 없음**: 원본 데이터를 그대로 사용하고 자동 스케일링
- **mmap 사용**: 메모리 효율적인 파일 읽기
- **IBM/IEEE float 지원**: SEG-Y 포맷의 다양한 데이터 형식 지원
- **간단한 인터페이스**: 핵심 기능에만 집중

## 요구사항

- Python 3.7+
- numpy
- PyQt5
- PyOpenGL
