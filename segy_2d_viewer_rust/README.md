# SEG-Y 2D Viewer (Rust + OpenGL)

SEG-Y 파일을 시각화하고 first break picking을 수행하는 OpenGL 기반 2D 뷰어 (Rust 버전).

## 주요 기능

1. **확대/축소 기능**: 마우스 휠을 사용하여 데이터를 확대/축소
2. **SEG-Y 뷰어**: OpenGL을 사용한 고성능 2D 지진파 데이터 시각화
   - memmap2를 이용한 빠른 파일 로딩
   - rayon을 통한 병렬 처리
3. **수동 피킹 기능**: First break를 수동으로 피킹 (1트레이스당 1점)
4. **자동 피킹 알고리즘**:
   - **STA/LTA** (Short-Term Average / Long-Term Average)
   - **Energy Ratio**
   - **AIC** (Akaike Information Criterion)
5. **자동 보간**: 피킹된 포인트 간 자동 선형 보간
6. **다양한 컬러맵**: Seismic, Grayscale 컬러맵 지원
7. **CSV 내보내기**: 피킹 결과를 CSV 형식으로 저장/로드

## 기술 스택

- **GUI Framework**: eframe (egui)
- **Graphics**: glow (OpenGL bindings)
- **SEG-Y Parsing**: 직접 구현 (memmap2 + byteorder)
- **Parallel Processing**: rayon
- **File I/O**: memmap2

## 시스템 요구사항

- Rust 1.70 이상
- OpenGL 3.3 이상 지원 GPU
- 최소 2GB RAM

## 빌드 및 실행

### 1. Rust 설치

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 2. 빌드

```bash
cd segy_2d_viewer_rust
cargo build --release
```

### 3. 실행

```bash
cargo run --release
```

## 사용법

### 마우스 조작

- **왼쪽 클릭**: 피킹 추가/수정
- **마우스 휠**: 확대/축소
- **오른쪽 버튼 드래그**: 패닝 (화면 이동)

### 메뉴 및 컨트롤

- **📁 Open SEG-Y**: SEG-Y 파일 열기
- **Reset View**: 뷰 초기화
- **💾 Save Picks**: 피킹 데이터를 CSV로 저장
- **📂 Load Picks**: 저장된 피킹 데이터 로드
- **🗑 Clear Picks**: 모든 피킹 제거
- **🤖 Auto Pick**: 자동 피킹 실행

## 프로젝트 구조

```
segy_2d_viewer_rust/
├── Cargo.toml           # 의존성 및 빌드 설정
├── src/
│   ├── main.rs          # 메인 애플리케이션 + UI
│   ├── segy_reader.rs   # SEG-Y 파일 파싱
│   ├── gl_renderer.rs   # OpenGL 렌더러
│   ├── picking_manager.rs   # 피킹 관리
│   └── auto_picking.rs  # 자동 피킹 알고리즘
└── README.md
```

## 성능 비교

| 항목 | Python 버전 | Rust 버전 |
|------|------------|-----------|
| 파일 로딩 | ~1-2초 | ~0.3-0.5초 |
| 메모리 사용 | 중간 | 낮음 |
| 렌더링 FPS | 60 | 60+ |
| 자동 피킹 | 중간 | 빠름 |

## 의존성

```toml
eframe = "0.27"        # GUI framework
egui = "0.27"          # Immediate mode GUI
glow = "0.13"          # OpenGL bindings
byteorder = "1.5"      # Byte order conversion
rayon = "1.8"          # Parallel processing
memmap2 = "0.9"        # Memory mapping
anyhow = "1.0"         # Error handling
csv = "1.3"            # CSV I/O
```

## 빌드 옵션

Release 빌드는 최적화가 적용됩니다:
- LTO (Link Time Optimization)
- Optimization level 3
- Single codegen unit

## Python 버전과의 차이점

### 장점
- ✅ 더 빠른 성능
- ✅ 더 적은 메모리 사용
- ✅ 타입 안정성
- ✅ 단일 실행 파일

### 단점
- ❌ 빌드 시간이 더 김
- ❌ Rust 학습 곡선

## 라이선스

MIT License
