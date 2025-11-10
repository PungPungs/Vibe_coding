# Quick Start Guide

## 설치 및 실행

```bash
# 1. 의존성 설치
cd radex_style_viewer
pip install -r requirements.txt

# 2. 테스트 데이터 생성
python create_test_segy.py

# 3. 프로그램 실행
cd src
python main.py
```

## 파일 열기

### 테스트 파일 사용
1. "Open SEG-Y" 클릭
2. `radex_style_viewer/test_data.sgy` 선택

### 실제 SEG-Y 파일 사용
1. "Open SEG-Y" 클릭
2. `.sgy` 또는 `.segy` 파일 선택

## 주요 변경사항

### ✅ 모든 Trace 표시
- **제한 제거**: 이제 모든 trace를 한번에 표시합니다
- **스케일 자동 조정**: RMS 기반 자동 스케일링
- **더 나은 시각화**: Wiggle amplitude를 1.5로 증가

### ✅ 향상된 Navigation
- **화살표 키**: 상하좌우 Pan 이동
- **+/- 키**: Zoom in/out (trace window 대신)
- **Home**: 수평 위치 리셋
- **R**: 전체 뷰 리셋

### ✅ Display Modes
모두 정상 작동:
- Wiggle
- Variable Area
- Variable Density
- Wiggle + VA (기본값)
- Wiggle + VD

## 문제 해결

### 데이터가 안 보이는 경우

1. **콘솔 출력 확인**:
```
[Display] Data set: 600 x 100
[Display] Range: [최소값, 최대값]
[Wiggle] Rendering 100 traces, RMS scale: 스케일값
```

2. **Display Mode 변경**:
   - Control Panel에서 다른 모드 시도
   - "Variable Density"는 항상 뭔가 보임

3. **AGC 활성화**:
   - Control Panel에서 "Enable AGC" 체크
   - Window를 100-200으로 조정

4. **Wiggle Amplitude 조정**:
   - Control Panel에서 Amplitude 슬라이더 조정
   - 더 크게 설정 (1.0 - 2.0)

5. **Zoom 조정**:
   - `+` 키로 zoom in
   - 마우스 휠로 zoom in/out

### 색상이 이상한 경우

1. **Colormap 변경** (Variable Density 모드):
   - Seismic (기본)
   - Grayscale
   - Jet

2. **Clip Percentile 조정**:
   - 99% (기본)
   - 95% 또는 98%로 낮추기

## 데이터 분석

### Trace 선택
1. 좌클릭으로 trace 선택
2. Header Viewer에서 상세 정보 확인
   - Trace Header 탭 확인
   - Source/Receiver 좌표 등

### Header 정보
1. "Header Information" 패널 열기
2. 탭 선택:
   - Text Header: 파일 설명
   - Binary Header: 파일 메타데이터
   - Trace Header: 선택한 trace 정보

## 성능 팁

### 대용량 파일 (>1000 traces)
1. **Variable Density 모드** 사용 (더 빠름)
2. **Zoom In** 해서 부분만 보기
3. **AGC 비활성화** (처리 시간 단축)

### 고해상도 표시
1. **Wiggle + VA 모드**
2. **Zoom In** (마우스 휠 또는 +)
3. **Amplitude 조정**

## Export

1. "Export Image" 클릭 (또는 Ctrl+E)
2. 파일 형식 선택:
   - PNG (권장)
   - JPEG
3. 저장 위치 선택

현재 화면 그대로 이미지로 저장됩니다.
