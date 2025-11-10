# Performance Optimizations

## Viewport-Based Rendering (최종 버전)

RadExPro 스타일 viewer에 viewport-based 렌더링 방식을 적용했습니다.

## 주요 최적화

### 1. Viewport-Based Rendering (화면 기반 렌더링)
- **문제**: 전체 trace (9,834개)를 한 번에 렌더링하면 매우 느림
- **해결**: 화면에 맞는 수만큼만 동적으로 렌더링 (예: 화면 너비가 1000px이면 ~50개 trace)
- **결과**: 렌더링 속도 20배 이상 향상 + 화면에 딱 맞는 표시

```python
# Before (느림)
for trace_idx in range(num_traces):  # 9,834 traces
    render_trace(trace_idx)

# After (빠르고 자동 조절)
traces_per_screen = widget_width / pixels_per_trace  # 화면 크기에 맞춤
for i in range(traces_per_screen):
    render_trace(view_offset + i)
```

### 2. Smooth Scrolling (부드러운 스크롤)
- **Left/Right 화살표**: 10% 단위로 부드럽게 스크롤 (예: 50개 trace 중 5개씩 이동)
- **Home/End**: 처음/끝으로 빠르게 점프
- **Fractional offset**: 소수점 단위 offset으로 매끄러운 스크롤 효과
- **부드러운 전환**: Texture 업데이트만으로 빠른 전환

### 3. Dynamic Viewport Calculation
- **Window resize 자동 대응**: 창 크기 변경시 자동으로 traces_per_screen 재계산
- **Pixels per trace**: 기본 20px로 설정 (조절 가능)
- **최소 보장**: 최소 10개 trace는 항상 표시

### 4. Texture Caching
- Viewport가 변경되지 않으면 texture 재사용
- `_texture_dirty` 플래그로 불필요한 업데이트 방지

### 5. RMS-based Scaling
- 전체 데이터 대신 현재 viewport만 스케일링 계산
- 3-sigma 방식으로 outlier에 robust

## 성능 비교

### 대용량 파일 (9,834 traces x 1,792 samples)

| 방식 | 렌더링 시간 | 메모리 사용 | Navigation | 화면 적응 |
|------|------------|------------|------------|---------|
| **전체 렌더링** | ~2-3초 | 높음 | 느림 | 없음 |
| **Fixed Slice** | ~0.1초 | 낮음 | 점프식 | 없음 |
| **Viewport-Based** | ~0.1초 | 낮음 | 부드러움 | 자동 |

## 사용 방법

### 기본 Navigation
```
← : 부드럽게 왼쪽으로 스크롤 (화면의 10% 이동)
→ : 부드럽게 오른쪽으로 스크롤 (화면의 10% 이동)
Home : 첫 번째 위치로 점프
End : 마지막 위치로 점프
```

### 현재 위치 확인
콘솔에서 현재 viewport 정보 확인:
```
[Display] Viewport mode: ~50 traces fit on screen
[Nav] Viewing traces 1 - 50 / 9834
[Nav] Viewing traces 6 - 55 / 9834
[Nav] Viewing traces 11 - 60 / 9834
...
```

### 화면당 Trace 수 조정 (필요시)
`gl_display.py`에서:
```python
self.pixels_per_trace = 20  # 기본값
# 더 많이 보기: 10 (trace가 좁아짐)
# 더 선명하게: 30 (trace가 넓어짐)
```

## 기술적 세부사항

### 1. _calculate_traces_per_screen() 메서드
```python
def _calculate_traces_per_screen(self):
    """Calculate how many traces fit on screen"""
    widget_width = max(self.width(), 800)  # Minimum 800px
    self.traces_per_screen = max(10, int(widget_width / self.pixels_per_trace))
```

### 2. _get_visible_traces() 메서드
```python
def _get_visible_traces(self) -> tuple:
    """Get currently visible trace range"""
    start_trace = self.view_offset  # Can be fractional (e.g., 12.5)
    end_trace = min(start_trace + self.traces_per_screen, self.num_traces)

    # Get integer indices with buffer for smooth scrolling
    start_idx = max(0, int(np.floor(start_trace)))
    end_idx = min(self.num_traces, int(np.ceil(end_trace)) + 1)

    return start_idx, end_idx, start_trace
```

### 3. paintGL() - Fractional Offset 처리
```python
def paintGL(self):
    # Get visible trace range
    start_idx, end_idx, start_trace = self._get_visible_traces()

    # Create viewport coordinate system
    gluOrtho2D(0, self.traces_per_screen, self.num_samples, 0)

    # Account for fractional offset (smooth scrolling)
    fractional_offset = start_trace - start_idx
    glTranslatef(-fractional_offset, 0.0, 0.0)
```

### 4. Coordinate System
- **화면 좌표**: [0, traces_per_screen] x [0, num_samples]
- **실제 데이터**: view_offset + display_idx (fractional 가능)
- **자동 변환**: OpenGL projection + modelview matrix

## 추가 최적화 가능

### 1. Downsampling (미래)
- 줌 아웃 시 trace decimation
- LOD (Level of Detail) 시스템

### 2. GPU 가속 (미래)
- Compute shader로 AGC 계산
- Vertex shader로 wiggle 생성

### 3. Background Loading (미래)
- 다음 slice 미리 로드
- 스레드 기반 비동기 처리

## 참고

- **segy_viz**: https://github.com/valohai/segy_viz
- **Slice-based rendering**: OpenGL texture streaming 방식
- **Performance target**: <100ms per frame
