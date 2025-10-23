# React Bookstore App

React + TypeScript로 제작된 도서 검색 앱

## 화면 구성

### 1. 로그인 화면
- 아이디/비밀번호 입력 필드
- 로그인 버튼 (검색 화면으로 이동)
- 회원가입 버튼
- Figma 디자인 기반 UI (베이지/갈색 톤)
- 책 아이콘 로고

### 2. 검색 화면 (로그인 후)
- 카테고리 선택 드롭다운 (전체, 소설, 기술, 요리, 자기계발, 여행)
- 검색어 입력 필드
- 검색 버튼
- 검색 결과 리스트
- 로그아웃 버튼

## 기술 스택

- **React 18** - UI 라이브러리
- **TypeScript** - 타입 안정성
- **Vite** - 빌드 도구
- **Tailwind CSS** - 스타일링
- **Lucide React** - 아이콘

## 프로젝트 구조

```
react_bookstore_app/
├── src/
│   ├── components/
│   │   └── ui/           # 재사용 가능한 UI 컴포넌트
│   │       ├── card.tsx
│   │       ├── input.tsx
│   │       ├── button.tsx
│   │       ├── label.tsx
│   │       └── select.tsx
│   ├── pages/
│   │   ├── LoginPage.tsx    # 로그인 화면
│   │   └── SearchPage.tsx   # 검색 화면
│   ├── App.tsx              # 메인 앱 (화면 전환 관리)
│   ├── main.tsx             # React 엔트리 포인트
│   └── index.css            # 글로벌 스타일
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
└── postcss.config.js
```

## 설치 및 실행

### 1. 의존성 설치
```bash
cd react_bookstore_app
npm install
```

### 2. 개발 서버 실행
```bash
npm run dev
```

브라우저에서 `http://localhost:5173` 접속

### 3. 빌드
```bash
npm run build
```

빌드된 파일은 `dist/` 폴더에 생성됩니다.

## 주요 기능

### 화면 전환
- 로그인 버튼 클릭 시 검색 화면으로 전환
- 로그아웃 버튼 클릭 시 로그인 화면으로 복귀
- `useState`를 사용한 간단한 상태 관리

### 검색 기능
- 카테고리별 필터링
- 제목/저자 검색
- 실시간 검색 결과 표시
- Enter 키 검색 지원

### 샘플 데이터
8권의 샘플 도서 데이터 포함:
- 소설, 기술, 요리, 자기계발, 여행 카테고리

## 참고사항

- **UI만 구성**되어 있습니다
- 실제 로그인/회원가입 API 연동은 구현되지 않았습니다
- 도서 데이터는 하드코딩된 샘플 데이터입니다
- 실제 서비스를 위해서는 백엔드 API 연동이 필요합니다

## 다음 구현 사항 (백엔드 연동)

1. 사용자 인증 API (로그인/회원가입)
2. 도서 검색 API
3. 상태 관리 라이브러리 (Redux, Zustand 등)
4. API 클라이언트 (Axios, React Query 등)
5. 라우팅 (React Router)
