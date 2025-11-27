# Flutter Login App

Flutter로 제작된 로그인 및 검색 화면 UI

## 화면 구성

### 1. 초기 화면 (로그인 화면)
- ID 텍스트 박스
- Password 텍스트 박스
- 로그인 버튼
- 회원가입 버튼

### 2. 로그인 후 화면 (검색 화면)
- 카테고리 선택 콤보 박스 (DropdownButton)
- 검색어 입력 텍스트 박스
- 검색 버튼
- 검색 결과 리스트뷰

## 파일 구조

```
flutter_login_app/
├── lib/
│   ├── main.dart                 # 앱 진입점
│   └── screens/
│       ├── login_screen.dart     # 로그인 화면
│       └── home_screen.dart      # 검색 화면 (로그인 후)
├── pubspec.yaml                  # 프로젝트 설정
└── README.md
```

## 실행 방법

```bash
cd flutter_login_app
flutter pub get
flutter run
```

## 참고사항

- 현재는 **UI만 구성**되어 있습니다
- 실제 로그인 로직, 회원가입 로직, 검색 API 연동은 구현되지 않았습니다
- 화면 전환과 샘플 데이터 표시만 가능합니다

## 다음 구현 사항 (함수 제작)

1. 로그인 API 연동
2. 회원가입 API 연동
3. 검색 API 연동
4. 상태 관리 (Provider, Bloc 등)
5. 데이터 모델 정의
