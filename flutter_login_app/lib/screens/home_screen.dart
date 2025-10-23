import 'package:flutter/material.dart';

// 도서 모델
class Book {
  final int id;
  final String title;
  final String author;
  final String category;
  final int year;

  Book({
    required this.id,
    required this.title,
    required this.author,
    required this.category,
    required this.year,
  });
}

// 샘플 도서 데이터
final List<Book> mockBooks = [
  Book(id: 1, title: '별이 빛나는 밤에', author: '김하늘', category: '소설', year: 2023),
  Book(id: 2, title: '코딩의 즐거움', author: '이개발', category: '기술', year: 2022),
  Book(id: 3, title: '요리하는 즐거움', author: '박요리', category: '요리', year: 2024),
  Book(id: 4, title: '마음의 평화', author: '정마음', category: '자기계발', year: 2023),
  Book(id: 5, title: '여행의 발견', author: '최여행', category: '여행', year: 2024),
  Book(id: 6, title: '프로그래밍 입문', author: '이개발', category: '기술', year: 2021),
  Book(id: 7, title: '달빛 아래서', author: '김하늘', category: '소설', year: 2022),
  Book(id: 8, title: '행복한 하루', author: '정마음', category: '자기계발', year: 2024),
];

class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key}) : super(key: key);

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final TextEditingController _searchController = TextEditingController();

  // 카테고리 선택을 위한 변수
  String _selectedCategory = '전체';
  final List<String> _categories = [
    '전체',
    '소설',
    '기술',
    '요리',
    '자기계발',
    '여행',
  ];

  // 검색 결과를 저장할 리스트
  List<Book> _filteredBooks = mockBooks;

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  void _handleSearch() {
    setState(() {
      List<Book> results = mockBooks;

      // 카테고리 필터
      if (_selectedCategory != '전체') {
        results = results.where((book) => book.category == _selectedCategory).toList();
      }

      // 검색어 필터
      if (_searchController.text.trim().isNotEmpty) {
        final searchText = _searchController.text.toLowerCase();
        results = results.where((book) =>
          book.title.toLowerCase().contains(searchText) ||
          book.author.toLowerCase().contains(searchText)
        ).toList();
      }

      _filteredBooks = results;
    });
  }

  @override
  Widget build(BuildContext context) {
    // 색상 정의 (React 디자인 기반)
    const Color bgStart = Color(0xFFF5E6D3);
    const Color bgMid = Color(0xFFF9F0E3);
    const Color bgEnd = Color(0xFFE8D5C4);
    const Color brownText = Color(0xFF8B6F47);
    const Color brownLight = Color(0xFFA0826D);
    const Color brownBorder = Color(0xFFD4B896);
    const Color brownBadge = Color(0xFFF5E6D3);
    const Color buttonStart = Color(0xFFC9A67A);
    const Color buttonEnd = Color(0xFFB08D5F);

    return Scaffold(
      backgroundColor: bgStart,
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [bgStart, bgMid, bgEnd],
          ),
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(32.0),
            child: Column(
              children: [
                // 헤더
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Expanded(
                      child: Column(
                        children: [
                          Text(
                            '📚 도서 검색',
                            style: TextStyle(
                              fontSize: 40,
                              color: brownText,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          SizedBox(height: 8),
                          Text(
                            '원하는 책을 찾아보세요!',
                            style: TextStyle(
                              fontSize: 20,
                              color: brownLight,
                            ),
                          ),
                        ],
                      ),
                    ),
                    // 로그아웃 버튼
                    OutlinedButton.icon(
                      onPressed: () => Navigator.pop(context),
                      icon: const Icon(Icons.logout, size: 16),
                      label: const Text('로그아웃'),
                      style: OutlinedButton.styleFrom(
                        foregroundColor: brownText,
                        side: const BorderSide(color: brownBorder),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 48),

                // 검색 영역 카드
                Container(
                  padding: const EdgeInsets.all(32),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.8),
                    borderRadius: BorderRadius.circular(24),
                    border: Border.all(color: brownBorder, width: 2),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.1),
                        blurRadius: 20,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // 카테고리 선택
                      const Text(
                        '📂 카테고리',
                        style: TextStyle(
                          fontSize: 20,
                          color: brownText,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 12),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 16),
                        decoration: BoxDecoration(
                          color: Colors.white,
                          border: Border.all(color: brownBorder, width: 2),
                          borderRadius: BorderRadius.circular(16),
                        ),
                        child: DropdownButton<String>(
                          value: _selectedCategory,
                          isExpanded: true,
                          underline: const SizedBox(),
                          style: const TextStyle(fontSize: 20, color: Colors.black),
                          items: _categories.map((String category) {
                            return DropdownMenuItem<String>(
                              value: category,
                              child: Text(category),
                            );
                          }).toList(),
                          onChanged: (String? newValue) {
                            setState(() {
                              _selectedCategory = newValue!;
                            });
                          },
                        ),
                      ),
                      const SizedBox(height: 24),

                      // 검색어 입력
                      const Text(
                        '🔍 검색어',
                        style: TextStyle(
                          fontSize: 20,
                          color: brownText,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 12),
                      TextField(
                        controller: _searchController,
                        style: const TextStyle(fontSize: 20),
                        decoration: InputDecoration(
                          hintText: '책 제목이나 저자를 입력하세요...',
                          filled: true,
                          fillColor: Colors.white,
                          contentPadding: const EdgeInsets.symmetric(
                            horizontal: 24,
                            vertical: 20,
                          ),
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(16),
                            borderSide: const BorderSide(color: brownBorder, width: 2),
                          ),
                          enabledBorder: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(16),
                            borderSide: const BorderSide(color: brownBorder, width: 2),
                          ),
                          focusedBorder: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(16),
                            borderSide: const BorderSide(color: brownText, width: 2),
                          ),
                        ),
                        onSubmitted: (_) => _handleSearch(),
                      ),
                      const SizedBox(height: 24),

                      // 검색 버튼
                      SizedBox(
                        width: double.infinity,
                        child: ElevatedButton(
                          onPressed: _handleSearch,
                          style: ElevatedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(vertical: 20),
                            backgroundColor: buttonStart,
                            foregroundColor: Colors.white,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(16),
                            ),
                            elevation: 4,
                          ),
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: const [
                              Icon(Icons.search, size: 24),
                              SizedBox(width: 12),
                              Text(
                                '검색하기',
                                style: TextStyle(
                                  fontSize: 24,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 32),

                // 검색 결과 헤더
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Text(
                      '📖 검색 결과',
                      style: TextStyle(
                        fontSize: 28,
                        color: brownText,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    Text(
                      '총 ${_filteredBooks.length}권',
                      style: const TextStyle(
                        fontSize: 24,
                        color: brownLight,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),

                // 검색 결과 리스트
                Expanded(
                  child: Container(
                    padding: const EdgeInsets.all(24),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.6),
                      borderRadius: BorderRadius.circular(24),
                      border: Border.all(color: brownBorder, width: 2),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.1),
                          blurRadius: 20,
                          offset: const Offset(0, 4),
                        ),
                      ],
                    ),
                    child: _filteredBooks.isEmpty
                        ? const Center(
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Text(
                                  '🔍',
                                  style: TextStyle(fontSize: 64),
                                ),
                                SizedBox(height: 16),
                                Text(
                                  '검색 결과가 없습니다',
                                  style: TextStyle(
                                    fontSize: 24,
                                    color: brownLight,
                                  ),
                                ),
                                SizedBox(height: 8),
                                Text(
                                  '다른 검색어를 입력해보세요',
                                  style: TextStyle(
                                    fontSize: 20,
                                    color: Color(0xFFB8A088),
                                  ),
                                ),
                              ],
                            ),
                          )
                        : ListView.builder(
                            itemCount: _filteredBooks.length,
                            itemBuilder: (context, index) {
                              final book = _filteredBooks[index];
                              return Container(
                                margin: const EdgeInsets.only(bottom: 16),
                                padding: const EdgeInsets.all(24),
                                decoration: BoxDecoration(
                                  color: Colors.white,
                                  borderRadius: BorderRadius.circular(16),
                                  border: Border.all(
                                    color: const Color(0xFFE8D5C4),
                                    width: 2,
                                  ),
                                ),
                                child: Row(
                                  children: [
                                    Expanded(
                                      child: Column(
                                        crossAxisAlignment: CrossAxisAlignment.start,
                                        children: [
                                          Text(
                                            book.title,
                                            style: const TextStyle(
                                              fontSize: 24,
                                              color: brownText,
                                              fontWeight: FontWeight.bold,
                                            ),
                                          ),
                                          const SizedBox(height: 8),
                                          Text(
                                            '저자: ${book.author}',
                                            style: const TextStyle(
                                              fontSize: 20,
                                              color: brownLight,
                                            ),
                                          ),
                                          const SizedBox(height: 12),
                                          Row(
                                            children: [
                                              Container(
                                                padding: const EdgeInsets.symmetric(
                                                  horizontal: 16,
                                                  vertical: 4,
                                                ),
                                                decoration: BoxDecoration(
                                                  color: brownBadge,
                                                  borderRadius: BorderRadius.circular(20),
                                                ),
                                                child: Text(
                                                  book.category,
                                                  style: const TextStyle(
                                                    fontSize: 18,
                                                    color: brownText,
                                                  ),
                                                ),
                                              ),
                                              const SizedBox(width: 12),
                                              Text(
                                                '${book.year}년',
                                                style: const TextStyle(
                                                  fontSize: 18,
                                                  color: Color(0xFFB8A088),
                                                ),
                                              ),
                                            ],
                                          ),
                                        ],
                                      ),
                                    ),
                                    const Text(
                                      '📕',
                                      style: TextStyle(fontSize: 40),
                                    ),
                                  ],
                                ),
                              );
                            },
                          ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
