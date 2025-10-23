import 'package:flutter/material.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key}) : super(key: key);

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final TextEditingController _searchController = TextEditingController();

  // 카테고리 선택을 위한 변수
  String? _selectedCategory;
  final List<String> _categories = [
    '전체',
    '전자제품',
    '의류',
    '도서',
    '식품',
    '스포츠',
  ];

  // 검색 결과를 저장할 리스트
  List<String> _searchResults = [];

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  void _handleSearch() {
    // 검색 로직은 여기에 추가하세요
    // 현재는 화면만 구성되어 있습니다
    setState(() {
      // 샘플 데이터 - 실제로는 검색 API를 호출해야 합니다
      _searchResults = [
        '검색 결과 1 - ${_searchController.text}',
        '검색 결과 2 - ${_searchController.text}',
        '검색 결과 3 - ${_searchController.text}',
        '검색 결과 4 - ${_searchController.text}',
        '검색 결과 5 - ${_searchController.text}',
      ];
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('검색'),
        centerTitle: true,
        backgroundColor: Colors.blue,
        actions: [
          IconButton(
            icon: const Icon(Icons.logout),
            onPressed: () {
              Navigator.pop(context);
            },
            tooltip: '로그아웃',
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // 카테고리 선택 콤보 박스 (DropdownButton)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12),
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey),
                borderRadius: BorderRadius.circular(4),
              ),
              child: DropdownButton<String>(
                value: _selectedCategory,
                hint: const Text('카테고리를 선택하세요'),
                isExpanded: true,
                underline: const SizedBox(),
                items: _categories.map((String category) {
                  return DropdownMenuItem<String>(
                    value: category,
                    child: Text(category),
                  );
                }).toList(),
                onChanged: (String? newValue) {
                  setState(() {
                    _selectedCategory = newValue;
                  });
                },
              ),
            ),
            const SizedBox(height: 16),

            // Item 텍스트 박스
            TextField(
              controller: _searchController,
              decoration: const InputDecoration(
                labelText: 'Item',
                hintText: '검색할 항목을 입력하세요',
                prefixIcon: Icon(Icons.search),
                border: OutlineInputBorder(),
              ),
              onSubmitted: (_) => _handleSearch(),
            ),
            const SizedBox(height: 16),

            // 검색 버튼
            ElevatedButton(
              onPressed: _handleSearch,
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
                backgroundColor: Colors.blue,
                foregroundColor: Colors.white,
              ),
              child: const Text(
                '검색',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
            ),
            const SizedBox(height: 24),

            // 검색 결과 헤더
            if (_searchResults.isNotEmpty)
              const Text(
                '검색 결과',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            const SizedBox(height: 8),

            // 검색 목록을 나열하는 리스트뷰
            Expanded(
              child: _searchResults.isEmpty
                  ? Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: const [
                          Icon(
                            Icons.search_off,
                            size: 64,
                            color: Colors.grey,
                          ),
                          SizedBox(height: 16),
                          Text(
                            '검색 결과가 없습니다',
                            style: TextStyle(
                              fontSize: 16,
                              color: Colors.grey,
                            ),
                          ),
                        ],
                      ),
                    )
                  : ListView.builder(
                      itemCount: _searchResults.length,
                      itemBuilder: (context, index) {
                        return Card(
                          margin: const EdgeInsets.symmetric(vertical: 4),
                          child: ListTile(
                            leading: CircleAvatar(
                              backgroundColor: Colors.blue,
                              child: Text(
                                '${index + 1}',
                                style: const TextStyle(color: Colors.white),
                              ),
                            ),
                            title: Text(_searchResults[index]),
                            subtitle: Text('카테고리: ${_selectedCategory ?? "미선택"}'),
                            trailing: const Icon(Icons.arrow_forward_ios),
                            onTap: () {
                              // 아이템 클릭 로직은 여기에 추가하세요
                              ScaffoldMessenger.of(context).showSnackBar(
                                SnackBar(
                                  content: Text('${_searchResults[index]} 선택됨'),
                                ),
                              );
                            },
                          ),
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }
}
