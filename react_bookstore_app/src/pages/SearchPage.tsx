import { useState } from 'react';
import { Search, LogOut } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { Input } from '../components/ui/input';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';

interface Book {
  id: number;
  title: string;
  author: string;
  category: string;
  year: number;
}

const mockBooks: Book[] = [
  { id: 1, title: '별이 빛나는 밤에', author: '김하늘', category: '소설', year: 2023 },
  { id: 2, title: '코딩의 즐거움', author: '이개발', category: '기술', year: 2022 },
  { id: 3, title: '요리하는 즐거움', author: '박요리', category: '요리', year: 2024 },
  { id: 4, title: '마음의 평화', author: '정마음', category: '자기계발', year: 2023 },
  { id: 5, title: '여행의 발견', author: '최여행', category: '여행', year: 2024 },
  { id: 6, title: '프로그래밍 입문', author: '이개발', category: '기술', year: 2021 },
  { id: 7, title: '달빛 아래서', author: '김하늘', category: '소설', year: 2022 },
  { id: 8, title: '행복한 하루', author: '정마음', category: '자기계발', year: 2024 },
];

interface SearchPageProps {
  onLogout: () => void;
}

export default function SearchPage({ onLogout }: SearchPageProps) {
  const [category, setCategory] = useState<string>('all');
  const [searchText, setSearchText] = useState<string>('');
  const [filteredBooks, setFilteredBooks] = useState<Book[]>(mockBooks);

  const handleSearch = () => {
    let results = mockBooks;

    // 카테고리 필터
    if (category !== 'all') {
      results = results.filter(book => book.category === category);
    }

    // 검색어 필터
    if (searchText.trim()) {
      results = results.filter(book =>
        book.title.toLowerCase().includes(searchText.toLowerCase()) ||
        book.author.toLowerCase().includes(searchText.toLowerCase())
      );
    }

    setFilteredBooks(results);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#f5e6d3] via-[#f9f0e3] to-[#e8d5c4] p-8">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Gamja+Flower&display=swap');

        * {
          font-family: 'Gamja+Flower', cursive !important;
        }
      `}</style>

      <div className="max-w-4xl mx-auto">
        {/* 헤더 */}
        <div className="text-center mb-12 relative">
          <h1 className="text-5xl mb-2 text-[#8b6f47]">📚 도서 검색</h1>
          <p className="text-xl text-[#a0826d]">원하는 책을 찾아보세요!</p>

          {/* 로그아웃 버튼 */}
          <Button
            onClick={onLogout}
            variant="outline"
            className="absolute top-0 right-0 border-[#d4b896] text-[#8b6f47] hover:bg-[#f5e6d3]"
          >
            <LogOut className="mr-2 h-4 w-4" />
            로그아웃
          </Button>
        </div>

        {/* 검색 영역 */}
        <Card className="bg-white/80 backdrop-blur-sm border-2 border-[#d4b896] shadow-xl p-8 mb-8 rounded-3xl">
          <div className="space-y-6">
            {/* 카테고리 선택 */}
            <div>
              <label className="block text-xl mb-3 text-[#8b6f47]">📂 카테고리</label>
              <Select value={category} onValueChange={setCategory}>
                <SelectTrigger className="w-full h-14 text-xl border-2 border-[#d4b896] rounded-2xl bg-white">
                  <SelectValue placeholder="카테고리를 선택하세요" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all" className="text-xl">전체</SelectItem>
                  <SelectItem value="소설" className="text-xl">소설</SelectItem>
                  <SelectItem value="기술" className="text-xl">기술</SelectItem>
                  <SelectItem value="요리" className="text-xl">요리</SelectItem>
                  <SelectItem value="자기계발" className="text-xl">자기계발</SelectItem>
                  <SelectItem value="여행" className="text-xl">여행</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* 검색어 입력 */}
            <div>
              <label className="block text-xl mb-3 text-[#8b6f47]">🔍 검색어</label>
              <Input
                type="text"
                placeholder="책 제목이나 저자를 입력하세요..."
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                onKeyPress={handleKeyPress}
                className="w-full h-14 text-xl border-2 border-[#d4b896] rounded-2xl px-6"
              />
            </div>

            {/* 검색 버튼 */}
            <Button
              onClick={handleSearch}
              className="w-full h-14 text-2xl bg-gradient-to-r from-[#c9a67a] to-[#b08d5f] hover:from-[#b08d5f] hover:to-[#9a7a4f] text-white rounded-2xl shadow-lg transition-all duration-300"
            >
              <Search className="mr-3 h-6 w-6" />
              검색하기
            </Button>
          </div>
        </Card>

        {/* 검색 결과 */}
        <div>
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-3xl text-[#8b6f47]">📖 검색 결과</h2>
            <span className="text-2xl text-[#a0826d]">총 {filteredBooks.length}권</span>
          </div>

          <div className="bg-white/60 backdrop-blur-sm border-2 border-[#d4b896] rounded-3xl p-6 shadow-xl min-h-[400px]">
            {filteredBooks.length > 0 ? (
              <div className="space-y-4">
                {filteredBooks.map((book) => (
                  <div
                    key={book.id}
                    className="bg-white border-2 border-[#e8d5c4] rounded-2xl p-6 hover:shadow-lg hover:border-[#c9a67a] transition-all duration-300 cursor-pointer"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <h3 className="text-2xl mb-2 text-[#8b6f47]">{book.title}</h3>
                        <p className="text-xl text-[#a0826d] mb-1">저자: {book.author}</p>
                        <div className="flex gap-3 items-center">
                          <span className="inline-block px-4 py-1 bg-[#f5e6d3] text-[#8b6f47] rounded-full text-lg">
                            {book.category}
                          </span>
                          <span className="text-lg text-[#b8a088]">{book.year}년</span>
                        </div>
                      </div>
                      <div className="text-4xl">📕</div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-[350px] text-center">
                <div className="text-6xl mb-4">🔍</div>
                <p className="text-2xl text-[#a0826d]">검색 결과가 없습니다</p>
                <p className="text-xl text-[#b8a088] mt-2">다른 검색어를 입력해보세요</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
