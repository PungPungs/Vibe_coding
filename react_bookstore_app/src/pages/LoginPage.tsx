import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Button } from '../components/ui/button';
import { BookOpen } from 'lucide-react';

interface LoginPageProps {
  onLogin: () => void;
}

export default function LoginPage({ onLogin }: LoginPageProps) {
  const [id, setId] = useState('');
  const [password, setPassword] = useState('');

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    console.log('로그인 시도:', { id, password });
    // 실제 로그인 로직은 여기에 구현
    onLogin(); // 로그인 성공 시 검색 화면으로 전환
  };

  const handleSignup = () => {
    console.log('회원가입 페이지로 이동');
    // 회원가입 페이지 이동 로직
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-amber-50 via-orange-50 to-yellow-50 px-4">
      <Card className="w-full max-w-md bg-gradient-to-b from-amber-50/80 to-white border-amber-200/50 shadow-lg">
        <CardHeader className="space-y-4 text-center pb-6">
          <div className="flex justify-center">
            <div className="w-20 h-20 bg-gradient-to-br from-amber-100 to-orange-100 rounded-full flex items-center justify-center shadow-md">
              <BookOpen className="w-10 h-10 text-amber-800" />
            </div>
          </div>
          <CardTitle className="text-3xl font-bold text-amber-900 leading-tight">
            책과 함께 찾는<br />나만의 서점
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-2">
          <form onSubmit={handleLogin} className="space-y-5">
            <div className="space-y-2">
              <Label htmlFor="id" className="font-bold text-amber-900 text-lg">아이디</Label>
              <Input
                id="id"
                type="text"
                placeholder="아이디를 입력하세요"
                value={id}
                onChange={(e) => setId(e.target.value)}
                className="bg-white/70 border-amber-200 focus:border-amber-400 text-lg"
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="password" className="font-bold text-amber-900 text-lg">비밀번호</Label>
              <Input
                id="password"
                type="password"
                placeholder="비밀번호를 입력하세요"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="bg-white/70 border-amber-200 focus:border-amber-400 text-lg"
                required
              />
            </div>
            <div className="space-y-2 pt-3">
              <Button type="submit" className="w-full font-bold bg-amber-700 hover:bg-amber-800 text-white text-lg">
                로그인
              </Button>
              <Button
                type="button"
                variant="outline"
                className="w-full font-bold border-amber-300 text-amber-900 hover:bg-amber-50 text-lg"
                onClick={handleSignup}
              >
                회원가입
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
