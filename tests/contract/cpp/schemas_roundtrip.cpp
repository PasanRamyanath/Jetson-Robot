// Contract test driver: writes a det-like message to stdout, then parses stdin back and echoes a summary.
#include <cstdio>
#include <iostream>
#include <iterator>
#include <string>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include "schemas.hpp"

int main(int argc, char** argv) {
#ifdef _WIN32  // msgpack/TS bytes on stdin/stdout: no CRLF translation
  _setmode(_fileno(stdin), _O_BINARY), _setmode(_fileno(stdout), _O_BINARY);
#endif
  using namespace beni;
  if (argc > 1 && std::string(argv[1]) == "write") {
    MsgWriter w;
    w.envelope("vision", 1).key("frames").arr(1);
    w.map(4).key("cam").i(1).key("fn").i(123456).key("ts").i(-5000000000LL).key("objs").arr(1);
    const unsigned char emb[4] = {0, 1, 2, 255};
    w.map(6).key("tid").i(70000).key("cls").i(0).key("gie").i(1).key("conf").fr(0.87654, 3);
    w.key("bbox").arr(4).fr(10.26, 1).fr(-3.0, 1).fr(300.04, 1).f32(0.5f).key("emb").bin(emb, 4);
    std::fwrite(w.buf.data(), 1, w.buf.size(), stdout);
    return 0;
  }
  std::string in((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
  Value v;
  if (!unpack(in.data(), in.size(), v)) return 2;
  const Value* big = v.get("big");
  const Value* arr = v.get("arr");
  std::printf("op=%s n=%.3f neg=%lld big=%lld arr=%zu s=%zu ok=%d\n", v.text("op").c_str(), v.num("n"),
              (long long)v.get("neg")->i, (long long)(big ? big->i : 0), arr ? arr->a.size() : 0,
              v.get("long")->s.size(), int(v.get("flag")->i));
  return 0;
}
