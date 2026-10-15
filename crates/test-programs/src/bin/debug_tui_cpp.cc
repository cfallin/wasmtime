//! flags = ["-fno-exceptions", "-lc++", "-lc++abi"]

// Exercises pretty-printing of C++ standard-library types in the
// `--debug-tui` debugger.

#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct Point {
  int x;
  int y;
};

// A type named like a container that the pretty-printers know, but with
// a different layout: printing must fall back to the raw view.
struct Weird {};
namespace std {
template <> class vector<Weird, allocator<Weird>> {
public:
  int not_a_vector = 7;
};
} // namespace std

template <typename T> __attribute__((noinline)) void consume(const T &) {}

int main() {
  std::vector<int> vec = {1, 2, 3};
  std::vector<Point> points = {{1, 2}, {3, 4}};
  std::string short_str = "hi";
  std::string long_str = "a string too long for the small-string buffer";
  std::map<int, std::string> map = {{1, "one"}, {2, "two"}, {3, "three"}};
  std::set<int> set = {5, 3, 9, 1};
  std::unordered_map<std::string, int> umap = {{"x", 10}, {"y", 20}};
  std::unordered_set<int> uset = {7, 8};
  std::unique_ptr<Point> uptr = std::make_unique<Point>(Point{5, 6});
  std::unique_ptr<Point> null_uptr;
  std::shared_ptr<Point> sptr = std::make_shared<Point>(Point{7, 8});
  std::shared_ptr<Point> sptr2 = sptr;
  const char *cstr = "a C string";
  std::vector<Weird> weird;

  consume(vec); // break here
  consume(points);
  consume(short_str);
  consume(long_str);
  consume(map);
  consume(set);
  consume(umap);
  consume(uset);
  consume(uptr);
  consume(null_uptr);
  consume(sptr2);
  consume(weird);
  printf("%zu %s %zu\n", vec.size(), long_str.c_str(), map.size());
  return 0;
}
