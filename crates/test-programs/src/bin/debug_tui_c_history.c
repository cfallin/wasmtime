//! flags = []

// A test program for watchpoints and reverse execution in the built-in
// debugger.

#include <stdio.h>

int counter = 0;

struct Acc {
  int total;
  int count;
};

int square(int x) {
  int r = x * x;
  return r;
}

void add(struct Acc *acc, int v) {
  acc->total += v;
  acc->count++;
  counter++;
}

int main() {
  struct Acc acc = {0, 0};
  for (int i = 1; i <= 4; i++) {
    int sq = square(i);
    add(&acc, sq);
  }
  printf("total = %d count = %d counter = %d\n", acc.total, acc.count, counter);
  return 0;
}
