//! flags = ["-O1"]

// Built with optimization so that some variables live in Wasm locals
// rather than linear memory (for the `location` command).

#include <stdio.h>
#include <string.h>

unsigned char global_buf[16];

__attribute__((noinline)) int checksum(const unsigned char *data, int n) {
  int total = 0;
  for (int i = 0; i < n; i++) {
    total += data[i] * (i + 1); // loop body
  }
  return total;
}

int main() {
  memcpy(global_buf, "0123456789abcdef", 16);
  int c = checksum(global_buf, 16);
  printf("%d\n", c);
  return 0;
}
