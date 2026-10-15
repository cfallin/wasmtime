//! flags = []

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

struct Point {
  int x;
  int y;
};

enum Color { RED, GREEN, BLUE };

struct Shape {
  const char *name;
  enum Color color;
  struct Point points[3];
  struct Shape *next;
  char tag[8];
  double scale;
  unsigned flags : 3;
};

union Num {
  int i;
  float f;
};

int global_counter = 42;
static struct Point origin = {0, 0};

int area(struct Shape *s) {
  int a = s->points[1].x - s->points[0].x;
  int b = s->points[2].y - s->points[0].y;
  int result = a * b / 2;
  return result;
}

int main() {
  struct Shape tri;
  struct Shape other = {"other", BLUE, {{1, 1}, {2, 2}, {3, 3}}, NULL, "oth", 0.5, 1};
  tri.name = "triangle";
  tri.color = GREEN;
  tri.points[0].x = 0;
  tri.points[0].y = 0;
  tri.points[1].x = 4;
  tri.points[1].y = 0;
  tri.points[2].x = 0;
  tri.points[2].y = 3;
  tri.next = &other;
  strcpy(tri.tag, "tri");
  tri.scale = 1.5;
  tri.flags = 5;
  union Num n;
  n.i = 0x3f800000;
  char *heap = malloc(16);
  strcpy(heap, "on the heap");
  int arr[5] = {10, 20, 30, 40, 50};
  global_counter++;
  int r = area(&tri);
  printf("area = %d %s %d %d %d\n", r, heap, arr[2], n.i, origin.x);
  return 0;
}
