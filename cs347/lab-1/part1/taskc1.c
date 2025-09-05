#include <malloc.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  int *arr[5];
  int **b = arr;
  for (int i = 1; i <= 5; i++) {
    *b = (int *)malloc(sizeof(int));
    **b = i;
    b++;
  }

  b = arr;
  for (int i = 1; i <= 5; i++) {
    printf("%d ", **b);
    b++;
  }

  b = arr;
  for (int i = 1; i <= 5; i++) {
    free(*b);
    b++;
  }

  return 0;
}
