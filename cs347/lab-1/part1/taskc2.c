#include <malloc.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  int *mat[5];
  for (int i = 0; i < 5; i++) {
    mat[i] = (int *)malloc(sizeof(int) * 5);
  }

  int val = 1;
  int(*ptr)[5];
  for (int i = 0; i < 5; i++) {
    ptr = (int(*)[5])mat[i]; // cast ptr to a pointer to an array of len 5
    for (int j = 0; j < 5; j++) {
      mat[i][j] = val++;
    }
  }

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      printf("%d ", mat[i][j]);
    }
    printf("\n");
  }

  for (int i = 0; i < 5; i++) {
    free(mat[i]);
  }

  return 0;
}
