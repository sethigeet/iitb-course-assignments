#include <malloc.h>
#include <stdio.h>

#define INITIAL_CAPACITY 10

typedef struct {
  int *data;
  int len;
  int capacity;
} vector;

void initialize(vector **vec) {
  *vec = (vector *)malloc(sizeof(vector));
  (*vec)->len = 0;
  (*vec)->capacity = INITIAL_CAPACITY;

  (*vec)->data = (int *)malloc(INITIAL_CAPACITY * sizeof(int));
}

void push_back(vector *vec, int data) {
  if (vec->len == vec->capacity) {
    vec->capacity *= 2;
    vec->data = (int *)realloc(vec->data, vec->capacity * sizeof(int));
  }

  vec->data[vec->len] = data;
  vec->len++;
}

int back(vector *vec) {
  if (vec->len == 0)
    return -1;

  return vec->data[vec->len - 1];
}

int get_index(vector *vec, int index) {
  if (vec->len <= index)
    return -1;

  return vec->data[index];
}

void destroy(vector **vec) {
  free((*vec)->data);
  free(*vec);
}

int main(int argc, char *argv[]) {
  vector *v = NULL;

  int op, arg;
  int run = 1;
  while (run) {
    scanf("%d", &op);

    switch (op) {
    case 1:
      initialize(&v);
      break;
    case 2:
      scanf("%d", &arg);
      push_back(v, arg);
      break;
    case 3:
      printf("%d\n", back(v));
      break;
    case 4:
      scanf("%d", &arg);
      printf("%d\n", get_index(v, arg));
      break;
    case 5:
      destroy(&v);
      run = 0;
      break;
    }
  }

  return 0;
}
