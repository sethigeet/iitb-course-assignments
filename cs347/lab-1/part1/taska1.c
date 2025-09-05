#include <stdio.h>

int main(int argc, char *argv[]) {
  printf("Hello World!\n");

  char name[100];
  scanf("%s", name);
  int len = 0;
  while (name[len] != '\0')
    len++;
  printf("Hello World %s %d", name, len);

  return 0;
}
