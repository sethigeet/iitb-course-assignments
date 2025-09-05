#include <assert.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
  int len;
  scanf("%d", &len);

  char name[len + 1];
  int bytesRead = read(0, name, len);
  assert(bytesRead == len);

  fprintf(stdout, "Hello %s!\n", name);
  char greeting[] = "Logged in with the name ";
  write(1, greeting, sizeof(greeting) - 1);
  write(1, name, sizeof(name) - 1);
  write(1, "!", 2);

  return 0;
}
