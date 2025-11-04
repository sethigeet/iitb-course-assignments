#include "types.h"
#include "user.h"

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf(1, "Usage: strace <program>");
    exit();
  }

  int pid = fork();
  if (pid == 0) {
    trace(getpid());
    exec(argv[1], argv+1);
  } else {
    wait();
  }

  exit();
}
