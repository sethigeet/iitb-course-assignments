#include "types.h"
#include "user.h"

int main() {
  int pid = fork();
  if (pid > 0) {
    int pid2 = fork();
    if (pid2 > 0) {
      pstree();
      // getchildren();
      printf(0, "Is 'me' valid: %d\n", is_proc_valid(getpid()));
      printf(0, "Is pid valid: %d\n", is_proc_valid(pid));
      printf(0, "Is pid2 valid: %d\n", is_proc_valid(pid2));
      printf(0, "Is random valid: %d\n", is_proc_valid(1234));
      wait();
      wait();
    } else {
      // getsiblings();
      if (fork() > 0) {
        wait();
      };
      sleep(1);
    }
  } else {
    // getsiblings();
    if (fork() > 0) {
      wait();
    };
    sleep(1);
  }

  exit();
}
