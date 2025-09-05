#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>

int main() {
  pid_t pid = fork();

  if (pid < 0) {
    // Fork failed
    perror("fork failed!\n");
    return 1;
  } else if (pid == 0) {
    // Child process
    printf("Child (PID: %d): Parent PID is %d\n", getpid(), getppid());

    sleep(8);

    printf("Child (PID: %d): I am now an orphan, new Parent PID is %d\n",
           getpid(), getppid());
  } else {
    // Parent process
    printf("Parent (PID: %d): Child PID is %d\n", getpid(), pid);

    // Parent exits immediately to make child an orphan
    return 0;
  }

  return 0;
}
