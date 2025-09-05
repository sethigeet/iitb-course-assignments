#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

int main() {
  pid_t pid = fork();

  if (pid < 0) {
    // Fork failed
    perror("fork failed!\n");
    return 1;
  } else if (pid == 0) {
    // Child process
    printf("Child: My process ID is: %d\n", getpid());
    printf("Child: My parent process ID is: %d\n", getppid());
    return 42;
  } else {
    // Parent process
    printf("Parent: My process ID is: %d\n", getpid());
    printf("Parent: The child process ID is: %d\n", pid);

    pid_t status;
    wait(&status);
    if (status < 0) {
      perror("wait failed!\n");
      return 1;
    }
    printf("Parent: Child process exited with status: %d\n",
           WEXITSTATUS(status));
  }

  return 0;
}
