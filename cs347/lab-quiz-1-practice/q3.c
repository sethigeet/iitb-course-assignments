#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int main() {
  int pipefd[2]; // pipefd[0] is read end, pipefd[1] is write end

  if (pipe(pipefd) == -1) {
    perror("pipe");
    exit(EXIT_FAILURE);
  }

  // First child: runs `ls`
  pid_t child1 = fork();
  if (child1 == -1) {
    perror("fork");
    exit(EXIT_FAILURE);
  }

  if (child1 == 0) {
    // Redirect stdout to write end of pipe
    dup2(pipefd[1], STDOUT_FILENO);
    close(pipefd[0]); // Close unused read end
    close(pipefd[1]); // Close original write end
    execlp("ls", "ls", NULL);
    perror("execlp ls");
    exit(EXIT_FAILURE);
  }

  // Second child: runs `wc -l`
  pid_t child2 = fork();
  if (child2 == -1) {
    perror("fork");
    exit(EXIT_FAILURE);
  }

  if (child2 == 0) {
    // Redirect stdin to read end of pipe
    dup2(pipefd[0], STDIN_FILENO);
    close(pipefd[1]); // Close unused write end
    close(pipefd[0]); // Close original read end
    execlp("wc", "wc", "-l", NULL);
    perror("execlp wc");
    exit(EXIT_FAILURE);
  }

  // Parent: close both ends and wait
  close(pipefd[0]);
  close(pipefd[1]);

  waitpid(child1, NULL, 0);
  waitpid(child2, NULL, 0);

  return 0;
}
