#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
  int fd1 = open("test1.txt", O_RDONLY);
  if (fd1 < 0) {
    perror("unable to open file1.txt\n");
    return 1;
  }

  int fd2 = open("test2.txt", O_RDONLY);
  if (fd2 < 0) {
    perror("unable to open file2.txt\n");
    close(fd1);
    return 1;
  }

  char msg[128];
  snprintf(msg, sizeof(msg), "test1.txt fd: %d\n", fd1);
  write(STDOUT_FILENO, msg, strlen(msg));

  snprintf(msg, sizeof(msg), "test2.txt fd: %d\n", fd2);
  write(STDOUT_FILENO, msg, strlen(msg));

  if (close(fd1) < 0) {
    perror("unable to close test1.txt\n");
    close(fd2);
    return 1;
  }
  write(STDOUT_FILENO, "closed test1.txt\n", 18);

  int fd3 = open("test3.txt", O_RDONLY);
  if (fd3 < 0) {
    perror("unable to open file3.txt\n");
    close(fd1);
    close(fd2);
    return 1;
  }

  snprintf(msg, sizeof(msg), "test3.txt fd: %d\n", fd3);
  write(STDOUT_FILENO, msg, strlen(msg));

  pid_t pid = fork();
  if (pid < 0) {
    perror("fork failed\n");
    close(fd2);
    close(fd3);
    return 1;
  } else if (pid == 0) {
    // Child process
    char buffer[128];
    int bytesRead = read(fd2, buffer, sizeof(buffer) - 1);
    if (bytesRead < 0) {
      perror("unable to read test2.txt\n");
      close(fd2);
      close(fd3);
      return 1;
    }

    // Null-terminate the string and print it
    buffer[bytesRead] = '\0';
    snprintf(msg, sizeof(msg), "Child read from test2.txt: %s\n", buffer);
    write(STDOUT_FILENO, msg, strlen(msg));

    // Close file descriptors in the child process
    if (close(fd2) < 0) {
      perror("unable to close test2.txt\n");
      close(fd3);
      return 1;
    }
    write(STDOUT_FILENO, "Child: closed test2.txt\n", 24);

    if (close(fd3) < 0) {
      perror("unable to close test3.txt\n");
      return 1;
    }
    write(STDOUT_FILENO, "Child: closed test3.txt\n", 24);
  } else {
    // Parent process
    if (wait(NULL) < 0) {
      perror("error while waiting for child\n");
      close(fd2);
      close(fd3);
      return 1;
    }

    write(STDOUT_FILENO, "Parent: Child process finished\n", 31);

    if (close(fd2) < 0) {
      perror("unable to close test2.txt\n");
      close(fd3);
      return 1;
    }
    write(STDOUT_FILENO, "Parent: closed test2.txt\n", 25);

    if (close(fd3) < 0) {
      perror("unable to close test3.txt\n");
      return 1;
    }
    write(STDOUT_FILENO, "Parent: closed test3.txt\n", 25);
  }

  return 0;
}
