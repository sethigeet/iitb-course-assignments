#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define BUFFER_SIZE 1024

int main(int argc, char *argv[]) {
  if (argc != 2) {
    write(STDERR_FILENO, "Usage: ./q2 <filename>\n", 24);
    exit(1);
  }

  // Open the file for writing (create if not exist, truncate if exists)
  int file_fd = open(argv[1], O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (file_fd < 0) {
    perror("unable to open file");
    exit(1);
  }

  char buffer[BUFFER_SIZE];
  ssize_t bytes_read;

  // Read from stdin
  while ((bytes_read = read(STDIN_FILENO, buffer, BUFFER_SIZE)) > 0) {
    // Write to stdout
    if (write(STDOUT_FILENO, buffer, bytes_read) != bytes_read) {
      perror("unable to write to stdout");
      close(file_fd);
      exit(1);
    }

    // Write to file
    if (write(file_fd, buffer, bytes_read) != bytes_read) {
      perror("unable to write to file");
      close(file_fd);
      exit(1);
    }
  }

  if (bytes_read < 0) {
    perror("error while reading from stdin");
  }

  close(file_fd);
  return 0;
}
