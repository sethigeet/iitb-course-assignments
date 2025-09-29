#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define BUFFER_SIZE 1024

int main(int argc, char *argv[]) {
  if (argc != 2) {
    write(STDERR_FILENO, "Usage: ./q1 <filename>\n", 23);
    exit(1);
  }

  int fd = open(argv[1], O_RDONLY);
  if (fd < 0) {
    write(STDERR_FILENO, "Error: Could not open file\n", 27);
    exit(1);
  }

  char buffer[BUFFER_SIZE];
  ssize_t bytesRead;

  int lines = 0, words = 0, characters = 0;
  int in_word = 0;

  while ((bytesRead = read(fd, buffer, BUFFER_SIZE)) > 0) {
    characters += bytesRead;
    for (ssize_t i = 0; i < bytesRead; i++) {
      char c = buffer[i];

      if (c == '\n') {
        lines++;
      }

      if (c == ' ') {
        in_word = 0;
      } else if (!in_word) {
        in_word = 1;
        words++;
      }
    }
  }

  if (bytesRead < 0) {
    write(2, "Error: Failed to read file\n", 27);
    close(fd);
    exit(1);
  }

  close(fd);

  char output[128];
  int len =
      snprintf(output, sizeof(output), "Lines: %d\nWords: %d\nCharacters: %d\n",
               lines, words, characters);
  write(STDOUT_FILENO, output, len);

  return 0;
}
