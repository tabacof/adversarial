#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

FILE   *fp = NULL;
size_t  fp_pos = 0;

int init(const char *fpath) {
   fp = fopen(fpath, "r");
   fp_pos = 0;
   if (fp == NULL) {
      printf("ERROR: could not find the file (%s)\n", fpath);
      return 1;
   }
   return 0;
}

int close(void) {
   if (fp != NULL) {
      fclose(fp);
      fp = NULL;
   }
   return 0;
}

int read(float *storage, long offset, long length) {
   const int wordsize = sizeof(float);

   if (fp != NULL) {
      if (fp_pos != offset*wordsize)
        fseek(fp, offset*wordsize, SEEK_SET);
      assert(fread(storage, wordsize, length, fp) == length);
      fp_pos += length*wordsize;
      return 0;
   } else {
      printf("ERROR: could not read the file pointer\n");
      return 1;
   }
}

int print(long length) {
   const int wordsize = sizeof(float);
   int i;
   float buffer;

   for (i = 0; i < length; i++) {
      if (fp == NULL) {
         return 1;
      } else {
         fread(&buffer, wordsize, 1, fp);
         printf("%f\n", buffer);
      }
   }
   return 0;
}
