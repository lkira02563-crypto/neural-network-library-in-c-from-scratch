#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

typedef struct {
  size_t rows;
  size_t cols;
  float *data;
  size_t stride;
} matrix;

#define MAT_AT(m, i, j) (m).data[(i) * (m).stride + (j)]

matrix mat_alloc(size_t rows, size_t cols);
void mat_free(matrix *m);

void mat_print(matrix m);

float rand_float(void);
void mat_rand(matrix m, float min, float max);
void mat_fill(matrix m, float val);
void mat_cpy(matrix dst, matrix src);
void mat_sum(matrix dst, matrix a);
void mat_subs(matrix dst, matrix a);
void mat_dot(matrix dst, matrix a, matrix b);

#endif
