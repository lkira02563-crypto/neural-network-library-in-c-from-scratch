#include "matrix.h"
#include "prng.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

matrix mat_alloc(size_t rows, size_t cols) {
  matrix m;
  m.rows = rows;
  m.cols = cols;
  m.stride = cols;
  m.data = malloc(rows * cols * sizeof(*m.data));
  assert(m.data != NULL);
  return m;
}

void mat_free(matrix *m) {
  free(m->data);
  m->data = NULL;
  m->rows = 0;
  m->cols = 0;
}
// void mat_cpy(matrix dst, matrix src) {
//   assert(dst.rows == src.rows);
//   assert(dst.cols == src.cols);
//   for (size_t i = 0; i < dst.rows; i++) {
//     for (size_t j = 0; j < dst.cols; j++) {
//       MAT_AT(dst, i, j) = MAT_AT(src, i, j);
//     }
//   }
// }
void mat_print(matrix m) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      printf("%f ", MAT_AT(m, i, j));
    }
    printf("\n");
  }
}

float rand_float(void) { return (float)rand() / RAND_MAX; }

void mat_rand(matrix m, float min, float max) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      MAT_AT(m, i, j) = rand_float() * (max - min) + min;
    }
  }
}

void mat_fill(matrix m, float val) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      MAT_AT(m, i, j) = val;
    }
  }
}
void mat_sum(matrix dst, matrix a) {
  assert(dst.rows == a.rows);
  assert(dst.cols == a.cols);

  for (size_t i = 0; i < dst.rows; i++) {
    for (size_t j = 0; j < dst.cols; j++) {
      MAT_AT(dst, i, j) += MAT_AT(a, i, j);
    }
  }
}

void mat_dot(matrix dst, matrix a, matrix b) {
  assert(dst.rows == a.rows);
  assert(dst.cols == b.cols);
  assert(a.cols == b.rows);

  for (size_t i = 0; i < dst.rows; i++) {
    for (size_t j = 0; j < dst.cols; j++) {

      MAT_AT(dst, i, j) = 0.0f;

      for (size_t k = 0; k < a.cols; k++) {
        MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
      }
    }
  }
}
