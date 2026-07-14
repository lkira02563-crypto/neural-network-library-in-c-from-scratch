#include "matrix.h"
#include "prng.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define sigf(x) (1.0f / (1.0f + expf(-(x))))
typedef struct {
  matrix x;
  matrix w1;
  matrix b1;
  matrix a1;
  matrix w2;
  matrix b2;
  matrix a2;
  matrix y;

} xor;
xor xor_alloc() {
  xor m;
  m.x = mat_alloc(1, 3);
  m.w1 = mat_alloc(3, 4);
  m.b1 = mat_alloc(1, 4);
  m.a1 = mat_alloc(1, 4);
  m.w2 = mat_alloc(4, 1);
  m.b2 = mat_alloc(1, 1);
  m.a2 = mat_alloc(1, 1);
  m.y = mat_alloc(1, 1);

  return m;
}
void gardient_descent(xor m, float rate, float eps, matrix train_in,
                      matrix train_out, size_t iter);
void finit_differnces(xor m, xor grad, float eps, matrix train_in,

                      matrix train_out);
float train_set[] = {0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0,
                     1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1};
float mse(xor m, matrix train_in, matrix train_out);
void forward(xor xor);
void mat_sigf(matrix a);
void mat_cpy(matrix dst, matrix src);
matrix mat_row(matrix m, size_t row);
int main(void) {

  matrix train_in = {.cols = 3, .rows = 8, .stride = 4, .data = train_set};

  matrix train_out = {.cols = 1, .rows = 8, .stride = 4, .data = train_set + 3};

  srand(time(NULL));

  xor xor = xor_alloc();
  mat_rand(xor.w1, 0.0f, 1.0f);
  mat_rand(xor.b1, 0.0f, 1.0f);
  mat_rand(xor.w2, 0.0f, 1.0f);
  mat_rand(xor.b2, 0.0f, 1.0f);
  printf("MSE BEFORE: %f\n", mse(xor, train_in, train_out));
  gardient_descent(xor, 5e-1, 1e-1, train_in, train_out, 1000 * 1000);
  printf("MSE  AFTER: %f\n", mse(xor, train_in, train_out));

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) {
      for (size_t k = 0; k < 2; k++) {
        MAT_AT(xor.x, 0, 0) = i;
        MAT_AT(xor.x, 0, 1) = j;
        MAT_AT(xor.x, 0, 2) = k;
        forward(xor);

        printf("%zu * %zu * %zu = %f\n", i, j, k, MAT_AT(xor.y, 0, 0));
      }
    }
  }
  return 0;
}

void mat_sigf(matrix a) {
  for (size_t i = 0; i < a.rows; i++) {

    for (size_t j = 0; j < a.cols; j++) {
      MAT_AT(a, i, j) = sigf(MAT_AT(a, i, j));
    }
  }
}
void mat_cpy(matrix dst, matrix src) {
  assert(dst.rows == src.rows);
  assert(dst.cols == src.cols);
  for (size_t i = 0; i < dst.rows; i++) {
    for (size_t j = 0; j < dst.cols; j++) {
      MAT_AT(dst, i, j) = MAT_AT(src, i, j);
    }
  }
}
void forward(xor xor) {
  mat_dot(xor.a1, xor.x, xor.w1);
  mat_sum(xor.a1, xor.b1);
  mat_sigf(xor.a1);

  mat_dot(xor.a2, xor.a1, xor.w2);
  mat_sum(xor.a2, xor.b2);
  mat_sigf(xor.a2);
  mat_cpy(xor.y, xor.a2);
}
matrix mat_row(matrix m, size_t row) {

  return (matrix){.rows = 1,
                  .cols = m.cols,
                  .stride = m.stride,
                  .data = &MAT_AT(m, row, 0)};
}
float mse(xor m, matrix train_in, matrix train_out) {
  assert(train_in.rows == train_out.rows);
  assert(train_in.cols == m.x.cols);
  assert(train_out.cols == m.y.cols);

  float result = 0.0f;
  for (size_t i = 0; i < train_in.rows; i++) {
    matrix x = mat_row(train_in, i);
    matrix y = mat_row(train_out, i);
    mat_cpy(m.x, x);
    forward(m);
    for (size_t j = 0; j < train_out.cols; j++) {
      float diff = MAT_AT(m.y, 0, j) - MAT_AT(y, 0, j);
      result += diff * diff;
    }
  }
  return result / train_in.rows;
}

void finit_differnces(xor m, xor grad, float eps, matrix train_in,
                      matrix train_out) {

  float loss = mse(m, train_in, train_out);
  float saved;
  for (size_t i = 0; i < m.w1.rows; ++i) {
    for (size_t j = 0; j < m.w1.cols; j++) {

      saved = MAT_AT(m.w1, i, j);
      MAT_AT(m.w1, i, j) += eps;
      MAT_AT(grad.w1, i, j) = (mse(m, train_in, train_out) - loss) / eps;
      MAT_AT(m.w1, i, j) = saved;
    }
  }
  for (size_t i = 0; i < m.b1.rows; i++) {
    for (size_t j = 0; j < m.b1.cols; j++) {
      saved = MAT_AT(m.b1, i, j);
      MAT_AT(m.b1, i, j) += eps;
      MAT_AT(grad.b1, i, j) = (mse(m, train_in, train_out) - loss) / eps;
      MAT_AT(m.b1, i, j) = saved;
    }
  }
  for (size_t i = 0; i < m.w2.rows; i++) {
    for (size_t j = 0; j < m.w2.cols; j++) {
      saved = MAT_AT(m.w2, i, j);
      MAT_AT(m.w2, i, j) += eps;
      MAT_AT(grad.w2, i, j) = (mse(m, train_in, train_out) - loss) / eps;
      MAT_AT(m.w2, i, j) = saved;
    }
  }
  for (size_t i = 0; i < m.b2.rows; i++) {
    for (size_t j = 0; j < m.b2.cols; j++) {
      saved = MAT_AT(m.b2, i, j);
      MAT_AT(m.b2, i, j) += eps;
      MAT_AT(grad.b2, i, j) = (mse(m, train_in, train_out) - loss) / eps;
      MAT_AT(m.b2, i, j) = saved;
    }
  }
}
void gardient_descent(xor m, float rate, float eps, matrix train_in,
                      matrix train_out, size_t iter) {
  xor grad = xor_alloc();
  for (size_t i = 0; i < iter; i++) {
    finit_differnces(m, grad, eps, train_in, train_out);
    for (size_t j = 0; j < m.w1.rows; j++) {
      for (size_t k = 0; k < m.w1.cols; k++) {
        MAT_AT(m.w1, j, k) -= rate * MAT_AT(grad.w1, j, k);
      }
    }
    for (size_t j = 0; j < m.b1.rows; j++) {
      for (size_t k = 0; k < m.b1.cols; k++) {
        MAT_AT(m.b1, j, k) -= rate * MAT_AT(grad.b1, j, k);
      }
    }
    for (size_t j = 0; j < m.w2.rows; j++) {
      for (size_t k = 0; k < m.w2.cols; k++) {
        MAT_AT(m.w2, j, k) -= rate * MAT_AT(grad.w2, j, k);
      }
    }
    for (size_t j = 0; j < m.b2.rows; j++) {
      for (size_t k = 0; k < m.b2.cols; k++) {
        MAT_AT(m.b2, j, k) -= rate * MAT_AT(grad.b2, j, k);
      }
    }
  }
}
