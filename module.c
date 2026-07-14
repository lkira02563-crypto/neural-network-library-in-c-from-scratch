/*
 * module.c - simple 2-layer neural network (XOR example)
 * Location: /home/jeanmath/projects_c/machine_learning
 *
 * This file implements a tiny feedforward network, MSE loss and a numeric
 * gradient-based trainer (finite differences). It's intended as a minimal
 * educational example — not optimized for performance.
 */

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
} xor_net;

xor_net xor_alloc(void) {
  xor_net m;
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

/* Forward pass */
void mat_sigf(matrix a) {
  for (size_t i = 0; i < a.rows; i++)
    for (size_t j = 0; j < a.cols; j++)
      MAT_AT(a, i, j) = sigf(MAT_AT(a, i, j));
}

void mat_cpy(matrix dst, matrix src) {
  assert(dst.rows == src.rows && dst.cols == src.cols);
  for (size_t i = 0; i < dst.rows; i++)
    for (size_t j = 0; j < dst.cols; j++)
      MAT_AT(dst, i, j) = MAT_AT(src, i, j);
}

void forward(xor_net *xn) {
  mat_dot(xn->a1, xn->x, xn->w1);
  mat_sum(xn->a1, xn->b1);
  mat_sigf(xn->a1);
  mat_dot(xn->a2, xn->a1, xn->w2);
  mat_sum(xn->a2, xn->b2);
  mat_sigf(xn->a2);
  mat_cpy(xn->y, xn->a2);
}

matrix mat_row(matrix m, size_t row) {
  return (matrix){.rows = 1, .cols = m.cols, .stride = m.stride,
                   .data = &MAT_AT(m, row, 0)};
}

float mse(xor_net *xn, matrix train_in, matrix train_out) {
  assert(train_in.rows == train_out.rows);
  assert(train_in.cols == xn->x.cols);
  assert(train_out.cols == xn->y.cols);
  float result = 0.0f;
  for (size_t i = 0; i < train_in.rows; i++) {
    matrix x = mat_row(train_in, i);
    matrix y = mat_row(train_out, i);
    mat_cpy(xn->x, x);
    forward(xn);
    for (size_t j = 0; j < train_out.cols; j++) {
      float diff = MAT_AT(xn->y, 0, j) - MAT_AT(y, 0, j);
      result += diff * diff;
    }
  }
  return result / (float)train_in.rows;
}

/* Numeric gradient (finite differences) */
void finite_differences(xor_net *xn, xor_net *grad, float eps,
                        matrix train_in, matrix train_out) {
  float base = mse(xn, train_in, train_out);
  float saved;
  for (size_t i = 0; i < xn->w1.rows; ++i)
    for (size_t j = 0; j < xn->w1.cols; ++j) {
      saved = MAT_AT(xn->w1, i, j);
      MAT_AT(xn->w1, i, j) = saved + eps;
      MAT_AT(grad->w1, i, j) = (mse(xn, train_in, train_out) - base) / eps;
      MAT_AT(xn->w1, i, j) = saved;
    }

  for (size_t i = 0; i < xn->b1.rows; ++i)
    for (size_t j = 0; j < xn->b1.cols; ++j) {
      saved = MAT_AT(xn->b1, i, j);
      MAT_AT(xn->b1, i, j) = saved + eps;
      MAT_AT(grad->b1, i, j) = (mse(xn, train_in, train_out) - base) / eps;
      MAT_AT(xn->b1, i, j) = saved;
    }

  for (size_t i = 0; i < xn->w2.rows; ++i)
    for (size_t j = 0; j < xn->w2.cols; ++j) {
      saved = MAT_AT(xn->w2, i, j);
      MAT_AT(xn->w2, i, j) = saved + eps;
      MAT_AT(grad->w2, i, j) = (mse(xn, train_in, train_out) - base) / eps;
      MAT_AT(xn->w2, i, j) = saved;
    }

  for (size_t i = 0; i < xn->b2.rows; ++i)
    for (size_t j = 0; j < xn->b2.cols; ++j) {
      saved = MAT_AT(xn->b2, i, j);
      MAT_AT(xn->b2, i, j) = saved + eps;
      MAT_AT(grad->b2, i, j) = (mse(xn, train_in, train_out) - base) / eps;
      MAT_AT(xn->b2, i, j) = saved;
    }
}

/* Gradient descent using numeric gradients */
void gradient_descent(xor_net *xn, float rate, float eps, matrix train_in,
                      matrix train_out, size_t iter) {
  xor_net grad = xor_alloc();
  for (size_t it = 0; it < iter; ++it) {
    finite_differences(xn, &grad, eps, train_in, train_out);
    for (size_t i = 0; i < xn->w1.rows; ++i)
      for (size_t j = 0; j < xn->w1.cols; ++j)
        MAT_AT(xn->w1, i, j) -= rate * MAT_AT(grad.w1, i, j);

    for (size_t i = 0; i < xn->b1.rows; ++i)
      for (size_t j = 0; j < xn->b1.cols; ++j)
        MAT_AT(xn->b1, i, j) -= rate * MAT_AT(grad.b1, i, j);

    for (size_t i = 0; i < xn->w2.rows; ++i)
      for (size_t j = 0; j < xn->w2.cols; ++j)
        MAT_AT(xn->w2, i, j) -= rate * MAT_AT(grad.w2, i, j);

    for (size_t i = 0; i < xn->b2.rows; ++i)
      for (size_t j = 0; j < xn->b2.cols; ++j)
        MAT_AT(xn->b2, i, j) -= rate * MAT_AT(grad.b2, i, j);
  }
}

/* Example runner (keeps previous behavior) */
float train_set[] = {
    0.0f, 0.0f, 0.0f, 0.0f, /* in0 */
    0.0f, 1.0f, 0.0f, 0.0f, /* in1 */
    0.0f, 0.0f, 1.0f, 0.0f, /* in2 */
    0.0f, 1.0f, 1.0f, 1.0f, /* in3 */
    1.0f, 0.0f, 0.0f, 0.0f, /* ... */
    1.0f, 0.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 0.0f, 0.0f,
    1.0f, 1.0f, 1.0f, 1.0f};

int main(void) {
  matrix train_in = {.cols = 3, .rows = 8, .stride = 4, .data = train_set};
  matrix train_out = {.cols = 1, .rows = 8, .stride = 4, .data = train_set + 3};

  srand((unsigned)time(NULL));
  xor_net xn = xor_alloc();
  mat_rand(xn.w1, -1.0f, 1.0f);
  mat_rand(xn.b1, -1.0f, 1.0f);
  mat_rand(xn.w2, -1.0f, 1.0f);
  mat_rand(xn.b2, -1.0f, 1.0f);

  printf("MSE BEFORE: %f\n", mse(&xn, train_in, train_out));
  gradient_descent(&xn, 5e-1f, 1e-3f, train_in, train_out, 2000);
  printf("MSE AFTER: %f\n", mse(&xn, train_in, train_out));

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 2; ++k) {
        MAT_AT(xn.x, 0, 0) = (float)i;
        MAT_AT(xn.x, 0, 1) = (float)j;
        MAT_AT(xn.x, 0, 2) = (float)k;
        forward(&xn);
        printf("%d * %d * %d = %f\n", i, j, k, MAT_AT(xn.y, 0, 0));
      }
  return 0;
}
