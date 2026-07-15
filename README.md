# Machine Learning (C)

A tiny, educational machine-learning library and examples written in C — by jeanmath.

Overview

I'm a C programmer and this repo contains a minimal feedforward neural-network example
(and small supporting utilities) intended for learning and experimentation, not
production use. The core demonstrates a 2-layer network, a forward pass, MSE loss,
and a simple numeric-gradient trainer (finite differences).

Environment / Context (my local dev machine)

- OS: Linux Ubuntu 24.04.4 LTS (amd64)
- User: jeanmath
- Working directory when editing: /home/jeanmath/projects_c/machine_learning
- Typical dev shell: bash 5.2.21(1)-release

Files

- module.c      — 2-layer neural net example (forward, MSE, numeric gradients, trainer)
- matrix.c/h    — matrix utilities used by the network (matrix.h is part of the public headers)
- xor.c         — XOR example harness
- README.md     — this document

Build (Make)

Run:

  make

This builds the example binary `ml_example` from the core sources.

Makefile (targets):
- all / ml_example — build the example
- clean — remove object files and binaries

Manual gcc command:

  gcc -O2 -std=c11 -Wall -Wextra module.c matrix.c prng.c -o ml_example -lm

Usage

Run the example binary to print MSE before/after training and sample predictions:

  ./ml_example

Notes

- The trainer uses finite differences to compute gradients. It's extremely slow
  for larger networks — this is deliberate for clarity. Replace with analytic
  backpropagation for performance.- The code uses single-precision floats. Adjust types if you need double precision.
- For brevity some allocations are not freed; if integrating into long-running
  applications, add proper free() calls or use an arena allocator.



