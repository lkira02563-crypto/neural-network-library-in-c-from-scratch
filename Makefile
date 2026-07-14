# Simple Makefile for the tiny ML C project

CC = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra
LDFLAGS = -lm
SRCS = module.c matrix.c 
OBJS = $(SRCS:.c=.o)
TARGET = ml_example

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
