#!/bin/bash

./sssp -f ../piccolo/data/uci-uni/uci-uni.el -r 1024108 -n 1 -v -o ../piccolo/data/uci-uni
wait

./sssp -f ../piccolo/data/sinaweibo/sinaweibo.el -r 147870 -n 1 -v -o ../piccolo/data/sinaweibo
wait

./sswp -f ../piccolo/data/uci-uni/uci-uni.el -r 1024108 -n 1 -v -o ../piccolo/data/uci-uni
wait

./sswp -f ../piccolo/data/sinaweibo/sinaweibo.el -r 147870 -n 1 -v -o ../piccolo/data/sinaweibo
wait

./sswp -f ../piccolo/data/konect/konect.el -r 14502684 -n 1 -o ../piccolo/data/konect

