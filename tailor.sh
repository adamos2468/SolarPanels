#!/bin/bash
python SolarDetection.py
python Failure.py -p ./Results/rotated.jpg ./Results/where.txt
