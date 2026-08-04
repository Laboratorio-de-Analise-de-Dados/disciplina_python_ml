#!/bin/bash

cmd //c tree //F >> directories.txt 

git add .
git commit -m "$1"
git push -u origin main