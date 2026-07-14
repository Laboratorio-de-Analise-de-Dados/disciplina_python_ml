#!/bin/bash

# tree -a -f > directories.txt

git add .
git commit -m "$1"
git push -u origin main