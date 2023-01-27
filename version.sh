#!/bin/bash
git status
time = $(date +"%T")
echo "The current time is $time"
git commit -m "updating current time"
git add version.sh
git push

