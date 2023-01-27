#!/bin/bash
bash version.sh
git status
git commit -m "updating current time"
git add version.sh
git push

