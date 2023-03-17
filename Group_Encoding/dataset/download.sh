#!/bin/bash

FILEID="1fBcerdcidxcKVDztHqUswhIQzx-py7Q1"
FILENAME="output_npy.zip"

curl -sc ~/cookie.txt "https://drive.google.com/uc?export=download&id=${FILEID}" > /dev/null
curl -Lb ~/cookie.txt "https://drive.google.com/uc?export=download&confirm=`awk '/_warning_/ {print $NF}' ~/cookie.txt`&id=${FILEID}" -o ${FILENAME}
unzip ${FILENAME}