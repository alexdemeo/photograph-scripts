#!/bin/zsh

HDR_SCRIPT_OUTPUT_DIR=~/Desktop/ca-hdr
EXTRACT_SCRIPT_OUTPUT_DIR=~/Desktop/ca-hdr-2

for f in $(find $HDR_SCRIPT_OUTPUT_DIR | grep mertens.JPG); do
    p="${f%/*}" # remove the file part
    p="${p##*/}" # get the last part of the path
    cp -p $f "$EXTRACT_SCRIPT_OUTPUT_DIR/$p.JPG"
done
