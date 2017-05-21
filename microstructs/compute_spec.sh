#!/bin/bash

if [ $# -eq 0 ]
then
    echo "USAGE: ${0} filelist indir outdir tmpdir"
    exit 0
fi

filelist = ${1}
indir = ${2}
outdir = ${3}
tmpdir = ${4}

echo "Extracting features from ${filelist}"

w2feat = ./wave2feat/wave2feat


