#!/bin/bash

# Generate ctl files for force aligned phones/words.
# wzhao1 cs cmu edu
# 05/27/2017

if [[ $# -lt 6 ]]; then
    echo "USAGE: $0 [-indir] dir containing aligned phones/words
    [-type] phone or word
    [-phase] train or test
    [-ext] extension for phone/word file
    [-tmpdir] temporary dir
    [-outdir] dir to store outputed phone/word ctls
    "
    exit 0
fi

while [[ $# -gt 1 ]]; do
    key="$1"
    case "$key" in
        -indir)
            INDIR="$2"
            shift
            ;;
        -type)
            TYPE="$2"
            shift
            ;;
        -phase)
            PHASE="$2"
            shift
            ;;
        -ext)
            EXT="$2"
            shift
            ;;
        -tmpdir)
            TMPDIR="$2"
            shift
            ;;
        -outdir)
            OUTDIR="$2"
            shift
            ;;
        *)
            echo "Unknown option!"
            ;;
    esac
    shift
done

[[ -e "${INDIR}/${PHASE}" ]] || (echo "${INDIR}/${PHASE} not found"; exit 1)
# Make list of aligned phones/words to tmp file
tmp_ls=$( mktemp --tmpdir=${TMPDIR} tmp.XXXX ) # make tmp file

find "${INDIR}/${PHASE}" -name "*.${EXT}" > "$tmp_ls" # find in train or test

tmpf=$( mktemp --tmpdir=${TMPDIR} tmp.XXXX )

function clean_up { # clean up tmp file
    rm -rf $tmp_ls $tmpf
    exit 0
}
trap clean_up 0 1 2 3 # clean up on exit

# Collect phones/words and corresponding aligned frames
for f in $( cat "$tmp_ls" ); do
    awk -v a=$f 'NR>1 && NF>=4 {print a,$1,$2,$4;}' $f | \
        sed -e "s+${INDIR}\/${PHASE}++g" -e "s+\.${EXT}++g" \
        >> $tmpf
done

echo "Successfully collected ${TYPE}s and aligned frames to $tmpf."

# Replace special characters in <s>, <sil>, </s> and X()
# if this is disabled the words above appear as-is, e.g. <sil>.ctl, ZERO(2).ctl
awk \
    '{gsub("\\(","@"); gsub("\\)","@"); gsub("<s>","bSIL"); \
    gsub("<sil>","midSIL"); gsub("</s>","eSIL"); print $0;}' \
    $tmpf > "${tmpf}_2"

mv "${tmpf}_2" "$tmpf"

# Sort phones/words and collect to CTLs
[[ -e $OUTDIR ]] || mkdir -p "$OUTDIR"

cnt=0
for phn in $( awk '{print $4;}' $tmpf | sort -u ); do
    echo "working on $phn ..."
    awk -v a=$phn '{if ($NF==a) print $1,$2,$3;}' $tmpf \
        > "${OUTDIR}/${phn}.ctl"
    cnt=$(( cnt+1 ))
done

echo "Successfully collected all $cnt ${TYPE}s to ${OUTDIR}."

exit 0
