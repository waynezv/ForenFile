#!/bin/bash

# Generate ctl files for force aligned phone segments.
# wzhao1 cs cmu edu
# 05/27/2017

if [[ $# -lt 3 ]]; then
    echo "USAGE: $0 [-indir] dir containing phone segments
    [-tmpdir]
    [-outdir]
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

# Make list of phone segments
tmp_phseg_ls=$( mktemp --tmpdir=${TMPDIR} tmp.XXXX ) # make tmp file

find $INDIR -name '*.phseg' > "$tmp_phseg_ls"

tmpf=$( mktemp --tmpdir=${TMPDIR} tmp.XXXX ) # make tmp file

exit 0

# Clean up tmp files on exit
function clean_up { # clean up tmp file
    rm -rf "$tmp_phseg_ls" "${tmpf}" # quote to avoid spaces and *
    exit 0
}

trap clean_up 0 1 2 3 # clean up on exit

# Collect phones and frames
for f in $( cat "$tmp_phseg_ls" ); do
    awk -v a=$f 'NR>1 && NF>+4 {print a,$1,$2,$4;}' $f | \
        sed -e 's+\.phseg++g' \
        >> $tmpf
done

exit 0

# Replace special characters in <s>, <sil>, </s> and X()
# if this is disabled the words above appear as-is, e.g. <sil>.ctl, ZERO(2).ctl
awk \
    '{gsub("\\(","@"); gsub("\\)","@"); gsub("<s>","bSIL"); \
    gsub("<sil>","midSIL"); gsub("</s>","eSIL"); print $0;}' \
    $TMPFILE > "${TMPFILE}_2"

mv "${TMPFILE}_2" $TMPFILE

[[ -e $OUTDIR ]] || mkdir -p "$OUTDIR"

for phn in $(`awk '{print $4;}' $TMPFILE | sort -u`); do
    echo working on $phn...
    # No spreading of boundaries
    awk -v a=$phn '{if($NF==a) print $1,$2,$3;}' $TMPFILE \
        >! "${OUTDIR}/${phn}.ctl"

    # Boundaries spread by 10 frames
    # awk -v a=$phn '{if($NF==a) print $1,$2-10,$3+10;}' tmp.$$ >! phonectls/$phn.ctl

    # Correct the filenames in ctls if required
    # awk '{if (FILENAME==ARGV[1]) {split($NF,t,"/"); fn = sprintf("%s",t[length(t)]); idx[fn] = $0; } else { if ($1 in idx) print idx[$1],$2,$3;}}' $refctl phonectls/${phn}.ctl >! phonectls/${phn}_full.ctl
done

echo done...

exit 0

clean_up
