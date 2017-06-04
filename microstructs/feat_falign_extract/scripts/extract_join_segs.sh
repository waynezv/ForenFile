#!/bin/bash

# Extract wav segments from aligned phone/word CTL, optionally join them.

if [[ $# -lt 7 ]]; then
    echo "USAGE: $0
    [-ctl] CTL file containing wav path and aligned frames
    [-inwav] Dir containing wavs
    [-ext] Wav file extension
    [-outwav] Dir to store outputed wavs
    [-join] true or false to join
    [-jpath] path to store joined wav
    [-jnum] number of wavs to join (in order)"
    exit 0
fi

while [[ $# -gt 1 ]]; do
    key="$1"
    case $key in
        -ctl)
            CTL="$2"
            shift
            ;;
        -inwav)
            INWAV="$2"
            shift
            ;;
        -ext)
            EXT="$2"
            shift
            ;;
        -outwav)
            OUTWAV="$2"
            shift
            ;;
        -join)
            JN="$2"
            shift
            ;;
        -jpath)
            JPATH="$2"
            shift
            ;;
        -jnum)
            JNUM="$2"
            shift
            ;;
        *)
            echo "Unknown option!"
            ;;
    esac
    shift
done

[[ -e "$CTL" ]] || (echo "$CTL not found"; exit 1)
[[ -e "$INWAV" ]] || (echo "$INWAV not found"; exit 1)

basen=$( basename "$CTL" | sed "s+.ctl++g")
OUTDIR="$OUTWAV/$basen"
[[ -e "$OUTDIR" ]] || mkdir -p "$OUTDIR"

# Extract wav segments in CTL
awk -v a=$INWAV -v b=$EXT -v c="$OUTDIR" \
    '{cmd = sprintf("sox %s%s.%s %s/%s.wav \
    trim %s %s fade t %s %s %s\n", \
    a, $1, b, c, NR, \
    $2/100, ($3-$2)/100, ($3-$2)/1200, ($3-$2)/100, ($3-$2)/1200); \
    print cmd; \
    system(cmd); \
    }' $CTL

echo "Processed $( ls $OUTDIR | wc -l ) wav files."

# Join wav files in CTL
if [[ $JN == "true" ]]; then
    [[ -e "$JPATH" ]] || (echo "$JPATH not found"; exit 1)

    files=( $( awk -v c=$OUTDIR '{printf("%s/%s.wav ", c, NR);}' $CTL | sort -n ) )

    idx=0
    while [[ $idx -lt $JNUM ]]; do
        jn_ls+=( ${files[idx++]} )
        # jn_ls[idx++]=${files[idx]}
    done

sox ${jn_ls[@]} -t .wav ${JPATH}/${basen}.wav

echo "Successfully join ${JNUM} wav segments for $basen to ${JPATH}/${basen}.wav."

fi

exit 0
