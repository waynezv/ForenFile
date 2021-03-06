#!/bin/bash

# Compute MFCC / log spectrogram features from audios.
# wzhao1 cs cmu edu
# 05/20/2017

if [[ $# -lt 5 ]]; then
    echo "USAGE: $0 [--type] feat_type (logspec / mfcc)
    [-f] ctl file containing list of wavs to be processed
    [-i] indir [-o] outdir [-t] tmpdir"

    exit 0
fi

# Pass arguments
while [[ $# -gt 1 ]]; do # if $# -gt 0, then can deal with single arg
    key="$1"

    case "$key" in
        --type)
            FEATTYPE="$2" # type of feature to extract
            shift # past argument
            ;;
        -f)
            FILELIST="$2" # files to be processed
            shift
            ;;
        -i)
            INDIR="$2" # input dir
            shift
            ;;
        -o)
            OUTDIR="$2" # output dir
            shift
            ;;
        -t)
            TMPDIR="$2" # temporary dir
            shift
            ;;
        *)
            echo "Unknown option!"
            ;;
    esac

    shift
done

# Check availability of necessary files
[[ -f "$FILELIST" ]] || ( echo "$FILELIST not found"; exit 1 )
[[ -d "$INDIR" ]] || ( echo "$INDIR not found"; exit 1 )
[[ -d "$OUTDIR" ]] || ( echo "$OUTDIR not found"; exit 1 )
[[ -d "$TMPDIR" ]]|| ( echo "$TMPDIR not found"; exit 1 )
[[ -e "./wave2feat/wave2feat" ]] || ( echo "wave2feat executable not found!"; exit 1 )

w2feat=./wave2feat/wave2feat # executable converting wav to feat

TMPFILE=$( mktemp --tmpdir=${TMPDIR} tmp.XXXX ) # make tmp file

function clean_up { # clean up tmp file
    rm -rf "$TMPFILE"
    exit 0
}
trap clean_up 0 1 2 3 # clean up on exit

echo "Extracting ${FEATTYPE} features for ${FILELIST} ..."

if [[ "$FEATTYPE" == "logspec" ]]; then
    for f in $( cat "${FILELIST}" ); do
        infile="${INDIR}/${f}"
        [ -f "${infile}" ] || ( echo "${infile} not found"; exit 1 )

        DIR=$( dirname "$f" )
        BASE=$( basename "$f" )
        outpath="${OUTDIR}/${DIR}"
        [[ -e "$outpath" ]] || mkdir -p "$outpath"
        outfile="${outpath}/${BASE%.*}.logspec"
        [ ! -f "${outfile}" ] || echo "${outfile} duplicated"

        # sox -t .flac ${f} -t .wav ${TMPFILE} # format conversion

        # NOTE: look at wave2feat for detailed instructions
        ${w2feat} -i ${infile} \
            -o ${outfile} \
            -nchans 1 \
            -mswav \
            -srate 16000 \
            -frate 100 \
            -alpha 0.970 \
            -wlen 0.0256 \
            -nfft 512 \
            -dither \
            -logspec

        echo "processed $infile to $outfile"
    done

elif [[ "$FEATTYPE" == "mfcc" ]]; then
    for f in $( cat "${FILELIST}" ); do
        infile="${INDIR}/${f}"
        [ -f "${infile}" ] || ( echo "${infile} not found"; exit 1 )

        DIR=$( dirname "$f" )
        BASE=$( basename "$f" )
        outpath="${OUTDIR}/${DIR}"
        [[ -e "$outpath" ]] || mkdir -p "$outpath"
        outfile="${outpath}/${BASE%.*}.80-7200-40f.mfc"
        [ ! -f "${outfile}" ] || echo "${outfile} duplicated"

        # sox -t .flac ${f} -t .wav ${TMPFILE} # format conversion

        ${w2feat} -i ${infile} \
            -o ${outfile} \
            -mswav \
            -srate 16000 \
            -frate 100 \
            -lowerf 80 \
            -upperf 7200 \
            -nfft 512 \
            -dither \
            -nfilt 40 \
            -ncep 13 # number of cep

        echo "processed $infile to $outfile"
    done
fi

exit 0
