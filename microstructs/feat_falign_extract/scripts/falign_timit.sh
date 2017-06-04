#!/bin/bash

# Align MFCCs with transcripts to decode the correspondence between words / phones
# and audio frames for TIMIT data.
# wzhao1 cs cmu edu
# 05/27/2017

if [[ $# -lt 11 ]]; then
    echo "USAGE: $0 [--part] part [--npart] npart
    [--listbasedir] dir containing list files
    [--ctlf] file containing data list
    [--transf] file containing transcripts
    [--dictf] file containing dict
    [--fillerf] file containing filler dict
    [--mfcdir] dir containing mfcc
    [--outputdir] dir to store outputed logs, ctls, etc
    [--outsegdir] dir to store outputed phone/word segments
    [--jobname] job name"

    exit 0
fi

while [[ $# -gt 1 ]]; do # if $# -gt 0, then can deal with single arg
    key="$1"

    case "${key}" in
        --part)
            PART="$2"
            shift
            ;;
        --npart)
            NPART="$2"
            shift
            ;;
        --listbasedir)
            LISTBASE="$2"
            shift
            ;;
        --ctlf)
            CTL="$2"
            shift
            ;;
        --transf)
            TRANS="$2"
            shift
            ;;
        --dictf)
            DICT="$2"
            shift
            ;;
        --fillerf)
            FILL="$2"
            shift
            ;;
        --mfcdir)
            MFC="$2"
            shift
            ;;
        --outputdir)
            OUTDIR="$2"
            shift
            ;;
        --outsegdir)
            OUTSEG="$2"
            shift
            ;;
        --jobname)
            JOB="$2"
            shift
            ;;
        *)
            echo "Unknown option!"
            ;;
    esac
    shift
done

echo "Preparing for force alignment ..."

# Input setup
CTLFN="$LISTBASE/$CTL"
TRANSFN="$LISTBASE/$TRANS"
DICTFN="$LISTBASE/$DICT"
FILLFN="$LISTBASE/$FILL"
MFCDIR="$MFC"
CEPEXT="80-7200-40f.mfc"

[[ -e "$CTLFN" ]] || (echo "$CTLFN not found"; exit 1)
[[ -e "$TRANSFN" ]] || (echo "$TRANSFN not found"; exit 1)
[[ -e "$DICTFN" ]] || (echo "$DICTFN not found"; exit 1)
[[ -e "$FILLFN" ]] || (echo "$FILLFN not found"; exit 1)
[[ -e "$MFCDIR" ]] || (echo "$MFCDIR not found"; exit 1)

# Output setup
[[ -e "$OUTDIR" ]] || mkdir -p "$OUTDIR"

outtransfn="${OUTDIR}/${JOB}.faligned-${PART}.trans"
outctlfn="${OUTDIR}/${JOB}.faligned-${PART}.ctl"
logfn="${OUTDIR}/${JOB}.faligned-${PART}.log"

for f in $( cat $CTLFN | awk '{print $1;}' ); do
    dirhead=$( dirname $f )
    [[ -e "${OUTSEG}/${dirhead}" ]] || mkdir -p "${OUTSEG}/${dirhead}"
done

# Job setup
nlines=$( wc $CTLFN | awk '{print $1;}' ) # number of lines in ctl file
ctloffset=$(( ( $nlines * ($PART - 1) ) / $NPART ))
ctlcount=$(( (($nlines * $PART) / $NPART) - $ctloffset ))
echo "Doing $ctlcount segments starting at number $ctloffset ..."

# Model setup
modeldir="./models/ads/ads.80-7200-40f.1-3/ads.80-7200-40f.1-3.ci_continuous.8gau"
mdeffn="./models/ads/ads.80-7200-40f.1-3.ci.mdef"
[[ -e "$modeldir" ]] || (echo "model not found"; exit 1)
[[ -e "$mdeffn" ]] || (echo "model def not found"; exit 1)

# Decoder setup
decoder="./decoderlatest/bin/linux/s3align"
[[ -e "$decoder" ]] || (echo "$decoder not found"; exit 1)

# Decode
echo "Decoding ..."

$decoder \
    -logbase 1.0001 \
    -mdeffn $mdeffn \
    -senmgaufn .cont. \
    -meanfn ${modeldir}/means \
    -varfn ${modeldir}/variances \
    -mixwfn ${modeldir}/mixture_weights \
    -tmatfn ${modeldir}/transition_matrices \
    -feat 1s_c_d_dd \
    -topn 32 \
    -beam 1e-80 \
    -dictfn $DICTFN \
    -fdictfn $FILLFN \
    -ctlfn $CTLFN \
    -ctloffset $ctloffset \
    -ctlcount $ctlcount \
    -cepdir $MFCDIR \
    -cepext $CEPEXT \
    -ceplen 13 \
    -agc none \
    -cmn current \
    -phsegdir $OUTSEG,CTL \
    -wdsegdir $OUTSEG,CTL \
    -insentfn $TRANSFN \
    -outsentfn $outtransfn\
    -outctlfn $outctlfn \
    -logfn $logfn

echo "Done decoding."
exit 0
