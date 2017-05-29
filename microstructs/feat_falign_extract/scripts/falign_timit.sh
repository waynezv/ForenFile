#!/bin/bash

# if [ $? PBS_O_WORKDIR ]; then
    # cd $PBS_O_WORKDIR
# fi

if [[ $# -lt 11 ]]; then
    echo "USAGE: $0 [--part] part [--npart] npart
    [--listbasedir] dir containing list files
    [--ctlf] file containing data list
    [--transf] file containing transcripts
    [--dictf] file containing dict
    [--fillerf] file containing filler dict
    [--mfcdir] dir containing mfcc
    [--outputdir] output dir
    [--phsegdir] dir for phone segments output
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
        --phsegdir)
            PHSEG="$2"
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

# Input setup
CTLFN="$LISTBASE/$CTL"
TRANSFN="$LISTBASE/$TRANS"
DICTFN="$LISTBASE/$DICT"
FILLFN="$LISTBASE/$FILL"
MFCDIR="$MFC"
CEPEXT="80-7200-40f.mfc"

# Output setup
BASEDIR="./"
[[ -e "$BASEDIR/$OUTDIR" ]] || mkdir -p "$BASEDIR/$OUTDIR"

outtransfn="$BASEDIR/$OUTDIR/$JOB.faligned-$PART.trans"
outctlfn="$BASEDIR/$OUTDIR/$JOB.faligned-$PART.ctl"
logfn="$BASEDIR/$OUTDIR/$JOB.faligned-$PART.log"

for f in $( cat $CTLFN | awk '{print $1}' ); do
    dirhead=$( dirname $f )
    [[ -e "$PHSEG/$dirhead" ]] || mkdir -p "$PHSEG/$dirhead"
done

# Job setup
nlines=$( wc $CTLFN | awk '{print $1}' ) # number of lines in ctl file
echo $nlines
ctloffset=$(( ( $nlines * ($PART - 1) ) / $NPART ))
ctlcount=$(( (($nlines * $PART) / $NPART) - $ctloffset ))
echo "Doing $ctlcount segments starting at number $ctloffset"

# Model setup
modeldir="./models/ads/ads.80-7200-40f.1-3/ads.80-7200-40f.1-3.ci_continuous.8gau"
mdeffn="./models/ads/ads.80-7200-40f.1-3.ci.mdef"

# Decoder setup
decoder="./decoderlatest/bin/linux/s3align"

# Decode
$decoder \
    -logbase 1.0001 \
    -mdeffn $mdeffn \
    -senmgaufn .cont. \
    -meanfn $modeldir/means \
    -varfn $modeldir/variances \
    -mixwfn $modeldir/mixture_weights \
    -tmatfn $modeldir/transition_matrices \
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
    -phsegdir $PHSEG,CTL \
    -wdsegdir $PHSEG,CTL \
    -insentfn $TRANSFN \
    -outsentfn $outtransfn\
    -outctlfn $outctlfn \
    -logfn $logfn

exit 0
