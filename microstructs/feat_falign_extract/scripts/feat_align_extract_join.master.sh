#!/bin/bash

# Master script to extract features, force align, extract aligned wav segments,
# and join wavs.
# wzhao1 cs cmu edu
# 06/01/2017

# 0. Opts
TASK="test"
DATE="060517"

TMPDIR="./tmp"

FEAT="mfcc"
FEATDIR="./mfcc"

OBJ="phone"
OBJEXT="phseg"
OBJOUTDIR="./phseg_out"
ALIGNOUTDIR="./falign_out"

CTLOUTDIR="./phctl_out"

SEGWAVDIR="./phsegwav_out"
JOIN="false"
JPATH="./phjoin_out"
JNUM=100

# 1. Extract features
echo "Extracting features ..."

# Run feature extraction script
./scripts/compute_feat.sh --type ${FEAT} -f ./lists/timit_${TASK}_wavlist.ctl \
    -i ../timit_data/timit -o ${FEATDIR} -t ${TMPDIR}

echo "Done extracting features."

# 2. Force align
echo "Starting force alignment ..."

./scripts/falign_timit.sh --part 1 --npart 1 \
    --listbasedir ./lists/transcripts --ctlf timit_${TASK}.ctl \
    --transf timit_correct_${TASK}.trans --dictf timit.dict --fillerf timit.fillerdict \
    --mfcdir ${FEATDIR} --outputdir ${ALIGNOUTDIR} --outsegdir ${OBJOUTDIR} \
    --jobname timit_falign_${TASK}_${DATE}

echo "Done force alignment."

# 3. Extract aligned wavs and join
echo "Start extracting aligned wavs and join them ..."

./scripts/makectl.sh  -indir ${OBJOUTDIR} -type ${OBJ} -phase ${TASK} \
    -ext ${OBJEXT} -tmpdir ${TMPDIR} -outdir ${CTLOUTDIR}/${TASK}

ctlout="${CTLOUTDIR}/${TASK}"
for f in $( ls "$ctlout" ); do
    ctl="${ctlout}/${f}"
    x=$( wc -l $ctl | awk '{print $1;}' )

    if [[ $x -gt 0 ]]; then
        ./scripts/extract_join_segs.sh -ctl $ctl \
            -inwav ../timit_data/timit/${TASK} -ext wav \
            -outwav ${SEGWAVDIR}/${TASK} \
            -join ${JOIN} -jpath ${JPATH}/${TASK} -jnum ${JNUM}
    fi
done

echo "All done successfully!"

exit 0
