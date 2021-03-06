                   Studio Quality Speaker-Independent 
                          Connected-Digit Corpus
                                (TIDIGITS)

                                CD-ROM Set

                   NIST Speech Discs 4-1, 4-2, and 4-3
                              February, 1991


This three-disc set of CD-ROMs contains a corpus of speech which was designed
and collected at Texas Instruments (TI) for the purpose of "designing and
evaluating algorithms for speaker-independent recognition of connected digit
sequences."[1]  The corpus contains read utterances from 326 speakers (111 men,
114 women, 50 boys, and 51 girls) each speaking approximately* 77 digit
sequences and has been divided into test and training subsets. 

The digit sequences were made up of the digits: "zero", "oh", "one", "two",
"three", "four", "five", "six", "seven", "eight", and "nine".  The digit
sequences spoken by each speaker can be broken down as follows:

     22 isolated digits (2 productions of each of 11 digits)
     11 2-digit sequences
     11 3-digit sequences
     11 4-digit sequences
     11 5-digit sequences
     11 7-digit sequences
     --
     77

Detailed information on the design and collection of the corpus can be found in
the file, "tidigits.doc" in the "doc" subdirectory, which contains the original
TI documentation. 

The corpus has been reformatted for CD-ROM by the National Institute of
Standards and Technology (NIST) and is distributed with TI's permission.  The
speech waveform files have been converted to the NIST SPHERE format and have a
".wav" filename extension.  Because of its large size, the corpus has been
distributed over three CD-ROMs as follows:

  CD4-1 : Men and Women training utterances
  CD4-2 : Men and Women test utterances
  CD4-3 : Boys and Girls test and training utterances

Each disc contains identical copies of all documentation for the user's
convenience. 


CD-ROM File and Directory Structure:
-----------------------------------
The speech corpus is organized on the discs as follows:

FILESPEC ::= /tidigits/<USAGE>/<SPEAKER-TYPE>/<SPEAKER-ID>/
                                               <DIGIT-STRING><PRODUCTION>.wav

where,

     USAGE ::= test | train
     SPEAKER-TYPE ::= man | woman | boy | girl
     SPEAKER-ID ::= aa | ab | ac | ... | tc
     DIGIT-STRING ::= <DIGIT> | <DIGIT><DIGIT> | <DIGIT><DIGIT><DIGIT> |
                      <DIGIT><DIGIT><DIGIT><DIGIT> |
                      <DIGIT><DIGIT><DIGIT><DIGIT><DIGIT> |
                      <DIGIT><DIGIT><DIGIT><DIGIT><DIGIT><DIGIT><DIGIT>
     where,

          DIGIT ::= z | o | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 

     PRODUCTION ::= a | b

The digit codes in a filename indicate the digit sequence spoken and can be
decoded as follows:
     z -> zero          3 -> three          7 -> seven
     o -> oh            4 -> four           8 -> eight
     1 -> one           5 -> five           9 -> nine
     2 -> two           6 -> six

Note: two productions (a,b) were collected for all 11 single-digit strings and
two productions were randomly collected for a few multi-digit strings.

Example:
     /tidigits/train/man/fd/6z97za.wav

     "tidigits" corpus, training material, adult male, speaker code "fd",
     digit sequence "six zero nine seven zero", first production, NIST
     SPHERE file.

Example:
     /tidigits/test/woman/pf/1b.wav

     "tidigits" corpus, test material, adult female, speaker code "pf",
     digit sequence "one", second production, NIST SPHERE file.


Online Documentation 
-------------------- 
The following documentation files have been included on each CD-ROM and are
located in the directory, "/tidigits/doc":

     dialects.txt - dialect codes and description
     spkrinfo.txt - speaker codes and their attributes
     tidigits.doc - original TI documentation for the corpus


(* Note: 6 utterances have been removed from the corpus because they contained
   egregious speaking errors.)


References
----------
1.  Leonard, R. G., "A Database for Speaker-Independent Digit Recognition", 
    Proc. ICASSP 84, Vol. 3, p. 42.11, 1984.
    [Text is identical to that in the file, "tidigits.doc" in the "doc"
     subdirectory.]
