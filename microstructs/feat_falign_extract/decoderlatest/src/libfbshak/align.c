/*
 * align.c -- Viterbi time align transcripts to speech.
 * 
 * **********************************************
 * CMU ARPA Speech Project
 *
 * Copyright (c) 1996 Carnegie Mellon University.
 * ALL RIGHTS RESERVED.
 * **********************************************
 * 
 * HISTORY
 * 
 * 13-Sep-96	M K Ravishankar (rkm@cs.cmu.edu) at Carnegie Mellon University
 * 		Changed align_sen_active to flag active senones instead of building a list
 * 		of them.
 * 
 * 24-Jul-96	M K Ravishankar (rkm@cs.cmu.edu) at Carnegie Mellon University
 * 		Stripped alternative pronunciation spec in input transcript to obtain
 * 		base word.
 * 
 * 15-Jul-96	M K Ravishankar (rkm@cs.cmu.edu) at Carnegie Mellon University
 * 		Created.
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <libutil/libutil.h>
#include <s3.h>
#include "s3types.h"
#include "mdef.h"
#include "tmat.h"
#include "dict.h"
#include "logs3.h"
#include "align.h"


/*
 * SOME ASSUMPTIONS
 *   - All phones (ciphones and triphones) have same HMM topology with n_state states.
 *   - Initial state = state 0; final state = state n_state-1.
 *   - Final state is a non-emitting state with no arcs out of it.
 *   - Some form of Bakis topology (ie, no cycles, except for self-transitions).
 */


/*
 * Phone-level sentence HMM structures:
 *     pnode_t: nodes of phones forming sentence HMM.
 *     plink_t: a link between two pnode_t nodes.
 * A phone node may have multiple successors and/or predecessors because of multiple
 * alternative pronunciations for a word, as well as the presence of OPTIONAL filler
 * words.
 * 
 * Assumptions:
 *   - No cycles in phone level sentence HMM.
 */

typedef struct pnode_s {
    s3wid_t wid;	/* Parent word id */
    s3cipid_t ci;	/* CI phone id corresponding to this node */
    s3cipid_t lc;	/* Left context CI phone */
    s3cipid_t rc;	/* Right context CI phone */
    int8 pos;		/* Phone position within word for this node */
    s3pid_t pid;	/* Triphone id for this node */
    int32 id;		/* Unique id for identifying node, debugging */
    struct plink_s *succlist;	/* Links to successor nodes */
    struct plink_s *predlist;	/* Links to predecessor nodes */
    struct pnode_s *next;	/* For building various lists of nodes */
    struct pnode_s *alloc_next;	/* Linear list of all allocated structures */
    struct snode_s *startstate;	/* Start state of underlying HMM */
} pnode_t;

/*
 * A may have links (transitions) to several successor or predecessor nodes.
 * They are captured by a list of the following plink_t type.
 */
typedef struct plink_s {
    pnode_t *node;		/* Target node for this link for a given parent node */
    struct plink_s *next;	/* Next link for same parent node */
} plink_t;


/*
 * Viterbi search history for each state at each time.
 */
typedef struct history_s {
    int32 score;
    struct snode_s *snode;		/* State for which this history node created */
    struct history_s *pred;		/* Previous frame history */
    struct history_s *alloc_next;	/* Linear list of all allocated history nodes */
} history_t;
static history_t *hist_head;		/* Head of list of all history nodes */

/*
 * State DAG structures similar to phone DAG structures.
 */
typedef struct snode_s {
    pnode_t *pnode;		/* Parent phone node */
    struct slink_s *succlist;	/* List of successor states */
    struct slink_s *predlist;	/* List of predecessor states */
    int32 score, newscore;	/* Score at start, end of each frame */
    history_t *hist, *newhist;	/* Path history at start, end of each frame */
    int32 active_frm;		/* Frame no. most recently active */
    s3senid_t sen;		/* Senone id, BAD_SENID if dummy node (head/tail) */
    int8 state;			/* Local state no. (within parent HMM) */
} snode_t;

typedef struct slink_s {
    snode_t *node;
    struct slink_s *next;
    int32 prob;
} slink_t;

static pnode_t phead, ptail;	/* Dummies at the beginning and end of the sent hmm */
static pnode_t *pnode_list;	/* List of all dynamically allocated pnodes */
static int32 n_pnode;		/* #pnodes allocated (used to ID each pnode) */

static snode_t shead, stail;	/* State-level DAG head and tail */

static dict_t *dict;		/* The dictionary */
static mdef_t *mdef;		/* Model definition */
static tmat_t *tmat;		/* Transition probability matrices */

static s3wid_t silwid, startwid, finishwid;	/* Base wids */
static s3wid_t *fillwid;	/* BAD_WID terminated array of optional filler basewid */

static snode_t **cur_active;	/* NULL-terminated active state list for current frame */
static snode_t **next_active;	/* Similar list for next frame */
static int32 active_list_size = 0;
#define ACTIVE_LIST_SIZE_INCR	16380
static int32 n_active;

static int32 curfrm;		/* Current frame */
static int32 beam;		/* Pruning beamwidth */
static int32 *score_scale;	/* Score by which state scores scaled in each frame */

/* Lists of state, phone and word-level alignments for most recent utterance */
static align_stseg_t *align_stseg;
static align_phseg_t *align_phseg;
static align_wdseg_t *align_wdseg;

static int32 ctr_nstate;	/* Counter: #states active */


/* Free all allocated pnodes */
static void pnodes_free ( void )
{
    pnode_t *p;
    
    while (pnode_list) {
	p = pnode_list->alloc_next;
	listelem_free ((char *) pnode_list, sizeof(pnode_t));
	pnode_list = p;
    }
}


/* Free the specified set of plinks */
static void plinks_free (plink_t *l)
{
    plink_t *tmp;
    
    while (l) {
	tmp = l->next;
	listelem_free ((char *) l, sizeof(plink_t));
	l = tmp;
    }
}


/*
 * Append a pnode to a list of pnodes (maintained in a list of plinks).
 */
static plink_t *append_pnode (plink_t *list, pnode_t *node)
{
    plink_t *l;
    
    l = (plink_t *) listelem_alloc (sizeof(plink_t));
    l->node = node;
    l->next = list;
    return l;
}


/*
 * Allocate a pnode with the given attributes and automatically link it to the global
 * list.  Return the allocated node pointer.
 */
static pnode_t *alloc_pnode (s3wid_t w, int32 pos,
			     s3cipid_t ci, s3cipid_t lc, s3cipid_t rc,
			     word_posn_t wpos)
{
    pnode_t *p;

    p = (pnode_t *) listelem_alloc (sizeof(pnode_t));
    p->wid = w;
    p->ci = ci;
    p->lc = lc;
    p->rc = rc;
    p->pos = pos;

    p->pid = mdef_phone_id_nearest (mdef, ci, lc, rc, wpos);

    p->succlist = NULL;
    p->predlist = NULL;
    p->next = NULL;
    
    p->id = n_pnode++;
    
    p->startstate = NULL;
    
    p->alloc_next = pnode_list;
    pnode_list = p;
    
    return p;
}


/*
 * Link source and destination phone HMM nodes.
 */
static void link_pnodes (pnode_t *src, pnode_t *dst)
{
    src->succlist = append_pnode (src->succlist, dst);
    dst->predlist = append_pnode (dst->predlist, src);
}


/*
 * Create phone-level HMM for a single word and append to partial sentence HMM.
 * Replicate HMM nodes as needed to account for all possible left and right context
 * phones.
 * Return a list of the final HMM nodes for the single word appended.
 */
static pnode_t *append_word (s3wid_t w,
			     pnode_t *prev_end,
			     s3cipid_t *pred_ci,
			     s3cipid_t *succ_ci)
{
    int32 i, M, N, m, n, pronlen, pron;
    pnode_t *node, *nodelist, *p;
    plink_t *l, *new_end;
    
    for (i = 0; IS_CIPID(pred_ci[i]); i++);
    M = (i > 0) ? i : 1;	/* #predecessor CI phones */
    
    for (i = 0; IS_CIPID(succ_ci[i]); i++);
    N = (i > 0) ? i : 1;	/* #successor CI phones */
    
    if ((pronlen = dict->word[w].pronlen) == 1) {
	/* Single phone case; replicated MxN times for all possible contexts */
	nodelist = NULL;

	for (m = 0; m < M; m++) {
	    for (n = 0; n < N; n++) {
		node = alloc_pnode (w, 0,
				    dict->word[w].ciphone[0], pred_ci[m], succ_ci[n],
				    WORD_POSN_SINGLE);
		/* Link to all predecessor nodes matching context requirements */
		for (p = prev_end; p; p = p->next) {
		    if ((p->ci == node->lc) &&
			((NOT_CIPID(p->rc)) || (p->rc == node->ci))) {
			link_pnodes (p, node);
		    }
		}

		node->next = nodelist;
		nodelist = node;
	    }
	}
	
	return nodelist;
    }

    /* Multi-phone case.  First phone, replicated M times */
    nodelist = NULL;
    for (m = 0; m < M; m++) {
	node = alloc_pnode (w, 0,
			    dict->word[w].ciphone[0],
			    pred_ci[m],
			    dict->word[w].ciphone[1],
			    WORD_POSN_BEGIN);
	/* Link to predecessor node(s) matching context requirements */
	for (p = prev_end; p; p = p->next) {
	    if ((p->ci == node->lc) &&
		((NOT_CIPID(p->rc)) || (p->rc == node->ci))) {
		link_pnodes (p, node);
	    }
	}
	node->next = nodelist;
	nodelist = node;
    }

    /* Intermediate phones */
    for (pron = 1; pron < pronlen-1; pron++) {
	node = alloc_pnode (w, pron,
			    dict->word[w].ciphone[pron],
			    dict->word[w].ciphone[pron-1],
			    dict->word[w].ciphone[pron+1],
			    WORD_POSN_INTERNAL);
	for (p = nodelist; p; p = p->next)
	    link_pnodes (p, node);
	nodelist = node;
    }

    /* Final phone, replicated N times */
    prev_end = nodelist;
    nodelist = NULL;
    for (n = 0; n < N; n++) {
	node = alloc_pnode (w, pron,
			    dict->word[w].ciphone[pron],
			    dict->word[w].ciphone[pron-1],
			    succ_ci[n],
			    WORD_POSN_END);
	for (p = prev_end; p; p = p->next)
	    link_pnodes (p, node);
	node->next = nodelist;
	nodelist = node;
    }

    return nodelist;
}


static void build_pred_ci (pnode_t *nodelist, s3cipid_t *pred_ci)
{
    int32 i, p;
    pnode_t *node;
    
    for (p = 0; p < mdef->n_ciphone; p++)
	pred_ci[p] = 0;

    for (node = nodelist; node; node = node->next)
	pred_ci[node->ci] = 1;

    i = 0;
    for (p = 0; p < mdef->n_ciphone; p++) {
	if (pred_ci[p])
	    pred_ci[i++] = p;
    }
    pred_ci[i] = BAD_CIPID;
}


static void build_succ_ci (s3wid_t w, int32 append_filler, s3cipid_t *succ_ci)
{
    int32 i, p;
    pnode_t *node;
    
    for (p = 0; p < mdef->n_ciphone; p++)
	succ_ci[p] = 0;

    for (; IS_WID(w); w = dict->word[w].alt)
	succ_ci[dict->word[w].ciphone[0]] = 1;
    
    if (append_filler) {
	for (i = 0; IS_WID(fillwid[i]); i++)
	    for (w = fillwid[i]; IS_WID(w); w = dict->word[w].alt)
		succ_ci[dict->word[w].ciphone[0]] = 1;
    }

    i = 0;
    for (p = 0; p < mdef->n_ciphone; p++) {
	if (succ_ci[p])
	    succ_ci[i++] = p;
    }
    succ_ci[i] = BAD_CIPID;
}


/*
 * Append a new word to partially build phone-level sentence HMM.  (Handle alternative
 * pronunciations.)  Link new word to end phones of previous words.
 * Append optional filler words before w, if indicated.
 * Also Link prev_end into the global node list.
 * Return value: list of end phone nodes for w.  (NOTE: these are not yet linked into
 * the global node list.)
 */
static pnode_t *append_transcript_word
    (s3wid_t w,			/* Transcript word to be appended */
     pnode_t *prev_end,		/* Previous end points to be attached to w */
     s3wid_t nextw,		/* Next word to follow w (ignoring optional fillers) */
     int32 prefix_filler,	/* Whether optional filler words to precede w */
     int32 append_filler)	/* Whether optional filler words to follow w */
{
    int32 i, p;
    pnode_t *new_end, *tmp_end, *node;
    s3cipid_t pred_ci[256], succ_ci[256];
    s3wid_t fw;
    
    if (mdef->n_ciphone >= 256)
	E_FATAL("Increase pred_ci, succ_ci array sizes to > #CIphones (%d)\n",
		mdef->n_ciphone);
    assert (prev_end != NULL);
    
    /* Add optional silence/filler words before w, if indicated */
    if (prefix_filler) {
	build_pred_ci (prev_end, pred_ci);	/* Predecessor CI list for fillers */
	build_succ_ci (w, 0, succ_ci);		/* Successor CI list for fillers */
	
	new_end = NULL;
	for (i = 0; IS_WID(fillwid[i]); i++) {
	    for (fw = fillwid[i]; IS_WID(fw); fw = dict->word[fw].alt) {
		tmp_end = append_word (fw, prev_end, pred_ci, succ_ci);

		for (node = tmp_end; node->next; node = node->next);
		node->next = new_end;
		new_end = tmp_end;
	    }
	}
	
	/* Augment prev_end with new_end for filler words added above */
	for (node = prev_end; node->next; node = node->next);
	node->next = new_end;
    }

    /* Add w */
    build_pred_ci (prev_end, pred_ci);			/* Predecessor CI list for w */
    build_succ_ci (nextw, append_filler, succ_ci);	/* Successor CI list for w */

    new_end = NULL;
    for (; IS_WID(w); w = dict->word[w].alt) {
	tmp_end = append_word (w, prev_end, pred_ci, succ_ci);

	for (node = tmp_end; node->next; node = node->next);
	node->next = new_end;
	new_end = tmp_end;
    }
    
    return (new_end);
}


#if _DEBUG_ALIGN_

static void dump_pnode_info (pnode_t *p)
{
    if (NOT_WID(p->wid))
	printf ("%s", (p->id == -1) ? "<head>" : "<tail>");
    else
	printf ("%s.%d.",
		dict_wordstr(p->wid), p->pos, mdef_ciphone_str (mdef, p->ci));
    printf ("%s", IS_CIPID(p->lc) ? mdef_ciphone_str (mdef, p->lc) : "-");
    printf ("(%s)", IS_CIPID(p->ci) ? mdef_ciphone_str (mdef, p->ci) : "-");
    printf ("%s", IS_CIPID(p->rc) ? mdef_ciphone_str (mdef, p->rc) : "-");
}


static void dump_pnode_succ_dag (pnode_t *p)
{
    plink_t *l;

    for (l = p->succlist; l; l = l->next) {
	dump_pnode_info (p);
	printf ("\t\t");
	dump_pnode_info (l->node);
	printf (";\n");
    }
}


static void dump_pnode_succ (pnode_t *p)
{
    plink_t *l;
    
    printf ("  %5d", p->id);
    if (IS_WID(p->wid))
	printf (" %20s %02d %6d %4s",
		dict_wordstr(p->wid), p->pos, p->pid, mdef_ciphone_str (mdef, p->ci));
    else
	printf (" %20s %02d %6d %4s",
		"<phead>", 0, BAD_PID, "");
    printf (" %4s %4s",
	    IS_CIPID(p->lc) ? mdef_ciphone_str (mdef, p->lc) : "-",
	    IS_CIPID(p->rc) ? mdef_ciphone_str (mdef, p->rc) : "-");
    printf ("\t");

    for (l = p->succlist; l; l = l->next)
	printf (" %5d", l->node->id);
    
    printf ("\n");
}


static void dump_pdag ( void )
{
    pnode_t *p;
    
    printf ("SUCCESSOR LIST (DAG format):\n");
    printf (".GS 5 5 fill\n");
    dump_pnode_succ_dag (&phead);
    for (p = pnode_list; p; p = p->alloc_next)
	dump_pnode_succ_dag (p);
    printf (".GE\n");
    
    printf ("SUCCESSOR LIST:\n");
    dump_pnode_succ (&phead);
    for (p = pnode_list; p; p = p->alloc_next)
	dump_pnode_succ (p);
}

#endif


/*
 * Append an snode to a list of snodes (maintained in a list of slinks).
 */
static slink_t *append_snode (slink_t *list, snode_t *node, int32 prob)
{
    slink_t *l;
    
    l = (slink_t *) listelem_alloc (sizeof(slink_t));
    l->node = node;
    l->next = list;
    l->prob = prob;

    return l;
}


/*
 * Link source and destination state nodes.
 */
static void link_snodes (snode_t *src, snode_t *dst, int32 prob)
{
    src->succlist = append_snode (src->succlist, dst, prob);
    dst->predlist = append_snode (dst->predlist, src, prob);
}


/*
 * Remove src->dst link and return the associated prob.
 */
static int32 un_slink_succ (snode_t *src, snode_t *dst)
{
    slink_t *l, *prevl;
    int32 prob;
    
    /* Find link from src to dst */
    prevl = NULL;
    for (l = src->succlist; l && (l->node != dst); l = l->next)
	prevl = l;
    assert (l);	/* The link must exist */
    
    if (! prevl)
	src->succlist = l->next;
    else
	prevl->next = l->next;
    
    prob = l->prob;
    listelem_free ((char *)l, sizeof(slink_t));
    
    return prob;
}


static void slinks_free (slink_t *l)
{
    slink_t *tmp;
    
    while (l) {
	tmp = l->next;
	listelem_free ((char *) l, sizeof(slink_t));
	l = tmp;
    }
}


/*
 * Build a state-level DAG from the phone-level one.  This DAG is the one actually
 * searched.
 */
static int32 build_state_dag ( void )
{
    pnode_t *p;
    plink_t *pl;
    snode_t *s, *fs;
    slink_t *sl;
    int32 n_state, final_state;
    int32 i, j;
    int32 **tp, prob;
    
    n_state = mdef->n_emit_state + 1;
    final_state = n_state - 1;
    
    for (p = pnode_list; p; p = p->alloc_next) {
	/* Allocate states for p */
	s = (snode_t *) ckd_calloc (n_state, sizeof(snode_t));
	p->startstate = s;
	
	for (i = 0; i < n_state; i++) {
	    s[i].pnode = p;
	    s[i].succlist = NULL;
	    s[i].predlist = NULL;
	    s[i].score = LOGPROB_ZERO;
	    s[i].hist = NULL;
	    s[i].active_frm = -1;
	    s[i].sen = mdef->phone[p->pid].state[i];
	    s[i].state = i;
	}
	
	/* Create transitions between states */
	tp = tmat->tp[mdef->phone[p->pid].tmat];
	for (i = 0; i < final_state; i++) {	/* #from states excludes final state */
	    for (j = 0; j < n_state; j++) {
		if (tp[i][j] > LOGPROB_ZERO)	/* Link from i to j */
		    link_snodes (s+i, s+j, tp[i][j]);
	    }
	}
    }

    /* Eliminate non-emitting nodes (final states of HMMs) from state DAG structure */
    for (p = pnode_list; p; p = p->alloc_next) {
	fs = p->startstate + final_state;
	assert (! fs->succlist);
	
	/*
	 * Link predecessor states of fs to start states of followers of parent pnode
	 * with the appropriate prob.  (Avoid linking to the dummy node ptail).
	 */
	for (sl = fs->predlist; sl; sl = sl->next) {
	    /* Unlink successor link between this predecessor and final state */
	    prob = un_slink_succ (sl->node, fs);
	    
	    /* Link this predecessor to start states of successor phones */
	    for (pl = p->succlist; pl; pl = pl->next) {
		if (pl->node->startstate)
		    link_snodes (sl->node, pl->node->startstate, prob);
		else
		    link_snodes (sl->node, &stail, prob);
	    }
	}

	slinks_free (fs->predlist);
	fs->predlist = NULL;
    }

    /* Link shead to initial states */
    for (pl = phead.succlist; pl; pl = pl->next)
	link_snodes (&shead, pl->node->startstate, 0);
    
    return 0;
}


static void destroy_state_dag ( void )
{
    pnode_t *p;
    snode_t *s;
    int32 i, n_state;
    
    n_state = mdef->n_emit_state + 1;

    for (p = pnode_list; p; p = p->alloc_next) {
	if ((s = p->startstate) != NULL) {	/* Maybe NULL if state dag not built */
	    for (i = 0; i < n_state; i++) {
		slinks_free (s[i].succlist);
		slinks_free (s[i].predlist);
	    }
	    
	    ckd_free (s);
	    p->startstate = NULL;
	}
    }

    slinks_free (shead.succlist);
    slinks_free (stail.predlist);
}


#if _DEBUG_ALIGN_

static void dump_snode_succ (snode_t *s)
{
    slink_t *l;
    pnode_t *p;
    
    p = s->pnode;
    printf (" %5d.%d\t", p->id, s->state);
    for (l = s->succlist; l; l = l->next)
	printf (" %5d.%d(%7d)", l->node->pnode->id, l->node->state, l->prob);
    printf ("\n");
}


static void dump_sdag ( void )
{
    pnode_t *p;
    snode_t *s;
    int32 i;
    
    printf ("STATE DAG:\n");
    for (p = pnode_list; p; p = p->alloc_next) {
	s = p->startstate;
	for (i = 0; i <= mdef->n_emit_state; i++)
	    dump_snode_succ (s+i);
    }
}


static void dump_sent_hmm ( void )
{
    dump_pdag ();
    dump_sdag ();
    E_INFO("%d pnodes, %d snodes\n", n_pnode, n_pnode * mdef->n_emit_state);
}

#endif


/*
 * Build a sentence HMM for the given transcription (wordstr).  A two-level DAG is
 * built: phone-level and state-level.
 *   - <s> and </s> always added at the beginning and end of sentence to form an
 *     augmented transcription.
 *   - Optional <sil> and noise words added between words in the augmented
 *     transcription.
 * wordstr must contain only the transcript; no extraneous stuff such as utterance-id.
 * Phone-level HMM structure has replicated nodes to allow for different left and right
 * context CI phones; hence, each pnode corresponds to a unique triphone in the sentence
 * HMM.
 * Return 0 if successful, <0 if any error (eg, OOV word encountered).
 */
int32 align_build_sent_hmm (char *wordstr)
{
    s3wid_t w, nextw;
    int32 k, oov;
    pnode_t *word_end, *node;
    char *wd, delim, *wdcopy;
    
    /* Initialize dummy head and tail entries of sent hmm */
    phead.wid = BAD_WID;
    phead.ci = BAD_CIPID;
    phead.lc = BAD_CIPID;	/* No predecessor */
    phead.rc = BAD_CIPID;	/* Any phone can follow head */
    phead.pid = BAD_PID;
    phead.succlist = NULL;
    phead.predlist = NULL;
    phead.next = NULL;		/* Will ultimately be the head of list of all pnodes */
    phead.id = -1;		/* Hardwired */
    phead.startstate = NULL;
    
    ptail.wid = BAD_WID;
    ptail.ci = BAD_CIPID;
    ptail.lc = BAD_CIPID;	/* Any phone can precede tail */
    ptail.rc = BAD_CIPID;	/* No successor */
    ptail.pid = BAD_PID;
    ptail.succlist = NULL;
    ptail.predlist = NULL;
    ptail.next = NULL;
    ptail.id = -2;		/* Hardwired */
    ptail.startstate = NULL;

    n_pnode = 0;
    pnode_list = NULL;
    oov = 0;
    
    /* State-level DAG initialization should be here in case the build is aborted */
    shead.pnode = &phead;
    shead.succlist = NULL;
    shead.predlist = NULL;
    shead.sen = BAD_SENID;
    shead.state = mdef->n_emit_state;
    shead.hist = NULL;

    stail.pnode = &ptail;
    stail.succlist = NULL;
    stail.predlist = NULL;
    stail.sen = BAD_SENID;
    stail.state = 0;
    stail.hist = NULL;
    
    /* Obtain the first transcript word */
    k = nextword (wordstr, " \t\n", &wd, &delim);
    if (k < 0)
	nextw = finishwid;
    else {
	wordstr = wd + k;
	wdcopy = ckd_salloc (wd);
	*wordstr = delim;
	nextw = dict_wordid (wdcopy);
	if (IS_WID(nextw))
	    nextw = dict_basewid (nextw);
    }
    
    /* Create node(s) for <s> before any transcript word */
    word_end = append_transcript_word (startwid, &phead, nextw, 0, 1);

    /* Append each word in transcription to partial sent HMM created so far */
    while (k >= 0) {
	w = nextw;
	if (NOT_WID(w)) {
	    E_ERROR("%s not in dictionary\n", wdcopy);
	    oov = 1;
	    /* Hack!! Temporarily set w to some dummy just to run through sentence */
	    w = finishwid;
	}
	ckd_free (wdcopy);
	
	k = nextword (wordstr, " \t\n", &wd, &delim);
	if (k < 0)
	    nextw = finishwid;
	else {
	    wordstr = wd + k;
	    wdcopy = ckd_salloc (wd);
	    *wordstr = delim;
	    nextw = dict_wordid (wdcopy);
	    if (IS_WID(nextw))
		nextw = dict_basewid (nextw);
	}

	word_end = append_transcript_word (w, word_end, nextw, 1, 1);
    }
    if (oov)
	return -1;
    
    /* Append phone HMMs for </s> at the end; link to tail node */
    word_end = append_transcript_word (finishwid, word_end, BAD_WID, 1, 0);
    for (node = word_end; node; node = node->next)
	link_pnodes (node, &ptail);
    
    /* Build state-level DAG from the phone-level one */
    build_state_dag ();
    /* Dag must begin and end at shead and stail, respectively */
    assert (shead.succlist);
    assert (stail.predlist);
    assert (! shead.predlist);
    assert (! stail.succlist);

#if _DEBUG_ALIGN_
    dump_sent_hmm ();	/* For debugging */
#endif

    k = n_pnode * mdef->n_emit_state;
    if (k > active_list_size) {	/* Need to grow active list arrays */
	if (active_list_size > 0) {
	    ckd_free (cur_active);
	    ckd_free (next_active);
	}
	for (; active_list_size <= k; active_list_size += ACTIVE_LIST_SIZE_INCR);
	cur_active = (snode_t **) ckd_calloc (active_list_size, sizeof(snode_t *));
	next_active = (snode_t **) ckd_calloc (active_list_size, sizeof(snode_t *));
    }
    
    return 0;
}


int32 align_destroy_sent_hmm ( void )
{
    pnode_t *p;
    
    destroy_state_dag ();
    
    for (p = pnode_list; p; p = p->alloc_next) {
	plinks_free (p->succlist);
	plinks_free (p->predlist);
    }
    pnodes_free ();
    
    plinks_free (phead.succlist);
    plinks_free (ptail.predlist);
    
    return 0;
}


static history_t *lat_entry (snode_t *s)
{
    history_t *h;
    
    h = (history_t *) listelem_alloc (sizeof(history_t));
    h->snode = s;
    h->score = s->newscore;
    h->pred = s->newhist;
    
    h->alloc_next = hist_head;
    hist_head = h;
    
    return h;
}


static void activate (snode_t *s, int32 frm)
{
    if (s->active_frm != frm) {
	assert (s->active_frm < frm);

	s->active_frm = frm;
	next_active[n_active++] = s;
    }
}


/*
 * Flag the active senones.
 */
void align_sen_active (s3senid_t *senlist, int32 n_sen)
{
    int32 i, sen;
    
    for (sen = 0; sen < n_sen; sen++)
	senlist[sen] = 0;
    
    for (i = 0; cur_active[i]; i++) {
	assert (IS_SENID(cur_active[i]->sen));
	senlist[cur_active[i]->sen] = 1;
    }
}


/*
 * Start Viterbi alignment using the sentence HMM previously built.
 * Assumes that each utterance will only be aligned once; state member variables
 * initialized during sentence HMM building.
 */
int32 align_start_utt (char *uttid)
{
    slink_t *l;
    
    curfrm = 0;
    shead.score = 0;
    shead.hist = NULL;
    hist_head = NULL;
    
    n_active = 0;
    for (l = shead.succlist; l; l = l->next) {
	assert (l->node->active_frm < 0);
	l->node->active_frm = 0;
	cur_active[n_active++] = l->node;
    }
    cur_active[n_active++] = NULL;
    
    return 0;
}


/*
 * One frame of Viterbi time alignment.
 */
int32 align_frame (int32 *senscr)
{
    int32 i, n, scr, tmpbest, bestscore, nf, thresh;
    snode_t *s, *ps;
    slink_t *l;
    history_t *tmphist;
    snode_t **tmpswap;
    
    nf = curfrm + 1;
    n_active = 0;

    /* For each active state update state score and history */
    bestscore = (int32) 0x80000000;
    for (i = 0; cur_active[i]; i++) {
	s = cur_active[i];
	assert (IS_SENID(s->sen));

	tmpbest = (int32) 0x80000000;
	for (l = s->predlist; l; l = l->next) {
	    ps = l->node;

	    if (ps->active_frm == curfrm) {
		scr = ps->score + l->prob;
		
		if (scr > tmpbest) {
		    tmpbest = scr;
		    tmphist = ps->hist;
		}
	    }
	}
	/* HACK BY BHIKSHA: DONT DIE, RETURN ERROR */
	if (tmpbest  <= (int32) 0x80000000){
	    return -1;
	}
/*	assert (tmpbest > (int32) 0x80000000); COMMENTED OUT BY BHIKSHA */
	
	s->newscore = tmpbest + senscr[s->sen];
	s->newhist = tmphist;

	if (s->newscore > bestscore)
	    bestscore = s->newscore;
    }
    
    counter_increment (ctr_nstate, i);
    
    if (bestscore <= LOGPROB_ZERO)
	E_ERROR("Bestscore= %d in frame %d\n", bestscore, curfrm);
    score_scale[curfrm] = bestscore;
    thresh = bestscore + beam;
    
    /* Update history lattice for each active state */
    for (i = 0; cur_active[i]; i++) {
	s = cur_active[i];

	if (s->newscore >= thresh) {
	    s->newscore -= bestscore;	/* Scale, to avoid underflow */
	    s->score = s->newscore;

	    s->hist = lat_entry (s);
	    activate (s, nf);

	    /* Also activate successor nodes of s as they are reachable next frame */
	    for (l = s->succlist; l; l = l->next) {
		if (IS_SENID(l->node->sen))
		    activate (l->node, nf);
	    }
	} else {
	    s->score = LOGPROB_ZERO;
	    s->hist = NULL;
	}
    }

    /* Update active state list */
    next_active[n_active] = NULL;
    tmpswap = cur_active;
    cur_active = next_active;
    next_active = tmpswap;

    curfrm = nf;

    return 0;
}


static void build_stseg (history_t *rooth)
{
    history_t *h, *prevh;
    align_stseg_t *stseg, *tail;
    int32 f, prevscr;
    
    assert (align_stseg == NULL);

    prevscr = 0;
    prevh = NULL;
    for (f = 0, h = rooth; h; h = h->pred, f++) {
	stseg = (align_stseg_t *) listelem_alloc (sizeof(align_stseg_t));
	if (! align_stseg)
	    align_stseg = stseg;
	else
	    tail->next = stseg;
	tail = stseg;
	stseg->next = NULL;
	
	stseg->pid = h->snode->pnode->pid;
	stseg->state = h->snode->state;
	stseg->start = ((! prevh) || (prevh->snode->pnode->id != h->snode->pnode->id));
	stseg->score = h->score - prevscr + score_scale[f];
	stseg->bsdiff = h->score;

	prevscr = h->score;
	prevh = h;
    }
}


static void build_phseg (history_t *rooth)
{
    history_t *h, *nh;
    align_phseg_t *phseg, *tail;
    int32 f, prevf, prevscr, scale, bsdiff;
    
    assert (align_phseg == NULL);

    prevscr = 0;
    bsdiff = 0;
    scale = 0;
    prevf = -1;
    
    for (f = 0, h = rooth; h; h = h->pred, f++) {
	bsdiff += h->score;
	scale += score_scale[f];
	
	nh = h->pred;
	if ((! nh) || (nh->snode->pnode->id != h->snode->pnode->id)) {
	    phseg = (align_phseg_t *) listelem_alloc (sizeof(align_phseg_t));
	    if (! align_phseg)
		align_phseg = phseg;
	    else
		tail->next = phseg;
	    tail = phseg;
	    phseg->next = NULL;
	    
	    phseg->pid = h->snode->pnode->pid;
	    phseg->sf = prevf+1;
	    phseg->ef = f;
	    phseg->score = h->score - prevscr + scale,
	    phseg->bsdiff = bsdiff;
	    
	    bsdiff = 0;
	    scale = 0;
	    prevscr = h->score;
	    prevf = f;
	}
    }
}


static void build_wdseg (history_t *rooth)
{
    history_t *h, *nh;
    align_wdseg_t *wdseg, *tail;
    int32 f, prevf, prevscr, scale, bsdiff;
    
    assert (align_wdseg == NULL);

    prevscr = 0;
    bsdiff = 0;
    scale = 0;
    prevf = -1;
    
    for (f = 0, h = rooth; h; h = h->pred, f++) {
	bsdiff += h->score;
	scale += score_scale[f];
	
	nh = h->pred;
	if ((! nh) ||
	    ((nh->snode->pnode->id != h->snode->pnode->id) &&
	     (nh->snode->pnode->pos == 0))) {	/* End of current word */

	    wdseg = (align_wdseg_t *) listelem_alloc (sizeof(align_wdseg_t));
	    if (! align_wdseg)
		align_wdseg = wdseg;
	    else
		tail->next = wdseg;
	    tail = wdseg;
	    wdseg->next = NULL;
	    
	    wdseg->wid = h->snode->pnode->wid;
	    wdseg->sf = prevf+1;
	    wdseg->ef = f;
	    wdseg->score = h->score - prevscr + scale,
	    wdseg->bsdiff = bsdiff;
	    
	    bsdiff = 0;
	    scale = 0;
	    prevscr = h->score;
	    prevf = f;
	}
    }
}


/*
 * All frames consumed.  Trace back best Viterbi state sequence and dump it out.
 */
int32 align_end_utt (align_stseg_t **stseg_out,
		     align_phseg_t **phseg_out,
		     align_wdseg_t **wdseg_out)
{
    slink_t *l;
    snode_t *s;
    history_t *h, *ph, *nh;
    align_stseg_t *stseg;
    align_phseg_t *phseg;
    align_wdseg_t *wdseg;
    
    /* Free up previous result, if any */
    while (align_stseg) {
	stseg = align_stseg->next;
	listelem_free ((char *) align_stseg, sizeof(align_stseg_t));
	align_stseg = stseg;
    }
    while (align_phseg) {
	phseg = align_phseg->next;
	listelem_free ((char *) align_phseg, sizeof(align_phseg_t));
	align_phseg = phseg;
    }
    while (align_wdseg) {
	wdseg = align_wdseg->next;
	listelem_free ((char *) align_wdseg, sizeof(align_wdseg_t));
	align_wdseg = wdseg;
    }
    
    /* First find best ending history and link to stail */
    stail.score = (int32)0x80000000;
    stail.hist = NULL;
    for (l = stail.predlist; l; l = l->next) {
	s = l->node;
	if ((s->active_frm == curfrm) && (s->score + l->prob > stail.score)) {
	    stail.score = s->score + l->prob;
	    stail.hist = s->hist;
	}
    }
    
    if (stail.hist) {
	/* Reverse the best Viterbi path (back trace) so it is forward in time */
	nh = NULL;
	for (h = stail.hist; h; h = ph) {
	    ph = h->pred;
	    h->pred = nh;
	    nh = h;
	}
	
	/* Trace state, phone, and word segmentations */
	build_stseg (nh);
	build_phseg (nh);
	build_wdseg (nh);
    }
    
    *stseg_out = align_stseg;
    *phseg_out = align_phseg;
    *wdseg_out = align_wdseg;
    
    /* delete history list */
    while (hist_head) {
	h = hist_head->alloc_next;
	listelem_free ((char *) hist_head, sizeof(history_t));
	hist_head = h;
    }

    return (stail.hist ? 0 : -1);
}


int32 align_init ( void )
{
    int32 k;
    s3wid_t w;
    float64 *f64arg;
    
    mdef = mdef_getmdef ();
    tmat = tmat_gettmat ();
    dict = dict_getdict ();
    
    assert (mdef && tmat && dict);
    
    startwid = dict_wordid (START_WORD);
    finishwid = dict_wordid (FINISH_WORD);
    silwid = dict_wordid (SILENCE_WORD);
    
    if ((NOT_WID(startwid)) || (NOT_WID(finishwid)))
	E_FATAL("%s or %s not in dictionary\n", START_WORD, FINISH_WORD);
    if (NOT_WID(silwid))
	E_ERROR("%s not in dictionary; no optional silence inserted between words\n",
	       SILENCE_WORD);

    /* Create list of optional filler words to be inserted between transcript words */
    fillwid = (s3wid_t *) ckd_calloc ((dict->filler_end - dict->filler_start + 3),
				      sizeof(s3wid_t));
    k = 0;
    if (IS_WID(silwid))
	fillwid[k++] = silwid;
    for (w = dict->filler_start; w <= dict->filler_end; w++) {
	if ((dict_basewid (w) == w) &&
	    (w != silwid) && (w != startwid) && (w != finishwid))
	    fillwid[k++] = w;
    }
    fillwid[k] = BAD_WID;

    f64arg = (float64 *) cmd_ln_access ("-beam");
    beam = logs3 (*f64arg);
    E_INFO ("logs3(beam)= %d\n", beam);

    score_scale = (int32 *) ckd_calloc (S3_MAX_FRAMES, sizeof(int32));
    
    hist_head = NULL;
    
    align_stseg = NULL;
    align_phseg = NULL;
    align_wdseg = NULL;

    ctr_nstate = counter_new ("NS");
    
    return 0;
}

/*
 * ADDED BY BHIKSHA
 * Align failed. Clean up
 */
int32 align_kill_utt ()
{
    slink_t *l;
    snode_t *s;
    history_t *h, *ph, *nh;
    align_stseg_t *stseg;
    align_phseg_t *phseg;
    align_wdseg_t *wdseg;
    
    /* Free up previous result, if any */
    while (align_stseg) {
	stseg = align_stseg->next;
	listelem_free ((char *) align_stseg, sizeof(align_stseg_t));
	align_stseg = stseg;
    }
    while (align_phseg) {
	phseg = align_phseg->next;
	listelem_free ((char *) align_phseg, sizeof(align_phseg_t));
	align_phseg = phseg;
    }
    while (align_wdseg) {
	wdseg = align_wdseg->next;
	listelem_free ((char *) align_wdseg, sizeof(align_wdseg_t));
	align_wdseg = wdseg;
    }
    
    /* delete history list */
    while (hist_head) {
	h = hist_head->alloc_next;
	listelem_free ((char *) hist_head, sizeof(history_t));
	hist_head = h;
    }

    return (-1);
}

