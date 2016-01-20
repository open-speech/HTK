/* ----------------------------------------------------------- */
/*                                                             */
/*                          ___                                */
/*                       |_| | |_/   SPEECH                    */
/*                       | | | | \   RECOGNITION               */
/*                       =========   SOFTWARE                  */ 
/*                                                             */
/*                                                             */
/* ----------------------------------------------------------- */
/* developed at:                                               */
/*                                                             */
/*      Machine Intelligence Laboratory                        */
/*      Department of Engineering                              */
/*      University of Cambridge                                */
/*      http://mi.eng.cam.ac.uk/                               */
/*                                                             */
/* ----------------------------------------------------------- */
/*         Copyright:                                          */
/*         2002-2003  Cambridge University                     */
/*                    Engineering Department                   */
/*                                                             */
/*   Use of this software is governed by a License Agreement   */
/*    ** See the file License for the Conditions of Use  **    */
/*    **     This banner notice must not be removed      **    */
/*                                                             */
/* ----------------------------------------------------------- */
/*         File: HLVRec.h Decoding related data types for      */
/*                        HTK LV Decoder                       */
/* ----------------------------------------------------------- */

/*
it might be worthwhile to make Tokens smaller, i.e. move as much info
as possible into WordendHyp (e.g. most likelihoods? all except tot or ac?).
Recombination and pruning is based on totlike + LMlookahead 
*/

#ifndef _HLVREC_H_
#define _HLVREC_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "HShell.h"     /* for HTime */
#include "HMem.h"       /* for Ptr */
#include "HMath.h"      /* for LogFloat */
#include "HLVNet.h"     /* for LexNet */
#include "HLVLM.h"      /* for LMState */
#include "HLVModel.h"


typedef struct _Token Token;            /* Info about partial hypothesis */
typedef struct _RelToken RelToken;      /* Info about partial hypothesis relative to main token */

typedef struct _WordendHyp WordendHyp;      /* records word level tracback */
typedef struct _AltWordendHyp AltWordendHyp;/* records alternatives for lattice tracback */

#ifdef MODALIGN
typedef struct _ModendHyp ModendHyp;    /* records model level tracback */

struct _ModendHyp {             /* stores info about one model(end) */
   ModendHyp *prev;             /* previous model info */
   LexNode *ln;                 /* lexnode that finished */
   short frame;                 /* end frame number of this model */
};
#endif


struct _RelToken {
   LMState lmState;
   void *we_tag;
   RelTokScore  delta;          /* delta score relative to main token, 
                                   normally negative (including LM lookahaed) */

   LMTokScore lmscore;          /* LM lookahead score for current word */
   WordendHyp *path;            /* word level path for traceback */
#ifdef MODALIGN
   ModendHyp *modpath;          /* model level traceback */
#endif
};

struct _Token {
   LMState lmState;
   TokScore  score;             /* current likelihood of token, 
                                   i.e. (ac + lm + pron) */
   LMTokScore lmscore;          /* LM lookahead score for current word */
   WordendHyp *path;            /* word level path for traceback */
};

struct _WordendHyp {            /* stores info about one word(end) */
   WordendHyp *prev;            /* previous word info */
   PronId pron;                 /* pronunciation chosen */
   short frame;                 /* end frame number of this word */
   TokScore score;              /* total likelihood at end of word (time t) */
   LMTokScore lm;               /* LM likelihood of this word given history */
                                /* don't need pron like, it's in pron->prob */
   AltWordendHyp *alt;          /* alternative paths for lattice traceback */
   int user;                  /* general user info; #### get rid of this! */
#ifdef MODALIGN
   ModendHyp *modpath;          /* model level traceback */
#endif
};

struct _AltWordendHyp {         /* stores info about N-best word(end) for lattice traceback */
   WordendHyp *prev;            /* previous word info */
   /* no need for pron (same as main weHyp) */
   TokScore score;              /* total likelihood at end of word (time t) */
   LMTokScore lm;               /* LM likelihood of this word given history */
                                /* don't need pron like, it's in pron->prob */
   /* no need for frame */
   AltWordendHyp *next;
#ifdef MODALIGN
   ModendHyp *modpath;          /* model level traceback */
#endif
};

/* macros to compare whether two tokens are quivalent accoring to the LM. */

#define TOK_LMSTATE_LT(t1,t2)   (((t1)->lmState <  (t2)->lmState) ||   \
                                 (((t1)->lmState == (t2)->lmState) &&  \
                                  ((t1)->we_tag < (t2)->we_tag)))
#define TOK_LMSTATE_EQ(t1,t2)   (((t1)->lmState == (t2)->lmState) &&   \
                                 ((t1)->we_tag == (t2)->we_tag))

#if 0   /* not used -- #### out of date! */
  #define TOK_LMSTATE_LE(t1,t2)   ((t1)->lmState <= (t2)->lmState)
  #define TOK_LMSTATE_GE(t1,t2)   ((t1)->lmState >= (t2)->lmState)
  #define TOK_LMSTATE_GT(t1,t2)   ((t1)->lmState >  (t2)->lmState)
#endif

typedef struct _TokenSet TokenSet;      /* contains n tokens with different LM states */

struct _TokenSet {
   TokScore score;
   RelToken *relTok;            /*# sorted by LMState? */
   unsigned short n;
   unsigned int id;             /*####  should be only 2byte short! */
};
   
struct _LexNodeInst {           /* attached to active LexNode's, contains info about tokens */
   LexNode *node;
   TokenSet *ts;                /* array of TokenSets; one per state (incl. entry and exit) */
   TokScore best;               /* score of best token in any HMM state and LM state,
                                   used for pruning */
   LexNodeInst *next;           /* next instance in linked linst for this layer */
                                /*#### need LM lookahead info, i.e. list of (LMState, LogFloat) */
};
   

#ifdef COLLECT_STATS
#ifdef COLLECT_STATS_ACTIVATION
#  define  STATS_MAXT 100
#endif
typedef struct _Stats Stats;    /* statistics about pruning etc. */
struct _Stats {
   unsigned long nTokSet;
   unsigned long sumTokPerTS;
   unsigned long nActive;
   unsigned long nActivate;
   unsigned long nDeActivate;
   unsigned long nFrames;
   unsigned long nLMlaCacheHit;
   unsigned long nLMlaCacheMiss;
#ifdef COLLECT_STATS_ACTIVATION
   unsigned long lnDeadT[STATS_MAXT+1];
   unsigned long lnLiveT[STATS_MAXT+1];
   unsigned long lnINF;
#endif
};
#endif
   

/**** LM lookahead cache */

typedef  struct _LMLACacheEntry LMLACacheEntry;

struct _LMLACacheEntry {
   LMState src;
   unsigned int idx;
   LMTokScore prob;
   struct _LMLACacheEntry *next;
};

#define LMLA_CACHE_SIZE 1000001
#define LMLA_HASH(src,idx) ((((unsigned int) src)^((unsigned int) idx)) % LMLA_CACHE_SIZE)

/* output prob cache */

typedef struct _OutPCache OutPCache;
struct _OutPCache {
   int block;
   int nMix;
   int nStates;
   int *stateT;
   int *mixT;
   LogFloat *stateOutP;
   LogFloat *mixOutP;
   int cacheHit;
   int cacheMiss;
};


#if 0
/* LM prob cache (includes lookahead and full probs) */

/* LMCacheTrans -- caches value of LMTransProb() */
typedef struct _LMCacheTrans LMCacheTrans;
struct _LMCacheTrans {
   PronId pronid;
   LMState dest;
   LMTokScore prob;
   /*    LMCacheTrans *next; */
};

#define LMCACHE_NTRANS 101         /* number of LMCacheTrans entries */

/* LMCacheLA  -- caches values of LMLookAheadProb() */
typedef struct _LMCacheLA LMCacheLA;
struct _LMCacheLA {
   int idx;
   LMTokScore prob;
   /*    LMCacheLA *next; */
};

#define LMCACHE_NLA 10007       /* number of LMCacheLA entries */


/* LMStateCache -- one of these per LMState */
typedef struct _LMStateCache LMStateCache;
struct _LMStateCache {
   LMState *src;
   int t;               /* last access time, used for aging */
   LMCacheTrans trans[LMCACHE_NTRANS];
   LMCacheLA la[LMCACHE_NLA];
   LMStateCache *next;
};

#define LMCACHE_NSTATE 503    /* number of LMStateCaches */

#endif


/* LMCacheLA  -- caches values of LMLookAheadProb() */
typedef struct _LMCacheLA LMCacheLA;
struct _LMCacheLA {
   LMState *src;
   LMTokScore prob;
   /*    LMCacheLA *next; */
};

#define LMCACHE_NLA 64    /* number of different LMStates kept per NodeCache */

/* LMNodeCache -- one of these for each LexNode with a unique lmlaIdx */
typedef struct _LMNodeCache LMNodeCache;
struct _LMNodeCache {
   int idx;
   int t;
   int nEntries;
   int nextFree;
   int size;
   LMCacheLA la[LMCACHE_NLA];
};


typedef struct _LMCache LMCache;
struct _LMCache {
   MemHeap nodeHeap;            /* MHEAP for LMNodeCache entries */
#if 0
   MemHeap transHeap;           /* MHEAP for LMCacheTrans entries */
   MemHeap laHeap;              /* MHEAP for LMCacheLA entries */
#endif   
   int nNode;
   LMNodeCache **node;
   int transHit;
   int transMiss;
   int laHit;
   int laMiss;
};




/**** decoder instance */

typedef struct _DecoderInst DecoderInst;  /* contains all state information about one instance
                                             of the decoder 
                                             the aim is to share as much info as possible across
                                             instances (i.e. the LexNet), but this is tough... */

struct _DecoderInst {
   LexNet *net;                 /* network, contains pointers to Vocab and HMMSet */
                /* current frame?
                   info about current utterance (source, filename, length)?
                */
   HMMSet *hset;
   FSLM *lm;

   MemHeap heap;                /* MSTACK for general allocation */
   MemHeap nodeInstanceHeap;    /* MHEAP for LexNodeInsts */
   MemHeap weHypHeap;           /* MHEAP for word end hyps */
   MemHeap altweHypHeap;        /* MHEAP for alt word end hyps (for latgen) */
   MemHeap *tokSetHeap;         /* MHEAPs for N TokenSet arrays */
   MemHeap relTokHeap;          /* MHEAP for RelToken arrays (dec->nTok-1 elements) */
   MemHeap lrelTokHeap;         /* MHEAP for larger size RelToken arrays (e.g. 6 * dec->nTok-1 elements) */

   TokenSet **tempTS;           /* temp tokset arrays for PropagateInternal() */
   RelToken *winTok;            /* RelTok array fro MergeTokSet() */

   int maxNStates;              /* max number of states in a HMM in HMMSet */
   int nLayers;                 /* nuber of node layers */
   LexNodeInst **instsLayer;    /* array of pointers to the linked list of 
                                   active LexNodeInsts in each layer */
   char *utterFN;               /* name of current utterance */
   Observation *obs;            /* Observation for current frame */
   Observation *obsBlock[MAXBLOCKOBS]; /* block of current and future Observations */
   int nObs;                    /* num of valid obs in bock */
   HTime frameDur;              /* Duration of one frame in seconds */
   int frame;                   /* current frame number */

   int nTok;                    /* max number of tokens per state */

   Boolean latgen;              /* generate lattices or just 1-bet? */
   LogFloat bestScore;          /* score of best token */
   LexNodeInst *bestInst;       /* instance containing best token */

   int maxModel;                /* for max model pruning (set by -u cmd line option) */
   TokScore beamWidth;          /* max beamWidth main beam (set by -t cmd line option) */
   TokScore weBeamWidth;        /* wordend beam width (set by -v cmd line option) */
   TokScore zsBeamWidth;        /* Z-S beam width (set by -v cmd line option) */
   TokScore curBeamWidth;       /* current dynamic beamWidth (due to max model pruning) */
   TokScore beamLimit;          /* threshold of the main beam (bestScore - beamWidth) */

   RelTokScore relBeamWidth;    /* beamWidth of relative tokenset  beam */
                                /* (set by -t BW RBW cmd line option) */

   LogFloat insPen;             /* word insertion penalty */
   float acScale;               /* acoustic scaling factor */
   float pronScale;             /* pronunciation scaling factor */
   float lmScale;               /* LM scaling factor */

   float maxLMLA;               /* maximum jump in LM lookahead per model */

   Boolean fastlmla;            /* use fast LM lookahead, i.e. back-off to bigram states */
   LogFloat fastlmlaBeam;       /* beam in which to use full lmla */
   
   Boolean useHModel;           /* use normal HModel OutP() functions? */
   /*    outP cache */
   OutPCache *outPCache;        /* cache of outP values for block of observations */

   /* LM lookahead cache */
   LMCache *lmCache;

   /* relToken set identifier */
   unsigned int tokSetIdCount;/* max id used so far for token sets */

   StateInfo_lv *si;

#ifdef MODALIGN
   Boolean modAlign;
   MemHeap modendHypHeap;       /* MHEAP for word end hyps */
#endif

#ifdef COLLECT_STATS
   Stats stats;                 /* statistics about pruning etc. */
#endif

   /* Phone posterior info */
   int nPhone;
   LabId monoPhone[100];           /* #### hard limit -- fix this */
   LogDouble *phonePost;
   int *phoneFreq;
};


/*
#define TOK_TOTSCORE(t) ((t)->score + (t)->lmscore)
#define TOK_LMSCORE(t) ((t)->lmscore)
*/

void InitLVRec(void);

DecoderInst *CreateDecoderInst(HMMSet *hset, FSLM *lm, int nTok, Boolean latgen, 
                               Boolean useHModel,
                               int outpBlocksize, Boolean doPhonePost,
                               Boolean modAlign);
void InitDecoderInst (DecoderInst *dec, LexNet *net, HTime sampRate, LogFloat beamWidth, 
                      LogFloat relBeamWidth, LogFloat weBeamWidth, LogFloat zsBeamWidth,
                      int maxModel,
                      LogFloat insPen, float acScale, float pronScale, float lmScale,
                      LogFloat fastlmlaBeam);

void CleanDecoderInst (DecoderInst *dec);
void ProcessFrame (DecoderInst *dec, Observation **obsBlock, int nObs,
                   AdaptXForm *xform);

Transcription *TraceBack (MemHeap *heap, DecoderInst *dec);
Lattice *LatTraceBack (MemHeap *heap, DecoderInst *dec);

void ReFormatTranscription(Transcription *trans,HTime frameDur,
                           Boolean states,Boolean models,Boolean triStrip,
                           Boolean normScores,Boolean killScores,
                           Boolean centreTimes,Boolean killTimes,
                           Boolean killWords,Boolean killModels);

#ifdef __cplusplus
}
#endif

#endif  /* _HLVREC_H_ */

/*  CC-mode style info for emacs
 Local Variables:
 c-file-style: "htk"
 End:
*/
