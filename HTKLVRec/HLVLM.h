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
/*         File: HLVLM.h Language model data types for         */
/*                       HTK LV Decoder                        */
/* ----------------------------------------------------------- */

#ifndef _HLVLM_H_
#define _HLVLM_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "HMem.h"       /* for Ptr */
#include "HDict.h"      /* for Vocab */
#include "HLVNet.h"     /* for PronId */
#include "HNet.h"       /* for Lattice */

typedef Ptr LMState;

typedef struct _FSLM FSLM;

/* the following definitions should probably be private */

typedef struct _FSLM_ngram FSLM_ngram;
typedef struct _FSLM_latlm FSLM_latlm;

typedef enum {fslm_ngram, fslm_latlm} FSLMType;

struct _FSLM {
   FSLMType type;
   union {
      FSLM_ngram *nglm;
      FSLM_latlm *latlm;
   } data;
   LMState initial;
   PronId startPronId;
   PronId endPronId;
   char *name;
   MemHeap *heap;
   LogFloat (*lookahead) (FSLM *lm, LMState src, PronId minPron, PronId maxPron);
   LogFloat (*transProb) (FSLM *lm, LMState src, PronId pronId, LMState *dest);
};


/**********  Lattice LM  */

typedef struct _FSLM_LatNode  FSLM_LatNode;

struct _FSLM_latlm {
   int nnodes;
   FSLM_LatNode *fslmln;
};

typedef struct _FSLM_LatArc {
   PronId word;                 /* _Pron_ id !! */
   float prob;                  /* probability */
   FSLM_LatNode *dest;
} FSLM_LatArc;

struct _FSLM_LatNode {
   Word word;
   int nfoll;
   FSLM_LatArc *foll;
};



/* #### the following is mostly the HLM implementation -- fix! */

#define MAX_LMID 65534          /* Max number of words */
#define NSIZE 4                 /* Max length of ngram 2==bigram etc */

/* two different types of identifiers are used:

   LMId         one unique identifier per word
   PronId       one unique identifier per (word,pron) pair.
                ordered so that lookahead only needs interval of 
                _consecutive_ PronIds

   #PronId >= #LMId

   In the NEntry history entries LMIds are used to save space and simplify 
   the LM reading code. 
   In the sparse arrays for the predicted word PronIds are used.

   The LMIds are based on the order in the on-disk LM. The alternative would 
   have been to introduce some kind of "normalised PronId" (e.g. the first PronId 
   of a word), this would only complicated the LM reading code and not actually save
   anything (i.e. we need a mapping table in either case.


*/
/* now to be set in the config.h files */
/* #define LM_NGRAM_INT */

#ifdef LM_NGRAM_INT
   typedef unsigned short NGLM_Prob;
#define NGLM_PROB_TO_FLOAT(x) (((float)(x))* -0.001)
#define FLOAT_TO_NGLM_PROB(x) ((NGLM_Prob) ((-x)/0.001))
#define NGLM_PROB_LZERO 65535
#define NGLM_PROB_ZERO 0
#define NGLM_PROB_GREATER(x,y) ((x)<(y))
#define NGLM_PROB_ADD(x,y)((((NGLM_Prob)((x)+(y))<(x))||((NGLM_Prob)((x)+(y))<(y)))?NGLM_PROB_LZERO:((x)+(y)))
#else
   typedef float NGLM_Prob;
#define NGLM_PROB_TO_FLOAT(x) (x)
#define FLOAT_TO_NGLM_PROB(x) (x)
#define NGLM_PROB_LZERO LZERO
#define NGLM_PROB_ZERO 0.0
#define NGLM_PROB_GREATER(x,y) ((x)>(y))
#endif

typedef struct sentry {         /* HLM NGram probability */
   PronId word;                 /* _Pron_ id !! */
   NGLM_Prob prob;              /* probability */
} SEntry;

typedef struct nentry {         /* HLM NGram history */
   LMId word[NSIZE-1];          /* Word history representing this entry */
   LMId nse;                    /* Number of ngrams for this entry */
   NGLM_Prob bowt;              /* Back-off weight */
   SEntry *se;                  /* Array[0..nse-1] of ngram probabilities */
   struct nentry *nebo;                /* NEntry for back-off */
   struct nentry *link;         /* Next entry in hash table */
} NEntry;

struct _FSLM_ngram {
   MemHeap *heap;
   int nsize;                   /* Unigram==1, Bigram==2, Trigram==3 */
   unsigned int hashsize;       /* Size of hashtab (adjusted by lm counts) */
   NEntry **hashtab;            /* Hash table for finding NEntries */
   int counts[NSIZE+1];         /* Number of [n]grams */
   Vocab *vocab;                /* Vocab used to find prons of words */
   int vocSize;                 /* Core LM size */
   NGLM_Prob *unigrams;         /* Unigram probabilities indexed by PronId! */
   LabId *lablist;              /* Lookup table for LabIds from LMId */ 
   Word *wordlist;              /* Lookup table for Words from LMId */
   LMId *pronId2LMId;           /* PronId -> LMId mapping array [1..voc->nprons] 
                                   needed for LM histories */
};

/*------------------------*/


#define BIN_ARPA_HAS_BOWT 1
#define BIN_ARPA_INT_LMID 2


void InitLVLM (void);
FSLM *CreateLMfromLat (MemHeap *heap, char *latfn, Lattice *lat, Vocab *vocab);
FSLM *CreateLM (MemHeap *heap, char *fn, char *startWord, char *endWord, Vocab *vocab);

LogFloat LMTransProb (FSLM *lm, LMState src, PronId word, LMState *dest); 

LMState LMInitial (FSLM *lm);

LogFloat LMLookAhead (FSLM *lm, LMState src, PronId minPron, PronId maxPron);

LMState Fast_LMLA_LMState (FSLM *lm, LMState src);


#ifdef __cplusplus
}
#endif

#endif  /* _HLVLM_H_ */


/*  CC-mode style info for emacs
 Local Variables:
 c-file-style: "htk"
 End:
*/
