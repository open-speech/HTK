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
/*         File: HLVNet.c Network data types for HTK LV Decoder*/
/* ----------------------------------------------------------- */

/*
   every LexNet has a single root node


   LexNode types:
	- root
	    single root node, providing single starting point for
	    Token propagation, i.e. during utterance initialisation a
	    single token is put here.
	- word end
	    corresponds to a uniqe pronunciation
	- context
	    special node to connect two large sets of nodes,
            taking only |A| + |B| links instead of |A| * |B|
	- model
	    "normal" node corresponding to a (context-dependent) HMM,
	    storing TokenSets for all states

*/

#ifndef _HLVNET_H_
#define _HLVNET_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "HLVLM.h"      /* for LMId */
#include "HNet.h"       /* for Lattice */



typedef struct _LexNode LexNode;
typedef struct _LexNodeInst LexNodeInst;        /* structure defined in HLVRec */

typedef struct _LMlaTree LMlaTree;


typedef enum _LexNodeType {
  LN_WORDEND, LN_CON, LN_MODEL
} LexNodeType;


struct _LexNode {
   LexNodeInst *inst;		/* model instance or NULL if inactive */
   union {
      HLink hmm;                /* #### switch to HMM Ids (2 byte ints) */
      PronId pron;
   } data;
   unsigned char type;          /* using char instead of enum can save 4 bytes! */
   unsigned char flags;         /* flags like LEFT_RIGHT */
   short nfoll;
   LexNode **foll;		/* array of following nodes */
   unsigned int lmlaIdx;        /* Idx of corresponding node in compressed LM 
                                   lookahead tree. can be 0 */
#ifdef COLLECT_STATS_ACTIVATION
   int eventT;          /* frame # of last (de)activation */
#endif
};

/* ####GE  maybe we can get rid of the links alltogether? we don't
           have any extra info (e.g. scores) on the links */
struct _LexLink {
   LexNode *end;
};

typedef struct _LexNet {
   LexNode *node;               /* pointer to array of LexNodes */
   int nNodes;

   int nLayers;                 /* nuber of node layers */
   int wordEndLayerId;          /* id of layer where token's time and score are copied to weHyp */
   int spSkipLayer;             /* id of layer where tokens get pronprob added and bypass sp */
   HLink hmmSP;                 /* pointer to "sp" hmm for spSkipLayer handling */
   LexNode *lnSEsp;             /* sp lexnode leading to SENT_END */
   LexNode *lnSEsil;            /* sil lexnode leading to SENT_END */
   
   LexNode **layerStart;        /* array of pointers to the first node in each layer */

   MemHeap *heap;
   Vocab *voc;
   char *vocabFN;
   Boolean silDict;             /* does dict contain -/sp/sil variants and pronprobs? */
   HMMSet *hset;

   LexNode *start;              /* single start node */
   LexNode *end;                /* single end node */

   PronId startPron;            /* pron of <s> */
   PronId endPron;              /* pron of </s> */

   Pron *pronlist;              /* array [1..voc->nprons]  of Prons for given PronId */

   LMlaTree *laTree;            /* look ahead tree */
} LexNet;



#define LEX_CON_HASH_SIZE 337
#define LEX_MOD_HASH_SIZE 337

#define NLAYERS 9

enum _LayerId {
   /*   LAYER_A=0, LAYER_AB, LAYER_BY, LAYER_WE, LAYER_YZ, LAYER_Z, LAYER_ZS, LAYER_SIL, LAYER_SA */
   LAYER_Z=0, LAYER_ZS=1, LAYER_SIL=2, LAYER_SA=3, LAYER_A=4, LAYER_AB=5,
   LAYER_BY=6, LAYER_WE=7, LAYER_YZ=8
} LayerId;


/*   temporary lexicon network structure used for creating the net,
     more explicit info and easier to manipulate data structures.
     first the temp LexNet is created and then converted into the more
     compact LexNet used for decoding.
*/

/* Temp Connection (Null) Node in lexicon network
   identified by right and left (1 phone) context
   this is only used during nextwork construction.
*/

typedef struct _TLexLink TLexLink;
typedef struct _TLexNode TLexNode;
typedef struct _TLexNode TLexConNode;

struct _TLexLink {
   TLexLink *next;              /* next link from this start node */
   TLexNode *start;
   TLexNode *end;
};

struct _TLexNode {
   TLexNode *next;               /* next node in hash table for some net part (A, B..Y or Z) */
   TLexNode *chain;              /* global chain of all TLexNodes */

   LexNodeType type;
   HLink hmm;
   Pron pron;
   LabId lc;            /* left context phone */
   LabId rc;            /* right context phone */
   int nlinks;
   TLexLink *link;		/* linked list of  TLexLinks to successor nodes */
   LexNode *ln;        /* the actual Lexicon Node */
   int layerId;

   int loWE;            /* lowest WE LMId reachable from here */
   int hiWE;            /* highest WE LMId reachable from here */
   int lmlaIdx;         /* index of node in (compressed) LMlaTree */
};


#if 0
typedef struct _TLexConNode TLexConNode;

struct _TLexConNode {
   TLexConNode *next;   /* next node in hash table chain */
   LabId lc;            /* left context phone */
   LabId rc;            /* right context phone */
   int nlinks;
   TLexLink *link;		/* linked list of  TLexLinks to successor nodes */
   TLexNode *ln;        /* the actual Lexicon Node */
};
#endif

typedef struct _TLexNet {
   MemHeap *heap;
   Vocab *voc;
   HMMSet *hset;
   LabId startId;       /* id of STARTWORD (from config) */
   LabId endId;         /* id of ENDWORD (from config) */

   TLexNode *start;     /* start node of network */
   TLexNode *end;       /* end node of network */

   TLexNode *root;	/* global chain of all nodes */
   Boolean silDict;     /* does dict contain -/sp/sil variants and pronprobs? */
   TLexNode *lnSEsp;             /* sp lexnode leading to SENT_END */
   TLexNode *lnSEsil;            /* sil lexnode leading to SENT_END */

   int nlexA;           /* array of initial phones */
   LabId *lexA;
   int nlexZ;           /* array of final phones */
   LabId *lexZ;

   int nlexP;           /* array of phones in single phone words*/
   LabId *lexP;

   /* the following all correspond to real nodes in the final LexNet */
   int nlexAB;          /* hastable of A-B contexts */
   TLexNode *lexABhash[LEX_CON_HASH_SIZE];
   int nlexYZ;         /* hastable of Y-Z contexts */
   TLexNode *lexYZhash[LEX_CON_HASH_SIZE];
   int nlexZS;         /* hastable of Z - (A+'sil') contexts */
   TLexNode *lexZShash[LEX_CON_HASH_SIZE];
   int nlexSA;         /* hastable of (Z+'sil') - A contexts */
   TLexNode *lexSAhash[LEX_CON_HASH_SIZE];

   int nNodeA;         /* hastable of A LexNodes */
   TLexNode *nodeAhash[LEX_MOD_HASH_SIZE];
   int nNodeZ;         /* hastable of Z LexNodes */
   TLexNode *nodeZhash[LEX_MOD_HASH_SIZE];

   int nNodeBY;         /* linked list of nodes in main prefix tree B -- Y */
   TLexNode *nodeBY;
   int nNodeSIL;         /* linked list of silencs (sil/sp) nodes between ZS and SA */
   TLexNode *nodeSIL;

   int nNodesLayer[NLAYERS]; /* number of nodes in each layer */

   int nPronIds;        /* number of wordend Ids assigned, should be = voc->nprons */
   int lmlaCount;       /* number of unique look ahead intervals */
} TLexNet;


/* LM look ahead tree.

     Each node in the prefix part of the LexNet points to an entry
     in LMlaTree. An entry of 0 indicates that the lookahead
     information doesn't need to be updated. This occur if the set
     of reachable word ends is the same as for the predecessor node,
     e.g.:
               3-->0
              /
     1-->0-->0
              \
               2
*/

/* simple LMlaNode corresponding to a continuous range of PronIds */
typedef struct _LMlaNode {
   LMId loWE;
   LMId hiWE;
} LMlaNode;


/* complex LMlaNode: corresponds to a set of lmlaNodes (potentially complex?) 
     used in the A Layer (and possibly in the YZ, Z, ZS, SA layers?) 
*/
typedef struct _CompLMlaNode {
   int n;
   int *lmlaIdx;
} CompLMlaNode;

struct _LMlaTree {
   int nNodes;                  /* number of nodes in LM LA tree */
   LMlaNode *node;              /* [0..nNodes-1] arry of entries */
   int nCompNodes;              /* number of complex nodes in LM LA tree */
   CompLMlaNode *compNode;      /* [0..nCompNodes-1] arry of entries */
};


void InitLVNet(void);

/* build lexicon network for recognition for Vocab and HMMSet */

LexNet *CreateLexNet (MemHeap *heap, Vocab *voc, HMMSet *hset, 
                      char *startWord, char *endWord, Boolean silDict);


void ConvertSilDict (Vocab *voc, LabId spLab, LabId silLab, 
                     LabId startLab, LabId endLab);
void MarkAllProns (Vocab *vocab);
void MarkAllWords (Vocab *vocab);
void UnMarkAllWords (Vocab *vocab);
void MarkAllWordsfromLat (Vocab *voc, Lattice *lat, Boolean silDict);


#ifdef __cplusplus
}
#endif

#endif  /* _HLVNET_H_ */

/*  CC-mode style info for emacs
 Local Variables:
 c-file-style: "htk"
 End:
*/
