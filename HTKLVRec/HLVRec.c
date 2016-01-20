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
/*         File: HLVRec.h Viterbi recognition engine for       */
/*                        HTK LV Decoder                       */
/* ----------------------------------------------------------- */


char *hlvrec_version = "!HVER!HLVRec:   3.4.1 [GE 12/03/09]";
char *hlvrec_vc_id = "$Id: HLVRec.c,v 1.1.1.1 2006/10/11 09:54:56 jal58 Exp $";


#include "HShell.h"
#include "HMem.h"
#include "HMath.h"
#include "HWave.h"
#include "HLabel.h"
#include "HAudio.h"
#include "HParm.h"
#include "HDict.h"
#include "HModel.h"
#include "HUtil.h"
#include "HNet.h"       /* for Lattice -- move to HLattice? */
#include "HAdapt.h"

#include "config.h"

#include "HLVNet.h"
#include "HLVRec.h"
#include "HLVModel.h"

#include <string.h>
#include <assert.h>

#define PRUNE

/* ----------------------------- Trace Flags ------------------------- */

#define T_TOP 0001         /* Trace  */
#define T_BEST 0002        /* print best token in each frame */
#define T_WORD 0004        /* word end handling */
#define T_TOKSTATS 0010    /* print token/active node stats in each frame */
#define T_PRUNE 0020       /* pruning (need to define DEBUG_TRACE for this!) */
#define T_ACTIV 0040       /* node activation (need to define DEBUG_TRACE for this!) */
#define T_PROP 0100        /* details of token propagation (need to define DEBUG_TRACE for this!) */
#define T_LAT 0200         /* details of lattice generation */
#define T_GC 0400          /* details of garbage collection */
#define T_MEM 01000          /* details of memory usage */

static int trace=0;
static ConfParam *cParm[MAXGLOBS];      /* config parameters */
static int nParm = 0;

static LogFloat maxLMLA = -LZERO; /* maximum jump in LM lookahead per model */

static Boolean buildLatSE = FALSE;/* build lat from single tok in SENTEND node */

static Boolean forceLatOut = TRUE;/* always output lattice, even when no token survived */

static int gcFreq = 100;          /* run Garbage Collection every gcFreq frames */

static Boolean pde = FALSE;      /* partial distance elimination */

static Boolean useOldPrune = FALSE;     /* backward compatibility for max model and reltok pruning etc. */
static Boolean mergeTokOnly = TRUE;     /* if merge token set with pruning */
static float maxLNBeamFlr = 0.8;        /* maximum percentile of glogal beam for max model pruning */
static float dynBeamInc = 1.3;          /* dynamic beam increment for max model pruning */
#define LAYER_SIL_NTOK_SCALE 6          /* SIL layer re-adjust token set size e.g. 6 */

/* -------------------------- Global Variables --------------------- */

RelToken startTok = {NULL, NULL, 0.0, 0.0, NULL};
#if 0
Token nullTok = {NULL, LZERO, LZERO, NULL};
Token debug_nullTok = {(void *) 7, LZERO, LZERO, NULL};
#endif
MemHeap recCHeap;                       /* CHEAP for small general allocation */
                                        /* avoid wherever possible! */
static AdaptXForm *inXForm;

/* --------------------------- Prototypes ---------------------- */

/* HLVRec.c */

void InitLVRec(void);
DecoderInst *CreateDecoderInst(HMMSet *hset, FSLM *lm, int nTok, Boolean latgen, 
                               Boolean useHModel,
                               int outpBlocksize, Boolean doPhonePost,
                               Boolean modAlign);
static Boolean CheckLRTransP (SMatrix transP);
void InitDecoderInst (DecoderInst *dec, LexNet *net, HTime sampRate, LogFloat beamWidth, 
                      LogFloat relBeamWidth, LogFloat weBeamWidth, LogFloat zsBeamWidth,
                      int maxModel, 
                      LogFloat insPen, float acScale, float pronScale, float lmScale,
                      LogFloat fastlmlaBeam);
void CleanDecoderInst (DecoderInst *dec);
static TokenSet *NewTokSetArray(DecoderInst *dec, int N);
static TokenSet *NewTokSetArrayVar(DecoderInst *dec, int N, Boolean isSil);
static LexNodeInst *ActivateNode (DecoderInst *dec, LexNode *ln);
static void DeactivateNode (DecoderInst *dec, LexNode *ln);
static void PruneTokSet (DecoderInst *dec, TokenSet *ts);
void ReFormatTranscription(Transcription *trans,HTime frameDur,
                           Boolean states,Boolean models,Boolean triStrip,
                           Boolean normScores,Boolean killScores,
                           Boolean centreTimes,Boolean killTimes,
                           Boolean killWords,Boolean killModels);

/* HLVRec-propagate.c */
static int winTok_cmp (const void *v1,const void *v2);
static void MergeTokSet (DecoderInst *dec, TokenSet *src, TokenSet *dest, 
                         LogFloat score, Boolean prune);
static void PropagateInternal (DecoderInst *dec, LexNodeInst *inst);
#ifdef MODALIGN
void UpdateModPaths (DecoderInst *dec, TokenSet *ts, LexNode *ln);
#endif
static void PropIntoNode (DecoderInst *dec, TokenSet *ts, LexNode *ln, Boolean updateLMLA);
static void PropagateExternal (DecoderInst *dec, LexNodeInst *inst, 
                               Boolean handleWE, Boolean wintTree);
static void HandleWordend (DecoderInst *dec, LexNode *ln);
static void UpdateWordEndHyp (DecoderInst *dec, LexNodeInst *inst);
static void AddPronProbs (DecoderInst *dec, TokenSet *ts, int var);
void HandleSpSkipLayer (DecoderInst *dec, LexNodeInst *inst);
void ProcessFrame (DecoderInst *dec, Observation **obsBlock, int nObs,
                   AdaptXForm *xform);

/* HLVRec-LM.c */
static void UpdateLMlookahead(DecoderInst *dec, LexNode *ln);
static LMCache *CreateLMCache (DecoderInst *dec, MemHeap *heap);
static void FreeLMCache (LMCache *cache);
static void ResetLMCache (LMCache *cache);
static int LMCacheState_hash (LMState lmstate);
LMNodeCache* AllocLMNodeCache (LMCache *cache, int lmlaIdx);
static LMTokScore LMCacheTransProb (DecoderInst *dec, FSLM *lm, 
                                    LMState src, PronId pronid, LMState *dest);
LMTokScore LMLA_nocache (DecoderInst *dec, LMState lmState, int lmlaIdx);
static LMTokScore LMCacheLookaheadProb (DecoderInst *dec, LMState lmState, 
                                        int lmlaIdx, Boolean fastlmla);
/* HLVRec-traceback.c */
static void PrintPath (DecoderInst *dec, WordendHyp *we);
static void PrintTok(DecoderInst *dec, Token *tok);
static void PrintRelTok(DecoderInst *dec, RelToken *tok);
static void PrintTokSet (DecoderInst *dec, TokenSet *ts);
TokenSet *BestTokSet (DecoderInst *dec);
Transcription *TraceBack(MemHeap *heap, DecoderInst *dec);
static void LatTraceBackCount (DecoderInst *dec, WordendHyp *path, int *nnodes, int *nlinks);
static void Paths2Lat (DecoderInst *dec, Lattice *lat, WordendHyp *path,
                       int *na);
Lattice *LatTraceBack (MemHeap *heap, DecoderInst *dec);
#ifdef MODALIGN
LAlign *LAlignFromModpath (DecoderInst *dec, MemHeap *heap,
                           ModendHyp *modpath, int wordStart, short *nLAlign);
LAlign *LAlignFromAltModpath (DecoderInst *dec, MemHeap *heap,
                              ModendHyp *modpath, ModendHyp *mainModpath,
                              int wordStart, short *nLAlign);
void PrintModPath (DecoderInst *dec, ModendHyp *m);
void CheckLAlign (DecoderInst *dec, Lattice *lat);
#endif
AltWordendHyp *FakeSEpath (DecoderInst *dec, RelToken *tok, Boolean useLM);
WordendHyp *AltPathList2Path (DecoderInst *dec, AltWordendHyp *alt, PronId pron);
WordendHyp *BuildLattice (DecoderInst *dec);
AltWordendHyp *BuildLatAltList (DecoderInst *dec, TokenSet *ts, Boolean useLM);
WordendHyp *BuildForceLat (DecoderInst *dec);

/* HLVRec-GC.c */
#ifdef MODALIGN
static void MarkModPath (ModendHyp *m);
#endif
static void MarkPath (WordendHyp *path);
static void MarkTokSet (TokenSet *ts);
static void SweepPaths (MemHeap *heap);
static void SweepAltPaths (MemHeap *heap);
#ifdef MODALIGN
static void SweepModPaths (MemHeap *heap);
#endif
static void GarbageCollectPaths (DecoderInst *dec);


/* HLVRec-outP.c */
static void ResetOutPCache (OutPCache *cache);
static OutPCache *CreateOutPCache (MemHeap *heap, HMMSet *hset, int block);
LogFloat SOutP_ID_mix_Block(HMMSet *hset, int s, Observation *x, StreamElem *se);
static LogFloat cOutP (DecoderInst *dec, Observation *x, HLink hmm, int state);
void OutPBlock_HMod (StateInfo_lv *si, Observation **obsBlock, 
                     int n, int sIdx, float acScale, LogFloat *outP, int id);


/* HLVRec-misc.c */
void CheckTokenSetOrder (DecoderInst *dec, TokenSet *ts);
static void CheckTokenSetId (DecoderInst *dec, TokenSet *ts1, TokenSet *ts2);
static WordendHyp *CombinePaths (DecoderInst *dec, RelToken *winner, RelToken *loser, LogFloat diff);
void Debug_DumpNet (LexNet *net);
void Debug_Check_Score (DecoderInst *dec);
void InitPhonePost (DecoderInst *dec);
void CalcPhonePost (DecoderInst *dec);
void AccumulateStats (DecoderInst *dec);


/* --------------------------- Initialisation ---------------------- */


#include "HLVRec-propagate.c"
#include "HLVRec-LM.c"
#include "HLVRec-traceback.c"
#include "HLVRec-GC.c"
#include "HLVRec-outP.c"
#include "HLVRec-misc.c"


/* EXPORT->InitLVRec: register module & set configuration parameters */
void InitLVRec(void)
{
   double f;
   int i;
   Boolean b;
   
   Register (hlvrec_version, hlvrec_vc_id);
   nParm = GetConfig("HLVREC", TRUE, cParm, MAXGLOBS);
   if (nParm>0){
      if (GetConfInt(cParm,nParm,"TRACE",&i)) trace = i;
      if (GetConfFlt (cParm, nParm, "MAXLMLA", &f))
         maxLMLA = f;
      if (GetConfBool (cParm, nParm, "BUILDLATSENTEND",&b)) buildLatSE = b;
      if (GetConfBool (cParm, nParm, "FORCELATOUT",&b)) forceLatOut = b;
      if (GetConfInt (cParm, nParm,"GCFREQ", &i)) gcFreq = i;
      if (GetConfBool (cParm, nParm, "PDE",&b)) pde = b;
      if (GetConfBool (cParm, nParm, "USEOLDPRUNE",&b)) useOldPrune = b;
      if (GetConfBool (cParm, nParm, "MERGETOKONLY",&b)) mergeTokOnly = b;
      if (GetConfFlt (cParm, nParm, "MAXLNBEAMFLR", &f)) maxLNBeamFlr = f;
      if (GetConfFlt (cParm, nParm, "DYNBEAMINC", &f)) dynBeamInc = f;

      if (useOldPrune) {
         mergeTokOnly = FALSE; maxLNBeamFlr = 0.0; dynBeamInc = 1.1;
      }
   }

#if 0
   printf ("sizeof(Token) = %d\n", sizeof (Token));
   printf ("sizeof(LexNodeInst) = %d\n", sizeof (LexNodeInst));
   printf ("sizeof(WordendHyp) = %d\n\n", sizeof (WordendHyp));
#endif

   CreateHeap (&recCHeap, "Decoder CHEAP", CHEAP, 1, 1.5, 10000, 100000);
}


/* --------------------------- the real code  ---------------------- */


/* CreateDecoderInst

     Create a new instance of the decoding engine. All state information is stored
     here. 
     #### Ideally instances should share other structures (i.e.
          LexNets) this is not implemented, yet.
*/
DecoderInst *CreateDecoderInst(HMMSet *hset, FSLM *lm, int nTok, Boolean latgen, 
                               Boolean useHModel,
                               int outpBlocksize, Boolean doPhonePost,
                               Boolean modAlign)
{
   DecoderInst *dec;
   int i, N;
   char buf[MAXSTRLEN];

   dec = (DecoderInst *) New (&recCHeap, sizeof (DecoderInst));

   dec->lm = lm;
   dec->hset = hset;
   dec->useHModel = useHModel;
   /*    dec->net = net; */

   /* create compact State info. This can change number of shared states! */
   /* #### this is ugly as we end up doing this twice, if we use adaptation! */
   dec->si = ConvertHSet (&gcheap, hset, dec->useHModel);

   CreateHeap (&dec->heap, "Decoder Instance heap", MSTAK, 1, 1.5, 10000, 100000);

   CreateHeap (&dec->nodeInstanceHeap, "Decoder NodeInstance heap", 
               MHEAP, sizeof (LexNodeInst), 1.5, 1000, 10000);


   dec->nTok = nTok;
   dec->latgen = latgen;
   dec->nLayers = 0;
   dec->instsLayer = NULL;

   /* alloc & init Heaps for TokenSets */
   N = MaxStatesInSet (dec->hset);
   dec->maxNStates = N;

   dec->tokSetHeap = (MemHeap *) New (&dec->heap, N * sizeof (MemHeap));


   /* #### make initial size of heap blocks smaller,
      or don't alloc unneeded ones in the first place (scan HMMSet) */
   for (i = 0; i < N; ++i) {
   sprintf (buf, "Decoder %d TokenSet heap", i+1);
      CreateHeap (&dec->tokSetHeap[i], buf, 
                  MHEAP, (i+1) * sizeof (TokenSet), 9, 10, 5000);
   }   

   dec->tempTS = (TokenSet **) New (&dec->heap, N * sizeof (TokenSet *));


   /* alloc Heap for RelToken arrays */
   CreateHeap (&dec->relTokHeap, "Decoder RelToken array heap",
               MHEAP, dec->nTok * sizeof (RelToken), 1, 1000, 5000);

   CreateHeap (&dec->lrelTokHeap, "Decoder RelToken array heap",
               MHEAP, LAYER_SIL_NTOK_SCALE * dec->nTok * sizeof (RelToken), 1, 1000, 5000);   
   
   /* alloc heap for word end hyps */
   CreateHeap (&dec->weHypHeap, "WordendHyp heap", MHEAP, sizeof (WordendHyp), 
               1.0, 80000, 800000);
   if (dec->latgen) {
      CreateHeap (&dec->altweHypHeap, "AltWordendHyp heap", MHEAP, 
                  sizeof (AltWordendHyp), 1.0, 8000, 80000);
   }
#ifdef MODALIGN
   dec->modAlign = modAlign;

   if (dec->modAlign) {
      CreateHeap (&dec->modendHypHeap, "ModendHyp heap", MHEAP, 
                  sizeof (ModendHyp), 1.0, 80000, 800000);
   }
#else
   if (modAlign)
      HError (9999, "CreateDecoderInst: model alignment not supported; recompile with MODALIGN");
#endif

   /* output probability cache */

   dec->outPCache = CreateOutPCache (&dec->heap, dec->hset, outpBlocksize);

   /* cache debug code */
#if 0
   printf (" %d %d \n", dec->hset->numStates, dec->nCacheFlags);
   for (i = 0; i < dec->nCacheEntries; ++i)
      printf ("i %d  cacheFlags %lu\n", i, dec->cacheFlags[i]);

   for (i = 0; i < dec->hset->numStates; ++i) {
      assert (!CACHE_FLAG_GET(dec,i));
      CACHE_FLAG_SET(dec, i);
      assert (CACHE_FLAG_GET(dec,i));
   }

   /*      printf ("i %d  C_G %lu\n", i, CACHE_FLAG_GET(dec,i)); */
#endif 


   /* tag left-to-right models */
   {
      HMMScanState hss;

      NewHMMScan(dec->hset,&hss);
      do {
         /* #### should check each tidX only once! */
         /*     if (!IsSeenV(hss.hmm->transP)) { */
         if (CheckLRTransP (hss.hmm->transP))
            hss.hmm->tIdx *= -1;
         /*            TouchV(hss.hmm->transP); */
      }
      while(GoNextHMM(&hss));
      EndHMMScan(&hss);
   
   }

   if (doPhonePost)
      InitPhonePost (dec);
   else
      dec->nPhone = 0;

   return dec;
}

/* CheckLRTransP

     determine wheter transition matrix is left-to-right, i.e. no backward transitions
*/
static Boolean CheckLRTransP (SMatrix transP)
{
   int r,c,N;

   N = NumCols (transP);
   assert (N == NumRows (transP));

   for (r = 1; r <= N; ++r) {
      for (c = 1; c < r; ++c) {
         if (transP[r][c] > LSMALL)
            return FALSE;
      }
      for (c = r+2; c < r; ++c) {
         if (transP[r][c] > LSMALL)
            return FALSE;
      }
   }

   return TRUE;   
}

/* InitDecoderInst

     Initialise previously created decoder instance. This needs to be
     called before each utterance.
*/
void InitDecoderInst (DecoderInst *dec, LexNet *net, HTime sampRate, LogFloat beamWidth, 
                      LogFloat relBeamWidth, LogFloat weBeamWidth, LogFloat zsBeamWidth,
                      int maxModel, 
                      LogFloat insPen, float acScale, float pronScale, float lmScale,
                      LogFloat fastlmlaBeam)
{       
   int i;

   dec->net = net;

   if (dec->nLayers) {
      Dispose (&dec->heap,dec->instsLayer);
   }

   /* alloc InstsLayer start pointers */
   dec->nLayers = net->nLayers;
   dec->instsLayer = (LexNodeInst **) New (&dec->heap, net->nLayers * sizeof (LexNodeInst *));

   /* reset inst (i.e. reset pruning, etc.)
      purge all heaps
   */

   ResetHeap (&dec->nodeInstanceHeap);
   ResetHeap (&dec->weHypHeap);
   if (dec->latgen)
      ResetHeap (&dec->altweHypHeap);
#ifdef MODALIGN
   if (dec->modAlign)
      ResetHeap (&dec->modendHypHeap);
#endif
   ResetHeap (&dec->nodeInstanceHeap);
   for (i = 0; i < dec->maxNStates; ++i) 
      ResetHeap (&dec->tokSetHeap[i]);
   ResetHeap (&dec->relTokHeap);
   ResetHeap (&dec->lrelTokHeap);

   if (trace & T_MEM) {
      printf ("memory stats at start of recognition\n");
      PrintAllHeapStats ();
   }

   dec->frame = 0;
   dec->frameDur = sampRate / 1.0e7;
   dec->maxModel = maxModel;
   dec->beamWidth = beamWidth;
   dec->weBeamWidth = weBeamWidth;
   dec->zsBeamWidth = zsBeamWidth;
   dec->curBeamWidth = dec->beamWidth;
   dec->relBeamWidth = - relBeamWidth;
   dec->beamLimit = LZERO;

   if (fastlmlaBeam < -LSMALL) {
      dec->fastlmla = TRUE;
      dec->fastlmlaBeam = - fastlmlaBeam;
   }
   else {
      dec->fastlmla = FALSE;
      dec->fastlmlaBeam = LZERO;
   }

   dec->tokSetIdCount = 0;

   dec->insPen = insPen;
   dec->acScale = acScale;
   dec->pronScale = pronScale;
   dec->lmScale = lmScale;

   dec->maxLMLA = dec->lmScale * maxLMLA;

   /*      HRec computes interval of possible predecessor states j for each 
      destination state i in each transition matrix (seIndexes).
   */
   

   /* alloc temp tokenset arrays for use in PropagateInternal */
   for (i=1; i <= dec->maxNStates; ++i) 
      dec->tempTS[i] = NewTokSetArrayVar (dec, i, TRUE);

   /* alloc winTok array for MergeTokSet */
   dec->winTok = (RelToken *) New (&dec->heap, LAYER_SIL_NTOK_SCALE * dec->nTok * sizeof (RelToken));


   /* init lists of active LexNode Instances  */
   for (i = 0; i < dec->nLayers; ++i)
      dec->instsLayer[i] = NULL;

   /* deactivate all nodes */
   for (i = 0; i < dec->net->nNodes; ++i) {
      dec->net->node[i].inst = NULL;
#ifdef COLLECT_STATS_ACTIVATION
      dec->net->node[i].eventT = -1;
#endif
   }

   ActivateNode (dec, dec->net->start);
   dec->net->start->inst->ts[0].n = 1;
   dec->net->start->inst->ts[0].score = 0.0;
   dec->net->start->inst->ts[0].relTok[0] = startTok;
   dec->net->start->inst->ts[0].relTok[0].lmState = LMInitial (dec->lm);

#ifdef COLLECT_STATS
   dec->stats.nTokSet = 0;
   dec->stats.sumTokPerTS = 0;
   dec->stats.nActive = 0;
   dec->stats.nActivate = 0;
   dec->stats.nDeActivate = 0;
   dec->stats.nFrames = 0;
   dec->stats.nLMlaCacheHit = 0;
   dec->stats.nLMlaCacheMiss = 0;
#ifdef COLLECT_STATS_ACTIVATION

   dec->stats.lnINF = 0;
   {
      int i;
      for (i = 0; i <= STATS_MAXT; ++i)
         dec->stats.lnDeadT[i] = dec->stats.lnLiveT[i] = 0;
   }
#endif
#endif

   /* LM lookahead cache */
   dec->lmCache = CreateLMCache (dec, &dec->heap);

   /* invalidate OutP cache */
   ResetOutPCache (dec->outPCache);
}

void CleanDecoderInst (DecoderInst *dec)
{
   FreeLMCache (dec->lmCache);
}


/* NewTokSetArray

*/
static TokenSet *NewTokSetArray(DecoderInst *dec, int N)
{
   TokenSet *ts;
   int i;

   ts= (TokenSet *) New (&dec->tokSetHeap[N-1], N * sizeof (TokenSet));

   /* clear token set */
   for (i = 0; i < N; ++i) {
      ts[i].score = 0.0;
      ts[i].n = 0;
      ts[i].id = 0;             /* id=0 means empty TokSet */
      ts[i].relTok = (RelToken *) New (&dec->relTokHeap, dec->nTok * sizeof (RelToken));
   }
   return ts;
}

/* NewTokSetArrayVar:

   allocating rel token array for lex node, larger array size
   for silence layer nodes
*/
static TokenSet *NewTokSetArrayVar(DecoderInst *dec, int N, Boolean isSil)
{
   TokenSet *ts;
   int i;

   ts= (TokenSet *) New (&dec->tokSetHeap[N-1], N * sizeof (TokenSet));

   /* clear token set */
   for (i = 0; i < N; ++i) {
      ts[i].score = 0.0;
      ts[i].n = 0;
      ts[i].id = 0;             /* id=0 means empty TokSet */
      ts[i].relTok = (isSil) ? (RelToken *) New (&dec->lrelTokHeap, LAYER_SIL_NTOK_SCALE * dec->nTok * sizeof (RelToken)) : (RelToken *) New (&dec->relTokHeap, dec->nTok * sizeof (RelToken));
   }
   return ts;
}


/* ActivateNode

     Allocate and init new Instance for given node.

*/
static LexNodeInst *ActivateNode (DecoderInst *dec, LexNode *ln)
{
   LexNodeInst *inst;
   int N;               /* number of states in HMM for this node */
   int l;

#ifdef COLLECT_STATS
   ++dec->stats.nActivate;
#endif
#ifdef COLLECT_STATS_ACTIVATION
   if (ln->eventT != -1) {
      int t;
      t = dec->frame - ln->eventT;
      if (t > STATS_MAXT)
         t = STATS_MAXT;
      ++dec->stats.lnDeadT[t];
   }
   ln->eventT = dec->frame;
#endif

   assert (!ln->inst);

   inst = (LexNodeInst *) New (&dec->nodeInstanceHeap, 0);

   inst->node = ln;
   ln->inst = inst;

   switch (ln->type) {
   case LN_MODEL:
      N = ln->data.hmm->numStates;
      break;
   case LN_CON:
   case LN_WORDEND:
      N = 1;
      break;
   default:
      abort ();
      break;
   }

   /* alloc N tokensets */
/*    inst->ts = NewTokSetArray (dec, N);          /\*  size is  N * sizeof (TokenSet) *\/ */

   inst->best = LZERO;

   /* add new instance to list of active nodes in the right place */
   /* find right layer */
   l = dec->nLayers-1;
   while (dec->net->layerStart[l] > ln) {
      --l;
      assert (l >= 0);
   }
   inst->ts = NewTokSetArrayVar (dec, N, (l == LAYER_SIL));          /*  size is  N * sizeof (TokenSet) */
#ifdef DEBUG_TRACE
   if (trace & T_ACTIV)
      printf ("allocating %d tokens in array for node in layer %d\n", ((l == LAYER_SIL) ? LAYER_SIL_NTOK_SCALE : 1) * dec->nTok, l);
#endif

   /* add to linked list */
   inst->next = dec->instsLayer[l];
   dec->instsLayer[l] = inst;

#ifdef DEBUG_TRACE
   if (trace & T_ACTIV)
      printf ("activated node in layer %d\n", l);
#endif

   return (inst);
}

static void DeactivateNode (DecoderInst *dec, LexNode *ln)
{
   int N, i, l;

#ifdef COLLECT_STATS
   ++dec->stats.nDeActivate;
#endif
#ifdef COLLECT_STATS_ACTIVATION
   if (ln->eventT != -1) {
      int t;
      t = dec->frame - ln->eventT;
      if (t > STATS_MAXT)
         t = STATS_MAXT;
      ++dec->stats.lnLiveT[t];
   }
   ln->eventT = dec->frame;
#endif

   assert (ln->inst);
   
   switch (ln->type) {
   case LN_MODEL:
      N = ln->data.hmm->numStates;
      break;
   case LN_CON:
   case LN_WORDEND:
      N = 1;
      break;
   default:
      abort ();
      break;
   }
   
#if 1
   /* find right layer */
   l = dec->nLayers-1;
   while (dec->net->layerStart[l] > ln) {
      --l;
      assert (l >= 0);
   }
   for (i = 0; i < N; ++i) {
      if (l == LAYER_SIL) Dispose (&dec->lrelTokHeap, ln->inst->ts[i].relTok);
      else Dispose (&dec->relTokHeap, ln->inst->ts[i].relTok);
   }

   Dispose (&dec->tokSetHeap[N-1], ln->inst->ts);
   Dispose (&dec->nodeInstanceHeap, ln->inst);
#endif

   ln->inst = NULL;
}


/* PruneTokSet

     apply global and relative beams to a tokenset

     #### this is rather inefficient
*/
static void PruneTokSet (DecoderInst *dec, TokenSet *ts)
{
   RelTokScore deltaLimit;
   RelToken *tok, *dest;
   int i, newN;

   return; /* ########  TEST */

#if 0
   /* only apply relative beam */
   deltaLimit = dec->relBeamWidth;
#else   /* main beam pruning for reltoks */
   /* main and relative beam pruning */
   deltaLimit = dec->beamLimit - ts->score;
   if (dec->relBeamWidth > deltaLimit)
      deltaLimit = dec->relBeamWidth;
#endif
   
   if (deltaLimit > 0) {        /* prune complete TokeSet */
      ts->n = 0;
      ts->id = 0;
      return;
   }

   /* #### maybe don't perform relTok pruning to keep relTokID the same?  */

   newN = 0;            /* number of relToks kept */
   for (i = 0, dest = tok = ts->relTok; i < ts->n; ++i, ++tok) {
      if (tok->delta > deltaLimit) {   /* keep */
         if (dest != tok)
            *dest = *tok;
         ++dest;
         ++newN;
      }
   }

   /* #### could calculate newN from difference between dest and ts->tok */
   if (newN != ts->n) {         /* some RelToks got pruned! */
      ts->n = newN;
      ts->id = ++dec->tokSetIdCount;

   }
}


/* stolen from HRec.c */

/* EXPORT->FormatTranscription: Format transcription prior to output */
void ReFormatTranscription(Transcription *trans,HTime frameDur,
                         Boolean states,Boolean models,Boolean triStrip,
                         Boolean normScores,Boolean killScores,
                         Boolean centreTimes,Boolean killTimes,
                         Boolean killWords,Boolean killModels)
{
   LabList *ll;
   LLink lab;
   HTime end;
   char buf[MAXSTRLEN],*p,tail[64];
   int lev,j,frames;
   
   if (killScores) {
      for (lev=1;lev<=trans->numLists;lev++) {
         ll=GetLabelList(trans,lev);
         for(lab=ll->head->succ;lab->succ!=NULL;lab=lab->succ) {
            lab->score=0.0;
            for (j=1;j<=ll->maxAuxLab;j++)
               lab->auxScore[j]=0.0;
         }
      }
   }
   if (triStrip) {
      for (lev=1;lev<=trans->numLists;lev++) {
         ll=GetLabelList(trans,lev);
         for(lab=ll->head->succ;lab->succ!=NULL;lab=lab->succ) {
            if (states && !models) {
               strcpy(buf,lab->labid->name);
               if ((p=strrchr(buf,'['))!=NULL) {
                  strcpy(tail,p);
                  *p=0;
               }
               else
                  *tail=0;
               TriStrip(buf); strcat(buf,tail);
               lab->labid=GetLabId(buf,TRUE);
            }
            else {
               strcpy(buf,lab->labid->name);
               TriStrip(buf); lab->labid=GetLabId(buf,TRUE);
            }
            for (j=1;j<=ll->maxAuxLab;j++) {
               if (lab->auxLab[j]==NULL) continue;
               strcpy(buf,lab->auxLab[j]->name);
               TriStrip(buf); lab->auxLab[j]=GetLabId(buf,TRUE);
            }
         }
      }
   }
   if (normScores) {
      for (lev=1;lev<=trans->numLists;lev++) {
         ll=GetLabelList(trans,lev);
         for(lab=ll->head->succ;lab->succ!=NULL;lab=lab->succ) {
            frames=(int)floor((lab->end-lab->start)/frameDur + 0.4);
            if (frames==0) lab->score=0.0;
            else lab->score=lab->score/frames;
            if (states && models && ll->maxAuxLab>0 && lab->auxLab[1]!=NULL) {
               end=AuxLabEndTime(lab,1);
               frames=(int)floor((end-lab->start)/frameDur + 0.4);
               if (frames==0) lab->auxScore[1]=0.0;
               else lab->auxScore[1]=lab->auxScore[1]/frames;
            }
         }
      }
   }
   if (killTimes) {
      for (lev=1;lev<=trans->numLists;lev++) {
         ll=GetLabelList(trans,lev);
         for(lab=ll->head->succ;lab->succ!=NULL;lab=lab->succ) {
            lab->start=lab->end=-1.0;
         }
      }
   }
   if (centreTimes) {
      for (lev=1;lev<=trans->numLists;lev++) {
         ll=GetLabelList(trans,lev);
         for(lab=ll->head->succ;lab->succ!=NULL;lab=lab->succ) {
            lab->start+=frameDur/2;
            lab->end-=frameDur/2;
         }
      }
   }
   if (killWords) {
      for (lev=1;lev<=trans->numLists;lev++) {
         ll=GetLabelList(trans,lev);
         if (ll->maxAuxLab>0)
            for(lab=ll->head->succ;lab->succ!=NULL;lab=lab->succ)
               lab->auxLab[ll->maxAuxLab]=NULL;
      }
   }
   if (killModels && models && states) {
      for (lev=1;lev<=trans->numLists;lev++) {
         ll=GetLabelList(trans,lev);
         if (ll->maxAuxLab==2)
            for(lab=ll->head->succ;lab->succ!=NULL;lab=lab->succ) {
               lab->auxLab[1]=lab->auxLab[2];
               lab->auxScore[1]=lab->auxScore[2];
               lab->auxLab[2]=NULL;
            }
      }
   }
}





/*  CC-mode style info for emacs
 Local Variables:
 c-file-style: "htk"
 End:
*/
