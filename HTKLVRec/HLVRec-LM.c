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
/*           Machine Intelligence Laboratory                   */
/*           Department of Engineering                         */
/*           University of Cambridge                           */
/*           http://mi.eng.cam.ac.uk/                          */
/*                                                             */
/* ----------------------------------------------------------- */
/*           Copyright: Cambridge University                   */
/*                      Engineering Department                 */
/*            2002-2015 Cambridge, Cambridgeshire UK           */
/*                      http://www.eng.cam.ac.uk               */
/*                                                             */
/*   Use of this software is governed by a License Agreement   */
/*    ** See the file License for the Conditions of Use  **    */
/*    **     This banner notice must not be removed      **    */
/*                                                             */
/* ----------------------------------------------------------- */
/*  File: HLVRec-LM.c  Update LM lookahead for HTK LV decoder  */
/* ----------------------------------------------------------- */

char *hlvrec_lm_version = "!HVER!HLVRec-LM:   3.5.0 [CUED 12/10/15]";
char *hlvrec_lm_vc_id = "$Id: HLVRec-LM.c,v 1.1.1.1 2006/10/11 09:54:55 jal58 Exp $";


/* UpdateLMlookahead

     update LM lookahead info for all RelToks in entry state of ln

     This might change the best score in ts-score, but will not affect the
     tokenset identity ts->id. All tokensets we might later want to merge with
     will be in the same node and have thus undergone the same LMla change.
*/
static void UpdateLMlookahead(DecoderInst *dec, LexNode *ln)
{
   int i;
   TokenSet *ts;
   RelToken *tok;
   unsigned int lmlaIdx;
   LMTokScore lmscore;
   RelTokScore bestDelta;

   assert (ln->type != LN_WORDEND);

   lmlaIdx = ln->lmlaIdx;
   assert (lmlaIdx != 0);
   assert (lmlaIdx < dec->net->laTree->nNodes + dec->net->laTree->nCompNodes);
   
   ts = ln->inst->ts;
   assert (ts->n > 0);

   bestDelta = LZERO;

   for (i = 0, tok = ts->relTok; i < ts->n; ++i, ++tok) {

      if (!dec->fastlmla) {
         lmscore = LMCacheLookaheadProb (dec, tok->lmState, lmlaIdx, FALSE);
         /*      lmscore = LMLA_nocache (dec, tok->lmState, lmlaIdx); */
         
         assert (lmscore <= tok->lmscore + 0.1);   /* +0.1 because of accuracy problems (yuck!) */
      }
      else {    /* if we ever do fastLMLA, be careful as tok->lmscore might increase! */
         lmscore = LMCacheLookaheadProb (dec, tok->lmState, lmlaIdx, 
                                         tok->delta < dec->fastlmlaBeam);
         if (lmscore > tok->lmscore)    /* if lmla goes up, leave old estimate */
            lmscore = tok->lmscore;
      }

      if (lmscore > LSMALL &&  tok->lmscore - lmscore > dec->maxLMLA)
         lmscore = tok->lmscore - dec->maxLMLA;

      tok->delta += lmscore - tok->lmscore;     /* subtract previous lookahead */
      if (tok->delta > bestDelta)
         bestDelta = tok->delta;

      tok->lmscore = lmscore;   /* store current lookahead */
   }

   /* renormalise to new best score */
   assert (bestDelta <= 0.1);   /* 0.1 for accuracy reasons */

   if (bestDelta > LSMALL) {
      for (i = 0, tok = ts->relTok; i < ts->n; ++i, ++tok) {
         tok->delta -= bestDelta;
      }
      ts->score += bestDelta;

#if 0
      /* #### new id because we renrmalised. Is this necessary? */
      ts->id = ++dec->tokSetIdCount;

      PruneTokSet (dec, ts);
#endif
   }
   else {       /* short cut pruning for LMLA = LZERO */
      ts->n = 0;
      ts->score = LZERO;
      ts->id = 0;
   }
}





/******************* LM trans * lookahead caching */

static LMCache *CreateLMCache (DecoderInst *dec, MemHeap *heap)
{
   LMCache *cache;
   int i;

   cache = (LMCache *) New (heap, sizeof (LMCache));
   CreateHeap (&cache->nodeHeap, "LMNodeCache Heap", MHEAP, sizeof (LMNodeCache),
               1.0, 1000, 2000);

   cache->nNode = dec->net->laTree->nNodes + dec->net->laTree->nCompNodes;
   cache->node = (LMNodeCache **) New (heap, cache->nNode * sizeof (LMNodeCache *));
   for (i = 0; i < cache->nNode; ++i)
      cache->node[i] = NULL;

   cache->transHit = cache->transMiss = 0;
   cache->laHit = cache->laMiss = 0;
   return cache;
}

static void FreeLMCache (LMCache *cache)
{
   DeleteHeap (&cache->nodeHeap);
}

#if 0
static void CacheLMLAprob (DecoderInst *dec, LMState lmState, int lmlaIdx, 
                           int hash, LMTokScore lmscore)
{
   LMLACacheEntry *entry;
   
   /* enter lmscore in cache */
   entry = New (&dec->lmlaCacheHeap, sizeof (LMLACacheEntry));
   entry->src = lmState;
   entry->idx = lmlaIdx;
   entry->prob = lmscore;
   entry->next = dec->lmlaCache[hash];
   dec->lmlaCache[hash] = entry;
   ++dec->nLMLACacheEntries;
}
#endif


#if 0
static int LMCacheTrans_hash (PronId pron)
{
   return ((unsigned int) pron % LMCACHE_NTRANS);
}

static int LMCacheLA_hash (int idx)
{
   return ((unsigned int) idx % LMCACHE_NLA);
}
#endif

LMNodeCache* AllocLMNodeCache (LMCache *cache, int lmlaIdx)
{
   LMNodeCache *n;

#if 0
   printf ("new LMNodeCache %d\n", lmlaIdx);
#endif
   n = (LMNodeCache *) New (&cache->nodeHeap, sizeof (LMNodeCache));
   memset ((void *) n, 1, sizeof (LMNodeCache));  /* clear all src entries */
   n->idx = lmlaIdx;
   n->size = LMCACHE_NLA;
   n->nextFree = n->nEntries = 0;

   return n;
}

/* LMCacheTransProb

     return the (scaled!) LM transition prob and dest LMState for the given LMState and PronId
*/
static LMTokScore LMCacheTransProb (DecoderInst *dec, FSLM *lm, 
                                    LMState src, PronId pronid, LMState *dest)
{
   return dec->lmScale * LMTransProb (lm, src, pronid, dest);

#if 0
   LMCache *cache;
   int hash;
   LMStateCache *stateCache;
   LMCacheTrans *entry;

   cache = dec->lmCache;
   hash = LMStateCache_hash (src);

   for (stateCache = cache->state[hash]; stateCache; stateCache = stateCache->next)
      if (stateCache->src == src)
         break;
   
   if (stateCache) {  
      /* touch this state */
      stateCache->t = dec->frame;
      
      entry = &stateCache->trans[LMCacheTrans_hash (pronid)];
      
      if (entry->pronid == pronid) {
         ++cache->transHit;
         *dest = entry->dest;
         return entry->prob;
      }
   }
   else {
      /* alloc new LMStateCache */
      stateCache = AllocLMStateCache (cache, src);
      stateCache->next = cache->state[hash];
      cache->state[hash] = stateCache;

      entry = &stateCache->trans[LMCacheTrans_hash (pronid)];
   }

   /* now entry points to the place to store the prob we are about to calulate */

   ++cache->transMiss;
   entry->pronid = pronid;
   entry->prob = dec->lmScale * LMTransProb (lm, src, pronid, dest);
   entry->dest = *dest;
   return entry->prob;
#endif
}

LMTokScore LMLA_nocache (DecoderInst *dec, LMState lmState, int lmlaIdx)
{
   LMlaTree *laTree;
   LMTokScore lmscore;

   laTree = dec->net->laTree;
   if (lmlaIdx < laTree->nNodes) {        /* simple node */
      LMlaNode *laNode;
         
      laNode = &laTree->node[lmlaIdx];
      lmscore = dec->lmScale * LMLookAhead (dec->lm, lmState, 
                                            laNode->loWE, laNode->hiWE);
   }
   else {         /* complex node */
      CompLMlaNode *laNode;
      LMTokScore score;
      int i;
         
      laNode = &laTree->compNode[lmlaIdx - laTree->nNodes];
      
      lmscore = LZERO;
      for (i = 0; i < laNode->n; ++i) {
         score = LMLA_nocache (dec, lmState, laNode->lmlaIdx[i]);
         if (score > lmscore)
            lmscore = score;
      }
   }
   return lmscore;
}

/* LMCacheLookaheadProb

     return the (scaled!) LM lookahead score for the given LMState and lmlaIdx
*/
static LMTokScore LMCacheLookaheadProb (DecoderInst *dec, LMState lmState, 
                                        int lmlaIdx, Boolean fastlmla)
{
   LMCache *cache;
   LMTokScore lmscore;
   LMNodeCache *nodeCache;
   LMCacheLA *entry;
   int i;

   cache = dec->lmCache;
   assert (lmlaIdx < cache->nNode);
   nodeCache = cache->node[lmlaIdx];

   if (fastlmla) {      /* #### should only go to fast LMState if real one is not cahced */
      lmState = Fast_LMLA_LMState (dec->lm, lmState);
   }

   if (nodeCache) {  
      /* touch this state */
      nodeCache->t = dec->frame;
      
      for (i = 0; i < nodeCache->nEntries; ++i) {
         entry = &nodeCache->la[i];
         if (entry->src == lmState)
            break;
      }
      if (i < nodeCache->nEntries) {
         ++cache->laHit;
#if 0         /* #### very expensive sanity check */
         assert (entry->prob == LMLA_nocache (dec, lmState, lmlaIdx));
#endif
         return entry->prob;
      }
      entry = &nodeCache->la[nodeCache->nextFree];
      nodeCache->nextFree = (nodeCache->nextFree + 1) % nodeCache->size;
      if (nodeCache->nEntries < nodeCache->size)
         ++nodeCache->nEntries;
   }
   else {
      /* alloc new LMNodeCache */
      nodeCache = cache->node[lmlaIdx] = AllocLMNodeCache (cache, lmlaIdx);

      entry = &nodeCache->la[0];
   }
   
   /* now entry points to the place to store the prob we are about to calulate */
   {
      LMlaTree *laTree;
      
      laTree = dec->net->laTree;
      if (lmlaIdx < laTree->nNodes) {        /* simple node */
         LMlaNode *laNode;
         
         laNode = &laTree->node[lmlaIdx];
         ++cache->laMiss;
         lmscore = dec->lmScale * LMLookAhead (dec->lm, lmState, 
                                               laNode->loWE, laNode->hiWE);
      }
      else {         /* complex node */
         CompLMlaNode *laNode;
         LMTokScore score;
         int i;
         
         laNode = &laTree->compNode[lmlaIdx - laTree->nNodes];
         
         lmscore = LZERO;
         for (i = 0; i < laNode->n; ++i) {
            score = LMCacheLookaheadProb (dec, lmState, laNode->lmlaIdx[i], FALSE);
            if (score > lmscore)
               lmscore = score;
         }
      }
   }

   if (lmscore < LSMALL)
      lmscore = LZERO;
   
   entry->src = lmState;
   entry->prob = lmscore;

   /*    printf ("lmla %f\n", lmscore); */
   return lmscore;
}

/* ------------------------ End of HLVRec-LM.c ----------------------- */

