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
/*         File: HLVRec-propagate.c Viterbi recognition engine */
/*                                  for HTK LV Decoder, token  */
/*                                  propagation                */
/* ----------------------------------------------------------- */

char *hlvrec_prop_vc_id = "$Id: HLVRec-propagate.c,v 1.1.1.1 2006/10/11 09:54:56 jal58 Exp $";


static int winTok_cmp (const void *v1,const void *v2)
{
   RelToken **tok1,**tok2;

   tok1 = (RelToken **) v1;
   tok2 = (RelToken **) v2;
   return ((int) ((*tok2)->delta - (*tok1)->delta));   /* reverse! i.e. largest first */
}


/* stats for TokenSet Id optimisation */
static int mts_copy = 0;
static int mts_fast = 0;
static int mts_slow = 0;
static int mts_newid = 0;
static int mts_newidNTOK = 0;


/* MergeTokSet

     Merge TokenSet src into dest after adding score to all src scores
     by recombining tokens in the same LM state and keeping only the
     dec->nTok best tokens in different LM states.

*/
static void MergeTokSet (DecoderInst *dec, TokenSet *src, TokenSet *dest, 
                         LogFloat score, Boolean prune)
{
   int i, j;
   RelToken *srcTok, *destTok;

   assert (src->n > 0);
   assert (src->n <= dec->nTok);
   assert (dest->n <= dec->nTok);

   assert (!prune || (src->score >= dec->beamLimit));

   if (dest->n == 0) {   /* empty dest tokenset -> just copy src */
      dest->n = src->n;
      dest->score = src->score + score;
      dest->id = src->id;

      ++mts_copy;
      for (i = 0, srcTok = src->relTok, destTok = dest->relTok; i < src->n; ++i, ++srcTok, ++destTok)
         *destTok = *srcTok;
      /*         dest->relTok[i] = src->relTok[i]; */
      return;
   } else if (prune && src->score + score < dec->beamLimit) {
      /*       printf ("MergeTokSet: pruned src TS\n"); */
      return;
   }
#ifdef TSIDOPT
   else if (src->id == dest->id) {      /* TokenSet Id optimisation from [Odell:2000] */
      TokScore srcScore;

      ++mts_fast;
      /* only compare Tokensets' best scores and pick better */
      srcScore = src->score + score;
      
      if (dest->score > srcScore)
         return;
      else {
         /* id, n and all RelToks are the same anyway */
         dest->score = srcScore;
         return;
      }
   }
#endif
   else {    /* expensive MergeTokSet, #### move into separate function */
#if 1
      /* exploit & retain RelTok order (sorted by lmState?) */
      int srcTokCount, destTokCount, nWinTok;
      int nWin[2], winLoc;
      RelToken *winTok;
      TokScore winScore;
      RelTokScore srcCorr, destCorr, deltaLimit;

      ++mts_slow;

      winTok = dec->winTok;
      nWinTok = 0;

      srcTok = &src->relTok[0];
      destTok = &dest->relTok[0];
      srcTokCount = src->n;
      destTokCount = dest->n;

      /* #### first go at sorted Tok merge, 
         #### very explicit, no optimisation at all, yet! */
      

      /* find best score */
      if (src->score + score > dest->score) {
         winScore = src->score + score;
         srcCorr = - score;     /* avoid adding score twice! */
         destCorr = dest->score - winScore;
      } else {
         winScore = dest->score;
         srcCorr = src->score - winScore;
         destCorr = 0.0;
      }

      deltaLimit = dec->nTok * dec->relBeamWidth;  /* scaled relative beam, must initialize !!!*/;
      if (prune) {
      /* set pruning deltaLimit */
#if 0
      deltaLimit = dec->relBeamWidth;          /* relative beam */
#else
      deltaLimit = dec->beamLimit - winScore;     /* main beam */
      if (dec->relBeamWidth > deltaLimit)            
         deltaLimit = dec->relBeamWidth;          /* relative beam */
#endif
#ifdef DEBUG_TRACE
      printf("dec->beamLimit = %f, winScore = %f, dec->beamLimit - winScore = %f, dec->relBeamWidth = %f, deltaLimit = %f\n",
             dec->beamLimit, winScore, dec->beamLimit - winScore, dec->relBeamWidth, deltaLimit);
#endif
      }

      /* find winning tokens */
      nWin[0] = nWin[1] = 0;    /* location where winnning toks came from:
                                   0 == src   1 == dest */
      do {
         if (TOK_LMSTATE_EQ(srcTok, destTok)) {
            /* pick winner */
            if (src->score + srcTok->delta + score > dest->score + destTok->delta) {
               /* store srcTok */
               winTok[nWinTok] = *srcTok;
               winTok[nWinTok].delta += srcCorr + score;
               winLoc = 0;
            } else {
               /* store destTok */
               winTok[nWinTok] = *destTok;
               winTok[nWinTok].delta += destCorr;
               winLoc = 1;
            }
            ++srcTok;
            --srcTokCount;
            ++destTok;
            --destTokCount;
         } else if (TOK_LMSTATE_LT(srcTok, destTok)) {
            /* store srcTok */
            winTok[nWinTok] = *srcTok;
            winTok[nWinTok].delta += srcCorr + score;
            winLoc = 0;
            ++srcTok;
            --srcTokCount;
         } else {
            /* store destTok */
            winTok[nWinTok] = *destTok;
            winTok[nWinTok].delta += destCorr;
            winLoc = 1;
            ++destTok;
            --destTokCount;
         }
         
         if (winTok[nWinTok].delta >= deltaLimit) {      /* keep or prune? */
            ++nWinTok;
            ++nWin[winLoc];
         }
      } while (srcTokCount != 0 && destTokCount != 0);

      /* add left overs to winTok set 
         only at most one of the two loops will actually do something */
      for (i = srcTokCount; i > 0; --i, ++srcTok) {
         winTok[nWinTok] = *srcTok;
         winTok[nWinTok].delta += srcCorr + score;
         if (winTok[nWinTok].delta >= deltaLimit) {     /* keep or prune? */
            ++nWinTok;
            ++nWin[0];
         }
      }
      for (i = destTokCount; i > 0; --i, ++destTok) {
         winTok[nWinTok] = *destTok;
         winTok[nWinTok].delta += destCorr;
         if (winTok[nWinTok].delta >= deltaLimit) {     /* keep or prune? */
            ++nWinTok;
            ++nWin[1];
         }
      }

      /* prune and copy back */
      assert (nWinTok <= src->n + dest->n);

      if (nWinTok <= dec->nTok) {
         /* just copy */
         for (i = 0; i < nWinTok; ++i)
            dest->relTok[i] = winTok[i];
         dest->n = nWinTok;
         dest->score = winScore;

         /*          printf ("MTS: %d nWin  %d src  %d dest\n", nWinTok, nWin[0], nWin[1]); */
         if (nWin[0] == nWinTok)
            dest->id = src->id;          /* copy src->id */
         else if (nWin[1] == nWinTok)
            dest->id = dest->id;         /* copy dest->id */
         else {
            dest->id = ++dec->tokSetIdCount;    /* new id */
            ++mts_newid;
         }
      } else {
         /* perform Bucket sort/Histogram pruning to reduce to dec->nTok tokens */
#define NBINS 64

         int n[NBINS], binLimit;
         int nTok;

         LogFloat binWidth, limit;

         dest->id = ++dec->tokSetIdCount;    /* #### new id always necessary? */
         ++mts_newidNTOK;

         binWidth = deltaLimit*1.001 / NBINS;   /* handle delta==deltaLimit case */

         for (i = 0; i < NBINS; ++i)
            n[i] = 0;

         for (i = 0; i < nWinTok; ++i) {
            assert (((int) (winTok[i].delta / binWidth)) >= 0);
            assert (((int) (winTok[i].delta / binWidth)) < NBINS);
            ++n[(int) (winTok[i].delta / binWidth)];
         }
         
         nTok = 0;
         i = -1;
         while (nTok < dec->nTok) {
            ++i;
            nTok += n[i];
         }
         
         if (nTok == dec->nTok) {
            binLimit = i;
            limit = binWidth * (i+1);


            for (i = 0, j = 0; i < nWinTok; ++i) { 
               if ((int) (winTok[i].delta / binWidth) <= binLimit) {
                  dest->relTok[j] = winTok[i];
                  ++j;
               }
            }
            assert (j == dec->nTok);
         }
         else {
            int nBetter;
            LogFloat bestDelta;
            
            /* do not include last bin */
            limit = binWidth * i;
            nTok -= n[i]; 

            /* need to relax limit so that we get an extra (dec->nTok - nTok) tokens */
            /* #### very simplistic implementation -- imporve? */

            
            bestDelta = limit;
            do {
               limit = bestDelta;
               bestDelta = LZERO;
               nBetter = 0;
               for (i = 0, j = 0; i < nWinTok; ++i) {
                  if (winTok[i].delta >= limit)
                     ++nBetter;
                  else
                     if (winTok[i].delta > bestDelta)
                        bestDelta = winTok[i].delta;
               }
            } while (nBetter < dec->nTok);
            /*             printf ("nBetter %d\n", nBetter); */

            if (nBetter > dec->nTok) {  /* multiple tokens with delta == limit
                                           ==> delete some */
               for (i = 0; nBetter > dec->nTok; ++i)
                  if (winTok[i].delta == limit) {
                     winTok[i].delta = LZERO;
                     --nBetter;
                  }
            }
         
            for (i = 0, j = 0; i < nWinTok; ++i) { 
               if (winTok[i].delta >= limit) {
                  dest->relTok[j] = winTok[i];
                  ++j;
               }
            }
            assert (j == dec->nTok);
         }

         dest->n = dec->nTok;
         dest->score = winScore;
      }
#endif
      
#ifndef NDEBUG   /* sanity check for reltoks */
   for (i = 0; i < dest->n; ++i) {
      assert (dest->relTok[i].delta <= 0.01);
   }
   for (i = 0; i < dest->n; ++i)
      if (dest->relTok[i].delta >= -0.01 && dest->relTok[i].delta <= 0.01)
         return;

   abort ();
#endif
   }

#ifndef NDEBUG   /* sanity check for reltoks */
   for (i = 0; i < dest->n; ++i) {
      assert (dest->relTok[i].delta <= 0.01);
   }
   for (i = 0; i < dest->n; ++i)
      if (dest->relTok[i].delta >= -0.01 && dest->relTok[i].delta <= 0.01)
         return;

   abort ();
#endif
}


static int PI_LR = 0;
static int PI_GEN = 0;
/* PropagateInternal

     Internal token propagation
*/
static void PropagateInternal (DecoderInst *dec, LexNodeInst *inst)
{
   LexNode *ln;
   HLink hmm;
   int i, j, N;
   SMatrix trP;
   LogFloat outP, bestScore;
   TokenSet *instTS, *ts;

   ln = inst->node;
   hmm = ln->data.hmm;
   N = hmm->numStates;
   trP = hmm->transP;
   instTS = inst->ts;

   assert (ln->type == LN_MODEL);               /* Model node */


   /* LM lookahead has already been updated in PropIntoNode() !!! */

   /* main beam pruning: prune tokensets before propagation
      the beamLimit is the one found during the last frame */
   for (i = 1, ts = &instTS[0]; i < N; ++i, ++ts)
      if (ts->score < dec->beamLimit) {
         ts->n = 0;
         ts->id = 0;
      }

   /* optimised version for L-R models */
   if (hmm->tIdx < 0) {
      /*         PropagateInternal_LR (dec, inst);  */
      
      PI_LR++;
      bestScore = LZERO;
      
      /* loop transition for state N-1 (which has no forward trans) */
      instTS[N-2].score += trP[N-1][N-1];

      for (i = N-2; i >=2; --i) {             /* for all internal states, except N-1 */
         ts = &instTS[i-1];
         
         if (ts->n > 0) {   /*  && (trP[i][i+1] > LSMALL)) { */
            /* propagate forward from state i */
            /* for j */
            
            /* only propagate to next -- no skip!!! */
            MergeTokSet (dec, ts, ts+1 , trP[i][i+1], (!mergeTokOnly));
            
            /* loop transition */
            ts->score += trP[i][i];
         }
      }
      
      /* entry transition i=1 -> j=2 */
      if ((instTS[0].n > 0) && (trP[1][2] > LSMALL)) {
         MergeTokSet (dec, &instTS[0], &instTS[1], trP[1][2], (!mergeTokOnly));
      }
      
      /* output probabilities */
      for (i = 2; i < N; ++i) {             /* for all internal states */
         if (instTS[i-1].n > 0) {
            outP = cOutP (dec, dec->obs, hmm, i);
            instTS[i-1].score += outP;         /* only change top score */
            if (instTS[i-1].score > bestScore)
               bestScore = instTS[i-1].score;
         }
      }
      inst->best = bestScore;
      
      instTS[0].n = 0;             /* clear entry state */
      instTS[0].id = 0;
      
      instTS[N-1].n = 0;           /* clear exit state */
      instTS[N-1].id = 0;
      
      /* exit transition i=N-1 -> j=N */
      if (instTS[N-2].n > 0) {      /* && (trP[N-1][N] > LSMALL)) { */
         /* don't prune -- beam stil refers to last frame's scores, here we have 
            already added this frame's outP */
         MergeTokSet (dec, &instTS[N-2], &instTS[N-1], trP[N-1][N], FALSE);
      }
      
      if (bestScore > dec->bestScore) {
         dec->bestScore = bestScore;
         dec->bestInst = inst;
      }
      
#ifdef COLLECT_STATS
      {
         TokenSet *ts;
         for (i = 1, ts = &instTS[0]; i <= N; ++i, ++ts) {
            if (ts->n > 0) {
               ++dec->stats.nTokSet;
               dec->stats.sumTokPerTS += ts->n;
            }
         }
      }
#endif

      return;
   }

   /* general (not left-to-right) PropagateInternal   */
   {
      TokenSet *ts;
      TokenSet *tempTS;         /* temp storage for N tokensets 
                                   #### N-2 would be enough? */
      LogFloat score;
      
      tempTS = dec->tempTS[N];
      
      PI_GEN++;
#ifdef DEBUG_TRACE
      if (trace & T_PROP)
         printf ("#########################PropagateInternal hmm %p '%s':\n", inst->node,
                 FindMacroStruct (dec->net->hset, 'h', inst->node->data.hmm)->id->name);
#endif
      
      /* internal propagation; transition i -> j,  \forall 2 <= j <= N-1 */
      
      /* internal states */
      for (j = 2; j < N; ++j) {
         tempTS[j-1].score = 0.0;
         tempTS[j-1].n = 0;
         tempTS[j-1].id = 0;
         
         for (i = 1; i < N; ++i) {
#ifdef DEBUG_TRACE
            if (trace & T_PROP)
               printf ("transP[%d][%d] = %f\n", i, j, trP[i][j]);
#endif
            if ((instTS[i-1].n > 0) && (trP[i][j] > LSMALL)) {
               MergeTokSet (dec, &instTS[i-1], &tempTS[j-1], trP[i][j], (!mergeTokOnly));
            }
         }
         if (tempTS[j-1].n > 0) {
            outP = cOutP (dec, dec->obs, hmm, j);
            tempTS[j-1].score += outP;         /* only change top tok */
         }
#ifdef DEBUG_TRACE
         if (trace & T_PROP) {
            printf ("PropagateInternal: tokens in state %d after OutP: ", j);
            PrintTokSet (dec, &instTS[j-1]);
            printf ("-------------------\n");
         }
#endif
      }
      
      instTS[0].n = 0;          /* clear entry state */
      instTS[0].id = 0;
      
      /* internal states: copy temp array back and find best score */
      bestScore = LZERO;
      for (j = 2; j < N; ++j) {
         /* copy reltoks */
         int k;
         
         instTS[j-1].n = tempTS[j-1].n;
         instTS[j-1].id = tempTS[j-1].id;
         instTS[j-1].score = tempTS[j-1].score;
         for (k = 0; k < tempTS[j-1].n; ++k)
            instTS[j-1].relTok[k] = tempTS[j-1].relTok[k];
         
         if (instTS[j-1].n > 0) {
            score = instTS[j-1].score;
            if (score > bestScore)
               bestScore = score;
         }
      }
      inst->best = bestScore;
      
      /* exit state (j=N),
         merge directly into ints->ts */
      j = N;
      instTS[j-1].n = 0;
      instTS[j-1].id = 0;
      
      for (i = 2; i < N; ++i) {
#ifdef DEBUG_TRACE
         if (trace & T_PROP)
            printf ("transP[%d][%d] = %f\n", i, j,trP[i][j]);
#endif
         if ((instTS[i-1].n > 0) && (trP[i][j] > LSMALL))
            MergeTokSet (dec, &instTS[i-1], &instTS[j-1], trP[i][j], FALSE);
      }
      
      /* exit state score is some internal state score plus transP,
         thus can be ignored for findeing the best score */
      
#ifdef DEBUG_TRACE
      if (trace & T_PROP) {
         printf ("PropagateInternal: tokens in exit state %d: ", j);
         PrintTokSet (dec, &instTS[j-1]);
         printf ("-------------------\n");
      }
#endif
      
      /* update global best score */
      if (bestScore > dec->bestScore) {
         dec->bestScore = bestScore;
         dec->bestInst = inst;
      }
      
      /* # this only collects stats for the model nodes */
#ifdef COLLECT_STATS
      for (i = 1, ts = &instTS[0]; i <= N; ++i, ++ts) {
         if (ts->n > 0) {
            ++dec->stats.nTokSet;
            dec->stats.sumTokPerTS += ts->n;
         }
      }
#endif
      
      
#ifdef DEBUG_TRACE
      if (trace & T_PROP) {
         printf ("best %f\n", inst->best);
         for (i = 0; i < N; ++i)
            printf (" %d ",inst->ts[i].n);
         
         printf("\n");
      }
#endif
   }
}


#ifdef MODALIGN
void UpdateModPaths (DecoderInst *dec, TokenSet *ts, LexNode *ln)
{

   ModendHyp *m;
   RelToken *tok;
   int i;
   
   /* don't accumulate info for CON nodes */
   if (ln->type != LN_CON) {
      /* #### optimise by sharing ModendHyp's between tokens with
         same tok->modpath */
      for (i = 0, tok = ts->relTok; i < ts->n; ++i, ++tok) {
         m = New (&dec->modendHypHeap, sizeof (ModendHyp));
         m->frame = dec->frame;
         m->ln = ln;
         m->prev = tok->modpath;
         
         tok->modpath = m;
      }
   }
}
#endif   


/* PropIntoNode

     Propagate tokenset into entry state of LexNode, activating as necessary
*/
static void PropIntoNode (DecoderInst *dec, TokenSet *ts, LexNode *ln, Boolean updateLMLA)
{
   LexNodeInst *inst;
   TokScore best;

   if (!ln->inst)                /* activate if necessary */
      ActivateNode (dec, ln);

   inst = ln->inst;
         
   /* propagate tokens from ln's exit into follLN's entry state */
   MergeTokSet (dec, ts, &inst->ts[0], 0.0, TRUE);

   if (updateLMLA) {
      if (ln->type != LN_WORDEND && ln->lmlaIdx != 0)
         UpdateLMlookahead (dec, ln);
   }

   /* only update inst->best if no LMLA update necessary or already done! */
   if (!(ln->type != LN_WORDEND && ln->lmlaIdx != 0) || updateLMLA) {
      best = inst->ts[0].score;
      if (best > inst->best)
         inst->best = best;
   }
}

/* PropagateExternal

     External token propagation. Activate following nodes if necessary and propagate 
     token set into their entry states.
*/
static void PropagateExternal (DecoderInst *dec, LexNodeInst *inst, 
                               Boolean handleWE, Boolean wintTree)
{
   LexNode *ln, *follLN;
   int i, N;
   TokenSet *entryTS, *exitTS;
   
   ln = inst->node;
   entryTS = &inst->ts[0];

   /* handle tee transition and set N */
   if (ln->type == LN_MODEL) {
      HLink hmm;

      hmm = ln->data.hmm;
      N = hmm->numStates;
      exitTS = &inst->ts[N-1];

      /* main beam pruning: only propagate if above beamLimit
         #### maybe unnecessary? can be done in PropIntoNode */
      if ((entryTS->n > 0) && (hmm->transP[1][N] > LSMALL) && 
          (entryTS->score > dec->beamLimit)) {
#ifdef DEBUG_TRACE
         if (trace & T_PROP) {
            printf ("found tee HMM node '%s' transp %f\n",
                    FindMacroStruct (dec->net->hset, 'h', hmm)->id->name,
                    hmm->transP[1][N]);
            printf ("PropagateExternal: entry state:\n");
            PrintTokSet (dec, entryTS);
            printf ("----\n");
            printf ("PropagateExternal: exit state:\n");
            PrintTokSet (dec, exitTS);
            printf ("----\n");
         }
#endif
         MergeTokSet (dec, entryTS, exitTS, hmm->transP[1][N], TRUE);
#ifdef DEBUG_TRACE
         if (trace & T_PROP) {
            printf ("PropagateExternal: exit state after tee propagation:\n");
            PrintTokSet (dec, exitTS);
            printf ("----\n");
         }
#endif
      }
   }
   else {
      N = 1;    /* all other nodes are 1 state */
      exitTS = entryTS;

      /* for wordend nodes first apply LM and recombine */
      if (ln->type == LN_WORDEND) {
         /* for LAYER_WR call HandleWordend() on all WE nodes 
            in separate pass for WE-pruning */
         if (handleWE)
            HandleWordend (dec, ln);
      }      
      /* main beam pruning: only propagate if above beamLimit */
      if (exitTS->score < dec->beamLimit) {
         exitTS->n = 0;
         exitTS->id = 0;
         inst->best = LZERO;
      }
      else
         inst->best = exitTS->score;
   }

#if 1
   /* prune token sets in states 2..N-1 */
   /* ####RELTOK  don't do this to avoid changing tokSet->id?? */
   for (i = 2; i < N; ++i) {
      if (inst->ts[i-1].n > 0)
         PruneTokSet (dec, &inst->ts[i-1]);
   }
#endif

   /* prune exit state token set */
   if (exitTS->n > 0)
      PruneTokSet (dec, exitTS);


   /* any tokens in exit state? */
   if (exitTS->n > 0 && exitTS->score > dec->beamLimit) {
#ifdef MODALIGN
      if (dec->modAlign)
         UpdateModPaths (dec, exitTS, ln);
#endif
      /* loop over following nodes */
      for (i = 0; i < ln -> nfoll; ++i) {
         follLN = ln->foll[i];
         PropIntoNode (dec, exitTS, follLN, wintTree);
      } /* for i following nodes */
   }
}


/* HandleWordend

     update traceback, add LM, update LM state, recombine tokens

*/
static void HandleWordend (DecoderInst *dec, LexNode *ln)
{
   LexNodeInst *inst;
   WordendHyp *weHyp, *prev;
   TokenSet *ts;
   RelToken *tok, *tokJ;
   int i, j, newN;
   LMState dest;
   LMTokScore lmScore;
   PronId pronid;
   RelTokScore newDelta, bestDelta, deltaLimit;
#ifdef MODALIGN
   ModendHyp *modpath;
#endif

   assert (ln->type == LN_WORDEND);

   inst = ln->inst;
   assert (inst);
   ts = inst->ts;
   
   if (trace & T_WORD) {
      printf ("PropagateWordEnd: handleWordend '%s'\n", dec->net->pronlist[ln->data.pron]->word->wordName->name);
      printf ("before LM application:\n");
      PrintTokSet (dec, ts);
      printf ("++++++++++\n");
   }

#if 0
   /* apply only relative beam */
   deltaLimit = dec->relBeamWidth;
#else   /* main beam pruning for reltoks */
   /* main and relative beam pruning */
   deltaLimit = dec->beamLimit - ts->score;
   if (dec->relBeamWidth > deltaLimit)
      deltaLimit = dec->relBeamWidth;
#endif

   /* for each token i in set take transition in LM
      recombine tokens in same LMState 
      newN is (current) number of new tokens (newN <= ts->n)
   */

   pronid = (PronId) ln->data.pron;
   assert (ts->n >= 1);

   newN = 0;
   bestDelta = LZERO;
   for (i = 0; i < ts->n; ++i) {
      tok = &ts->relTok[i];

      if (tok->delta < deltaLimit)
         continue;      /* prune */

      lmScore = LMCacheTransProb (dec, dec->lm, tok->lmState, pronid, &dest);

      /* word insertion penalty */
      lmScore += dec->insPen;
      /* remember prev path now, as we might overwrite it below */
      prev = tok->path;
#ifdef MODALIGN
      modpath = tok->modpath;
#endif


      /* subtract lookahead which has already been applied */
      if (!dec->fastlmla) {
         assert (lmScore <= tok->lmscore + 0.1); /* +0.1 because of accuracy problems (yuck!) */
      }
      newDelta = tok->delta + (lmScore - tok->lmscore);

      if (newDelta < deltaLimit)
         continue;              /* prune */

      if (newDelta > bestDelta)
         bestDelta = newDelta;

#if 0   /* removed -- we store the info directly in tokJ below */
      tok->delta = newDelta;
      tok->lmState = dest;
      tok->lmscore = 0.0;       /* reset lookahead */
#endif 
      /* insert in list */
      for (j = 0; j < newN; ++j) {      /* is there already a token in state dest? */
         tokJ = &ts->relTok[j];
         if (tokJ->lmState == dest) {
            if (!dec->latgen) {
               if (newDelta > tokJ->delta) {        /* replace tokJ */
                  tokJ->delta = newDelta;
                  tokJ->lmscore = 0.0;     /* reset lookahead */
#ifdef MODALIGN
                  tokJ->modpath = modpath;
#endif
                  /* update path; 
                     weHyp exists, pron is the same anyway, update rest */
                  assert (tokJ->path->pron == ln->data.pron);
                  tokJ->path->prev = prev;
                  tokJ->path->score = ts->score + newDelta;
                  tokJ->path->lm = lmScore;
#ifdef MODALIGN
                  tokJ->path->modpath = modpath;
#endif
               }
               /* else just toss token */
            }
            else {      /* latgen */
               AltWordendHyp *alt;

               alt = (AltWordendHyp *) New (&dec->altweHypHeap, sizeof (AltWordendHyp));
               
               if (newDelta > tokJ->delta) {
                  /* move tokJ->path to alt */
                  alt->prev = tokJ->path->prev;
                  alt->score = tokJ->path->score;
                  alt->lm = tokJ->path->lm;
#ifdef MODALIGN
                  alt->modpath = tokJ->path->modpath;
#endif

                  /* replace tokJ */
                  tokJ->delta = newDelta;
                  tokJ->lmscore = 0.0;     /* reset lookahead */
#ifdef MODALIGN
                  tokJ->modpath = modpath;

                  tokJ->path->modpath = modpath;
#endif

                  /* store new tok info in path 
                     weHyp exists, pron is the same anyway, update rest */
                  assert (tokJ->path->pron == ln->data.pron);
                  tokJ->path->prev = prev;
                  tokJ->path->score = ts->score + newDelta;
                  tokJ->path->lm = lmScore;
               }
               else {
                  /* store new tok info in alt */
                  alt->prev = prev;
                  alt->score = ts->score + newDelta;
                  alt->lm = lmScore;
#ifdef MODALIGN
                  alt->modpath = modpath;
#endif
               }

               /* attach alt to tokJ's weHyp */
               alt->next = tokJ->path->alt;
               tokJ->path->alt = alt;
            }               
            break;      /* leave j loop */
         }
      }

      if (j == newN) {          /* no token in state dest yet */
         int k;

         /* find spot to insert LMState dest */
         for (j = 0; j < newN; ++j)
            if (ts->relTok[j].lmState > dest)
               break;

         /* move any following reltokens up one slot */
         for (k = newN ; k > j; --k)
            ts->relTok[k] = ts->relTok[k-1];
         
         tokJ = &ts->relTok[j];
         ++newN;

         /* new wordendHyp */
         weHyp = (WordendHyp *) New (&dec->weHypHeap, sizeof (WordendHyp));
      
         weHyp->prev = prev;
         weHyp->pron = ln->data.pron;
         weHyp->score = ts->score + newDelta;
         weHyp->lm = lmScore;
         weHyp->frame = dec->frame;
         weHyp->alt = NULL;
         weHyp->user = 0;
#ifdef MODALIGN
         weHyp->modpath = modpath;

         tokJ->modpath = modpath;
#endif

         tokJ->path = weHyp;
         /* only really necessary if (i!=j) i.e. (tok!=tokJ) */
         tokJ->delta = newDelta;
         tokJ->lmState = dest;
         tokJ->we_tag = (void *) ln;
         tokJ->lmscore = 0.0;   /* reset lookahead */
      }
   } /* for token i */

   ts->n = newN;

   if (newN > 0) {
      AltWordendHyp *alt;
      /* renormalise  to new best score */
      if (!dec->fastlmla) {
         assert (bestDelta <= 0.1);  /* 0.1 for accuracy reasons */
      }
      assert (bestDelta > LSMALL);
      for (i = 0; i < ts->n; ++i) {
         tok = &ts->relTok[i];
         tok->delta -= bestDelta;

         /* convert alt wordendHyp scores to deltas relativ to main weHyp */
         for (alt = tok->path->alt; alt; alt = alt->next) {
            alt->score = alt->score - tok->path->score;
            assert (alt->score <= 0.1);         /* 0.1 for accuracy reasons */
         }
      }
      ts->score += bestDelta;

      /* ####  TokSet id */
      ts->id = ++dec->tokSetIdCount;
   }
   else {
      ts->id = 0;
      ts->score = LZERO;
   }

#if 0
   CheckTokenSetOrder (dec, ts);
#endif

   inst->best = ts->score;

   if (trace & T_WORD) {
      printf ("after LM application:\n");
      PrintTokSet (dec, ts);
      printf ("++++++++++\n");
   }
}


/* UpdateWordEndHyp

     update wordend hyps of all tokens with current time and score
 */
static void UpdateWordEndHyp (DecoderInst *dec, LexNodeInst *inst)
{
   int i;
   TokenSet *ts;
   RelToken *tok;
   WordendHyp *weHyp, *oldweHyp;
   
   ts = inst->ts;
           
   for (i = 0; i < ts->n; ++i) {
      tok = &ts->relTok[i];

      oldweHyp = tok->path;

      /* don't copy weHyp, if it is up-to-date (i.e. for <s>) */
      if (oldweHyp->frame != dec->frame || oldweHyp->pron != dec->net->startPron) {
         weHyp = (WordendHyp *) New (&dec->weHypHeap, sizeof (WordendHyp));
         *weHyp = *oldweHyp;
         weHyp->score = ts->score + tok->delta;
         weHyp->frame = dec->frame;
#ifdef MODALIGN
         weHyp->modpath = tok->modpath;
#endif
         tok->path = weHyp;

         /* altweHyps don't need to be changed  here as they are relative to 
            the main weHyp's score*/
      }
#ifdef MODALIGN
      tok->modpath = NULL;
#endif
   }
}

static void AddPronProbs (DecoderInst *dec, TokenSet *ts, int var)
{
   /* #### this is pathetically slow!!
      #### fix by storing 3 lists parallel to pronlist with 
      #### prescaled pronprobs
   */
   int i;
   RelToken *tok;
   Pron pron;
   WordendHyp *path;
   RelTokScore bestDelta = LZERO;

   for (i = 0, tok = ts->relTok; i < ts->n; ++i, ++tok) {
      path = tok->path;
      pron = dec->net->pronlist[path->pron];
      if (var == 1)
         pron = pron->next;
      else if (var == 2)
         pron = pron->next->next;
      
      tok->delta += dec->pronScale * pron->prob;
      if (tok->delta > bestDelta)
         bestDelta = tok->delta;

      /* need to make copy of path before modifying it */
      if (path->user != var) {
         WordendHyp *weHyp;

         weHyp = (WordendHyp *) New (&dec->weHypHeap, sizeof (WordendHyp));
         *weHyp = *path;
         weHyp->user = var;
         tok->path = weHyp;
      }
   }

   /* renormalise token set */
   for (i = 0, tok = ts->relTok; i < ts->n; ++i, ++tok)
      tok->delta -= bestDelta;
   ts->score += bestDelta;
}


void HandleSpSkipLayer (DecoderInst *dec, LexNodeInst *inst)
{
   LexNode *ln;

   ln = inst->node;
   if (ln->nfoll == 1) {    /* sp is unique foll, for sil case there are two! */
      LexNode *lnSA;
      
      assert (ln->foll[0]->data.hmm == dec->net->hmmSP);
      assert (ln->foll[0]->nfoll == 1);

      /* bypass sp model for - variant */
      /*  propagate tokens to follower of sp (ln->foll[0]->foll[0]) */
      /*    adding - variant pronprob (pron->prob) */
      lnSA = ln->foll[0]->foll[0];
                  
      /* node should be either inactive or empty */
      assert (!lnSA->inst || lnSA->inst->ts[0].n == 0);
      
      PropIntoNode (dec, &inst->ts[0], ln->foll[0]->foll[0], FALSE);
      
      /* add pronprobs and keep record of variant in path->user */
      /*   user = 0: - variant, 1: sp, 2: sil */
      AddPronProbs (dec, &lnSA->inst->ts[0], 0);
      
      /* now add sp variant pronprob to token set and propagate as normal */
      AddPronProbs (dec, &inst->ts[0], 1);
      PropagateExternal (dec, inst, FALSE, FALSE);
   }
   else {   /* sil variant */
      int sentEnd = 0;
      TokenSet *tempTS;

      tempTS = dec->tempTS[1];
      tempTS->score = 0.0;
      tempTS->n = 0;
      tempTS->id = 0;

      assert (ln->nfoll == 2);  /* inter word sil and path to sent end */
      
      if (ln->foll[1]->type == LN_CON)  /* lnTime node in SA, see comment 
                                           in HLVNet:CreateStartEnd() */
         sentEnd = 1;
      
      /*   path to SENT_END */
      /* propagate to ln->foll[sentEnd] and add - var pronpob */
      MergeTokSet (dec, &inst->ts[0], tempTS, 0.0, FALSE);
      AddPronProbs (dec, tempTS, 0);
      if (tempTS->score >= dec->beamLimit)
         PropIntoNode (dec, tempTS, ln->foll[sentEnd], FALSE);
      /* propagate to SENT_END  sp */
      tempTS->score = 0.0; tempTS->n = 0; tempTS->id = 0;
      MergeTokSet (dec, &inst->ts[0], tempTS, 0.0, FALSE);
      AddPronProbs (dec, tempTS, 1);
      if (tempTS->score >= dec->beamLimit)
         PropIntoNode (dec, tempTS, dec->net->lnSEsp, FALSE);
      /* propagate to SENT_END  sil */
      tempTS->score = 0.0; tempTS->n = 0; tempTS->id = 0;
      MergeTokSet (dec, &inst->ts[0], tempTS, 0.0, FALSE);
      AddPronProbs (dec, tempTS, 2);
      if (tempTS->score >= dec->beamLimit)
         PropIntoNode (dec, tempTS, dec->net->lnSEsil, FALSE);
      
      
      /*   normal word loop */
      /* add sil variant pronprob to token set and propagate */
      AddPronProbs (dec, &inst->ts[0], 2);
      if (inst->ts[0].score < dec->beamLimit) { /* prune to keep MTS happy */
         inst->ts[0].n = 0;
         inst->ts[0].id = 0;
      }
      else {
         PropIntoNode (dec, &inst->ts[0], ln->foll[1 - sentEnd], FALSE);
      }
   }
}

/* ProcessFrame

     Takes the observation vector and propatagets all tokens and
     performs pruning as necessary.
*/
void ProcessFrame (DecoderInst *dec, Observation **obsBlock, int nObs, 
                   AdaptXForm *xform)
{
   int l, i;
   LexNodeInst *inst, *prevInst, *next;
   int nActive, modelActive;
   TokScore beamLimit;
   
   inXForm = xform; /* sepcifies the transform to use */
   
   dec->obs = obsBlock[0];
   dec->nObs = nObs;
   for (i = 0; i < nObs; ++i)
      dec->obsBlock[i] = obsBlock[i];
   dec->bestScore = LZERO;
   dec->bestInst = NULL;
   ++dec->frame;

   if (dec->frame % gcFreq == 0)
      GarbageCollectPaths (dec);

   mts_copy = mts_fast = mts_slow = 0;
   mts_newid = mts_newidNTOK = 0;

   if (trace & T_BEST) {
      printf ("frame: %d beamLimit: %f\n", dec->frame, dec->beamLimit);
   }

   /* internal token propagation:
      order doesn't really matter, but we use the same as for external propagation */
   modelActive = 0;
   for (l = 0; l < dec->nLayers; ++l) {
      nActive = 0;
      for (inst = dec->instsLayer[l]; inst; inst = inst->next) {
         ++nActive;
         ++modelActive;
#ifdef DEBUG_TRACE
         if (trace & T_ACTIV) {
            printf ("l %d active node %p ",l, inst);
            switch (inst->node->type) {
            case LN_MODEL:
               printf (" HMM '%s' ",
                       FindMacroStruct (dec->net->hset, 'h', inst->node->data.hmm)->id->name);
               break;
            case LN_WORDEND:
               printf (" WE '%s' ",
                       dec->net->pronlist[inst->node->data.pron]->word->wordName->name);
               break;
            }
            printf ("\n");
         }
#endif

         if (inst->node->type != LN_MODEL) {         /* Context or Wordend node */
            /* clear tokenset in preparation for external propagation*/
            inst->ts[0].n = 0;
            inst->best =  LZERO;
         }
         else {               /* Model node */
            PropagateInternal (dec, inst);
         }
         /*         printf ("BEST %p %f\n", inst->node, inst->best); */
      }
#if 0
   printf ("MTS_copy: %d MTS_fast: %d  MTS slow: %d ", mts_copy, mts_fast, mts_slow);
   printf ("MTS_newid: %d MTS_newidNTOK: %d\n", mts_newid, mts_newidNTOK);
#endif
      if (trace & T_TOKSTATS)
         printf ("Pass1: %d active nodes in layer %d\n", nActive, l);
   }

   if (trace & T_TOKSTATS)
      printf ("Sum Pass1: %d active models\n", modelActive);


   /* now for all LN_MODEL nodes inst->best is set, this is used to determine 
      the lower beam limit */

   dec->beamLimit = dec->bestScore - dec->curBeamWidth;

   if (trace & T_BEST)
      printf ("best token in HMM %p '%s' score %f %f\n", dec->bestInst->node, 
              FindMacroStruct (dec->net->hset, 'h', dec->bestInst->node->data.hmm)->id->name,
              dec->bestScore, dec->bestScore/dec->frame);


   /* beam pruning & external propagation */
   modelActive = 0;
   for (l = 0; l < dec->nLayers; ++l) {

#ifdef DEBUG_TRACE
      if (trace & T_ACTIV)
         printf ("external propagation for layer %d\n", l);
#endif
      nActive=0;

#if 1           /* # make this a command line option */
      /* update word end time and score in tok->path when passing
         through the appropriate layer */
      if (l == dec->net->wordEndLayerId) {
         for (inst = dec->instsLayer[l]; inst; inst = inst->next) {
            UpdateWordEndHyp (dec, inst);
         }
      }
#endif

      /*** wordend beam pruning ***/
      beamLimit = dec->beamLimit;
      if ((dec->weBeamWidth < dec->beamWidth) && 
          (l == LAYER_WE)) {
         TokScore bestWEscore = LZERO;
         
         prevInst = NULL;
         for (inst = dec->instsLayer[l]; inst; inst = next) {
            next = inst->next;     /* store now, we might free inst below! */
            assert (inst->node->type == LN_WORDEND);
            assert (inst->ts[0].n == 0 || inst->ts[0].score == inst->best);
            if (inst->best < beamLimit) {  /* global main beam */
               if (prevInst)
                  prevInst->next = inst->next;
               else                 /* first inst in layer */
                  dec->instsLayer[l] = inst->next;

#ifdef DEBUG_TRACE
            if (trace & T_PRUNE) {
               char *name = NULL;
               name = dec->net->pronlist[inst->node->data.pron]->word->wordName->name;
               printf ("pruning word end node %p '%s' score %f off beam limit %f\n", inst->node, name, inst->best, beamLimit);
               PrintTokSet (dec, inst->ts);
            }
#endif

               DeactivateNode (dec, inst->node);
            }
            else {
               HandleWordend (dec, inst->node);
               if (inst->best > bestWEscore)
                  bestWEscore = inst->best;
               prevInst = inst;
            }
         }
         beamLimit = bestWEscore - dec->weBeamWidth;
         if (dec->beamLimit > beamLimit)  /* global beam is tighter */
            beamLimit = dec->beamLimit; 
      }
#if 1
      else if ((dec->zsBeamWidth < dec->beamWidth) &&   /* Z..S layer pruning */
               (l == LAYER_ZS || l == LAYER_SA)) {
#if 0
               ((l == LAYER_YZ) || (l == LAYER_Z) || (l == LAYER_ZS) || 
                (l == LAYER_SIL) || (l == LAYER_SA))) {
#endif
         TokScore bestScore = LZERO;
         
         for (inst = dec->instsLayer[l]; inst; inst = inst->next) {
            if (inst->best > bestScore)
               bestScore = inst->best;
         }
         beamLimit = bestScore - dec->zsBeamWidth;
         if (dec->beamLimit > beamLimit)  /* global beam is tighter */
            beamLimit = dec->beamLimit; 
      }
#endif      

      /* Due to the layer-by-layer structure inst->best values for
         LN_CON and LN_WORDEND nodes are set before they are examined
         for pruning purposes, although they were not available when the 
         beamlimit  was set.
      */
      prevInst = NULL;
      for (inst = dec->instsLayer[l]; inst; inst = next) {
         next = inst->next;     /* store now, we might free inst below! */
         
         if (inst->node->type != LN_WORDEND && inst->node->lmlaIdx != 0 &&
             inst->ts->n > 0) {
            TokScore best;

#ifdef DEBUG_TRACE
            if (trace & T_PRUNE) {
               char *name = NULL;
               if (inst->node->type == LN_MODEL)
                  name = FindMacroStruct (dec->net->hset, 'h', inst->node->data.hmm)->id->name;
               else 
                  name = "NULL_NODE";
               printf ("before lmla, node %p '%s' score %f\n", inst->node, name, inst->ts[0].score);
            }
#endif

            if (inst->ts[0].score >= beamLimit)       /* don't bother if inst will be pruned anyway */
               UpdateLMlookahead (dec, inst->node);

            if (inst->ts->n > 0) {      /* UpLMLA might have killed the entire TS, esp. in latlm */
#ifdef DEBUG_TRACE
               if (trace & T_PRUNE) {
                  char *name = NULL;
                  if (inst->node->type == LN_MODEL)
                     name = FindMacroStruct (dec->net->hset, 'h', inst->node->data.hmm)->id->name;
                  else 
                  name = "NULL_NODE";
                  if (inst->ts[0].score > inst->best) 
                     printf ("after lmla, node %p '%s' score %f\n", inst->node, name, inst->ts[0].score);
               }
#endif
               best = inst->ts[0].score;
               if (best > inst->best)
                  inst->best = best;
            }
            else {
#ifdef DEBUG_TRACE
               if (trace & T_PRUNE) {
                  char *name = NULL;
                  if (inst->node->type == LN_MODEL)
                     name = FindMacroStruct (dec->net->hset, 'h', inst->node->data.hmm)->id->name;
                  else 
                     name = "NULL_NODE";
                  printf ("after lmla, node %p '%s' has no active token : %d\n", inst->node, name, inst->ts->n);
               }
#endif
            }
         }

         if (inst->best < beamLimit) {
#ifdef DEBUG_TRACE
            if (trace & T_PRUNE) {
               char *name = NULL;
               if (inst->node->type == LN_MODEL)
                  name = FindMacroStruct (dec->net->hset, 'h', inst->node->data.hmm)->id->name;
               else 
                  name = "NULL_NODE";
               printf ("pruning node %p '%s' score %f off beam limit %f\n", inst->node, name, inst->best, beamLimit);

               PrintTokSet (dec, inst->ts);
            }
#endif
            /* take inst out of instsLayer list and deactivate it */
            if (prevInst)
               prevInst->next = inst->next;
            else                 /* first inst in layer */
               dec->instsLayer[l] = inst->next;
            DeactivateNode (dec, inst->node);
         }
         else {         /* inst survived */
#ifdef COLLECT_STATS
            ++dec->stats.nActive;
#endif
            /* special code for pronprob handling before sil layer */
            if (dec->net->silDict && (l == dec->net->spSkipLayer) && inst->ts[0].n > 0) 
               HandleSpSkipLayer (dec, inst);
            else {      /* normal case: non silDict or non spSkipLayer */
               /* call HandleWordend, if we don't do we-pruning or we 
                  are in in LAYER_SIL and LAYER_AB (where we need to handle the 
                  wordend nodes for SENT_START and SENT_END).
                  ### fix this
               */
#if 1          /* experiment for richer lattices. keep sp and sil
                  variants distinct by marking sil in LSBit of
                  tok->we_tag */
               /* #### we need the equivalent for pronprob sildicts! */
               if (l == LAYER_SIL && inst->node->type == LN_MODEL) {
                  if (inst->node->data.hmm != dec->net->hmmSP) {
                     int N, i;
                     TokenSet *ts;
                     RelToken *tok;
                     N = inst->node->data.hmm->numStates;
                     ts = &inst->ts[N-1];
                     for (i = 0; i < ts->n; ++i) {
                        tok = &ts->relTok[i];
                        tok->we_tag = (void *) ((long) tok->we_tag | 1);
                     }
                  }
               }
#endif

               PropagateExternal (dec, inst, !(dec->weBeamWidth < dec->beamWidth) || 
                                  (l == LAYER_SIL) || (l == LAYER_AB),
                                  l == LAYER_BY);
            }

            
            prevInst = inst;
            ++nActive;
            ++modelActive;
         }
      } /* for inst */

#if 0
      printf ("MTS_copy: %d MTS_fast: %d  MTS slow: %d ", mts_copy, mts_fast, mts_slow);
      printf ("MTS_newid: %d MTS_newidNTOK: %d\n", mts_newid, mts_newidNTOK);
      printf ("LMCacheLA:  %d hits  %d misses\n", 
              dec->lmCache->laHit, dec->lmCache->laMiss);
#endif
      if (trace & T_TOKSTATS)
         printf ("Pass2: %d active nodes in layer %d\n", nActive, l);

   }    /* for layer */

   if (trace & T_TOKSTATS)
      printf ("Sum Pass2: %d %f \n", modelActive, dec->curBeamWidth);


#if 1   /* max model pruning (using histogram pruning) */
#define MMP_NBINS 128
   if (dec->maxModel > 0) {
      int i, bin, nhist, hist[MMP_NBINS];
      LogFloat binWidth;
      
      binWidth = dec->curBeamWidth / MMP_NBINS;
      nhist = 0;
      for (i = 0; i < MMP_NBINS; ++i)
         hist[i] = 0;
      
      
      /* fill histogram */
      for (l = 0; l < dec->nLayers; ++l) {
         for (inst = dec->instsLayer[l]; inst; inst = inst->next) {
            if (inst->best > LSMALL) { 
               bin = (dec->bestScore - inst->best) / binWidth;
               assert (bin >= 0);       /* best is either LZERO or <= dec->bestScore */
               if (bin < MMP_NBINS) {
                  ++hist[bin];
                  ++nhist;
               }
            }
         }
      }
      
#if DEBUG_MMP
      for (i = 0; i < MMP_NBINS; ++i)
         printf ("i %d  %d\n",  i, hist[i]);
#endif

      if (nhist > dec->maxModel) {
         int nMod = 0;
         i = -1;
         while (nMod < dec->maxModel) {
            ++i;
            assert (i < MMP_NBINS);
            nMod += hist[i];
         }
         
         if (trace & T_PRUNE)
            printf ("nMod: %d, dec->maxModel: %d\n", nMod, dec->maxModel);

         if (trace & T_PRUNE)
            printf ("beam old: %f  ", dec->curBeamWidth);
         dec->curBeamWidth = (binWidth * i > maxLNBeamFlr * dec->beamWidth) ? binWidth * i : maxLNBeamFlr * dec->beamWidth;
         if (trace & T_PRUNE)
            printf ("  new: %f\n", dec->curBeamWidth);
      }
      else {       /* modelActive < maxModel */
         /* slowly increase beamWidth again */
         dec->curBeamWidth *= dynBeamInc;
         if (dec->curBeamWidth > dec->beamWidth)
            dec->curBeamWidth = dec->beamWidth;
      }
   }
#endif

   dec->beamLimit = dec->bestScore - dec->curBeamWidth;


#ifdef COLLECT_STATS
   ++dec->stats.nFrames;
#endif

#if 0
   printf ("cacheHits: %d  cacheMisses: %d\n", 
           dec->outPCache->cacheHit, dec->outPCache->cacheMiss);
   printf ("MTS_copy: %d MTS_fast: %d  MTS slow: %d ", mts_copy, mts_fast, mts_slow);
   printf ("MTS_newid: %d MTS_newidNTOK: %d\n", mts_newid, mts_newidNTOK);
   printf ("tokSetIDcount: %d\n", dec->tokSetIdCount);
   printf ("PI_LR: %d  PI_GEN: %d\n", PI_LR, PI_GEN);
#endif
   PI_LR = PI_GEN = 0;
   dec->outPCache->cacheHit = dec->outPCache->cacheMiss = 0;

#if 0
   printf ("LMCacheTrans:  %d hits  %d misses\n", 
           dec->lmCache->transHit, dec->lmCache->transMiss);
   printf ("LMCacheLA:  %d hits  %d misses\n", 
           dec->lmCache->laHit, dec->lmCache->laMiss);
#endif
   dec->lmCache->transHit = dec->lmCache->transMiss = 0;
   dec->lmCache->laHit = dec->lmCache->laMiss = 0;

#if 0
   Debug_DumpNet (dec->net);
#endif
#if 0
   AccumulateStats (dec);
#endif

   if (dec->nPhone > 0)
      CalcPhonePost (dec);
}




/*  CC-mode style info for emacs
 Local Variables:
 c-file-style: "htk"
 End:
*/
