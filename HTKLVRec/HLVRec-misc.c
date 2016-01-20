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
/* File: HLVRec-misc.c  Miscellaneous functions for LV decoder */
/* ----------------------------------------------------------- */

char *hlvrec_misc_version = "!HVER!HLVRec-misc:   3.5.0 [CUED 12/10/15]";
char *hlvrec_misc_vc_id = "$Id: HLVRec-misc.c,v 1.1.1.1 2006/10/11 09:54:55 jal58 Exp $";

/* CheckTokenSetOrder

     check whether the relTokens are sorted by LMState order 
*/
void CheckTokenSetOrder (DecoderInst *dec, TokenSet *ts)
{
   int i;
   RelToken *prevTok;
   Boolean ok = TRUE;

   prevTok = &ts->relTok[0];
   for (i = 1; i < ts->n; ++i) {
      if (TOK_LMSTATE_LT(&ts->relTok[i], prevTok) || TOK_LMSTATE_EQ(&ts->relTok[i], prevTok))
         ok = FALSE;
      prevTok = &ts->relTok[i];
   }

   if (!ok) {
      printf ("XXXXX CheckTokenSetOrder \n");
      PrintTokSet (dec, ts);
      abort();
   }
}

/****           debug functions */

/*
  Debug_DumpNet

*/
void Debug_DumpNet (LexNet *net)
{
   int i, j, k, N;
   LexNode *ln;
   LexNodeInst *inst;
   TokenSet *ts;
   RelToken *rtok;
   FILE *debugFile;
   Boolean isPipe;
   static char *debug_net_fn = "net.dump";

   debugFile = FOpen (debug_net_fn, NoOFilter, &isPipe);
   if (!debugFile) {
      printf ("fopen failed\n");
      return;
   }

   fprintf (debugFile, "(LexNet *) %p\n", net);
   fprintf (debugFile, "nNodes %d\n", net->nNodes);

   for (i = 0; i < net->nNodes; ++i) {
      ln = &net->node[i];
      inst = ln->inst;
      if (inst) {
         fprintf (debugFile, "node %d  (LexNode *) %p", i, ln);
         fprintf (debugFile, " type %d nfoll %d", ln->type, ln->nfoll);
         if (ln->type == LN_MODEL)
            fprintf (debugFile, " name %s", 
                    FindMacroStruct (net->hset, 'h', ln->data.hmm)->id->name);
         fprintf (debugFile, "\n");
         
         assert (inst->node == ln);
         fprintf (debugFile, " (LexNodeInst *) %p", inst);
         fprintf (debugFile, "  best %f\n", inst->best);
         N = (ln->type == LN_MODEL) ? ln->data.hmm->numStates : 1;
         for (j = 0; j < N; ++j) {
            ts = &inst->ts[j];
            if (ts->n > 0) {
               fprintf (debugFile, "  state %d (TokenSet *) %p", j+1, ts);
               fprintf (debugFile, "   score %f\n", ts->score);
               for (k = 0; k < ts->n; ++k) {
                  rtok = &ts->relTok[k];
                  fprintf (debugFile, "   (RelToken *) %p", rtok);
                  fprintf (debugFile, "    delta %f  lmstate %p lmscore %f\n", 
                          rtok->delta, rtok->lmState, rtok->lmscore);
               }
            }
         }
      }
   }

   FClose (debugFile, isPipe);
}



#if 0
void Debug_Dump_LMLA_hastab(DecoderInst *dec)
{
   int i, n;
   LMLACacheEntry *e;

   FILE *debugFile;
   Boolean isPipe;
   static char *debug_net_fn = "lmla_cache.dump";

   debugFile = FOpen (debug_net_fn, NoOFilter, &isPipe);
   if (!debugFile) {
      printf ("fopen failed\n");
      return;
   }

   for (i = 0; i < dec->nLMLACacheBins; ++i) {
      n = 0;
      for (e = dec->lmlaCache[i]; e; e = e->next)
         ++n;
      fprintf (debugFile, "LMLA %d %d\n", i, n);
   }

   FClose (debugFile, isPipe);
}
#endif

/*

*/

void Debug_Check_Score (DecoderInst *dec)
{
   int l, j, N;
   LexNode *ln;
   LexNodeInst *inst;
   TokenSet *ts;

   for (l = 0; l < dec->nLayers; ++l) {
      int sumTok, maxTok, nTS;
      sumTok = maxTok = 0;
      nTS = 0;
      for (inst = dec->instsLayer[l]; inst; inst = inst->next) {
#if 0
         if (inst->best < dec->beamLimit)
            printf ("layer %d (LexNodeInst *) %p\n", l, inst);
#endif
         ln = inst->node;
         N = (ln->type == LN_MODEL) ? ln->data.hmm->numStates : 1;
         for (j = 0; j < N; ++j) {
            ts = &inst->ts[j];
            if (ts->n > 0) {
               sumTok += ts->n;
               ++nTS;
               if (ts->n > maxTok)
                  maxTok = ts->n;
#if 0
               for (k = 0; k < ts->n; ++k) {
                  rtok = &ts->relTok[k];
                  if (ts->score + rtok->delta < dec->beamLimit) {
                     printf ("l %d *((LexNodeInst *) %p) *((TokenSet *) %p)\n", l, inst, ts);
                  }
               }
#endif
            }
         }
      }
      printf ("l %d aveTok/TS %.3f maxTok %d\n",
              l, (float)sumTok/nTS, maxTok);
   }
}


/***************** phone posterior estimation *************************/

void InitPhonePost (DecoderInst *dec)
{
   HMMScanState hss;
   HLink hmm;
   MLink m;
   char buf[100];
   LabId phoneId;

   NewHMMScan (dec->hset, &hss);
   do {
      hmm = hss.hmm;
      assert (!hmm->hook);
      m = FindMacroStruct (dec->hset, 'h', hmm);
      assert (strlen (m->id->name) < 100);
      strcpy (buf, m->id->name);
      TriStrip (buf);
      phoneId = GetLabId (buf, TRUE);
      phoneId->aux = (Ptr) 0;
      hmm->hook = (Ptr) phoneId;
   } while(GoNextHMM(&hss));
   EndHMMScan(&hss);

   dec->nPhone = 0;
   /* count monophones -- #### make this more efficent! */
   NewHMMScan (dec->hset, &hss);
   do {
      hmm = hss.hmm;
      phoneId = (LabId) hmm->hook;
      if (!phoneId->aux) {
         ++dec->nPhone;
         phoneId->aux = (Ptr)(unsigned long int)dec->nPhone;

         assert (dec->nPhone < 100);
         dec->monoPhone[dec->nPhone] = phoneId;
      }
   } while(GoNextHMM(&hss));
   EndHMMScan(&hss);

   printf ("found %d monophones\n", dec->nPhone);

   dec->phonePost = (LogDouble *) New (&gcheap, (dec->nPhone+1) * sizeof (LogDouble));
   dec->phoneFreq = (int *) New (&gcheap, (dec->nPhone+1) * sizeof (int));
}

void CalcPhonePost (DecoderInst *dec)
{
   int l, N, i, j;
   LexNodeInst *inst;
   LabId phoneId;
   int phone;
   LogDouble *phonePost;
   int *phoneFreq;
   TokenSet *ts;
   RelToken *tok;
   LogDouble sum;

   phonePost = dec->phonePost;
   phoneFreq = dec->phoneFreq;
   for (i = 0; i <= dec->nPhone; ++i) {
      phonePost[i] = LZERO;
      phoneFreq[i] = 0;
   }

   for (l = 0; l < dec->nLayers; ++l) {
      for (inst = dec->instsLayer[l]; inst; inst = inst->next) {
         if (inst->node->type == LN_MODEL) {
            phoneId = (LabId) inst->node->data.hmm->hook;
            phone = (int)(unsigned long int)phoneId->aux;
            assert (phone >= 1 && phone <= dec->nPhone);
            
            N = inst->node->data.hmm->numStates;
            for (i = 1; i < N; ++i) {
               ts = &inst->ts[i];
               for (j = 0, tok = ts->relTok; j < ts->n; ++j, ++tok) {
                  phonePost[phone] = LAdd (phonePost[phone], ts->score + tok->delta);
                  ++phoneFreq[phone];
               }
            }
         }
      }
   }

   sum = LZERO;
   for (i = 0; i <= dec->nPhone; ++i)
      sum = LAdd (sum, phonePost[i]);
#if 0
   printf ("sum %f\n", sum);
#endif
   
   for (i = 0; i <= dec->nPhone; ++i) {
      if (phonePost[i] > LSMALL) {
         phonePost[i] = phonePost[i] - sum;
#if 0 
         printf ("phone %d freq %d post %f\n", i, phoneFreq[i],
                 phonePost[i]);
#endif
      } else 
         phonePost[i] = LZERO;
      
   }
}

/******************** Token Statistics ********************/

struct _LayerStats {
   int nInst;
   int nTS;
   int nTok;
   TokScore bestScore;
   TokScore worstScore;
};

typedef struct _LayerStats LayerStats;

void AccumulateStats (DecoderInst *dec)
{
   MemHeap statsHeap;
   LayerStats *layerStats;
   int l;

   CreateHeap (&statsHeap, "Token Stats Heap", MSTAK, 1, 1.5, 10000, 100000);

   layerStats = (LayerStats *) New (&statsHeap, dec->nLayers * sizeof (LayerStats));
   memset ((void *) layerStats, 0, dec->nLayers * sizeof (LayerStats));

   for (l = 0; l < dec->nLayers; ++l) {
      layerStats[l].nInst = layerStats[l].nTS = layerStats[l].nTok = 0;
      layerStats[l].bestScore = LZERO;
      layerStats[l].worstScore = - LZERO;
   }

   /* count inst/ts/tok, find best/worst scores */
   {
      int l, N, s, i;
      LayerStats *ls;
      LexNodeInst *inst;
      LexNode *ln;
      TokenSet *ts;
      RelToken *tok;
      TokScore score;
      TokScore instBest;

      for (l = 0; l < dec->nLayers; ++l) {
         ls = &layerStats[l];
         for (inst = dec->instsLayer[l]; inst; inst = inst->next) {
            ++ls->nInst;
            ln = inst->node;

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

            instBest = LZERO;
            for (s = 0; s < N; ++s) {   /* for each state/TokenSet */
               ts = &inst->ts[s];
               if (ts->n > 0) {
                  ++ls->nTS;
                  if (ts->score > instBest)
                     instBest = ts->score;
                  for (i = 0; i < ts->n; ++i) { /* for each token */
                     tok = &ts->relTok[i];
                     ++ls->nTok;
                     score = ts->score + tok->delta;
                     if (score > ls->bestScore)
                        ls->bestScore = score;
                     if (score < ls->worstScore)
                        ls->worstScore = score;
#if 0   /* sanity check for Lat Rescore */                     
                     if (tok->path && tok->lmState != (Ptr) 0xfffffffe ) {
                        assert (dec->net->pronlist[tok->path->pron]->word == 
                                ((FSLM_LatNode *) tok->lmState)->word);
                     }
#endif
                  }
               }
            }
            assert (instBest == inst->best);
            
         } /* for each inst */
         printf ("STATS layer %d  ", l);
         printf ("%5d Insts   %5d TokenSets   %6d Tokens  %.2f Tok/TS ",
                 ls->nInst, ls->nTS, ls->nTok, (float) ls->nTok / ls->nTS);
         printf ("best:  %.4f   ", ls->bestScore);
         printf ("worst: %.4f\n", ls->worstScore);

      } /* for each layer */
   }
   
   ResetHeap (&statsHeap);
   DeleteHeap (&statsHeap);
}

/* ------------------------ End of HLVRec-misc.c ----------------------- */

