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
/*    File: HLVRec-traceback.c  Traceback for HTK LV decoder   */
/* ----------------------------------------------------------- */

char *hlvrec_trace_version = "!HVER!HLVRec-traceback:   3.5.0 [CUED 12/10/15]";
char *hlvrec_trace_vc_id = "$Id: HLVRec-traceback.c,v 1.2 2015/10/12 12:07:24 cz277 Exp $";

/* Print Path
 */
static void PrintPath (DecoderInst *dec, WordendHyp *we)
{
   Pron pron;

   for (; we; we = we->prev) {
      pron = dec->net->pronlist[we->pron];
      if ((we->user & 3) == 1)
         pron = pron->next;             /* sp */
      else if ((we->user & 3) == 2)
         pron = pron->next->next;       /* sil */
      
      printf ("%s%d (%d %.3f %.3f)  ", pron->word->wordName->name, pron->pnum, 
              we->frame, we->score, we->lm);
   }
}

/* PrintRelTok
*/
static void PrintRelTok(DecoderInst *dec, RelToken *tok)
{
   printf ("lmState %p delta %f path %p ", tok->lmState, tok->delta, tok->path);
   PrintPath (dec, tok->path);
   printf ("\n");
}

/* PrintTokSet
 */
static void PrintTokSet (DecoderInst *dec, TokenSet *ts)
{
   int i;

   printf ("n = %d  score = %f id = %d\n", ts->n, ts->score, ts->id);

   for (i = 0; i < ts->n; ++i) {
      printf ("  %d ", i);
      PrintRelTok (dec, &ts->relTok[i]);
   }
}

/* BestTokSet

     returns best token set in network
*/

TokenSet *BestTokSet (DecoderInst *dec)
{
   LexNodeInst *bestInst;
   TokenSet *tsi, *ts;
   LogFloat best;
   int i, N;

   bestInst = dec->bestInst;
   ts = NULL;

   /* cz277 - best tok */
   if (bestInst != NULL) {
       switch (bestInst->node->type) {
       case LN_MODEL:
          N = bestInst->node->data.hmm->numStates;
          break;
       case LN_CON:
       case LN_WORDEND:
          N = 1;
          break;
       default:
          abort ();
          break;
       }
   
       /*ts = NULL;*/
       best = LZERO;
       for (i = 0; i < N; ++i) {
          tsi = &bestInst->ts[i];
          if (tsi->n > 0 && tsi->score > best)
             ts = tsi;
       }
   }

   return (ts);
}

/* TraceBack

     Finds best token in end state and returns path.

*/
Transcription *TraceBack(MemHeap *heap, DecoderInst *dec)
{
   Transcription *trans;
   LabList *ll;
   LLink lab, nextlab;
   WordendHyp *weHyp;
   TokenSet *ts;
   RelToken *bestTok=NULL;
   LogFloat prevScore, score;
   RelTokScore bestDelta;
   Pron pron;
   int i;
   HTime start;

   if (dec->net->end->inst && dec->net->end->inst->ts->n > 0)
      ts = dec->net->end->inst->ts;
   else {
      HError (-7820, "no token survived to sent end!");

      ts = BestTokSet (dec);
      if (!ts) {        /* return empty transcription */
         HError (-7820, "best inst is dead as well!");
         trans = CreateTranscription (heap);
         ll = CreateLabelList (heap, 0);
         AddLabelList (ll, trans);
         
         return trans;
      }
   }

   bestDelta = LZERO;
   for (i = 0; i < ts->n; ++i)
      if (ts->relTok[i].delta > bestDelta) {
         bestTok = &ts->relTok[i];
         bestDelta = bestTok->delta;
      }
   assert (bestDelta <= 0.1);   /* 0.1 for accuracy reasons */
   
   if (trace & T_PROP)
      PrintTokSet (dec, ts);

   if (trace & T_TOP) {
      printf ("best score %f\n", ts->score);
      PrintRelTok (dec, bestTok);
   }

   trans = CreateTranscription (heap);
   ll = CreateLabelList (heap, 0);

   if(bestTok==NULL)
     HError(7820,"best token not found");
   /* going backwards from </s> to <s> */
   for (weHyp = bestTok->path; weHyp; weHyp = weHyp->prev) {
      lab = CreateLabel (heap, ll->maxAuxLab);
      pron = dec->net->pronlist[weHyp->pron];
      if ((weHyp->user & 3) == 1)
         pron = pron->next;             /* sp */
      else if ((weHyp->user & 3) == 2)
         pron = pron->next->next;       /* sil */

      lab->labid = pron->outSym;
      lab->score = weHyp->score;
      lab->start = 0.0;
      lab->end = weHyp->frame * dec->frameDur * 1.0e7;
      lab->succ = ll->head->succ;
      lab->pred = ll->head;
      lab->succ->pred = lab->pred->succ = lab;
   }

   start = 0.0;
   prevScore = 0.0;
   for (lab = ll->head->succ; lab != ll->tail; lab = lab->succ) {
      lab->start = start;
      start = lab->end;
      score = lab->score - prevScore;
      prevScore = lab->score;
      lab->score = score;
   }

   for (lab = ll->head->succ; lab != ll->tail; lab = nextlab) {
      nextlab = lab->succ;
      if (!lab->labid)          /* delete words with [] outSym */
         DeleteLabel (lab);
   }

   AddLabelList (ll, trans);
   
   return trans;
}

/* LatTraceBackCount

     recursively assign numbers to wordendHyps (lattice nodes) and at the 
     same time count weHyps + altweHyps (lattice links)
*/
static void LatTraceBackCount (DecoderInst *dec, WordendHyp *path, int *nnodes, int *nlinks)
{
   AltWordendHyp *alt;

   if (!path)
      return;

   /* the pronvar is encoded in the user field: 
      0: -  1:sp  2: sil  */

   if (path->user < 4) {      /* not seen yet */

      /*      path->score *= -1.0; */

      ++(*nlinks);
      LatTraceBackCount (dec, path->prev, nnodes, nlinks);
      for (alt = path->alt; alt; alt = alt->next) {
         ++(*nlinks);
         LatTraceBackCount (dec, alt->prev, nnodes, nlinks);
      }
      ++(*nnodes);
      path->user += *nnodes * 4;         /* preserve pronvar bits */

#if 0
      {
         Pron pron;
         char *s;

         pron = dec->net->pronlist[path->pron];

         s = (pron->outSym) ? pron->outSym->name : "!NULL";

         printf ("   n %d t %.3f W %s\n", (int) path->user, path->frame*dec->frameDur, s);
      }
#endif

   }
}

/* Paths2Lat

     recursively create nodes and arcs for weHyp end and predecessors
*/
static void Paths2Lat (DecoderInst *dec, Lattice *lat, WordendHyp *path,
                       int *na)
{
   int s, n;
   AltWordendHyp *alt;
   TokScore prevScore;
   LNode *ln;
   LArc *la;
   Pron pron;

   if (!path)
      return;

   n = (int) (path->user / 4);  /* current node (end node of arcs) */

   if (!lat->lnodes[n].hook) {      /* not seen yet */
      ln = &lat->lnodes[n];
      ln->hook = (Ptr) path;

      ln->n = n;
      ln->time = path->frame*dec->frameDur;   /* fix frame duration! */
      pron = dec->net->pronlist[path->pron];

      if ((path->user & 3) == 1)
         pron = pron->next;             /* sp */
      else if ((path->user & 3) == 2)
         pron = pron->next->next;       /* sil */
      else 
         assert ((path->user & 3) == 0);
               

      ln->word = pron->word;
      ln->v = pron->pnum;

      if (trace & T_LAT)
         printf ("I=%d t=%.2f W=%d\n", n, path->frame*dec->frameDur, path->pron);

      la = &lat->larcs[*na];
      ++(*na);

      s = path->prev ? (int) (path->prev->user / 4) : 0;
      prevScore = path->prev ? path->prev->score : 0.0;

      la->start = &lat->lnodes[s];
      la->end = &lat->lnodes[n];
      /* add to linked lists  foll/pred */
      la->farc = la->start->foll;
      la->start->foll = la;
      la->parc = la->end->pred;
      la->end->pred = la;

      la->prlike = pron->prob;
      la->aclike = path->score - prevScore - path->lm 
         - la->prlike * dec->pronScale;
      la->lmlike = (path->lm - dec->insPen) / dec->lmScale;


#ifdef MODALIGN
      if (dec->modAlign) {
         int startFrame;

         startFrame = path->prev ? path->prev->frame : 0;
         la->lAlign = LAlignFromModpath (dec, lat->heap, path->modpath,
                                         startFrame, &la->nAlign);

#if 0   /* debug trace */
         printf ("%d  ", *na - 1);
         PrintModPath (dec, path->modpath);
#endif
      }
#endif

      if (trace & T_LAT)
         printf ("J=%d S=%d E=%d a=%f l=%f\n", *na, s, n, la->aclike, la->lmlike);

      Paths2Lat (dec, lat, path->prev, na);

      /* alternatives */
      for (alt = path->alt; alt; alt = alt->next) {

         la = &lat->larcs[*na];
         ++(*na);

         s = alt->prev ? (int) (alt->prev->user / 4) : 0;
         prevScore = alt->prev ? alt->prev->score : 0.0;

         la->start = &lat->lnodes[s];
         la->end = &lat->lnodes[n];
         /* add to linked lists  foll/pred */
         la->farc = la->start->foll;
         la->start->foll = la;
         la->parc = la->end->pred;
         la->end->pred = la;

         la->prlike = pron->prob;
         la->aclike = (path->score + alt->score) - prevScore - alt->lm - 
            la->prlike * dec->pronScale;
         la->lmlike = (alt->lm - dec->insPen) / dec->lmScale;
         
#ifdef MODALIGN
         if (dec->modAlign) {
            int startFrame;
            
            startFrame = alt->prev ? alt->prev->frame : 0;
            
            la->lAlign = LAlignFromAltModpath (dec, lat->heap, alt->modpath, path->modpath,
                                               startFrame, &la->nAlign);
#if 0           /* debug trace */
            printf ("%d ALT ", *na - 1);
            PrintModPath (dec, alt->modpath);
            printf ("     ");
            PrintModPath (dec, path->modpath);
#endif
         }
#endif
         if (trace & T_LAT)
            printf ("J=%d S=%d E=%d a=%f l=%f\n", *na, s, n, la->aclike, la->lmlike);
         
         Paths2Lat (dec, lat, alt->prev, na);
      }
   }
}


/* LatTraceBack

     produce Lattice from the wordEnd hypotheses recoded in dec
*/
Lattice *LatTraceBack (MemHeap *heap, DecoderInst *dec)
{
   Lattice *lat;
   int i, nnodes = 0, nlinks = 0;
   WordendHyp *sentEndWE;

   if (!dec->net->end->inst)
      HError (-7821, "LatTraceBack: end node not active");
   else
      printf ("found %d tokens in end state\n", dec->net->end->inst->ts->n);

   if (buildLatSE && dec->net->end->inst && dec->net->end->inst->ts->n == 1)
      sentEndWE = dec->net->end->inst->ts->relTok[0].path;
   else {
      if (buildLatSE)
         HError (-7821, "no tokens in sentend -- falling back to BUILDLATSENTEND = F");

      sentEndWE = BuildLattice (dec);
   }

   if (!sentEndWE) {
      HError (-7821, "LatTraceBack: no active sil wordend nodes");
      if (forceLatOut) {
         HError (-7821, "LatTraceBack: forcing lattice output");
#ifdef MODALIGN
         if (dec->modAlign) 
/*             HError (-9999, "LatTraceBack: forced lattice output not supported with model-alignment"); */
            sentEndWE = BuildForceLat (dec);
         else 
#endif
            sentEndWE = BuildForceLat (dec);
      }
   }
   if (!sentEndWE)
      return NULL;
   
   /* recursively number weHyps (nodes), count weHyp + altweHyp (links) */
   LatTraceBackCount (dec, sentEndWE, &nnodes, &nlinks);

   ++nnodes;    /* !NULL lattice start node */
   printf ("nnodes %d nlinks %d\n", nnodes, nlinks);

   /*# create lattice */
   lat = NewLattice (heap, nnodes, nlinks);

   /* #### fill in info (e.g. lmscale, inspen, models) */
   lat->voc = dec->net->voc;
   lat->utterance = dec->utterFN;
   lat->vocab = dec->net->vocabFN;
   lat->hmms = dec->hset->mmfNames ? dec->hset->mmfNames->fName : NULL;
   lat->net = dec->lm->name;
   lat->lmscale = dec->lmScale;
   lat->wdpenalty = dec->insPen;
   lat->prscale = dec->pronScale;
   lat->framedur = 1.0;
      
   for (i = 0; i < nnodes; ++i)
      lat->lnodes[i].hook = NULL;

   {
      int na;
      na = 0;
      /* create lattice nodes & arcs */
      Paths2Lat (dec, lat, sentEndWE, &na);
   }
   
#ifdef MODALIGN
   if (dec->modAlign)
      CheckLAlign (dec, lat);

#endif
   return lat;
}



/************      model-level traceback */

#ifdef MODALIGN
LAlign *LAlignFromModpath (DecoderInst *dec, MemHeap *heap,
                           ModendHyp *modpath, int wordStart, short *nLAlign)
{
   ModendHyp *m;
   LAlign *lalign;
   MLink ml;
   int n;
   int startFrame = 0;
   
   n = 0;
   for (m = modpath; m; m = m->prev)
      if (m->ln->type == LN_MODEL)
         ++n;

   lalign = New (heap, n * sizeof(LAlign));
   *nLAlign = n;

   for (m = modpath; m; m = m->prev) {
      if (m->ln->type == LN_MODEL) {
         startFrame = (m->prev ? m->prev->frame : wordStart);
         ml = FindMacroStruct (dec->hset, 'h', (Ptr) m->ln->data.hmm);
         if (!ml)
            HError (7822, "LAlignFromModpath: model not found!");

         assert (m->frame >= startFrame);
         --n;
         lalign[n].state = -1;
         lalign[n].like = 0.0;
         lalign[n].dur = (m->frame - startFrame) * dec->frameDur;
         /* sxz20 */
         if (m->ln->labid) assert(m->ln->labid->name);
         lalign[n].label = (m->ln->labid)? m->ln->labid : ml->id;
         /*lalign[n].label = ml->id;*/
      }
   }
   assert (n == 0);

   return (lalign);
}

LAlign *LAlignFromAltModpath (DecoderInst *dec, MemHeap *heap,
                              ModendHyp *modpath, ModendHyp *mainModpath,
                              int wordStart, short *nLAlign)
{
   ModendHyp *m, *nextM;
   LAlign *lalign;
   MLink ml;
   int n;
   int startFrame = 0;
   
   /* check for WE in main modpath */
   n = 0;
   for (m = mainModpath; m; m = m->prev)
      if (m->ln->type == LN_WORDEND)
         break;
      else {
         if (m->ln->type == LN_MODEL)
            ++n;
      }
   
   /* if there are no WE models in main modpath then we are looking at
      the </s> link and should just call the normal LAlignFromModpath() */
   if (!m)
      return LAlignFromModpath (dec, heap, modpath, wordStart, nLAlign);

   /* take the n first model entries from the main modpaths (upto the WE node)
      and then switch to the alt modpath */

   for (m = modpath; m; m = m->prev)
      if (m->ln->type == LN_MODEL)
         ++n;
   
   lalign = New (heap, n * sizeof(LAlign));
   *nLAlign = n;
   
   for (m = mainModpath; m; m = nextM) {
      nextM = m->prev;
      if (m->ln->type == LN_MODEL) {
         startFrame = (m->prev ? m->prev->frame : wordStart);
         ml = FindMacroStruct (dec->hset, 'h', (Ptr) m->ln->data.hmm);
         if (!ml)
            HError (7822, "LAlignFromModpath: model not found!");

         assert (m->frame >= startFrame);
         --n;
         assert (n >= 0);
         lalign[n].state = -1;
         lalign[n].like = 0.0;
         lalign[n].dur = (m->frame - startFrame) * dec->frameDur;
         /* sxz20 */
         if (m->ln->labid) assert(m->ln->labid->name);
         lalign[n].label = (m->ln->labid)? m->ln->labid : ml->id;
         /*lalign[n].label = ml->id;*/
      }
      else if (m->ln->type == LN_WORDEND)
         nextM = modpath;
   }
   assert (n == 0);

   return (lalign);
}

void PrintModPath (DecoderInst *dec, ModendHyp *m)
{
   MLink ml;
   char *s, *t;

   for (; m; m = m->prev) {
      s = "?";
      switch (m->ln->type) {
      case LN_WORDEND:
         t = "WE";
         s = dec->net->pronlist[m->ln->data.pron]->outSym->name;
         break;
      case LN_CON:
         t = "CON";
         s = "";
         break;
      case LN_MODEL:
         t = "MOD";
         ml = FindMacroStruct (dec->hset, 'h', (Ptr) m->ln->data.hmm);
         if (ml)
            s = ml->id->name;
      }
      printf ("(%d %s %s) ", m->frame, t, s);
   }
   printf ("\n");
}

/* Faking sentence end arc model alignment */
void FakeSEModelAlign(Lattice *lat, LArc *la)
{  
   la->nAlign = 1;   
      
   la->lAlign = New (lat->heap, sizeof(LAlign));
   la->lAlign->state = -1;
   la->lAlign->dur = la->end->time - la->start->time;
   la->lAlign->label = GetLabId("sil", FALSE);
}

void CheckLAlign (DecoderInst *dec, Lattice *lat)
{
   int i, j;
   LArc *la;
   float dur, laDur;
   Pron pron;

   for (i = 0, la = lat->larcs; i < lat->na; ++i, ++la) {
      if (la->nAlign == 0 || !la->lAlign) {
         if (forceLatOut) {
            /* Faking sentence end arc model alignment */
            FakeSEModelAlign(lat, la);
         }
         else {
            HError (7823, "CheckLAlign: empty model alignment for arc %d", i);
         }
      }

      for (pron = la->end->word->pron; pron; pron = pron->next)
         if (pron->pnum == la->end->v)
            break;
      assert (pron);

      laDur = (la->end->time - la->start->time);
      dur = 0.0;
      for (j = 0; j < la->nAlign; ++j) {
         dur += la->lAlign[j].dur;

#if 0   /* sanity checking -- does not work for non-sildicts */
         strcpy (buf, la->lAlign[j].label->name);
         TriStrip (buf);
         monolab = GetLabId (buf, FALSE);
         assert (pron->phones[j] == monolab);
#endif
      }
#if 0
      assert (la->nAlign == pron->nphones);
#endif

      if (fabs (dur - laDur) > dec->frameDur/2)
         printf ("CheckLAlign: MODALIGN Sanity check failed! %d laDur %.2f  dur %.2f\n", i, laDur, dur);
   }
}
#endif




/* FakeSEpath

     helper functions for BuildLattice and BuildForceLat.
     takens token and add LM transition to </s>

*/

AltWordendHyp *FakeSEpath (DecoderInst *dec, RelToken *tok, Boolean useLM)
{
   AltWordendHyp *alt = NULL;
   PronId endPronId;
   LMState dest;
   LMTokScore lmScore;
   
   endPronId = dec->net->end->data.pron;

   if (useLM)
      lmScore = LMCacheTransProb (dec, dec->lm, tok->lmState, endPronId, &dest);
   else
      lmScore = 0.0;
   if (lmScore > LSMALL) {  /* transition for END state possible? */
      /* cz277 - 64bit */
      /*assert (!useLM || dest == (Ptr) 0xfffffffe);*/
      assert (!useLM || dest == (Ptr) 0xfffffffffffffffe);

      lmScore += dec->insPen;

      alt = (AltWordendHyp *) New (&dec->altweHypHeap, sizeof (AltWordendHyp));
      alt->next = NULL;

      if (!dec->fastlmla) {
         assert (lmScore <= tok->lmscore + 0.1); /* might not be true for more aggressive LMLA? */
      }
      /* temporarily store full score in altWEhyp */
      alt->score = tok->delta + (lmScore - tok->lmscore);
      alt->lm = lmScore;
      alt->prev = tok->path;
   }
   
   return alt;
}

/* AltPathList2Path

     Create full WordendHyp with alternatives from list of AltWordendHyps

*/
WordendHyp *AltPathList2Path (DecoderInst *dec, AltWordendHyp *alt, PronId pron)
{
   WordendHyp *path;
   AltWordendHyp *bestAlt=NULL, *a;
   TokScore bestAltScore = LZERO;
   AltWordendHyp **pAlt;
   int i;

   /* find best */
   for (a = alt; a; a = a->next) {
      if (a->score > bestAltScore) {
         bestAltScore = a->score;
         bestAlt = a;
      }
   }
   if(bestAlt==NULL)
     HError(7823,"failed to find best alternative word end");

   /* create full WordendHyp for best */
   path = (WordendHyp *) New (&dec->weHypHeap, sizeof (WordendHyp));
   path->prev = bestAlt->prev;
   path->pron = pron;
   path->frame = dec->frame;
   path->score = bestAlt->score;
   path->lm = bestAlt->lm;
   path->user = 0;

   i = 0;
   pAlt = &path->alt;
   for ( ; alt; alt = alt->next) {
      if (alt != bestAlt) {
         ++i;
         *pAlt = alt;
         pAlt = &alt->next;
         alt->score = alt->score - path->score;
      }
   }
   *pAlt = NULL;

   printf ("found %d arcs\n", i);
   return path;
}


/* BuildLattice

     construct WordendHyp structure at the end of a sentence from all
     the tokensets in the final state of the sil Nodes in the SIL layer.
*/
WordendHyp *BuildLattice (DecoderInst *dec)
{
   int N, i;
   LexNodeInst *inst;
   LexNode *ln;
   TokenSet *ts;
   HLink hmmSP;
   AltWordendHyp *alt, *altPrev;
   WordendHyp *path;
   RelToken *tok;
#ifdef MODALIGN
   ModendHyp *silModend = NULL;
#endif

   alt = altPrev = NULL;
   hmmSP = dec->net->hmmSP;
   for (inst = dec->instsLayer[LAYER_SIL]; inst; inst = inst->next) {
      ln = inst->node;
      if (ln->type == LN_MODEL && ln->data.hmm != hmmSP) {
         N = ln->data.hmm->numStates;
         ts = &inst->ts[N-1];

         for (i = 0; i < ts->n; ++i) {
            tok = &ts->relTok[i];
            /* we have to update the path's score & frame, since 
               UpdateWordEndHyp() never got called on this path,
               A side effect is that the </s> link will have 
               aclike=0.0 and 1 frame length*/
            tok->path->score = ts->score + tok->delta;
            tok->path->frame = dec->frame - 1;
            
#ifdef MODALIGN
            /* skip the final (MOD sil) modpath entry.
               If the token is outside the beam we will not have added a (MOD SIL) entry
               in PropagateExternal(). Nevertheless we should use this token, as it might
               slip into the LATPRUNEBEAM due to the final </s> LM transition applied to
               all tokens. 
            */
            if (dec->modAlign) {
               if (tok->modpath->ln == ln)
                  tok->modpath = tok->modpath->prev;
               tok->path->frame = tok->modpath->frame;
               tok->path->modpath = tok->modpath;
               
               if (!silModend) {
                  silModend = New (&dec->modendHypHeap, sizeof (ModendHyp));
                  silModend->frame = dec->frame;
                  silModend->ln = ln;   /* dodgy, but we just need ln with 'sil' model... */
                  silModend->prev = NULL;
               }
            }
#endif
            alt = FakeSEpath (dec, tok, TRUE);

            if (alt) {
               alt->score += ts->score;
               alt->next = altPrev;
               altPrev = alt;
#ifdef MODALIGN
               alt->modpath = silModend;
#endif
            }
         } /* for tok */
      }
   } /* for inst */

   if (!alt)
      alt = altPrev;  /* make sure we don't end up with a NULL alt when there is a non-NULL alternative */

   if (!alt)   /* no token in sil models at all */
      return NULL;

   path = AltPathList2Path (dec, alt, dec->net->end->data.pron);
#ifdef MODALIGN
   path->modpath = silModend;
#endif

   return path;
}

AltWordendHyp *BuildLatAltList (DecoderInst *dec, TokenSet *ts, Boolean useLM)
{
   AltWordendHyp *alt, *altPrev;
   RelToken *tok;
   int i;
#ifdef MODALIGN
   ModendHyp *silModend = NULL;
#endif

   alt = altPrev = NULL;
   for (i = 0; i < ts->n; ++i) {
      tok = &ts->relTok[i];

      alt = FakeSEpath (dec, tok, useLM);
      if (alt) {
         alt->score += ts->score;
         alt->next = altPrev;
         altPrev = alt;
#ifdef MODALIGN
         alt->modpath = silModend;
#endif
      }
   }
   if (!alt)
      alt = altPrev;  /* make sure we don't end up with a NULL alt when there is a non-NULL alternative */
   return alt;
}


WordendHyp *BuildForceLat (DecoderInst *dec)
{
   TokenSet *ts;
   WordendHyp *path;
   AltWordendHyp *alt;
   RelToken *tok;
   int i;
#ifdef MODALIGN
   ModendHyp *silModend = NULL;
#endif

   ts = BestTokSet (dec);
   for (i = 0; i < ts->n; ++i) {
      tok = &ts->relTok[i];
      /* we have to update the path's score & frame, since 
         UpdateWordEndHyp() never got called on this path,
         A side effect is that the </s> link will have 
         aclike=0.0 and 1 frame length*/
      if (tok->path) {
         tok->path->score = ts->score + tok->delta;
         tok->path->frame = dec->frame - 1;
      }
   }

   alt = BuildLatAltList (dec, ts, TRUE);

   
   if (!alt) {  /* no valid LM transitions, try without */
      HError (-7820, "BuildForceLat: no tokens survived with valid LM transitions, inserting LM 0.0 arcs.");
      alt = BuildLatAltList (dec, ts, FALSE);
   }

   if (!alt) {   /* how can this happen? */
      HError (-7899, "BuildForceLat: unable to force building lattice, giving up. THIS SHOULDN'T HAPPEN!");
      return NULL;
   }

   path = AltPathList2Path (dec, alt, dec->net->end->data.pron);
#ifdef MODALIGN
   path->modpath = silModend;
#endif
   return path;
}


/* ------------------------ End of HLVRec-traceback.c ----------------------- */

