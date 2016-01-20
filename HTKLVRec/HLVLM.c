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
/*       File: HLVLM.c  Language model for HTK LV decoder      */
/* ----------------------------------------------------------- */

char *hlvlm_version = "!HVER!HLVLM:   3.5.0 [GE 12/10/15]";
char *hlvlm_vc_id = "$Id: HLVLM.c,v 1.1.1.1 2006/10/11 09:54:55 jal58 Exp $";

#include "HShell.h"
#include "HMem.h"
#include "HMath.h"
#include "HWave.h"
#include "HLabel.h"
#include "HDict.h"
#include "HAudio.h"
#include "HParm.h"
#include "HANNet.h"
#include "HModel.h"
#include "HNet.h"

#include "lvconfig.h"

/* #include "HLM.h" */
#include "HLVLM.h"

#include <assert.h>

/* ----------------------------- Trace Flags ------------------------- */

#define T_TOP 0001         /* Trace  */
#define T_ACCESS 0002      /* trace ever lm access */

static int trace=0;
static ConfParam *cParm[MAXGLOBS];      /* config parameters */
static int nParm = 0;
static Boolean rawMITFormat = FALSE;    /* by default do not use HTK quoting */

const double LN10 = 2.30258509299404568;    /* Defined to save recalculating it */

/* --------------------------- Prototypes ---------------------- */

FSLM_ngram *CreateBoNGram (MemHeap *heap, int vocSize, int counts[NSIZE]);
NEntry *GetNEntry (FSLM_ngram *nglm, LMId ndx[NSIZE], Boolean create);

void SetNEntryBO (FSLM *lm);

LogFloat LMTransProb_ngram (FSLM *lm, LMState src, PronId pronid, LMState *dest);
LogFloat LMLookAhead_2gram (FSLM *lm, LMState src, PronId minPron, PronId maxPron);
LogFloat LMLookAhead_3gram (FSLM *lm, LMState src, PronId minPron, PronId maxPron);
LogFloat LMLookAhead_ngram (FSLM *lm, LMState src, PronId minPron, PronId maxPron);
LogFloat LMTransProb_latlm (FSLM *lm, LMState src, PronId pronid, LMState *dest);
LogFloat LMLookAhead_latlm (FSLM *lm, LMState src, PronId minPron, PronId maxPron);
LMState Fast_LMLA_LMState (FSLM *lm, LMState src);
     
/* --------------------------- Initialisation ---------------------- */

/* EXPORT->InitLVLM: register module & set configuration parameters */
void InitLVLM (void)
{
   int i;
   Boolean b;

   Register (hlvlm_version, hlvlm_vc_id);
   nParm = GetConfig("HLVLM", TRUE, cParm, MAXGLOBS);
   if (nParm>0){
      if (GetConfInt (cParm, nParm, "TRACE", &i)) trace = i;
      if (GetConfBool (cParm, nParm, "RAWMITFORMAT", &b)) rawMITFormat = b;
   }

#if 0
   /* init HLM */
   InitLM ();
#endif

#ifdef LM_NGRAM_INT
   if (sizeof (SEntry) != 4)
      HError (7690, "strange size of SEntry structures (%d)", sizeof (SEntry));
#endif
}

/* --------------------------- the real code  ---------------------- */

/* ----------- ARPA LM handling ---------- */

#if 0
   float NGLM_global_min = 0.0;
#endif
/* GetProb

     Read one Float from src and convert from log_10 to log_e
*/
static LogFloat GetProb(Source *src, Boolean bin)
{
   float prob;

   if (!ReadFloat (src, &prob, 1, bin)) {
      char buf[MAXSTRLEN];
      HError (7613, "ReadARPAngram: failed reading lm prob at %s", 
              SrcPosition (*src, buf));
   }
#if 0   /* find global minimum prob/bowt for quantization */
   if (LN10*prob < -90.0)
      printf ("REALLY small: %f\n", LN10*prob);
   else {
      if (LN10*prob < NGLM_global_min)
         NGLM_global_min = LN10*prob;
   }
#endif

   return (LN10 * prob);
}

/* GetLMWord

     Read Word string from src, possibly in raw format

*/
static void GetLMWord (Source *src, char *buf, Boolean raw)
{
   if (raw) {
      if (!ReadRawString (src, buf))
         HError (7613, "ReadARPAngram: failed reading lm word at %s",
                 SrcPosition (*src, buf));
   }
   else
      if (!ReadString (src, buf))
         HError (7613, "ReadARPAngram: failed reading lm word at %s",
                 SrcPosition (*src, buf));
}


/* UnigramLMIdMapper

     convert string to LMId
     return 0 for unknown labels
*/
static LMId UnigramLMIdMapper(FSLM_ngram *nglm, char *w)
{
   LMId id;
   static unsigned long int idCounter = 0;
   LabId labId;
   Word word;

   /* map string to id */
   labId = GetLabId (w, FALSE);
   ++idCounter;
   if (labId) {
      if (labId->aux)
         HError ( 7620, "ReadARPAunigram: Duplicate word (%s) in 1-gram list",
                  labId->name);

      word = GetWord (nglm->vocab, labId, FALSE);
      if (!word) {
         HError (-7621, "ReadARPAunigram: unknown Word '%s' found in LM -- ignored\n", w);
         id = 0;
         labId = NULL;
      }
      else {
         labId->aux = (Ptr)idCounter;
         id = idCounter;
      }
   }
   else {       /* label unknown */
      /*      HError (-9999, "ReadARPAngram: unknown label '%s' found in LM -- ignored\n", w); */
      id = 0;
      word = NULL;
   }

   nglm->lablist[idCounter] = labId;      /* NULL for ignored words */
   nglm->wordlist[idCounter] = word;      /* NULL for ignored words */

   return id;
}

/* LMIdMapper

     look-up LMId of given string
     return 0 for unknown labels
*/
static LMId LMIdMapper(FSLM_ngram *nglm, char *w)
{
   LabId wdId;

   wdId = GetLabId (w, FALSE);
   if (!wdId) {
      /* warning only for non OOV symbols */
      if (strcmp(w, "!!UNK")!=0 && strcmp(w,"<unk>")!=0) {
      HError (-7621, "ReadARPAngram: unseen word '%s' in ngram", w);
      }
      return 0;
   }
   
   return ((int)(unsigned long int)wdId->aux);
}

/* se_cmp

     compare two SEntries
*/
static int se_cmp(const void *v1,const void *v2)
{
   SEntry *s1,*s2;

   s1 = (SEntry*) v1;
   s2 = (SEntry*) v2;
   return ((int) (s1->word - s2->word));
}

/* GetLMEntry

     read one LM entry (one line in ARPA LM file)
*/
static void GetLMEntry (FSLM_ngram *nglm, Source *src, Boolean bin, int n, LMId *ndx, 
                        LMId word2lmid(FSLM_ngram *, char*), 
                        LogFloat *prob, Boolean *hasBO, LogFloat *bo, Boolean *hasUNK)
{
   unsigned char flags;
   char buf[MAXSTRLEN];
   int i;
   unsigned short us;
   unsigned int ui;
   
   if (bin) {
      GetCh (src);
      flags = GetCh (src);
   }
   
   /* read probability */
   *prob = GetProb (src, bin);
   
   *hasUNK = FALSE;
   for (i = 0; i < n; ++i) {
      if (bin) {
         if (flags & BIN_ARPA_INT_LMID) {
            if (!ReadInt (src, (int *) &ui, 1, bin))
               HError (7613, "ReadARPAngram: failed reading int lm word id at %s",
                       SrcPosition (*src, buf));
            ndx[n-i-1] = (LMId) ui;
         }
         else {
            if (!ReadShort (src, (short *) &us, 1, bin))
               HError (7613, "ReadARPAngram: failed reading short lm word id at %s",
                       SrcPosition (*src, buf));
            ndx[n-i-1] = (LMId) us;
         }
      }
      else {
         GetLMWord (src, buf, rawMITFormat);
         ndx[n-i-1] = word2lmid (nglm, buf);
      }

      /* check whether word in vocab */
      if ((ndx[n-i-1] == 0) || !nglm->lablist[ndx[n-i-1]]) {
         *hasUNK = TRUE;
      }
   }

   /* maybe read back-off weight */
   *hasBO = FALSE;
   if (bin) {
      if (flags & BIN_ARPA_HAS_BOWT) {
         *hasBO = TRUE;
         *bo = GetProb (src, TRUE);
      }
   }
   else {
      SkipWhiteSpace (src);
      if (!src->wasNewline) {
         *hasBO = TRUE;
         *bo = GetProb (src, FALSE);
      }
   }
}

#define PROGRESS(g) \
   if (trace&T_TOP) { \
      if ((g%25000)==0) \
         printf(". "),fflush(stdout); \
      if ((g%800000)==0) \
         printf("\n   "),fflush(stdout); \
   }
                        

/* ReadARPAngram

     read one n-gram section from ARPA LM file
*/
static void ReadARPAngram (FSLM_ngram *nglm, Source *lmSrc, int n, int count, Boolean bin,
                           Vocab *vocab)
{
   LogFloat prob, bo;
   Boolean hasBO, hasUNK;
   int i;
   LMId ndx[NSIZE+1];
   NEntry *ne, *le = NULL;
   SEntry *tmpSE, *curtmpSE=NULL;
   int ntmpSE = 0;
   LMId (*word2lmid)(FSLM_ngram *, char *);
   Word word;
   Pron pron;
   PronId pronid;

   if (trace & T_TOP)
      printf("n%d ", n);
   
   for (i = 0; i <= NSIZE; ++i)
      ndx[i] = 0;

   /* for PronIds the # of SEntries is unkknown a-priori 
      alloc setnry arrays independently */
   tmpSE = (SEntry *) New (&gcheap, vocab->nprons * sizeof (SEntry));

   if (bin)
      word2lmid = NULL;
   else
      word2lmid = (n == 1) ?  UnigramLMIdMapper : LMIdMapper;

   for (i = 1; i <= count; ++i) {
      PROGRESS(i);
      GetLMEntry (nglm, lmSrc, bin, n, ndx, word2lmid, &prob, &hasBO, &bo, &hasUNK);

      if (hasUNK)       /* skip ngram if any of the words Labs is unknown */
         continue;

      /* store unigrams in array as well */
      if (n == 1) {
         /* prLMLA:         nglm->unigrams[ndx[0]] = prob; */
         /* store prob for each pron of word */
         word = nglm->wordlist[ndx[0]];
         if (word) {
            for (pron = word->pron; pron; pron = pron->next) {
	      pronid = (PronId) (unsigned long int)pron->aux;
               nglm->unigrams[pronid] = FLOAT_TO_NGLM_PROB(prob);
               /* #### add pron prob here */
               
               assert (pronid <= nglm->vocSize);
               nglm->pronId2LMId[pronid] = ndx[0];
            }
         }
         else {         /* skip unigram if word not in vocab */
            HError (7621, "ReadARPAngram: unigram: unknown word '%s' found in LM\n", 
                    nglm->lablist[ndx[0]]->name);
            continue;
         }
      }

      ne = GetNEntry (nglm,  ndx+1, FALSE);
      if (ne == NULL)
         HError(7622,"ReadNGrams: Backoff weight not seen for %dth %dGram", i, n);
      if (ne!=le) {
         if (le != NULL && ne->se != NULL)
            HError(7623,"ReadNGrams: %dth %dGrams out of order", i, n);
         if (le) {
            if (ntmpSE == 0) {
               abort ();
               le->se = NULL;
            }
            else {
               assert (ntmpSE <= vocab->nprons);
               qsort (tmpSE, ntmpSE, sizeof (SEntry), se_cmp);
               le->se = (SEntry *) New (nglm->heap, ntmpSE * sizeof (SEntry));
               memmove (le->se, tmpSE, ntmpSE * sizeof (SEntry));
               le->nse = ntmpSE;
               /* #### copy to two separate arrrays instead */
            }
         }
         curtmpSE = tmpSE;
         ntmpSE = 0;
         le = ne;
#if 0 /* pre LMLA */
         ne->se = se;
         ne->nse = 0;
         le = ne;
#endif
      }
      /* #### map to PronIds */
      word = nglm->wordlist[ndx[0]];
      if (!word)
         HError (7622, "ReadARPAngram: unknown word LMid %d\n", ndx[0]);
            
      for (pron = word->pron; pron; pron = pron->next) {
         pronid = (PronId) (unsigned long int) pron->aux;

         if (pronid > 0) {
            assert (ntmpSE <= vocab->nprons);
            curtmpSE->prob = FLOAT_TO_NGLM_PROB(prob);
            curtmpSE->word = pronid;
            ++ntmpSE;
            ++curtmpSE;
         }
      }
#if 0 /* pre LMLA */
      se->prob = prob;
      se->word = ndx[0];
      ne->nse++; 
      se++;
#endif 
      
      
      if (hasBO) {
         ne = GetNEntry (nglm, ndx, TRUE);
         ne->bowt = FLOAT_TO_NGLM_PROB(bo);
      }
   }    /* for i LM entries */


   /* fixup finale NEntry */
   /* #### duplicated from above */
   if (le) {
      if (ntmpSE == 0) {
         abort ();
         le->se = NULL;
      }
      else {
         assert (ntmpSE <= vocab->nprons);
         qsort (tmpSE, ntmpSE, sizeof (SEntry), se_cmp);
         le->se = (SEntry *) New (nglm->heap, ntmpSE * sizeof (SEntry));
         memmove (le->se, tmpSE, ntmpSE * sizeof (SEntry));
         le->nse = ntmpSE;
         /* #### copy to two separate arrrays instead */
      }
   }


   if (trace & T_TOP) {
      printf("\n");
      fflush (stdout);
   }

   Dispose (&gcheap, tmpSE);
}


/* ReadARPALM

     read ARPA LM from file
*/
FSLM *ReadARPALM(MemHeap *heap, char *lmfn, Vocab *vocab)
{
   FSLM *lm;
   Source lmSrc;
   char buf[MAXSTRLEN], fmt[MAXSTRLEN], ngFmtCh;
   int i, n;
   Boolean ngBin[NSIZE+1];
   int nng[NSIZE+1];  /* number of ngrams */

   if (InitSource (lmfn, &lmSrc, LangModFilter) < SUCCESS)
      HError (7613, "ReadARPALM: Cannot open lm file '%s'", lmfn);

   ReadUntilLine (&lmSrc, "\\data\\");
   
   n=0;
   for (i = 0; i <= NSIZE; ++i)
      nng[i] = 0;

   while (ReadLine (&lmSrc, buf) && strncmp (buf, "ngram", 5) == 0) {
      ++n;
      sprintf (fmt, "ngram %d%%c%%d", n);
      if (sscanf (buf, fmt, &ngFmtCh, &nng[n]) != 2)
         HError (7613, "ReadARPALM: error parsing LM");

      switch (ngFmtCh) {
      case '=':
         ngBin[n] = FALSE;
         break;
      case '~':
         ngBin[n] = TRUE;
         break;
      default:
         HError (7613, "ReadARPALM: unknown ngram format type '%c'", ngFmtCh);
      }
   }

   if (ngBin[1])
      HError (7613, "ReadARPALM: unigram must be stored as text");

   /* alloc LM structure 
      # stolen from HLM -- fix! */
   lm = (FSLM *) New (heap, sizeof(FSLM));
   lm->heap = heap;
   lm->name = CopyString (heap, lmfn);
   lm->type = fslm_ngram;

   /* PreLMla  lm->data.nglm = CreateBoNGram (heap, nng[1], nng); */
   lm->data.nglm = CreateBoNGram (heap, vocab->nprons, nng);
   lm->data.nglm->vocab = vocab;

   /* #### fixup nglm->lablist, so that ngrams with </s> in hist  get skipped? */

   /* read 1..n gram probs */
   for (i = 1; i <= n; ++i) {
      /* find beginning of i-gram */
      sprintf (buf, "\\%d-grams:", i);
      ReadUntilLine (&lmSrc, buf);

      ReadARPAngram (lm->data.nglm, &lmSrc, i, nng[i], ngBin[i], vocab);
   }

   CloseSource (&lmSrc);

   SetNEntryBO (lm);    /* init back-off pointers for all NEntries */

   return lm;
}


void SetStartEnd (FSLM *lm, char *startWord, char *endWord, Vocab *vocab)
{
   LabId startLabId, endLabId;
   Word word;

   /* find start and end Ids in LM */
   startLabId = GetLabId (startWord, FALSE);
   if (!startLabId)
      HError (7624, "HLVLM: cannot find STARTWORD '%s'\n", startWord);
   word = GetWord (vocab, startLabId, FALSE);
   if (!word)
      HError (7624, "HLVLM: cannot find STARTWORD '%s' in dict\n", startWord);

   lm->startPronId = (PronId) (unsigned long int) word->pron->aux;

   endLabId = GetLabId (endWord, FALSE);
   if (!endLabId)
      HError (7624, "HLVLM: cannot find ENDWORD '%s'\n", endWord);
   word = GetWord (vocab, endLabId, FALSE);
   if (!word)
      HError (7624, "HLVLM: cannot find ENDWORD '%s' in dict\n", endWord);

   lm->endPronId = (LMId) (unsigned long int) word->pron->aux;
}

/* CreateLM

     Read ARPA-style language model from File and return LM structure
*/
FSLM *CreateLM (MemHeap *heap, char *fn, char *startWord, char *endWord, Vocab *vocab)
{
   FSLM *lm;

   /*#### fix-up p(<s>) ?  it seems rather small... */

   lm = ReadARPALM (heap, fn, vocab);

   SetStartEnd (lm, startWord, endWord, vocab);

   lm->initial = (LMState) 0xffffffff;

   lm->lookahead = LMLookAhead_ngram;
   lm->transProb = LMTransProb_ngram;
   switch (lm->data.nglm->nsize) {
   case 2:
      lm->lookahead = LMLookAhead_2gram;
      lm->transProb = LMTransProb_ngram;
      break;
   case 3:
      lm->lookahead = LMLookAhead_3gram;
      lm->transProb = LMTransProb_ngram;
      break;
   }

   return (lm);
}



/* la_cmp

     compare two FSLM_LatArc entries
*/
static int la_cmp(const void *v1,const void *v2)
{
   FSLM_LatArc *l1,*l2;

   l1 = (FSLM_LatArc *) v1;
   l2 = (FSLM_LatArc *) v2;
   return ((int) (l1->word - l2->word));
}


FSLM *CreateLMfromLat (MemHeap *heap, char *latfn, Lattice *lat, Vocab *vocab)
{
   FSLM *lm;

   /* heap must be mstack, so that we can easily dispose of lattice again. */
   lm = (FSLM *) New (heap, sizeof(FSLM));
   lm->heap = heap;
   lm->name = CopyString (heap, latfn);
   lm->type = fslm_latlm;

   lm->data.latlm = (FSLM_latlm *) New (heap, sizeof (FSLM_latlm));
   
   {
      LNode *ln, *lnend = NULL;
      LArc *la;
      int i, p, npron;
      FSLM_latlm *latlm;
      FSLM_LatNode *fslmln, *lmstart = NULL;
      FSLM_LatArc *fslmla;
      Pron pron;

      latlm = lm->data.latlm;

      latlm->fslmln = (FSLM_LatNode *) New (heap, lat->nn * sizeof (FSLM_LatNode));
      latlm->nnodes = lat->nn;

      /* initialise LMState array for each latnode */
      for (i = 0; i < lat->nn; ++i) {
         ln = &lat->lnodes[i];
         fslmln = &latlm->fslmln[i];
         fslmln->word = ln->word;
      }

      /* convert arcs */
      for (i = 0; i < lat->nn; ++i) {
         ln = &lat->lnodes[i];
         fslmln = &latlm->fslmln[i];
         /* count number of prons following node */
         npron = 0;
         for (la = ln->foll; la; la = la->farc) {
            /*             npron += la->end->word->nprons; */
            for (pron = la->end->word->pron; pron; pron = pron->next)
               if (pron->aux > 0)
                  ++npron;
         }

         fslmla = (FSLM_LatArc *) New (heap, npron * sizeof (FSLM_LatArc));
         fslmln->nfoll = npron;
         fslmln->foll = fslmla;

         p = 0;
         for (la = ln->foll; la; la = la->farc) {
            for (pron = la->end->word->pron; pron; pron = pron->next) {
               if (pron->aux > 0) {
                  fslmla[p].word = (PronId) (unsigned long int) pron->aux;
                  fslmla[p].prob = la->lmlike;  /* #### add pron prob here? */
                  /*      fslmla[p].dest = &latlm->fslmln[(la->end - lat->lnodes)]; */
                  if (!la->end->foll ||  (la->end->foll->end->word == vocab->nullWord &&
                                          !la->end->foll->farc)) {
                     /* if arc leads to end node or only to !NULL node 
                        then label dest as leading to !SENT_END */
                     fslmla[p].dest = (Ptr) 0xfffffffe;
                  }
                  else {
                     fslmla[p].dest = &latlm->fslmln[(la->end - lat->lnodes)];
                     
                     assert (fslmla[p].dest >= &latlm->fslmln[0]);
                     assert (fslmla[p].dest <= &latlm->fslmln[lat->nn]);
                  }
                  ++p;
               }
            }
         }
         assert (p == npron);

         /* sort LatArc array */
         qsort (fslmla, npron, sizeof (FSLM_LatArc), la_cmp);
#if 1   /* sanity check: is latlm deterministic? */
         {
            int j;
            for (j = 1; j < fslmln->nfoll; ++j) {
               if (fslmln->foll[j-1].word == fslmln->foll[j].word
                   && fslmln->foll[j-1].prob != fslmln->foll[j].prob )
                  HError (7625, "CreateLMfromLat: lattice is not deterministic: [%s -> %s : %f] and [%s -> %s : %f]", 
                          fslmln->word->wordName->name, fslmln->foll[j-1].dest->word->wordName->name, 
                          fslmln->foll[j-1].prob,
                          fslmln->word->wordName->name, fslmln->foll[j].dest->word->wordName->name, 
                          fslmln->foll[j].prob);
            }
         }
#endif
         if (!ln->pred) {
            if (lmstart)
               HError (7625, "CreateLMfromLat: lattice has multiple start nodes");
            lmstart = fslmln;
         }
         if (!ln->foll) {
            if (lnend)
               HError (7625, "CreateLMfromLat: lattice has multiple end nodes");
            lnend = ln;
         }
      }

      if (!lmstart)
         HError (7625, "CreateLMfromLat: lattice has no start node");

      lm->initial = (LMState) lmstart;

      /*# check that there is ony one STARTWORD transition from lmstart */

      /*# check that there is a ENDWORD transition into lnend */
   }

   /*    SetStartEnd (lm, startWord, endWord, vocab); */

   lm->lookahead = LMLookAhead_latlm;
   lm->transProb = LMTransProb_latlm;


   return lm;
}


/* FindSEntry

     find SEntry for wordId in array using binary search
*/
static SEntry *FindSEntry (SEntry *se, PronId pronId, int l, int h)
{
   /*#### here l,h,c must be signed */
   int c;

   while (l <= h) {
      c = (l + h) / 2;
      if (se[c].word == pronId) 
         return &se[c];
      else if (se[c].word < pronId)
         l = c + 1;
      else
         h = c - 1;
   }

   return NULL;
}


/* LMTransProb

     return logprob of transition from src labelled word. Also return dest state.
*/
LogFloat LMTransProb (FSLM *lm, LMState src, PronId pronid, LMState *dest)
{
   switch (lm->type) {
   case fslm_ngram:
      return LMTransProb_ngram (lm, src, pronid, dest);
      break;
   case fslm_latlm:
      return LMTransProb_latlm (lm, src, pronid, dest);
      break;
   default:
      abort();
   }
   return LZERO;        /* make compiler happy */
}


/* LMTransProb_ngram

     return logprob of transition from src labelled word. Also return dest state.
     ngram case
*/
LogFloat LMTransProb_ngram (FSLM *lm, LMState src, PronId pronid, LMState *dest)
{
   FSLM_ngram *nglm;
   NGLM_Prob lmprob;
   LMId hist[NSIZE] = {0};      /* initialise whole array to zero! */
   int i, l;
   NEntry *ne;
   SEntry *se;

   assert (lm->type == fslm_ngram);

   assert (src != (Ptr) 0xfffffffe);

   if (trace & T_ACCESS)
      printf ("src %p PronId %u\n", src, (unsigned int) pronid);

   nglm = lm->data.nglm;

   if (pronid == 0 || pronid > nglm->vocSize) {
      HError (7626, "pron %d not in LM wordlist", pronid);
      *dest = NULL;
      return (LZERO);
   }

   /* from initial state only allow startword transition */
   if (src == (Ptr) 0xffffffff) {
      assert (pronid == lm->startPronId);

      hist[0] = nglm->pronId2LMId[pronid];
      *dest = (LMState) GetNEntry (nglm, hist, FALSE);
      return 0.0;
   }

   ne = src;
   
   if (!src) {          /* unigram case */
      lmprob = nglm->unigrams[pronid];
   }
   else {
      /* lookup prob p(word | src) */
      /* try to find pronid in SEntry array */
      se = FindSEntry (ne->se, pronid, 0, ne->nse - 1);

      assert (!se || (se->word == pronid));

      if (se)        /* found */
         lmprob = se->prob;
      else {             /* not found */
         lmprob = 0.0;
         l = 0;
         hist[NSIZE-1] = 0;
         for (i = 0; i < NSIZE-1; ++i) {
            hist[i] = ne->word[i];
            if (hist[i] != 0)
               l = i;
         } /* l is now the index of the last (oldest) non zero element */
         
         for ( ; l > 0; --l) {
            lmprob += ne->bowt;
            hist[l] = 0;   /* back-off: discard oldest word */
            ne = GetNEntry (nglm, hist, FALSE);
            /* try to find pronid in SEntry array */
            se = FindSEntry (ne->se, pronid, 0, ne->nse - 1);
            assert (!se || (se->word == pronid));
            if (se) { /* found it */
#ifdef LM_NGRAM_INT
	       lmprob = NGLM_PROB_ADD(lmprob,se->prob);
#else
               lmprob += se->prob;
#endif
               l = -1;
               break;
            }
         }
         if (l == 0) {          /* backed-off all the way to unigram */
            assert (!se);
#ifdef LM_NGRAM_INT
	    lmprob = NGLM_PROB_ADD(lmprob, ne->bowt);
	    lmprob = NGLM_PROB_ADD(lmprob, nglm->unigrams[pronid]);
#else
            lmprob += ne->bowt;
            lmprob += nglm->unigrams[pronid];
#endif
         }
      }
   }


   /* now determine dest state */
   if (pronid != lm->endPronId) {
      if (src) {
         ne = (NEntry *) src;
      
         l = 0;
         hist[NSIZE-1] = 0;
         for (i = 1; i < NSIZE-1; ++i) {
            hist[i] = ne->word[i-1];
            if (hist[i] != 0)
               l = i;
         } /* l is now the index of the last (oldest) non zero element */
      }
      else {
         for (i = 1; i < NSIZE-1; ++i)
            hist[i] = 0;
         l = 1;
      }
      
      hist[0] = nglm->pronId2LMId[pronid];
      
      ne = (LMState) GetNEntry (nglm, hist, FALSE);
      for ( ; !ne && (l > 0); --l) {
         hist[l] = 0;              /* back off */
         ne = (LMState) GetNEntry (nglm, hist, FALSE);
      }
      /* if we left the loop because l=0, then ne is still NULL, which is what we want */
      *dest = ne;
   }
   else {       /* SENT_END case */
      *dest = (Ptr) 0xfffffffe;
   }

   if (trace & T_ACCESS)
      printf ("lmprob = %f  dest %p\n", NGLM_PROB_TO_FLOAT(lmprob), *dest);

   return (NGLM_PROB_TO_FLOAT(lmprob));
}

/* LMInitial

     return initial state of FSM LM.
     for n-gram just make sure we back-off to unigrams
*/
LMState LMInitial (FSLM *lm)
{
   return lm->initial;         /* signals start of utterance */
}


/* FindMinSEntry

     find first SEntry >= minPron
     return NULL if there isn't one.
*/
SEntry *FindMinSEntry (SEntry *se, int nse, PronId minPron)
{
   int i;

   /* #### convert to binary search! */
   for (i = 0 ; i < nse; ++i, ++se)
      if (se->word >= minPron)
         return se;
   
   return NULL;
}

/* FindMinSEntryP

     find first SEntry >= minPron
     return NULL if there isn't one.
     uses binsearch
*/
static SEntry *FindMinSEntryP (SEntry *low, SEntry *hi, PronId minPron)
{
  SEntry *mid=NULL;

  if (minPron > hi->word)
    return NULL;

  while (low <= hi) {
    mid = low + (hi - low) / 2;   /*  (l + h) / 2; */
    if (mid->word == minPron) 
      return mid;
    else if (mid->word < minPron)
      low = mid + 1;
    else
      hi = mid - 1;
  }

  if(mid==NULL) HError(7626,"failed to find entry");
  if (mid->word >= minPron)
    return mid;
  else 
    return mid+1;
}

/* LMLookAhead

     return \max_{i=minWord}^{maxWord} p(w_i | src)
*/
LogFloat LMLookAhead (FSLM *lm, LMState src, PronId minPron, PronId maxPron)
{
   return (lm->lookahead) (lm, src, minPron, maxPron);
}

/* LMLookAhead_2gram

     optimised version for 2gram lookahead

     return \max_{i=minWord}^{maxWord} p(w_i | src)
     ngram case
*/
LogFloat LMLookAhead_2gram (FSLM *lm, LMState src, PronId minPron, PronId maxPron)
{
   NEntry *neSrc;
   SEntry *se, *seLast;
   PronId p, pend;
   NGLM_Prob *unigrams;
   NGLM_Prob maxScore = NGLM_PROB_LZERO;
   NGLM_Prob ug_maxScore = NGLM_PROB_LZERO;     /* maximum of the unigrams[p], i.e. missing bowt! */
   NGLM_Prob prob, bowt = 0;

   p = minPron;
   unigrams = lm->data.nglm->unigrams;

   if (src) {
      neSrc = (NEntry *) src;
      bowt = neSrc->bowt;
      
      if (neSrc->nse > 0) {     /* see comment in Debug_Check_LMhashtab */
         se = neSrc->se;
         seLast = &se[neSrc->nse - 1];
         se = FindMinSEntryP (se, seLast, minPron);
      
         if (se) {
            pend = maxPron;
            if (maxPron > seLast->word)
               pend = seLast->word;
            
            for ( ; p <= pend; ++p) {
               if (se->word != p) {
                  prob = unigrams[p];
                  if (NGLM_PROB_GREATER(prob,ug_maxScore))
                     ug_maxScore = prob;
               } else {
                  prob = se->prob;
                  ++se;
                  if (NGLM_PROB_GREATER(prob,maxScore))
                     maxScore = prob;
               }
            }
         }
      }
   }
   /* always backoff to unigrams for the rest */
   for ( ; p <= maxPron; ++p) {
      prob = unigrams[p];
      if (NGLM_PROB_GREATER(prob,ug_maxScore))
         ug_maxScore = prob;
   }

   /* add the back-off weight to ug_maxscore and combine with maxscore */
   if (NGLM_PROB_GREATER(ug_maxScore,NGLM_PROB_LZERO)) {
#ifdef LM_NGRAM_INT
      ug_maxScore = NGLM_PROB_ADD(ug_maxScore, bowt);
#else
      ug_maxScore += bowt;
#endif
      if (NGLM_PROB_GREATER(ug_maxScore, maxScore))
         maxScore = ug_maxScore;
   }
   
   return NGLM_PROB_TO_FLOAT(maxScore);
}

/* LMLookAhead_3gram

     optimised version for 3gram lookahead

     return \max_{i=minWord}^{maxWord} p(w_i | src)
     ngram case
*/
LogFloat LMLookAhead_3gram (FSLM *lm, LMState src, PronId minPron, PronId maxPron)
{
   NEntry *ne_tg, *ne_bg;
   SEntry *se_tg, *seLast_tg, *se_bg, *seLast_bg=NULL;
   PronId p, pend;
   NGLM_Prob *unigrams;
   NGLM_Prob maxScore = NGLM_PROB_LZERO;
   NGLM_Prob bg_maxScore = NGLM_PROB_LZERO;     /* maximum of the bigrams, missing bowt! */
   NGLM_Prob ug_maxScore = NGLM_PROB_LZERO;     /* maximum of the unigrams[p], i.e. missing bowt! */
   NGLM_Prob prob;
   NGLM_Prob bowt_ug = 0, bowt_bg = 0;          /* backoff-weight _to_ ug and bg resp. */

   p = minPron;
   unigrams = lm->data.nglm->unigrams;

   if (src) {
      ne_tg = (NEntry *) src;

      if (ne_tg->word[1] == 0)       /* this is a bigram NEntry */
         return LMLookAhead_2gram (lm, src, minPron, maxPron);
      
      bowt_bg = ne_tg->bowt;
      if (ne_tg->nse > 0) {       /* there are potential trigram entries */
         se_tg = ne_tg->se;
         seLast_tg = &se_tg[ne_tg->nse - 1];
         se_tg = FindMinSEntryP (se_tg, seLast_tg, minPron);
      }
      else
         se_tg = NULL;          /* force back-off to bigram */

      ne_bg = ne_tg->nebo;
#ifdef LM_NGRAM_INT
      bowt_ug = NGLM_PROB_ADD(bowt_bg, ne_bg->bowt);
#else
      bowt_ug = bowt_bg + ne_bg->bowt;
#endif
      if (ne_bg->nse > 0) {
         se_bg = ne_bg->se;
         seLast_bg = &se_bg[ne_bg->nse - 1];
         se_bg = FindMinSEntryP (se_bg, seLast_bg, minPron);
      }
      else
         se_bg = NULL;

#if 0   /*sanity check: trigram range is subset of bigram range */
      if (se_tg) {
         assert (se_tg->word >= se_bg->word);
         assert (seLast_tg->word <= seLast_bg->word);
      }
#endif

      if (se_tg) {              /* there are trigram entries to handle */
         pend = maxPron;
         if (maxPron > seLast_tg->word)
            pend = seLast_tg->word;
         
         for ( ; p <= pend; ++p) {
            if (se_tg->word != p) {     /* back-off to bigram */
               if (!se_bg) { /* back-off to unigram */
                  prob = unigrams[p];
                  if (NGLM_PROB_GREATER(prob,ug_maxScore))
                     ug_maxScore = prob; 
               } else {
                  if (se_bg->word != p) {  /* back-off to unigram */
                     prob = unigrams[p];
                     if (NGLM_PROB_GREATER(prob,ug_maxScore))
                        ug_maxScore = prob; 
                  }
                  else {                   /* bigram */
                  prob = se_bg->prob;
                  ++se_bg;
                  if (NGLM_PROB_GREATER(prob,bg_maxScore))
                     bg_maxScore = prob;
                  }
               }
            } else {                    /* tigram */
               prob = se_tg->prob;
#if 0           /* sanity check */
               assert (se_bg->word == se_tg->word);
#endif               
               ++se_tg;
	       if (se_bg) ++se_bg;
               if (NGLM_PROB_GREATER(prob,maxScore))
                  maxScore = prob;
            }
         }
      }
      /* always back-off to (at least) bigram for the rest */
      /* this is the core loop of LMLookAhead_2gram  */
      if (se_bg) {
         pend = maxPron;
         if (maxPron > seLast_bg->word)
            pend = seLast_bg->word;
         
         for ( ; p <= pend; ++p) {
            if (se_bg->word != p) {
               prob = unigrams[p];
               if (NGLM_PROB_GREATER(prob,ug_maxScore))
                  ug_maxScore = prob;
            } else {
               prob = se_bg->prob;
               ++se_bg;
               if (NGLM_PROB_GREATER(prob,bg_maxScore))
                  bg_maxScore = prob;
            }
         }
      }
   }
   /* always backoff to unigrams for the rest */
   for ( ; p <= maxPron; ++p) {
      prob = unigrams[p];
      if (NGLM_PROB_GREATER(prob,ug_maxScore))
         ug_maxScore = prob;
   }

   /* add the back-off weight to bg_maxscore and combine with maxscore */
   if (NGLM_PROB_GREATER(bg_maxScore,NGLM_PROB_LZERO)) {
#ifdef LM_NGRAM_INT
      bg_maxScore = NGLM_PROB_ADD(bg_maxScore, bowt_bg);
#else
      bg_maxScore += bowt_bg;
#endif
      if (NGLM_PROB_GREATER(bg_maxScore, maxScore))
         maxScore = bg_maxScore;
   }
   /* add the back-off weight to ug_maxscore and combine with maxscore */
   if (NGLM_PROB_GREATER(ug_maxScore,NGLM_PROB_LZERO)) {
#ifdef LM_NGRAM_INT
      ug_maxScore = NGLM_PROB_ADD(ug_maxScore, bowt_ug);
#else
      ug_maxScore += bowt_ug;
#endif
      if (NGLM_PROB_GREATER(ug_maxScore, maxScore))
         maxScore = ug_maxScore;
   }

#if 0   /* very expensive check */
   {
      if (fabs (NGLM_PROB_TO_FLOAT(maxScore) -
                LMLookAhead_ngram (lm, src, minPron, maxPron)) > 0.01)
         abort();
   }
#endif
   return NGLM_PROB_TO_FLOAT(maxScore);
}

LogFloat LMLookAhead_ngram (FSLM *lm, LMState src, PronId minPron, PronId maxPron)
{
   FSLM_ngram *nglm;
   NEntry *neSrc;
   int i;
   NGLM_Prob maxScore;

   nglm = lm->data.nglm;
   neSrc = (NEntry *) src;
   maxScore = NGLM_PROB_LZERO;

   if (neSrc) {
      /* # add special case minPron == maxPron */

      int hiIdx;             /* hist len of src - 1 */
      int l, ll, p;
      LMId hist[NSIZE-1];
      NEntry *ne[NSIZE-1];
      SEntry *se[NSIZE-1], *seEnd[NSIZE-1];
      NGLM_Prob bowt[NSIZE-1], prob;

#if 0           /* make debugging easier */
      for (l = 0; l < NSIZE-1; ++l) {
         ne[l] = NULL;
         se[l] = NULL;
         bowt[l] = NGLM_PROB_ZERO;
         hist[l] = 0;
      }
#endif
      /* # find histLen from neSrc */
      hiIdx = NSIZE;
      for (l = 0; l < NSIZE-1; ++l) {
         hist[l] = neSrc->word[l];
         if (hist[l] != 0)
            hiIdx = l;
      }

      assert (hiIdx < NSIZE);
      /* hiIdx points to highest index (oldest word) in hist array
         i.e. ne's history has hiIdx+1 words */
      
      ne[hiIdx] = neSrc;
      bowt[hiIdx] = NGLM_PROB_ZERO;
      /* initialise ne[] array */
      for (l = hiIdx - 1; l >= 0; --l) {
         /* #### optimise NEntry lookups by storing pointers
            in each NEntry to the next back-off entry */
         hist[l+1] = 0;
         ne[l] = GetNEntry (nglm, hist, FALSE);
         assert (ne[l]);
         /* set bowt[l] to sum of all required bowts if we use se[l] */
#ifdef LM_NGRAM_INT
	 bowt[l] = NGLM_PROB_ADD(bowt[l+1], ne[l+1]->bowt);
#else
         bowt[l] = bowt[l+1] + ne[l+1]->bowt;
#endif
      }
      
      for (l = hiIdx; l >= 0; --l) {
         /* find first SEntry >= minPron in ne[l] */
         if (ne[l]->nse > 0)         /* see comment for Debug_Check_LMhashtab */
            se[l] = FindMinSEntryP (ne[l]->se, 
                                    ne[l]->se + (ne[l]->nse - 1), minPron);
         else
            se[l] = NULL;
#if 0   /* sanity check of binary search implementation */
         assert (se[l] == FindMinSEntry (ne[l]->se, ne[l]->nse, minPron));
#endif             
         /* set seEnd[] sentinel */
         seEnd[l] = ne[l]->se + ne[l]->nse;
         
         assert (!se[l] || (ne[l]->se <= se[l] && se[l] < seEnd[l]));
      }
      
      maxScore = NGLM_PROB_LZERO;
      for (p = minPron; p <= maxPron; ++p) {
         for (l = hiIdx; l >= 0; --l) {
            if (se[l]) {     /* any entries left at this level? */
               if (se[l]->word == p) {
#ifdef LM_NGRAM_INT
		  prob = NGLM_PROB_ADD(se[l]->prob, bowt[l]);
#else
                  prob = se[l]->prob + bowt[l];
#endif
                  if (NGLM_PROB_GREATER(prob,maxScore))
                     maxScore = prob;
                  se[l]++;
                  if (se[l] >= seEnd[l])
                     se[l] = NULL;
                  /* advance se[] pointer for shorter hists */
                  for (ll = l-1; ll >= 0; --ll) {
                     while (se[ll]->word  <= p) {
                        if (se[ll] < seEnd[ll])
                           ++se[ll];
                        else {
                           se[ll] = NULL;
                           break;
                        }
                     }
                  }
                  break;
               }
            }
         }
         if (l < 0) {       /* not found => back-off to unigram */
#ifdef LM_NGRAM_INT
	    prob = NGLM_PROB_ADD(nglm->unigrams[p], bowt[0]);
	    prob = NGLM_PROB_ADD(prob, ne[0]->bowt);
#else
            prob = nglm->unigrams[p] + bowt[0] + ne[0]->bowt;
#endif
            if (NGLM_PROB_GREATER(prob,maxScore))
               maxScore = prob;
         }
#if 0           /* sanity check: compare with LMTransProb */
         {
            LMState dest;
            LogFloat prob_lmtrans;
            prob_lmtrans = LMTransProb (lm, src, p, &dest);
            assert (fabs (prob - prob_lmtrans) < 1.0e-4);
         }
#endif
      }
   } else {  /* loop over unigrams */
      for (i = minPron; i <= maxPron; ++i)
         if (NGLM_PROB_GREATER(nglm->unigrams[i],maxScore))
            maxScore = nglm->unigrams[i];
   }
   return NGLM_PROB_TO_FLOAT(maxScore);
}


/* Fast_LMLA_LMState

     back-off to a more "simple" state, i.e. for 3grams discard oldest word 
     and use 2gram state
*/
LMState Fast_LMLA_LMState (FSLM *lm, LMState src)
{
   /* #### currently only works for 3grams! FIX THIS! */
   
   NEntry *ne_tg;

   abort();
   if (src) {
      ne_tg = (NEntry *) src;
      
      if (ne_tg->word[1] == 0)       /* this is a bigram NEntry */
         return src;
      
      return (LMState) ne_tg->nebo;}

   return NULL;
}



void SetNEntryBO (FSLM *lm)
{
   FSLM_ngram *nglm;
   NEntry *ne;
   int hash, l, hiIdx;
   LMId hist[NSIZE-1];
   
   nglm = lm->data.nglm;
   
   for (hash = 0; hash < nglm->hashsize; hash++) {
      for (ne = nglm->hashtab[hash]; ne; ne=ne->link) {
         
         hiIdx = NSIZE;
         for (l = 0; l < NSIZE-1; ++l) {
            hist[l] = ne->word[l];
            if (hist[l] != 0)
               hiIdx = l;
         }
         /* hiIdx points to highest index (oldest word) in hist array
            i.e. ne's history has hiIdx+1 words */
         if (hiIdx == NSIZE) {  /* unigram */
            ne->nebo = NULL;
         }
         else if (hiIdx == 0) {      /* bigram */
            ne->nebo = NULL;
         }
         else {
            hist[hiIdx] = 0;    /* delete oldest word */
            ne->nebo = GetNEntry (nglm, hist, FALSE);
            assert (ne->nebo);
         }
      }
   }
}

/* PrintLMHashStats                             DEBUG

     print size of each slot in LM  NEntry hashtable
*/
void PrintLMHashStats(FSLM *lm)
{
   FSLM_ngram *nglm;
   NEntry *ne;
   int hash, n;
   
   nglm = lm->data.nglm;
   
   for (hash = 0; hash < nglm->hashsize; hash++) {
      n = 0;
      for (ne = nglm->hashtab[hash]; ne; ne=ne->link) {
         ++n;
      }
      printf ("bin %d has %d entries\n", hash, n);
   }
}

/* #### the following is adapted from HLM.c -- fix! */
/*------------------------- NEntry handling ---------------------------*/

static int hvs[]= { 165902236, 220889002, 32510287, 117809592,
                    165902236, 220889002, 32510287, 117809592 };

/* EXPORT->GetNEntry: Access specific NGram entry indexed by ndx */
NEntry *GetNEntry (FSLM_ngram *nglm, LMId ndx[NSIZE], Boolean create)
{
   NEntry *ne;
   unsigned int hash;
   int i;
   /* #define LM_HASH_CHECK */
  
   hash=0;
   for (i=0;i<NSIZE-1;i++)
      hash=hash+(ndx[i]*hvs[i]);
   hash=(hash>>7)&(nglm->hashsize-1);
  
   for (ne=nglm->hashtab[hash]; ne!=NULL; ne=ne->link) {
      if (ne->word[0]==ndx[0]
#if NSIZE > 2
          && ne->word[1]==ndx[1]
#endif
#if NSIZE > 3
          && ne->word[2]==ndx[2]
#endif
#if NSIZE > 4
          && ne->word[3]==ndx[3]
#endif
          )
         break;
   }

   if (ne==NULL && create) {
      ne=(NEntry *) New(nglm->heap,sizeof(NEntry));
      nglm->counts[0]++;
      
      for (i=0;i<NSIZE-1;i++)
         ne->word[i]=ndx[i];
      ne->nse=0;
      ne->se=NULL;;
      ne->bowt=0.0;
      ne->link=nglm->hashtab[hash];
      nglm->hashtab[hash]=ne;
   }

   return(ne);
}

#define NGHSIZE1 8192
#define NGHSIZE2 32768
#define NGHSIZE3 131072

/* EXPORT->CreateBoNGram: Allocate and create basic NGram structures */
FSLM_ngram *CreateBoNGram (MemHeap *heap, int vocSize, int counts[NSIZE])
{
   LMId ndx[NSIZE];
   int i,k;
   FSLM_ngram *nglm;

   nglm = (FSLM_ngram *) New (heap, sizeof(FSLM_ngram));

   nglm->heap = heap;

   for (i=0;i<=NSIZE;i++) nglm->counts[i]=0;
   for (i=1;i<=NSIZE;i++)
      if (counts[i]==0) break;
      else nglm->counts[i]=counts[i];
   nglm->nsize=i-1;

   /* Don't count final layer */
   for (k=0,i=1;i<nglm->nsize;i++) 
      k+=nglm->counts[i];
   /* Then use total to guess NEntry hash size */
   if (k<25000) 
      nglm->hashsize=NGHSIZE1;
   else if (k<250000) 
      nglm->hashsize=NGHSIZE2;
   else 
      nglm->hashsize=NGHSIZE3;

   nglm->hashtab=(NEntry **) New(nglm->heap,sizeof(NEntry*)*nglm->hashsize);
   for (i=0; i<nglm->hashsize; i++) 
      nglm->hashtab[i]=NULL;

   nglm->vocSize = vocSize;
   nglm->unigrams = (NGLM_Prob *) New (nglm->heap, (nglm->vocSize + 1) * sizeof (NGLM_Prob));
   for (i = 0; i < (nglm->vocSize + 1); i++){
     nglm->unigrams[i] = NGLM_PROB_LZERO;
   }
   nglm->pronId2LMId = (LMId *) New (nglm->heap, (nglm->vocSize + 1) * sizeof (LMId));

   nglm->lablist = (LabId *) New (nglm->heap, nglm->counts[1] * sizeof(LabId)); nglm->lablist--;
   nglm->wordlist = (Word *) New(nglm->heap,nglm->counts[1] * sizeof(Word)); nglm->wordlist--;
   for (i = 1; i <= nglm->counts[1]; i++) {
      nglm->lablist[i]=NULL;
      nglm->wordlist[i]=NULL;
   }

   for (i = 0; i < NSIZE; i++) 
      ndx[i]=0;
   GetNEntry(nglm,ndx,TRUE);

   return(nglm);
}   

/******* latlm code */

FSLM_LatArc *FindMinLatArc (FSLM_LatArc *low, FSLM_LatArc *hi, PronId minPron)
{
   FSLM_LatArc *mid=NULL;

   if (minPron > hi->word)
      return NULL;

   while (low <= hi) {
      mid = low + (hi - low) / 2;   /*  (l + h) / 2; */
      if (mid->word == minPron) 
         return mid;
      else if (mid->word < minPron)
         low = mid + 1;
      else
         hi = mid - 1;
   }

   if(mid==NULL) HError(7626,"failed to find entry");
   if (mid->word >= minPron)
      return mid;
   else 
      return mid+1;
   return NULL;
}

FSLM_LatArc *FindLatArc (FSLM_LatArc *low, FSLM_LatArc *hi, PronId pronId)
{
   FSLM_LatArc *mid;

   while (low <= hi) {
      mid = low + (hi - low) / 2;   /*  (l + h) / 2; */
      if (mid->word == pronId) 
         return mid;
      else if (mid->word < pronId)
         low = mid + 1;
      else
         hi = mid - 1;
   }
   return NULL;
}

/* LMTransProb_latlm

     return logprob of transition from src labelled word. Also return dest state.
     lattice case
*/
LogFloat LMTransProb_latlm (FSLM *lm, LMState src,
                            PronId pronid, LMState *dest)
{
   FSLM_LatNode *ln;
   FSLM_LatArc *la;

   ln = (FSLM_LatNode *) src;

   if (ln->nfoll > 0) {
      la = FindLatArc (ln->foll, ln->foll + (ln->nfoll - 1), pronid);
   
      assert (!la || (la->word == pronid));

      if (la) {
         *dest = la->dest;
         return la->prob;
      }
   }
   
   *dest = NULL;
   return LZERO;
}

/* LMLookAhead_ngram

     return \max_{i=minWord}^{maxWord} p(w_i | src)
     ngram case
*/
LogFloat LMLookAhead_latlm (FSLM *lm, LMState src, 
                            PronId minPron, PronId maxPron)
{
   FSLM_LatNode *ln;
   FSLM_LatArc *la, *laLast;
   LogFloat maxScore;

   ln = (FSLM_LatNode *) src;
   maxScore = LZERO;
   
   laLast = ln->foll + (ln->nfoll - 1);
   la = FindMinLatArc (ln->foll, laLast, minPron);

   if (la) {
      for ( ; la <= laLast; ++la) {
         if (la->word > maxPron)
            break;
         if (la->prob > maxScore)
            maxScore = la->prob;
      }
   }

   return maxScore;
}

/************* debug */

/* 

    check all NEntries for nse>0

    maybe this is too harsh: we might have tossed some n-grams, e.g. with
    word=UNK, but we still need the back-off weight.

    #### investigate this!

*/
void Debug_Check_LMhashtab(FSLM_ngram *nglm)
{
   int i;
   NEntry *ne;

   for (i = 0; i < nglm->hashsize; i++) {
      for (ne = nglm->hashtab[i]; ne; ne=ne->link) {
         assert (ne->nse > 0);
         assert (ne->se);
      }
   }
}

/* ------------------------ End of HLVLM.c ----------------------- */

