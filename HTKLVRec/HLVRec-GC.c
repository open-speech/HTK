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
/*   File: HLVRec-GC.c Garbage collection for HTK LV Decoder   */
/* ----------------------------------------------------------- */

char *hlvrec_gc_version = "!HVER!HLVRec-GC:   3.5.0 [CUED 12/10/15]";
char *hlvrec_gc_vc_id = "$Id: HLVRec-GC.c,v 1.2 2015/10/12 12:07:24 cz277 Exp $";

/* #### this should be generalised and moved into HMem as MHEAP-GC */

/* use highest bit in path->user for GC marking */
/* # alternative would be a bitmap for each block in HMem
   # advantage would be that the sweep phase would be trivial
   # but marking would be more expensive (finding the right block) 
*/
#define MARK_PATH_MASK  0x8000
#define MARK_PATH(p)            (p->user = (p)->user | MARK_PATH_MASK)
#define MARKED_PATH_P(p)        ((p)->user & MARK_PATH_MASK)
#define UNMARK_PATH(p)          (p->user = p->user & ~MARK_PATH_MASK)

/* mark AltWordendHyp in least significant bit of a->prev, which is normally 
   always 0, since pointers are aligned */
#define MARK_ALTPATH_MASK     0x0000000000000001UL
#define MARK_ALTPATH(a)         (a->prev = (WordendHyp *)((long)(a->prev) | MARK_ALTPATH_MASK))
#define MARKED_ALTPATH_P(a)     ((long)((a)->prev) & MARK_ALTPATH_MASK)
#define UNMARK_ALTPATH(a)       (a->prev = (WordendHyp *)((long)(a->prev) & ~MARK_ALTPATH_MASK))
#define GC_ALTPATH_PREV(a)      ((WordendHyp *) ((long)(a)->prev & ~MARK_ALTPATH_MASK))

#ifdef MODALIGN
/*#define MARK_MODPATH_MASK     0x00000001UL
#define MARK_MODPATH(m)         (m->ln = (int)((m)->ln) | MARK_MODPATH_MASK)
#define MARKED_MODPATH_P(m)     ((int)((m)->ln) & MARK_MODPATH_MASK)
#define UNMARK_MODPATH(m)       (m->ln = (int)((m)->ln) & ~MARK_MODPATH_MASK)
*/

/* cz277 - 64bit */
#define MARK_MODPATH_MASK       0x0000000000000001UL
#define MARK_MODPATH(m)         (m->ln = (LexNode *)((long)((m)->ln) | MARK_MODPATH_MASK))
#define MARKED_MODPATH_P(m)     ((long)((m)->ln) & MARK_MODPATH_MASK)
#define UNMARK_MODPATH(m)       (m->ln = (LexNode *)((long)((m)->ln) & ~MARK_MODPATH_MASK))

static void MarkModPath (ModendHyp *m)
{
   if (!m || MARKED_MODPATH_P(m))
      return;

   MARK_MODPATH(m);

   /*# mark prev (end-recursive -- use goto?) */
   MarkModPath (m->prev);
}
#endif

static void MarkPath (WordendHyp *path)
{
   AltWordendHyp *alt;

   assert (path);

   if (MARKED_PATH_P (path))
      return;

   /* mark this path */
   MARK_PATH (path);
#ifdef MODALIGN
   MarkModPath (path->modpath);
#endif

   /* mark paths leading to alternatives */
   for (alt = path->alt; alt; alt = alt->next) {
      MARK_ALTPATH(alt);
#ifdef MODALIGN
      MarkModPath (alt->modpath);
#endif
      if (alt->prev) {
         MarkPath (GC_ALTPATH_PREV(alt));
      }
   }

   /*# mark prev (end-recursive -- use goto?) */
   if (path->prev)
      MarkPath (path->prev);
}

static void MarkTokSet (TokenSet *ts)
{
   int i;
   RelToken *tok;

   for (i = 0; i < ts->n; ++i) {
      tok = &ts->relTok[i];
      if (tok->path)
         MarkPath (tok->path);
#ifdef MODALIGN
      if (tok->modpath)
         MarkModPath (tok->modpath);
#endif
   }
}

static void SweepPaths (MemHeap *heap)
{
   int i;
   BlockP b;
   WordendHyp *path;
   size_t elemSize;
   int iMapPos, iMapMask;
   int total, freed;

   assert (heap->type == MHEAP);

   total = freed = 0;
   elemSize = heap->elemSize;

   for (b = heap->heap; b; b = b->next) {
      for (i = 0, path = b->data; i < b->numElem; 
           ++i, path = (WordendHyp *) (((char *) path) + elemSize)) {
         iMapPos = i/8;
         iMapMask = 1 << (i&7);

         if (b->used[iMapPos] & iMapMask) {     /* currently assigned? */
            ++total;
            if (MARKED_PATH_P (path))
               UNMARK_PATH (path);
            else {         /* free it */
               ++freed;
#if 0
               path->prev = (WordendHyp *) 0xDEADBEEF;
               path->pron = -1;
               path->frame = -1;
               path->lm = 999.99;
               path->alt = (AltWordendHyp *) 0xDEADBEEF;
               path->user = 99;
#endif
               /* similar to Dispose (heap, path) */
               b->used[iMapPos] &= ~(iMapMask);
               if (i < b->firstFree) 
                  b->firstFree = i;
               b->numFree++; 
               heap->totUsed--;
            }
         }
      }

      /* end of block clean-up */
      if (b->numFree == b->numElem) { 
         if (trace&T_GC)
            printf ("should free whole block!\n");
      }
   }

   if (trace&T_GC)
      printf ("freed %d of %d Paths\n", freed, total);
}

static void SweepAltPaths (MemHeap *heap)
{
   int i;
   BlockP b;
   AltWordendHyp *path;
   size_t elemSize;
   int iMapPos, iMapMask;
   int total, freed;

   assert (heap->type == MHEAP);

   total = freed = 0;
   elemSize = heap->elemSize;

   for (b = heap->heap; b; b = b->next) {
      for (i = 0, path = b->data; i < b->numElem; 
           ++i, path = (AltWordendHyp *) (((char *) path) + elemSize)) {
         iMapPos = i/8;
         iMapMask = 1 << (i&7);

         if (b->used[iMapPos] & iMapMask) {     /* currently assigned? */
            ++total;
            if (MARKED_ALTPATH_P (path))
               UNMARK_ALTPATH (path);
            else {         /* free it */
               ++freed;
#if 0
               path->prev = (WordendHyp *) 0xDEADBEEF;
               path->lm = 999.99;
               path->score = 999.99;
               path->next = (AltWordendHyp *) 0xDEADBEEF;
#endif
               /* similar to Dispose (heap, path) */
               b->used[iMapPos] &= ~(iMapMask);
               if (i < b->firstFree) 
                  b->firstFree = i;
               b->numFree++; 
               heap->totUsed--;
            }
         }
      }

      /* end of block clean-up */
      if (b->numFree == b->numElem) { 
         if (trace&T_GC)
            printf ("should free whole block!\n");
      }
   }

   if (trace&T_GC)
      printf ("freed %d of %d AltPaths\n", freed, total);
}

#ifdef MODALIGN
static void SweepModPaths (MemHeap *heap)
{
   int i;
   BlockP b;
   ModendHyp *path;
   size_t elemSize;
   int iMapPos, iMapMask;
   int total, freed;

   assert (heap->type == MHEAP);

   total = freed = 0;
   elemSize = heap->elemSize;

   for (b = heap->heap; b; b = b->next) {
      for (i = 0, path = b->data; i < b->numElem; 
           ++i, path = (ModendHyp *) (((char *) path) + elemSize)) {
         iMapPos = i/8;
         iMapMask = 1 << (i&7);

         if (b->used[iMapPos] & iMapMask) {     /* currently assigned? */
            ++total;
            if (MARKED_MODPATH_P (path))
               UNMARK_MODPATH (path);
            else {         /* free it */
               ++freed;
#if 0
               path->prev = (ModendHyp *) 0xDEADBEEF;
               path->frame = -1;
               path->user = 99;
#endif
               /* similar to Dispose (heap, path) */
               b->used[iMapPos] &= ~(iMapMask);
               if (i < b->firstFree) 
                  b->firstFree = i;
               b->numFree++; 
               heap->totUsed--;
            }
         }
      }

      /* end of block clean-up */
      if (b->numFree == b->numElem) { 
         if (trace&T_GC)
            printf ("should free whole block!\n");
      }
   }

   if (trace&T_GC)
      printf ("freed %d of %d ModPaths\n", freed, total);
}
#endif

/* GarbageCollectPaths

     dispose all WordEndhyps that are not reachable by active tokens
     anymore
     uses simple mark & sweep GC
*/
static void GarbageCollectPaths (DecoderInst *dec)
{
   int i, l, N;
   LexNodeInst *inst;
   TokenSet *ts;

   if (trace&T_GC) {
      printf ("Garbage Collecting paths.\n");
      PrintHeapStats (&dec->weHypHeap);
      PrintHeapStats (&dec->altweHypHeap);
#ifdef MODALIGN
      if (dec->modAlign)
        PrintHeapStats (&dec->modendHypHeap);
#endif
   }
   /*# mark phase */
   for (l = 0; l < dec->nLayers; ++l) {
      if (trace&T_GC)
         printf (" layer %d...\n", l);

      for (inst = dec->instsLayer[l]; inst; inst = inst->next) {
         switch (inst->node->type) {
         case LN_MODEL:
            N = inst->node->data.hmm->numStates;
            break;
         default:       /* all others are single state */
            N = 1;
            break;
         }

         for (i = 0; i < N; ++i) {
            ts = &inst->ts[i];
            if (ts->n > 0)
               MarkTokSet (ts);
         }
      }
   }

   /* sweep phase */
   SweepPaths (&dec->weHypHeap);
   SweepAltPaths (&dec->altweHypHeap);
#ifdef MODALIGN
   if (dec->modAlign)
      SweepModPaths (&dec->modendHypHeap);
#endif

   if (trace&T_GC) {
      printf ("Garbage Collection finished.\n");
      PrintHeapStats (&dec->weHypHeap);
      PrintHeapStats (&dec->altweHypHeap);
#ifdef MODALIGN
   if (dec->modAlign)
      PrintHeapStats (&dec->modendHypHeap);
#endif
   }
}


/* ------------------------ End of HLVRec-GC.c ----------------------- */

