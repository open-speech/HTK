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
/*       File: HLVRec-outP.c OutP calculation and caching      */
/* ----------------------------------------------------------- */

char *hlvrec_outp_version = "!HVER!HLVRec-outP:   3.5.0 [CUED 12/10/15]";
char *hlvrec_outp_vc_id = "$Id: HLVRec-outP.c,v 1.2 2015/10/12 12:07:24 cz277 Exp $";

static void ResetOutPCache (OutPCache *cache)
{
   int i;

   
   if (cache->nStates > 0)
      for (i = 0; i <= cache->nStates; ++i)
         cache->stateT[i] = -1000;
   
   if (cache->nMix > 0)
      for (i = 0; i <= cache->nMix; ++i)
         cache->mixT[i] = -1000;

   cache->cacheHit = cache->cacheMiss = 0;
}

static OutPCache *CreateOutPCache (MemHeap *heap, HMMSet *hset, int block)
{
   OutPCache *cache;

   cache = New (heap, sizeof (OutPCache));

   cache->block = block;
   cache->nStates = hset->numSharedStates;
   cache->nMix = hset->numSharedMix;
   /* the sIdx values are 1..numSharedStates, thus the +1 below. Same for mIdx */
   
   cache->stateOutP = cache->mixOutP = NULL;
   if (cache->nStates > 0) {
      cache->stateT = (int *) New (heap, (cache->nStates + 1) * sizeof (int));
      cache->stateOutP = (LogFloat *) New (heap, (cache->nStates + 1) * cache->block * sizeof (LogFloat));
   }
   if (cache->nMix > 0) {
      cache->mixT = (int *) New (heap, (cache->nMix + 1) * sizeof (int));
      cache->mixOutP = (LogFloat *) New (heap, (cache->nMix + 1) * cache->block * sizeof (LogFloat));
   }

   return cache;
}

/* SOutP_ID_mix_Block: returns log prob of stream s of observation x */
LogFloat SOutP_ID_mix_Block(HMMSet *hset, int s, Observation *x, StreamElem *se)
{
   int vSize;
   LogDouble px;
   MixtureElem *me;
   MixPDF *mp;
   Vector v;
   LogFloat wt;

   assert (hset->hsKind == PLAINHS && hset->hsKind == SHAREDHS);
   
   v = x->fv[s];
   vSize = VectorSize(v);
   assert (vSize == hset->swidth[s]);
   me = se->spdf.cpdf+1;
   if (se->nMix == 1){     /* Single Mixture Case */
      mp = me->mpdf; 
      assert (mp->ckind == INVDIAGC);
      /*       px = IDOutP(v,vSize,mp); */
      {
         int i;
         float sum;
         float *mean, *ivar;

         mean = mp->mean;
         ivar = mp->cov.var;

         sum = mp->gConst;
         for (i=1 ; i <= vSize; i++) {
            sum += (v[i] - mean[i]) * (v[i] - mean[i]) * ivar[i];
         }
         px = -0.5*sum;
      }


      return px;
   } else {             /* Multi Mixture Case */
      LogDouble bx = LZERO;                   
      int m;

      for (m=1; m<=se->nMix; m++,me++) {
         wt = MixLogWeight(hset,me->weight);
         if (wt>LMINMIX) {  
            mp = me->mpdf; 
            /*       px = IDOutP(v,vSize,mp);   */
            {
               int i;
               float sum,xmm;
               
               sum = mp->gConst;
               for (i=1;i<=vSize;i++) {
                  xmm = v[i] - mp->mean[i];
                  sum += xmm*xmm*mp->cov.var[i];
               }
               px = -0.5*sum;
            }
            
            bx = LAdd(bx,wt+px);
         }
      }
      return bx;
   }
   return LZERO;;
}

#if 0           /* old OutPBlock()  copes with streams and non-diag outp's */
static void OutPBlock (DecoderInst *dec, Observation **obsBlock, 
                       int n, HLink hmm, int state, float acScale, LogFloat *outP)
{
   int i;
   
#if 0
   for (i = 0; i < n; ++i) {
      outP[i] = OutP (obsBlock[i], hmm, state);
   }

#else
   StateInfo *si;
   StreamElem *se;
   int s, S = obsBlock[0]->swidth[0];
   
   si = (hmm->svec+state)->info;
   se = si->pdf+1;
   
   if (S == 1 && !si->weights) {
      
      for (i = 0; i < n; ++i) {
         outP[i] = OutP_lv (dec->si, hmm->svec[state].info->sIdx, &obsBlock[i]->fv[1][1]);
#if 1   /* sanity checking */
         {
            LogFloat soutp;
            soutp = SOutP (hmm->owner, 1, obsBlock[i], se);
            assert (fabs (outP[i] - soutp) < 0.01);
         }
#endif
      }
   }
   else {       /* multi stream */
      Vector w;

      for (i = 0; i < n; ++i)
         outP[i] = 0.0;
      
      w = si->weights;
      for (s = 1; s <= S; s++, se++)
         for (i = 0; i < n; ++i)
            outP[i] += w[s] * SOutP (hmm->owner, s, obsBlock[i], se);
   }

   /* acoustic scaling */
   if (acScale != 1.0)
      for (i = 0; i < n; ++i)
         outP[i] *= acScale;
#endif
}

#endif


/* cOutP

     caching version of OutP from HModel. This only caches only on a state 
     level, not on a mixture level. 
*/
static LogFloat cOutP (DecoderInst *dec, Observation *x, HLink hmm, int state)
{
   int sIdx, n;
   LogFloat outP;
   OutPCache *cache;

   assert (x == dec->obsBlock[0]);

   cache = dec->outPCache;
   sIdx = hmm->svec[state].info->sIdx;

   assert (sIdx >= 0);
   assert (sIdx < cache->nStates);
   
   n = dec->frame - cache->stateT[sIdx];

   assert (n >= 0);

   if (n < cache->block) {
      outP = cache->stateOutP[sIdx * cache->block + n];
      ++cache->cacheHit;
#if 0
      /* the following is *very* expensive, it effectively disables the cache,
         use only for sanity checking! */
      assert (outP == dec->acScale * OutP (x, hmm, state));
#endif
   }
   else {
      ++cache->cacheMiss;
      if (!cache->mixOutP) {     /* don't bother caching mixtures */
         /* #### handle boundary case where we don't have cache->block obs left */

         if (dec->hset->hsKind == HYBRIDHS) {   /* cz277 - ANN */
            OutPBlock_Hybrid(dec->si, cache->block, sIdx, dec->acScale, &cache->stateOutP[sIdx * cache->block], dec);         
         }
         else {
            if (!dec->si->useHModel) 
               OutPBlock(dec->si, &dec->obsBlock[0], cache->block, sIdx, dec->acScale, &cache->stateOutP[sIdx * cache->block], dec);    /* cz277 - ANN */
            else
               OutPBlock_HMod(dec->si, &dec->obsBlock[0], cache->block, sIdx, dec->acScale, &cache->stateOutP[sIdx * cache->block], dec->frame, dec);   /* cz277 - ANN */
         }    

         cache->stateT[sIdx] = dec->frame;
         outP = cache->stateOutP[sIdx * cache->block];

#if 0   /* sanity checking for OutPBlock */
         {
            LogFloat safe_outP;
            safe_outP = dec->acScale * OutP (x, hmm, state);
            assert (fabs (outP - safe_outP) < 0.01);
         }
#endif
      }
      else {            /* cache mixtures (e.g. for soft-tied systems) */
         abort ();
         /*
x      outP = OutP (x, hmm, state);
x      dec->cacheOutP[sIdx] = outP;
x      CACHE_FLAG_SET(dec, sIdx);
         */
      }
   }      

   return outP;
}



/* outP caclulation for USEHMODEL=T case  */


/*******************************************************************************/
/*  outP calculation from HModel.c and extended for new adapt code */

/* cz277 - ANN */
static LogFloat SOutP_HMod (HMMSet *hset, int s, Vector v, StreamElem *se,
                            int id)
{
   int m;
   LogFloat bx,px,wt,det;
   MixtureElem *me;
   MixPDF *mp;
   Vector otvs;

   /* Note hset->kind == SHAREDHS */
   assert (hset->hsKind == SHAREDHS);

   /*v=x->fv[s];*/
   me=se->spdf.cpdf+1;
   if (se->nMix==1){     /* Single Mixture Case */
      bx= MOutP(ApplyCompFXForm(me->mpdf,v,inXForm,&det,id),me->mpdf);
      bx += det;
   } else if (!pde) {
      bx=LZERO;                   /* Multi Mixture Case */
      for (m=1; m<=se->nMix; m++,me++) {
         wt = MixLogWeight(hset,me->weight);
         if (wt>LMINMIX) {   
            px= MOutP(ApplyCompFXForm(me->mpdf,v,inXForm,&det,id),me->mpdf);
            px += det;
            bx=LAdd(bx,wt+px);
         }
      }
   } else {   /* Partial distance elimination */
      wt = MixLogWeight(hset,me->weight);
      mp = me->mpdf;
      otvs = ApplyCompFXForm(mp,v,inXForm,&det,id);
      px = IDOutP(otvs,VectorSize(otvs),mp);
      bx = wt+px+det;
      for (m=2,me=se->spdf.cpdf+2;m<=se->nMix;m++,me++) {
         wt = MixLogWeight(hset,me->weight);
	 if (wt>LMINMIX){
	    mp = me->mpdf;
	    otvs = ApplyCompFXForm(mp,v,inXForm,&det,id);
	    if (PDEMOutP(otvs,mp,&px,bx-wt-det) == TRUE)
	      bx = LAdd(bx,wt+px+det);
	 }
      }
   }

   return bx;
}

/* cz277 - ANN */
LogFloat POutP_HModel (HMMSet *hset,Observation *x, StateInfo *si, int id, DecoderInst *dec, int frameIdx)
{
   LogFloat bx;
   StreamElem *se;
   Vector w;
   int s, S = x->swidth[0];

   if (S == 1 && si->weights == NULL) {
      switch (dec->decodeKind) {
         case NORMALDK:
            return SOutP_HMod(hset, 1, x->fv[1], si->pdf + 1, id);
         case TANDEMDK:
            return SOutP_HMod(hset, 1, dec->cacheVec[frameIdx][1], si->pdf + 1, id);
         case HYBRIDDK:
         default:
            HError(7890, "POutP_HModel: Unsupported DecodeKind");
      }
   }
   bx = 0.0; se = si->pdf + 1; w = si->weights;
   switch (dec->decodeKind) {
      case NORMALDK:
         for (s = 1; s <= S; s++, se++) {
            bx += w[s] * SOutP_HMod(hset, s, x->fv[s], se, id);
         }
         break;
      case TANDEMDK:
         for (s = 1; s <= S; s++, se++) {
            bx += w[s] * SOutP_HMod(hset, s, dec->cacheVec[frameIdx][s], se, id);
         }
         break;
      case HYBRIDDK:
      default:
         HError(7890, "POutP_HModel: Unsupported DecodeKind");
   }

   return bx;
}

/* cz277 - ANN */
void OutPBlock_HMod (StateInfo_lv *si, Observation **obsBlock, 
                int n, int sIdx, float acScale, LogFloat *outP, int id, DecoderInst *dec)
{
   int i;

   assert  (si->useHModel);
   
   if (dec->decodeKind == NORMALDK) {
      for (i = 0; i < n; ++i) {
         outP[i] = POutP_HModel(si->hset, obsBlock[i], si->si[sIdx], id, dec, -1);
      }
      if (acScale != 1.0)
         for (i = 0; i < n; ++i)
            outP[i] *= acScale;
   }
   else if (dec->decodeKind == TANDEMDK) {
      outP[0] = POutP_HModel (si->hset, obsBlock[0], si->si[sIdx], id, dec, dec->cacheVecIdx) * acScale;
   }
   else {
      HError(7890, "OutPBlock_HMod: Funtion is designed only for NORMALDK and TANDEMDK");
   }
}

/* cz277 - ANN */
LogFloat POutP_Hybrid(HMMSet *hset, StateInfo *si, DecoderInst *dec, int frameIdx) {
    int s, S, targetIdx;
    StreamElem *se;
    Vector w;
    LogFloat bx;
    /*float targetPen;*/

    /* get the stream width */
    S = hset->swidth[0];
    /* get the stream element */
    se = si->pdf + 1;
    /* if single stream */
    if (S == 1) {
        targetIdx = se->targetIdx;
        return dec->cacheVec[frameIdx][1][targetIdx];
        /*targetPen = se->targetPen;
        return dec->cacheVec[frameIdx][1][targetIdx] + targetPen;*/
    }
    /* if multi-stream */
    bx = 0.0;
    w = si->weights;
    for (s = 1; s <= S; ++s, ++se) {
        targetIdx = se->targetIdx;
        bx += w[s] * dec->cacheVec[frameIdx][s][targetIdx];
        /*targetPen = se->targetPen;
        bx += w[s] * (dec->cacheVec[frameIdx][s][targetIdx] + targetPen);*/
    }

    return bx;
}

/* cz277 - ANN */
void OutPBlock_Hybrid(StateInfo_lv *si, int n, int sIdx, float acScale, LogFloat *outP, DecoderInst *dec) {

    assert(dec->decodeKind == HYBRIDDK);
    outP[0] = POutP_Hybrid(si->hset, si->si[sIdx], dec, dec->cacheVecIdx) * acScale;
}

/* ------------------------ End of HLVRec-outP.c ----------------------- */

