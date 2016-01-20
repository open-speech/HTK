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
/*         File: HLVmodel.c Model handling for HTK LV Decoder  */
/* ----------------------------------------------------------- */

char *hlvmodel_version = "!HVER!HLVmodel:   3.4.1 [GE 12/03/09]";
char *hlvmodel_vc_id = "$Id: HLVModel.c,v 1.1.1.1 2006/10/11 09:54:55 jal58 Exp $";


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

#include "HLVModel.h"


#include "config.h"

#include <assert.h>


/* ----------------------------- Trace Flags ------------------------- */

#define T_TOP 0001         /* top level Trace  */

static int trace=0;
static ConfParam *cParm[MAXGLOBS];      /* config parameters */
static int nParm = 0;

/* -------------------------- Global Variables etc ---------------------- */


/* --------------------------- Initialisation ---------------------- */

/* EXPORT->InitLvmodel: register module & set configuration parameters */
void InitLVModel(void)
{
   int i;
   
   Register(hlvmodel_version,hlvmodel_vc_id);
   nParm = GetConfig("HLVMODEL", TRUE, cParm, MAXGLOBS);
   if (nParm>0){
      if (GetConfInt(cParm,nParm,"TRACE",&i)) trace = i;
   }

}

/* --------------------------- the real code  ---------------------- */

size_t RoundAlign(size_t addr, size_t align)
{
   return ((addr % align) == 0) ? addr : (addr/align + 1) * align;
}

StateInfo_lv *ConvertHSet(MemHeap *heap, HMMSet *hset, Boolean useHModel)
{
   HMMScanState hss;
   StateInfo_lv *si;
   int minNMix = 1e6;
   int sIdx = 0;        /* next free sIdx, start at 0 !!! */

   assert (hset->swidth[0] == 1);

   si = (StateInfo_lv *) New (heap, sizeof (StateInfo_lv));

   si->hset = hset;
   si->nDim = hset->vecSize;
   si->nVec = RoundAlign (si->nDim, HLVMODEL_VEC_PAD) / HLVMODEL_VEC_PAD;
   si->floatsPerMix = 2 * (si->nVec * HLVMODEL_VEC_PAD) + 4;
   
   /* find block size and assign state indexes */
   NewHMMScan (hset, &hss);
   while (GoNextState (&hss, FALSE)) {
      hss.si->sIdx=-1;
      if (hss.si->pdf[1].nMix < minNMix)
         minNMix = hss.si->pdf[1].nMix;
   } 
   EndHMMScan (&hss);

   NewHMMScan (hset, &hss);
   while (GoNextState (&hss, FALSE)) {
      if (hss.si->sIdx == -1) {
         if (hss.si->pdf[1].nMix == minNMix) {
            hss.si->sIdx = sIdx;
            ++sIdx;
         } else {       /* need multiple blocks */
            hss.si->sIdx = sIdx;
            sIdx += RoundAlign (hss.si->pdf[1].nMix,minNMix) / minNMix;
         }
      }
      else {
         printf ("THIS SHOULD NOT HAPPEN!!!\n");
      }
   }
   EndHMMScan (&hss);

   /* create StateInfo_lv structure */
   si->mixPerBlock = minNMix;
   si->floatsPerBlock = si->mixPerBlock * si->floatsPerMix;
   si->nBlocks = sIdx;
   hset->numSharedStates = si->nBlocks;

   if (!useHModel) {
      si->base = (float *) New (heap, si->nBlocks * si->floatsPerBlock * sizeof (float));
      HLVMODEL_BLOCK_INVVAR_OFFSET(si) = HLVMODEL_BLOCK_MEAN_OFFSET(si) + si->nVec * HLVMODEL_VEC_PAD;
      
      NewHMMScan (hset, &hss);
      while (GoNextState (&hss, FALSE)) {
         StreamElem *se;
         MixtureElem *me;
         MixPDF *mp;
         LogFloat mixw;
         int m, i;
         float *base, *mean, *invVar;

         se = &hss.si->pdf[1];

         base = HLVMODEL_BLOCK_BASE(si, hss.si->sIdx);
         HLVMODEL_BLOCK_NMIX(si,base) = se->nMix;

         mean = base + HLVMODEL_BLOCK_MEAN_OFFSET(si);
         invVar = base + HLVMODEL_BLOCK_INVVAR_OFFSET(si);

         for (m = 1; m <= se->nMix; ++m) {
            me = &se->spdf.cpdf[m];
            mp = me->mpdf;
            assert (mp->ckind == INVDIAGC);

            HLVMODEL_BLOCK_GCONST(si, base) = mp->gConst;
            HLVMODEL_BLOCK_MPDF(si, base) = mp;
            mixw = MixLogWeight(hset,me->weight);
            assert (mixw > LSMALL);
            HLVMODEL_BLOCK_MIXW(si, base) = mixw;
         
            for (i = 1; i <= si->nDim;  ++i) {
               mean[i-1] = mp->mean[i];
               invVar[i-1] = mp->cov.var[i];
            }
            base += si->floatsPerMix;
            mean += si->floatsPerMix;
            invVar += si->floatsPerMix;
         }
      } 
      EndHMMScan (&hss);
   }
   else
      si->base = NULL;

   si->useHModel = useHModel;
#if 1   /* USEHMODEL=T */
   si->si = (StateInfo **) New (heap, sIdx * sizeof (StateInfo *));
   NewHMMScan (hset, &hss);
   while (GoNextState (&hss, FALSE)) {
      si->si[hss.si->sIdx] = hss.si;
   }
   EndHMMScan (&hss);
#endif

   return si;
}


void PrintState_lv (StateInfo_lv *si,  unsigned short s)
{
   int m, i;
   float *base;
   float *mean;
   float *invVar;
   float gc;
   int nMix;
   LogFloat mixw;

   base = HLVMODEL_BLOCK_BASE(si, s);
   nMix = HLVMODEL_BLOCK_NMIX(si,base);
   mean = base + HLVMODEL_BLOCK_MEAN_OFFSET(si);
   invVar = base + HLVMODEL_BLOCK_INVVAR_OFFSET(si);

   for (m = 1; m <= nMix; m++) {
      mixw = HLVMODEL_BLOCK_MIXW(si,base);
      gc = HLVMODEL_BLOCK_GCONST(si,base);
      
      printf ("mix %d  mixw %.2f gc %.2f  \n",m, mixw, gc);

         
      printf (" mean ");
      for (i = 0; i < si->nDim; ++i) {
         printf ("%.2f ", mean[i]);
      }
      printf ("\n invVar ");
      for (i = 0; i < si->nDim; ++i) {
         printf ("%.2f ", invVar[i]);
      }
      printf ("\n");
      base += si->floatsPerMix;
      mean += si->floatsPerMix;
      invVar += si->floatsPerMix;
   }
}

/* EXPORT-> OutP_lv: returns log prob for state s of observation x */
LogFloat OutP_lv (StateInfo_lv *si,  unsigned short s, float *x)
{
   int m, i;
   LogDouble bx;
   LogFloat px;
   float *base;
   float *mean;
   float *invVar;
   int nMix;
   LogFloat mixw, xmm;

   base = HLVMODEL_BLOCK_BASE(si, s);
   nMix = HLVMODEL_BLOCK_NMIX(si,base);
   mean = base + HLVMODEL_BLOCK_MEAN_OFFSET(si);
   invVar = base + HLVMODEL_BLOCK_INVVAR_OFFSET(si);

   bx = LZERO;                   /* Multi Mixture Case */
   for (m = 1; m <= nMix; m++) {
      mixw = HLVMODEL_BLOCK_MIXW(si,base);

      px = HLVMODEL_BLOCK_GCONST(si,base);
      for (i = 0; i < si->nDim; ++i) {
         xmm = x[i] - mean[i];
         px += xmm*xmm * invVar[i];
      }
      px = -0.5 * px;;
      
      bx = LAdd (bx, mixw + px);

      base += si->floatsPerMix;
      mean += si->floatsPerMix;
      invVar += si->floatsPerMix;
      
   }
   return bx;
}


void OutPBlock (StateInfo_lv *si, Observation **obsBlock, 
                int n, int sIdx, float acScale, LogFloat *outP)
{
   int i;

   for (i = 0; i < n; ++i) {
      outP[i] = OutP_lv (si, sIdx, &obsBlock[i]->fv[1][1]);
   }

   /* acoustic scaling */
   if (acScale != 1.0)
      for (i = 0; i < n; ++i)
         outP[i] *= acScale;
}

/*  CC-mode style info for emacs
 Local Variables:
 c-file-style: "htk"
 End:
*/
