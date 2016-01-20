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
/*         File: HLVmodel.h Model handling for HTK LV Decoder  */
/* ----------------------------------------------------------- */

#ifndef _HLVMODEL_H_
#define _HLVMODEL_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "HShell.h"
#include "HMem.h"
#include "HMath.h"
#include "HWave.h"
#include "HLabel.h"
#include "HAudio.h"
#include "HParm.h"
#include "HDict.h"
#include "HModel.h"

#include <assert.h>

/*
  The model representation in HModel is very general and extensible
  but in many cases this flexibility is not needed and by restricting
  the model structure a lot of memory and run-time overhead can be saved.

  This module defines a new data structure HMMSet_lv that is optimised for
  compact storage and especially for fast acces as needed in a large
  vocabulary decoder.

  The Function ConvertHMMSet2LV() takes an existing HMMSet and converts it 
  into the optimised representation. 

  ### at the momnet the original HMMSet is still needed after this conversion
  as for some HMMs it will not be possible to represent them in the compact
  form.


  A typical HMM set has the following properties.

    - many physical HMMs (e.g. 15k)
    - small number of tied transP matricies (e.g. 50)
    - most transP are left-to-right with no skips
    - tied-states
    - moderate number of states (e.g. 6k)
    - multi-mixture Gaussians
    - diagonal covariance


  For fast access we want:

    - vectors (means, vars, etc.) aligned on 16(?) Byte boundary for SSE
    - all vecotrs zero-padded to nearest multiple of 4 elements for SSE
    - trade off CPU for memory saving (calc loop transP as 1-step)
    - maybe: quantise 4byte-floats for storage into 16bit-ints?
    - store log values if we log the all the time anyway (mixweights, gConst)
    - assume fixed minimum number of mixes for all states. If a state has more mixes
      then skip stateIds and use multiple blocks. 


  We won't bother doing the following:

   - perform calculation based on (quantised) 8/16bit-ints
   - assume fixed number of states

*/

#define HLVMODEL_VEC_ALIGN 16
#define HLVMODEL_VEC_PAD 4


   /* the info about states is arranged in blocks, normally one state 
      corresponds to one block, but states with many mixes can use multiple blocks */


typedef struct _StateInfo_lv StateInfo_lv;

struct _StateInfo_lv {
   float *base;
   unsigned long mixPerBlock;
   unsigned long nBlocks;
   unsigned long nDim;         /* real number of dimensions, e.g. 39 */
   unsigned long nVec;         /* number of vecotrs (e.g. 10 = RoundAlign (nDim,  HLVMODEL_VEC_PAD) */
   size_t floatsPerMix;
   size_t floatsPerBlock;        /* mixPerBlock * floatsPerMix */
   /*   size_t meanOffset;  */
   size_t invVarOffset;         /* 4 + nDim * floatsPerMix*/

   HMMSet *hset;
   Boolean useHModel;
   StateInfo **si;              /* pointers to HModel:StateInfos  for USEHMODEL=T */
};

   /* layout of a block:
      for each of the mixPerBlock mixes:
        LogFloat gConst;
        LogFloat mixWeight;
        int nMix;               only valid for first mix 
        StreamElem *se; 
        float pad[HLVMODEL_VEC_PAD - 4];
        float mean[nDim];
        float invVar[nDim];

 relies on sizeof (int)==sizeof (float)  
      and  sizeof (MixPDF *)==sizeof (float)
   */

#define HLVMODEL_BLOCK_BASE(si, s)   ((si)->base + (s) * (si)->floatsPerBlock)
#define HLVMODEL_BLOCK_GCONST_OFFSET(si) (0)
#define HLVMODEL_BLOCK_MIXW_OFFSET(si) (1)
#define HLVMODEL_BLOCK_GCONST(si,base) (*((base) + 0))
#define HLVMODEL_BLOCK_MIXW(si,base) (*((base) + 1))
#define HLVMODEL_BLOCK_NMIX(si,base) (*((int *) ((base) + 2)))
#define HLVMODEL_BLOCK_MPDF(si,base) (*((MixPDF **) ((base) + 3)))
#define HLVMODEL_BLOCK_MEAN_OFFSET(si) (4)
#define HLVMODEL_BLOCK_INVVAR_OFFSET(si) ((si)->invVarOffset)




StateInfo_lv *ConvertHSet(MemHeap *heap, HMMSet *hset, Boolean useHModel);
LogFloat OutP_lv (StateInfo_lv *si,  unsigned short s, float *x);
void OutPBlock (StateInfo_lv *si, Observation **obsBlock, 
                int n, int sIdx, float acScale, LogFloat *outP);



#if 0   /* forget about the following for now, we don't need this in a state net */

/*
  assumptions made for compact model representation:

  - HLVMODEL_TRANSP_LR1   
    topology: only self loop and next transitions.
        <TRANSP> 5
         0.0  1.0  0.0  0.0  0.0
         0.0  1-#0 #1   0.0  0.0
         0.0  0.0  1-#2 #2   0.0
         0.0  0.0  0.0  1-#3 #3
         0.0  0.0  0.0  0.0  0.0
     here #n refers to the probs actually stored.

   - diagonal covariance

*/

typedef struct _State_lv State_lv;
typedef struct _TransP_lv TransP_lv;
typedef struct _HMM_lv HMM_lv;

/*#### modify this so that  mean and invVar are interleaved and aligned optimally */
stuct {
   LogFloat *mixWeight;         /* [0..nMix-1] log mixture weights */
   LogFloat *gConst;            /* [0..nMix-1] log precomputed part of outP */
   float *mean;                 /* nMix mean vectors (nDim dimensions) */
   float *invVar;               /* nMix inv diagC vectors (nDim dimensions) */
   unsigned short nMix;
} _State_lv;

stuct {
   LogFloat a[6];          /* log transition probs (see above) */
} _TransP_lv;

stuct {
  unsisgned short tIdx;         /* transP index */
  int sIdx;         /* state index */
} _HMM_lv;


typedef stuct {
   int nHMM;                    /* number of compact models */
   int nDim;                    /* dimensionality of feature spae = VectorSize(obs) */
   int nTransP;                 /* number of transition matrices */
   int nState;                  /* number of states */
   TransP_lv *transP;           /* [0..nTransP-1] compact transition matrices */
   State_lv *state;             /* [0..nState-1] compact state info */

} HMMSet_lv;

#endif

#ifdef __cplusplus
}
#endif

#endif  /* _HLVMODEL_H_ */


/*  CC-mode style info for emacs
 Local Variables:
 c-file-style: "htk"
 End:
*/
