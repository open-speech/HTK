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
/*           Speech Vision and Robotics group                  */
/*           (now Machine Intelligence Laboratory)             */
/*           Cambridge University Engineering Department       */
/*           http://mi.eng.cam.ac.uk/                          */
/*                                                             */
/* main authors:                                               */
/*           Valtcho Valtchev, Steve Young,                    */
/*           Julian Odell, Gareth Moore                        */
/*                                                             */
/* ----------------------------------------------------------- */
/*           Copyright: Cambridge University                   */
/*                      Engineering Department                 */
/*            1994-2015 Cambridge, Cambridgeshire UK           */
/*                      http://www.eng.cam.ac.uk               */
/*                                                             */
/*   Use of this software is governed by a License Agreement   */
/*    ** See the file License for the Conditions of Use  **    */
/*    **     This banner notice must not be removed      **    */
/*                                                             */
/* ----------------------------------------------------------- */
/*               File: LPMerge  LM interpolation               */
/* ----------------------------------------------------------- */

/* !HVER!LPMerge:   3.5.0 [CUED 12/10/15] */

/* ------------------- Model interpolation  ----------------- */

#ifndef _LPMERGE_H
#define _LPMERGE_H

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_LMODEL    32

typedef struct {
   char *fn;                /* LM filename */
   BackOffLM *lm;           /* the language model */
   float weight;            /* interpolation weight */
} LMInfo;

void InitPMerge(void);
/* 
   Initialise module 
*/

BackOffLM *MergeModels(MemHeap *heap, LMInfo *lmInfo, int nLModel, 
		       int nSize, WordMap *wList);
/*
   Interpolate models in lmInfo and return resulting model
*/

void NormaliseLM(BackOffLM *lm);
/* 
   Normalise probabilities and calculate back-off weights 
*/

#ifdef __cplusplus
}
#endif

#endif


/* -------------------- End of LPMerge.h ---------------------- */

