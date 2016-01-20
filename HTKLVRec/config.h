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
/*         2000-2003  Cambridge University                     */
/*                    Engineering Department                   */
/*                                                             */
/*   Use of this software is governed by a License Agreement   */
/*    ** See the file License for the Conditions of Use  **    */
/*    **     This banner notice must not be removed      **    */
/*                                                             */
/* ----------------------------------------------------------- */
/*         File: config.h: Global Configuration Options        */
/* ----------------------------------------------------------- */

#define NDEBUG
#define COLLECT_STATS
#undef COLLECT_STATS_ACTIVATION

#if 1
typedef unsigned short PronId;             /* uniquely identifies (word,pron) pair, i.e. 
                                   homophones have different Ids */
typedef unsigned short LMId;
#define LM_NGRAM_INT
#else
typedef unsigned int LMId;
typedef unsigned int PronId;
#undef LM_NGRAM_INT
#endif


/* types for scores at various levels */

/* typedef LogDouble TokScore; */
typedef LogFloat TokScore;
typedef LogFloat RelTokScore;

typedef LogFloat LMTokScore;

/* definitions for normal and mod versions of HDecode */
#ifdef HDECODE_MOD
#undef TSIDOPT
#define MODALIGN
#else
#define TSIDOPT
#undef MODALIGN
#endif

#define MAXBLOCKOBS 16

#undef USE_INTEL_SSE


#undef LEGACY_CUHTK2_MLLR


/* always disable TSIDOPT when we want to support MODALIGN, because
   TSIDOPT totally screws up model time traceback */
#ifdef MODALIGN
#undef TSIDOPT
#endif
