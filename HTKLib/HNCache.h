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
/* author:                                                     */
/*           Chao Zhang <cz277@cam.ac.uk>                      */
/*                                                             */
/* ----------------------------------------------------------- */
/*           Copyright: Cambridge University                   */
/*                      Engineering Department                 */
/*            2013-2015 Cambridge, Cambridgeshire UK           */
/*                      http://www.eng.cam.ac.uk               */
/*                                                             */
/*   Use of this software is governed by a License Agreement   */
/*    ** See the file License for the Conditions of Use  **    */
/*    **     This banner notice must not be removed      **    */
/*                                                             */
/* ----------------------------------------------------------- */
/*             File: HNCache.h  ANN model data cache           */
/* ----------------------------------------------------------- */

/* !HVER!HNCache:   3.5.0 [CUED 12/10/15] */

#ifndef _HNCACHE_H_
#define _HNCACHE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include "HMem.h"
#include "HParm.h"
#include <pthread.h>

/* ------------------------- Trace Flags ------------------------- */

/*
 The following types define the in-memory representation of a ANN
*/

/* ------------------------- Predefined Types ------------------------- */

#define MAXLATSUTT 1
#define MAXAUGFEAS 5

/* ------------------------- ANN Definition ------------------------- */


/* FRMVK (DataCache.frmOrder): DataCache.frmOrder != NULL, DataCache.frmPtr.frmOrderPtr, UttElem.frmOrder == NULL */
/* NONEVK (DataCache.uttOrder): DataCache.frmOrder == NULL, DataCache.uttOrderPtr, UttElem.frmOrder == NULL */
/* PLNONEVK (DataCache.uttOrder): DataCache.frmOrder == NULL, DataCache.ptrNum * DataCache.uttOrderPtr, UttElem.frmOrder == NULL */
/* PLUTTVK (DataCache.uttOrder): DataCache.frmOrder == NULL, DataCache.ptrNum * DataCache.uttOrderPtr, UttElem.frmOrder == NULL */
/* PLUTTFRMVK (DataCache.uttOrder & UttElem.frmOrder): DataCache.frmOrder == NULL, DataCache.ptrNum * DataCache.uttOrderPtr, UttElem.frmOrder != NULL */
/* UTTFRMVK (DataCache.uttOrder & UttElem.frmOrder): DataCache.frmOrder == NULL, DataCache.uttOrderPtr, UttElem.frmOrder != NULL */
/* UTTVK (DataCache.uttOrder): DataCache.frmOrder == NULL, DataCache.uttOrderPtr, UttElem.frmOrder == NULL */
enum _VisitKind {FRMVK, NONEVK, PLNONEVK, PLUTTVK, PLUTTFRMVK, UTTFRMVK, UTTVK};
typedef enum _VisitKind VisitKind;

enum _ShuffKind {KNUTHFSK, KNUTHRSK, QUICKNETSK};
typedef enum _ShuffKind ShuffKind;

enum _LabelKind {NULLLK = 0, FEALK = 1, LABLK = 2, LATLK = 4};
typedef enum _LabelKind LabelKind;

typedef struct _UttElem {
    char *uttName;              /* the name of the utterance, could be NULL (by saveUttName) */
    int uttLen;                 /* the length (frame number) of this utterance */
    int frmUsed;                /* the number of frames processed */
    float *frmMat;              /* the frame matrix; could be NULL if utterance not loaded */
    float *flabMat;		/* cz277 - FEALAB */
    Vector augFeaVec[MAXAUGFEAS + 1];	/* specify the maximum number of augmented feature vectors */
    int *frmOrder;              /* the frame visiting order within the utterance */
    int *labIdxes;              /* the vector with the indexes for all frames, could be NULL */
    Lattice *denLats[MAXLATSUTT];
    Lattice *numLats[MAXLATSUTT];
    Boolean *numInDen;	
    AdaptXForm *inXForm;
    AdaptXForm *paXForm;
    /* cz277 - xform */
    NMatBundle **curUttNMatRPLs;
    NVecBundle **curUttNVecRPLs; 
} UttElem;

typedef struct _FrmIndex {
    int uttIdx;                 /* the index of the relevant utterance in the array */
    int frmIdx;                 /* the index of current frame vector in the relevant utterance */
} FrmIndex;

typedef struct _LabelInfo {
    LabelKind labelKind;        /* the kind of the label */
    char *labFileMask;          /* mast for reading labels */
    char *labDir;               /* label file directory */
    char *labExt;               /* label file extension */
    /*Boolean countOcc;*/		/* whether count state occupancies or not */
    char *latFileMask;          /* lattice file mask, could be NULL */
    char *latMaskNum;           /* numerator lattice file mask, could be NULL */
    char **numLatDir;           /* numerator lattice directories */
    int nNumLats;               /* number of numerator lattice directories */
    char *numLatSubDirPat;      /* sub directory for numerator lattices */
    char *latMaskDen;           /* denominator lattice file mask, could be NULL */
    char **denLatDir;           /* denorminatory lattice directories */
    int nDenLats;               /* number of denominator lattice directories */
    char *denLatSubDirPat;      /* sub directory for denominator lattices */
    char *latExt;               /* lattice extension */
    FILE *scpFLab;
    int dimFLab;		/* cz277 - FEALAB */
    /*ParmBuf pbufFLab;*/		/* cz277 - FEALAB */
    Observation *obsFLab;	/* cz277 - FEALAB */ 
    Boolean incNumInDen;
    Boolean useLLF;		/* whether use LLF or not */
    Vocab *vocab;                  /* word list or directory pointer (Vocab *) */
    UPDSet uFlags;
} LabelInfo;

typedef struct _DataCache {
    /* basic elements */
    MemHeap *cmem;              /* the memory heap for this data cache */
    Boolean revisit;            /* whether the cache has been revisit or not */
    size_t cacheSamples;           /* the approx number of samples stored in this cache */
    int frmDim;                 /* the frame dimension */
    int tUttNum;                /* the total number of utterances */
    /* storage structure */
    int nxtUttPos;              /* the position of the next unloaded utterance */
    UttElem *uttElems;          /* the utterance information list */
    /* visit order kind */
    VisitKind visitKind;        /* the identifier for the random visiting method */
    /* utterance level visiting order */
    int stUttPos;               /* the start position of the utterance visiting index */
    int edUttPos;               /* the end position of the utterance visiting index */
    int *uttOrder;              /* the indicator for the visiting order */
    /* frame level visiting order */
    size_t frmNum;                 /* the number of frames in cache */
    size_t tFrmNum;                /* the total number of frames */
    size_t fvLen;                  /* the actual length of frmVisit */
    FrmIndex *frmOrder;         /* the indicator for the frame visiting order for FRMVK only */
    /* visiting pointers */
    size_t orderPtr;               /* could point to the next element in either uttOrder or frmOrder */
    int ptrNum;                 /* the number of pointers in cache.frmPtr */
    FrmIndex *frmPtrs;          /* NULL if FRMVK */
    /* label structure */
    LabelInfo *labelInfo;       /* the pointer to the label information, NULL if no label is available */
    /* structure for internal batch (without context expansion) */
    int batLen;               	/* the number of frames in the batch (frmBatch) */
    FrmIndex *frmBatch;         /* the internal batch of frmIndex */
    int *CMDVecPL;		/* the list for PL* visiting order, [..., -2]: do nothing; -1: clear; [0, batchSize): move to */
    /* auxiliary elements */
    FILE *scpFile;              /* the handler of the associated file */
    HMMSet *hmmSet;                 /* the hmmset associated to this cache (HMMSet *) */
    /*ParmBuf parmBuf;*/            /* the parm buf */
    Observation *obs;           /* the auxiliary structure for loading frame into cache */
    int streamIdx;              /* the stream index of the input feature */
    LELink outLayer;            /* the link to the out layer associated with this cache */
    NMatrix *labMat;            /* the label matrix associated with this data cache */
    IntVec labVec;              /* the vector contains the index of the reference targets */
    Boolean saveUttName;        /* whether saves each utterance name or not */
    int batchSamples;		/* the mini-batch size associated with this cache */
    XFInfo *xfInfo;
    pthread_t extThread;	/* cz277 - mtload */
    Boolean firstLoad;		/* cz277 - mtload */
} DataCache;

/* ------------------------ Global Settings ------------------------- */

int GetNMKLThreads(void);
size_t GetDefaultNCacheSamples(void);
VisitKind GetDefaultVisitKind(void);
void SetEpochIndex(int curEpochIdx);
void AccAllCacheSamples(size_t curCacheSamp);
void SetNeed2UnloadFlag(void);

void InitNCache(void);
DataCache *CreateCache(MemHeap *heap, FILE *scpFile, int scpCnt, HMMSet *hset, Observation *obs, int streamIdx, size_t cacheSamples, VisitKind visitKind, XFInfo *xfInfo, LabelInfo *labelInfo, Boolean saveUttName);
void InitCache(DataCache *cache);
Boolean FillAllInpBatch(DataCache *cache, int *nSamples, int *uttCnt);
/* cz277 - mtload */
void UpdateCacheStatus(DataCache *cache);
void LoadCacheData(DataCache *cache);
int UnloadCacheData(DataCache *cache);

char *GetCurUttName(DataCache *cache);
int GetCurUttLen(DataCache *cache);
UttElem *GetCurUttElem(DataCache *cache);
FBLatInfo *LoadXFormsFromUttElem(UttElem *uttElem, FBLatInfo *fbInfo);
FBLatInfo *LoadNumLatsFromUttElem(UttElem *uttElem, FBLatInfo *fbInfo);
FBLatInfo *LoadDenLatsFromUttElem(UttElem *uttElem, FBLatInfo *fbInfo);
void UpdateTargetLogPrior(DataCache *cache, float offset);

void ResetCache(DataCache *cache);
void FreeCache(DataCache *cache);
void ResetCacheHMMSetCfg(DataCache *cache, HMMSet *hset);

char *MakeNameNMatRPL(char *curSpkr, char *tgtMacro, char *RPLName);
char *MakeNameNVecRPL(char *curSpkr, char *tgtMacro, char *RPLName);
void SaveAllNMatRPLs(HMMSet *hset, FILE *script);
void SaveAllNVecRPLs(HMMSet *hset, FILE *script);
void InstallOneUttNMatRPLs(UttElem *uttElem);
void InstallOneUttNVecRPLs(UttElem *uttElem);
void ResetNMatRPL(void);
void ResetNVecRPL(void);


#ifdef __cplusplus
}
#endif

#endif  /* _HNCACHE_H_ */

/* ------------------------- End of HNCache.h ------------------------- */ 
