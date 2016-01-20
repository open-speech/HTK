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
/*             File: HNCache.c  ANN model data cache           */
/* ----------------------------------------------------------- */

char *hncache_version = "!HVER!HNCache:   3.5.0 [CUED 12/10/15]";
char *hncache_vc_id = "$Id: HNCache.c,v 1.0 2015/10/12 12:07:24 cz277 Exp $";

#include <time.h>
#include "config.h"
#include "HShell.h"
#include "HMem.h"
#include "HMath.h"
#include "HSigP.h"
#include "HWave.h"
#include "HAudio.h"
#include "HParm.h"
#include "HLabel.h"
#include "HANNet.h"
#include "HModel.h"
#include "HTrain.h"
#include "HUtil.h"
#include "HAdapt.h"
#include "HFB.h"
#include "HNet.h"
#include "HArc.h"
#include "HFBLat.h"
#include "HLM.h"
#include "HLat.h"
#include "HNCache.h"
#include <math.h>

/* ------------------------------ Trace Flags ------------------------------ */

static int trace = 0;

#define T_TOP 0001
#define T_CCH 0002
#define T_RPL 0004
#define MAX(a,b) ((a)>(b) ? (a):(b))


/* --------------------------- Memory Management --------------------------- */

static MemHeap pbufStack;
static MemHeap transStack;
/* static MemHeap latHeap; */

/* ----------------------------- Configuration ------------------------------*/


static ConfParam *cParm[MAXGLOBS];      /* config parameters */
static int nParm = 0;

static size_t defaultCacheSamples = 100000;          /* the number of samples in cached; 1 sample by defult */
/*static size_t defaultCacheSize = 134217728;*/     /* the size of the total size of the cache data block, in bytes; */
/*static size_t batchSamples = 1;*/                 /* the number of samples in batch; 1 sample by default */
static VisitKind defaultVisitKind = FRMVK;      /* the indicator for visiting order; frame level randomization by default */
static ShuffKind shuffKind = KNUTHRSK;          /* the way of shuffle the train set cache */
static unsigned int shuffSeed = 0;              /* the seed for generating random integers for shuffling the list */
/*static char *updtFlagStr = NULL;*/                /* the string pointer indicating the layers to update */
static size_t allCacheSamp = 0;                 /* the size of all data caches */
static Boolean need2Unload = TRUE;              /* the flag for whether the memory can hold all samples */

static int epochIdx = 1;                        /* the epoch number used by QuickNetShuffle */
static int QN_outsegno = 0;                     /* the output segment number used by QuickNetShuffle */
static const size_t QN_XORTable[32] = {0x1, 0x1, 0x3, 0x6, 0xc, 0x14, 0x30, 0x60, 0xb8, 0x110, 0x240, 0x500, 0xca0, 0x1b00, 0x3500, 0x6000, 0xb400, 0x12000, 0x20400, 0x72000, 0x90000, 0x140000, 0x300000, 0x420000, 0xd80000, 0x1200000, 0x3880000, 0x7200000, 0x9000000, 0x14000000, 0x32800000, 0x48000000};    /* the XOR table acquired from QuickNet (in QN_seqgen.cc) */

/* cz277 - mtload */
static Boolean extThreadLoad = FALSE;

static void UnloadOneUtt(DataCache *cache, int dstPos);


/* get the batch size */
/*int GetNBatchSamples(void) {
    return batchSamples;
}*/

/* set the batch size */
/*void SetNBatchSamples(int userBatchSamples) {
    batchSamples = userBatchSamples;
}*/

/* get current cache size */
size_t GetDefaultNCacheSamples(void) {
    return defaultCacheSamples;
}

/* get current cache visiting kind */
VisitKind GetDefaultVisitKind(void) {
    return defaultVisitKind;
}

/* set epoch index as a random seed */
void SetEpochIndex(int curEpochIdx) {
    epochIdx = curEpochIdx;
}


/* accumulate total cache size */
void AccAllCacheSamples(size_t curCacheSamp) {
    allCacheSamp += curCacheSamp;
}

/* set the need2unload flag */
void SetNeed2UnloadFlag(void) {
    if (allCacheSamp < defaultCacheSamples)
        need2Unload = FALSE;
}

/* do knuth shuffling */
static inline void KnuthShuffle(void *list, size_t stPos, size_t edPos, int unitLen) {
    size_t i, randPos, n, range;
    void *srcAddr, *dstAddr;
    char buf[MAXSTRLEN];  /* the temp structure */

    /* stPos, stPos + 1, ..., edPos - 1*/
    n = edPos - stPos;
    /* set the range of the random value */
    range = n;      /* for default KNUTHFSK */
    for (i = n - 1; i > 0; --i) {
        /* set the range of the random value */
        if (shuffKind == KNUTHRSK) {
            range = i + 1;
        }
        /* get the random position to switch */
        randPos = rand() % range;
        /* compute the address of the source and destinate items */
        srcAddr = list + (stPos + i) * unitLen;
        dstAddr = list + (stPos + randPos) * unitLen;
        /* swap the values */
        memcpy(buf, srcAddr, unitLen);
        memcpy(srcAddr, dstAddr, unitLen);
        memcpy(dstAddr, buf, unitLen);
    }
}

/* copied from QuickNet (in QN_seqgen.cc) */
static inline unsigned int GetLog2Ceil(unsigned int val) {
    unsigned int mask, clear;
    int topbits, count, j;

    mask = 0xffff0000u;
    topbits = 16;
    count = 0;
    val -= 1;
    for (j = 0; j < 4; ++j) {
        clear = val & mask;
        if (!clear) {
            count += topbits;
            val <<= topbits;
        }
        topbits >>= 1;
        mask <<= topbits;
    }
    clear = val & 0xc0000000;
    if (((int) clear) >= 0) {
        ++count;
    }
    if (clear == 0) {
        ++count;
    }
    return 32 - count;
}

/* do QuickNet shuffling, copied from QuickNet (in QN_seqgen.cc) */
static inline void QuickNetShuffle(void *list, int stPos, int edPos, int unitLen) {
    int i, n;
    void *srcVals, *srcAddr, *dstAddr;
    unsigned int maxVal, xorVal, nxtVal, curVal, newSeed;

    n = edPos - stPos;
    maxVal = n;
    xorVal = QN_XORTable[GetLog2Ceil(maxVal + 1)];
    if (maxVal < 1 && maxVal > 0x7fffffff) {
        HError(8920, "QuickNetShuffle: maxVal out of range");
    }
    newSeed = QN_outsegno + (12345 * epochIdx) + shuffSeed;
    nxtVal = (unsigned int) (newSeed % maxVal) + 1;
    srcVals = (void *) New(&gstack, unitLen * n);
    memcpy(srcVals, list + stPos * unitLen, unitLen * n);

    for (i = 0; i < n; ++i) {
        curVal = nxtVal;
        do {
            if (nxtVal & 1) {
                nxtVal = (nxtVal >> 1) ^ xorVal;
            }
            else {
                nxtVal = nxtVal >> 1;
            }
        } while (nxtVal > maxVal);
        curVal -= 1;
        /* rearrange the value */
        srcAddr = srcVals + ((int) curVal) * unitLen;
        dstAddr = list + (stPos + i) * unitLen;
        memcpy(dstAddr, srcAddr, unitLen);
    }

    Dispose(&gstack, srcVals);
}

/* shuffle a segment */
static inline void ShuffleSegment(void *list, size_t stPos, size_t edPos, int unitLen) {
    switch (shuffKind) {
        case KNUTHFSK:
        case KNUTHRSK:
            KnuthShuffle(list, stPos, edPos, unitLen);
            break;
        case QUICKNETSK:
            QuickNetShuffle(list, (int) stPos, (int) edPos, unitLen);
            break;
        default:
            HError(8991, "ShuffleSegment: Unknown shuffle kind");
    }
}

/*  */
void InitNCache(void)
{
    int intVal;
    char buf[MAXSTRLEN];
    Boolean boolVal;

    Register(hncache_version, hncache_vc_id);
    nParm = GetConfig("HNCACHE", TRUE, cParm, MAXGLOBS);

    if (nParm > 0) {
        if (GetConfInt(cParm, nParm, "TRACE", &intVal)) { 
            trace = intVal;
        }
        if (GetConfInt(cParm, nParm, "DATACACHESIZE", &intVal)) {
            defaultCacheSamples = intVal;
        }
        if (GetConfStr(cParm, nParm, "DATAACCESSKIND", buf)) {
            if (strcmp(buf, "FRAMERAND") == 0) 
                defaultVisitKind = FRMVK;
            else if (strcmp(buf, "UTTERANCERAND") == 0) 
                defaultVisitKind = UTTVK;
            else if (strcmp(buf, "PARALLELSTREAMORIGIN") == 0) 
                defaultVisitKind = PLNONEVK;
            else if (strcmp(buf, "PARALLELSTREAMRAND") == 0) 
                defaultVisitKind = PLUTTVK;
            else if (strcmp(buf, "ORIGINAL") == 0)
                defaultVisitKind = NONEVK;
            else 
                HError(8921, "InitNCache: Unknown data access kind");
            /*if (strcmp(buf, "FRAMERAND") == 0) {
                defaultVisitKind = FRMVK;
            }
            else if (strcmp(buf, "UTTRAND") == 0) {
                defaultVisitKind = UTTVK;
            }
            else if (strcmp(buf, "HIERAND") == 0) {
                defaultVisitKind = UTTFRMVK;
            }
            else if (strcmp(buf, "PIPLORIG") == 0) {
                defaultVisitKind = PLNONEVK;
            }
            else if (strcmp(buf, "PIPLRAND") == 0) {
                defaultVisitKind = PLUTTVK;
            }
            else if (strcmp(buf, "HIEPIPLRAND") == 0) {
                defaultVisitKind = PLUTTFRMVK;
            }
            else if (strcmp(buf, "ORIGINAL") == 0) {
                defaultVisitKind = NONEVK;
            }
            else {
                HError(9999, "InitNCache: Unknown visit kind");
            }*/
        }
        if (GetConfStr(cParm, nParm, "SHUFFLEKIND", buf)) {
            if (strcmp(buf, "KNUTHFIXED") == 0) {
                shuffKind = KNUTHFSK;
            }
            else if (strcmp(buf, "KNUTHRAND") == 0) {
                shuffKind = KNUTHRSK;
            }
            else if (strcmp(buf, "QUICKNET") == 0) {
                shuffKind = QUICKNETSK;
            }
            else {
                HError(8921, "InitNCache: Unknown shuffle kind");
            }
        }
        if (GetConfStr(cParm, nParm, "RANDSEED", buf)) {
            if (strcmp(buf, "CURRENTTIME") == 0) {
                shuffSeed = time(NULL);
            }
            else {
                shuffSeed = (unsigned int) atol(buf);
            }
            srand(shuffSeed);   /* set the seed */
        }
        /* cz277 - mtload */
        if (GetConfBool(cParm, nParm, "EXTRATHREADLOADING", &boolVal)) {
            extThreadLoad = boolVal;
            HError(-8921, "InitNCache: EXTRATHREADLOADING is disabled");
        }
    }

    /* initialise the stacks */
    CreateHeap(&pbufStack, "pbufStore", MSTAK, 1, 0.5, 1000, 10000);
    CreateHeap(&transStack, "labStore", MSTAK, 1, 0.5, 1000, 10000);
    /*CreateHeap(&latHeap, "latStore", CHEAP, 1, 0, 1000000, 10000000);*/

    if (TRUE) {
            /* GPU/MKL/CPU */              /* discard: should be set when compiling */
            /* THREADS */
            /* SGD/HF */
            /* LEARNING RATE SCHEDULE */
            /*     RELATED STUFFS */
    }
}

/* use to create a cache */
/* if labelInfo == NULL, then no label is available */
DataCache *CreateCache(MemHeap *heap, FILE *scpFile, int scpCnt, HMMSet *hset, Observation *obs, int streamIdx, size_t cacheSamples, VisitKind visitKind, XFInfo *xfInfo, LabelInfo *labelInfo, Boolean saveUttName) {
    int i;
    DataCache *cache;

    /* check the data heap type */
    if (heap->type != CHEAP) {
        HError(8992, "CreateCache: Only CHEAP is supported for cache");
    }
    /* check the correctness of the cache */
    if (obs == NULL) {
        HError(8900, "CreateCache: NULL observation pointer");
    }
    if ((obs->pk & BASEMASK) == DISCRETE || (obs->pk & HASVQ)) {
        HError(8901, "CreateCache: VQ feature is not supported");
    }
    if (streamIdx <= 0 || streamIdx > obs->swidth[0]) {
        HError(8990, "CreateCache: Illegal stream index");
    }
    /* initiate the cache structure first */
    cache = (DataCache *) New(heap, sizeof(DataCache));
    /* 0. set cmem and revisit */
    cache->cmem = heap;
    cache->revisit = FALSE;
    /* 1. set frmDim */
    cache->frmDim = obs->swidth[streamIdx];
    /* 2. set cacheSamples */
    cache->cacheSamples = MAX(cacheSamples, defaultCacheSamples);
    /* 3. set tUttNum */
    cache->tUttNum = scpCnt;
    /* 4. set nxtUttPos */
    cache->nxtUttPos = 0;    
    /* 5. initialise uttElems */
    cache->uttElems = (UttElem *) New(cache->cmem, cache->tUttNum * sizeof(UttElem));
    memset(cache->uttElems, 0, cache->tUttNum * sizeof(UttElem));
    /* 6. set visitKind */
    cache->visitKind = visitKind;
    /* 7. stUttPos == edUttPos means nothing loaded yet */
    cache->stUttPos = 0;
    cache->edUttPos = 0;
    /* 8. initialise uttOrder */
    cache->uttOrder = (int *) New(cache->cmem, cache->tUttNum * sizeof(int));
    /* 9. set frmNum and tFrmNum */
    cache->frmNum = 0;
    cache->tFrmNum = 0;
    /* 10. initialise frmOrder */
    cache->fvLen = 0;
    cache->frmOrder = NULL;
    /* 11. set order ptr */
    /*cache->orderPtr = -1;*/
    cache->orderPtr = 0;
    /* 12. initialise the frmPtrs */
    cache->batchSamples = GetNBatchSamples();
    if (visitKind == FRMVK) {
        cache->ptrNum = 0;
    }
    else if (visitKind == PLNONEVK || visitKind == PLUTTVK || visitKind == PLUTTFRMVK) {
        /* TODO: add some safety check for batch and cache size */
        cache->ptrNum = cache->batchSamples;
    }
    else {
        cache->ptrNum = 1;
    }
    cache->frmPtrs = (FrmIndex *) New(cache->cmem, cache->ptrNum * sizeof(FrmIndex));
    for (i = 0; i < cache->ptrNum; ++i) {
        cache->frmPtrs[i].uttIdx = -1;
        cache->frmPtrs[i].frmIdx = -1;
    }
    /* 13. basic label structure */
    /*cache->labelInfo = labelInfo;*/
    if (labelInfo != NULL) {
        cache->labelInfo = (LabelInfo *) New(cache->cmem, sizeof(LabelInfo));
        memcpy(cache->labelInfo, labelInfo, sizeof(LabelInfo));
    }
    else {
        cache->labelInfo = NULL;
    }
    /* 14. initialise batLen and frmBatch */
    cache->batLen = 0;
    cache->frmBatch = (FrmIndex *) New(cache->cmem, cache->batchSamples * sizeof(FrmIndex));
    /* cz277 - semi */
    /*if (visitKind == PLNONEVK || visitKind == PLUTTVK || visitKind == PLUTTFRMVK) {
        cache->CMDVecPL = (int *) New(cache->cmem, cache->batchSamples * sizeof(int));
    }
    else {
        cache->CMDVecPL = NULL;
    }*/
    if (visitKind == FRMVK) {
        cache->CMDVecPL = NULL;
    }
    else {
        cache->CMDVecPL = (int *) New(cache->cmem, cache->batchSamples * sizeof(int));
    }

    /* set the auxiliary structures */
    /* 1. set script file */
    cache->scpFile = scpFile;
    /* 2. set the HMMSet pointer */
    cache->hmmSet = hset;
    /* 3. initilise parmBuf */
    /*cache->parmBuf = NULL;*/
    /* 4. set obs */
    cache->obs = obs;
    /* 5. set streamIdx */
    cache->streamIdx = streamIdx;
    /* set outLayer and labMat */
    cache->outLayer = hset->annSet->outLayers[streamIdx];
    /*cache->labMatMapSum = NULL;*/
    if ((labelInfo != NULL) && (labelInfo->labelKind & FEALK) != 0) {
        cache->labVec = NULL;
        cache->labMat = cache->outLayer->trainInfo->labMat;
    }
    else if ((labelInfo != NULL) && (labelInfo->labelKind & LABLK) != 0) {
        cache->labVec = CreateIntVec(&gcheap, cache->batchSamples);
        cache->labMat = cache->outLayer->trainInfo->labMat;
    }
    else {
        cache->labVec = NULL;
        cache->labMat = NULL;
    }
    /* 6. set saveUttName */
    /*if (cache->visitKind != NONEVK && cache->visitKind != UTTFRMVK && cache->visitKind != UTTVK && saveUttName)
        HError(9999, "CreateCache: SaveUttName is only possible to be true for UTT series");*/
    cache->saveUttName = saveUttName;
    /* 7. set xfInfo */
    cache->xfInfo = xfInfo;

    return cache;
}

/* update the cache, when doing HMMSet reload */
void ResetCacheHMMSetCfg(DataCache *cache, HMMSet *hset) {
    if (cache->hmmSet != hset) 
        HError(8990, "ResetCacheConfig: New hset address does not equal to previous");
    /* reset hmmSet */
    /* cache->hmmSet = hset; */
    /* reset outLayer */
    cache->outLayer = hset->annSet->outLayers[cache->streamIdx];
    /* reset labMat */
    cache->labMat = cache->outLayer->trainInfo->labMat;
}

/* A function to release all current loaded utterances */
static void CleanCache(DataCache *cache) {
    int i;

    for (i = cache->stUttPos; i < cache->edUttPos; ++i) 
        UnloadOneUtt(cache, i);
}

/* reset to reuse the cache */
void ResetCache(DataCache *cache) {
    /* set revisit */
    cache->revisit = TRUE;
    /* release the rest loaded utterances */
    if (need2Unload) {
        CleanCache(cache);
    }
    /* reset frame items */
    if (cache->frmOrder != NULL) {
        Dispose(cache->cmem, cache->frmOrder);
        cache->frmOrder = NULL;
        cache->fvLen = 0;
    }
    if (need2Unload) {
        cache->frmNum = 0;
        cache->tFrmNum = 0;
    }
    cache->batLen = 0;
    /* reset utterance items */
    if (need2Unload) {
        cache->nxtUttPos = 0;
    }
    /*memset(cache->uttElems, 0, cache->tUttNum * sizeof(UttElem));*/
    cache->stUttPos = 0;
    cache->edUttPos = 0;
    /* reset file handler */
    if (need2Unload) {
        rewind(cache->scpFile);
    }
    /* initialise the cache */
    /*InitCache(cache);*/
}

/* A function to release the whole cache */
void FreeCache(DataCache *cache) {
    /* release the rest loaded utterances */
    CleanCache(cache);
    /* release uttElems */
    Dispose(cache->cmem, cache->uttElems);
    /* release uttOrder */
    Dispose(cache->cmem, cache->uttOrder);
    /* release frmPtrs */
    Dispose(cache->cmem, cache->frmPtrs);
    /* release frmBatch */
    Dispose(cache->cmem, cache->frmBatch);
    /* cz277 - semi, release CMDVecPL */
    if (cache->CMDVecPL != NULL) {
        Dispose(cache->cmem, cache->CMDVecPL);
    }
    /* release labelInfo */
    if (cache->labelInfo != NULL) {
        Dispose(cache->cmem, cache->labelInfo);
    }
}

/* cz277 - 150811 */
char *MakeNameNMatRPL(char *curSpkr, char *tgtMacro, char *RPLName) {
    char buf[MAXSTRLEN];

    strcpy(buf, tgtMacro);
    strcat(buf, "+");
    strcat(buf, curSpkr);
    strcpy(RPLName, buf);

    return RPLName;
}

/* cz277 - 150811 */
char *MakeNameNVecRPL(char *curSpkr, char *tgtMacro, char *RPLName) {
    char buf[MAXSTRLEN];

    strcpy(buf, tgtMacro);
    strcat(buf, "+");
    strcat(buf, curSpkr);
    strcpy(RPLName, buf);

    return RPLName;
}

/* cz277 - xform */
static void LoadOneUttNMatRPLs(HMMSet *hset, UttElem *uttElem) {
    int i;
    Boolean maskMatch;
    char curSpkr[MAXSTRLEN], curMacroName[MAXSTRLEN], curFN[MAXSTRLEN];
    RILink curRPLInfo;
    NMatBundle *bundle;

    /* handle input replaceable parts */
    i = 0;
    curRPLInfo = GetHeadNMatRPLInfo();
    while (curRPLInfo != NULL) {
        maskMatch = MaskMatch(curRPLInfo->inRPLMask, curSpkr, uttElem->uttName);
        if (maskMatch == FALSE)
            HError(8919, "LoadOneUttNMatRPLs: %s does not match %s", curRPLInfo->inRPLMask, uttElem->uttName);
        MakeNameNMatRPL(curSpkr, curRPLInfo->id->name, curRPLInfo->cacheInSpkr); 
        MakeFN(curRPLInfo->cacheInSpkr, NULL, curRPLInfo->inRPLExt, curMacroName);
        MakeFN(curSpkr, NULL, curRPLInfo->inRPLExt, curFN);
        bundle = LoadOneNMatRPL(hset, curRPLInfo->inRPLDir, curFN, curMacroName);
        if (bundle == NULL)
            HError(8919, "LoadOneUttNMatRPLs: %s does not have replaceable matrix %s", uttElem->uttName, curMacroName);
        AugHostNMatBundleByNMatBundle(hset->hmem, curRPLInfo->curNMat, bundle);
        /*uttElem->curUttNMatRPLs[i] = LoadOneNMatRPL(hset, curRPLInfo->inRPLDir, curFN, curMacroName);*/
        uttElem->curUttNMatRPLs[i] = bundle;
        if (uttElem->curUttNMatRPLs[i]->variables->rowNum != curRPLInfo->curNMat->variables->rowNum || 
            uttElem->curUttNMatRPLs[i]->variables->colNum != curRPLInfo->curNMat->variables->colNum)
            HError(8922, "LoadOneUttNMatRPLs: %s matrix dim inconsistent %s", uttElem->uttName, curMacroName);
        ++i;
        curRPLInfo = curRPLInfo->nextInfo;
    } 
}

static void LoadOneUttNVecRPLs(HMMSet *hset, UttElem *uttElem) {
    int i;
    Boolean maskMatch;
    char curSpkr[MAXSTRLEN], curMacroName[MAXSTRLEN], curFN[MAXSTRLEN];
    RILink curRPLInfo;
    NVecBundle *bundle;

    /* handle input replaceable parts */
    i = 0;
    curRPLInfo = GetHeadNVecRPLInfo();
    while (curRPLInfo != NULL) {
        maskMatch = MaskMatch(curRPLInfo->inRPLMask, curSpkr, uttElem->uttName);
        if (maskMatch == FALSE)
            HError(8919, "LoadOneUttNVecRPLs: %s does not match %s", curRPLInfo->inRPLMask, uttElem->uttName);
        MakeNameNVecRPL(curSpkr, curRPLInfo->id->name, curRPLInfo->cacheInSpkr);
        MakeFN(curRPLInfo->cacheInSpkr, NULL, curRPLInfo->inRPLExt, curMacroName);
        MakeFN(curSpkr, NULL, curRPLInfo->inRPLExt, curFN);
        bundle = LoadOneNVecRPL(hset, curRPLInfo->inRPLDir, curFN, curMacroName);
        if (bundle == NULL)
            HError(8919, "LoadOneUttNVecRPLs: %s does not have replaceable vector %s", uttElem->uttName, curMacroName);
        AugHostNVecBundleByNVecBundle(hset->hmem, curRPLInfo->curNVec, bundle);
        /*uttElem->curUttNVecRPLs[i] = LoadOneNVecRPL(hset, curRPLInfo->inRPLDir, curFN, curMacroName);*/
        uttElem->curUttNVecRPLs[i] = bundle;
        if (uttElem->curUttNVecRPLs[i]->variables->vecLen != curRPLInfo->curNVec->variables->vecLen)
            HError(8922, "LoadOneUttNVecRPLs: %s matrix dim inconsistent %s", uttElem->uttName, curMacroName);
        ++i;
        curRPLInfo = curRPLInfo->nextInfo;
    }
}


/* cz277 - xform */
void SaveAllNMatRPLs(HMMSet *hset, FILE *script) {
    Boolean maskMatch;
    char uttName[MAXSTRLEN], curMacroName[MAXSTRLEN], curFN[MAXSTRLEN];
    RILink curRPLInfo;
    NMatBundle *bundle;

    curRPLInfo = GetHeadNMatRPLInfo();
    while (curRPLInfo != NULL) {
        strcpy(curRPLInfo->cacheInSpkr, "");
        curRPLInfo = curRPLInfo->nextInfo;
    }
    /* for each word in the scp file, output when speaker change happens */
    rewind(script);
    while (GetNextScpWord(script, uttName) != NULL) {
        /* output the replaceable matrices */
        curRPLInfo = GetHeadNMatRPLInfo();
        while (curRPLInfo != NULL) {
            maskMatch = MaskMatch(curRPLInfo->inRPLMask, curRPLInfo->curInSpkr, uttName);
            if (maskMatch == FALSE)
                HError(8919, "SaveAllNMatRPLs: Input mask %s does not match the name %s", curRPLInfo->inRPLMask, uttName);
            if (strcmp(curRPLInfo->cacheInSpkr, curRPLInfo->curInSpkr) == 0) {
                MakeNameNMatRPL(curRPLInfo->curInSpkr, curRPLInfo->id->name, curMacroName);
                bundle = LoadOneNMatRPL(hset, NULL, NULL, curMacroName);
                MakeFN(curRPLInfo->curInSpkr, curRPLInfo->outRPLDir, curRPLInfo->outRPLExt, curFN);
                SaveOneNMatRPL(hset, bundle, curFN, curRPLInfo->saveBinary);
                strcpy(curRPLInfo->cacheInSpkr, curRPLInfo->curInSpkr);
            }
            curRPLInfo = curRPLInfo->nextInfo;
        } 
    }
}

void SaveAllNVecRPLs(HMMSet *hset, FILE *script) {
    Boolean maskMatch;
    char uttName[MAXSTRLEN], curMacroName[MAXSTRLEN], curFN[MAXSTRLEN];
    RILink curRPLInfo;
    NVecBundle *bundle;

    curRPLInfo = GetHeadNVecRPLInfo();
    while (curRPLInfo != NULL) {
        strcpy(curRPLInfo->cacheInSpkr, "");
        curRPLInfo = curRPLInfo->nextInfo;
    }
    /* for each word in the scp file, output when speaker change happens */
    rewind(script);
    while (GetNextScpWord(script, uttName) != NULL) {
        /* output the replaceable matrices */
        curRPLInfo = GetHeadNVecRPLInfo();
        while (curRPLInfo != NULL) {
            maskMatch = MaskMatch(curRPLInfo->inRPLMask, curRPLInfo->curInSpkr, uttName);
            if (maskMatch == FALSE)
                HError(8919, "SaveAllNVecRPLs: Input mask %s does not match the name %s", curRPLInfo->inRPLMask, uttName);
            if (strcmp(curRPLInfo->cacheInSpkr, curRPLInfo->curInSpkr) == 0) {
                MakeNameNVecRPL(curRPLInfo->curInSpkr, curRPLInfo->id->name, curMacroName);
                bundle = LoadOneNVecRPL(hset, NULL, NULL, curMacroName);
                MakeFN(curRPLInfo->curInSpkr, curRPLInfo->outRPLDir, curRPLInfo->outRPLExt, curFN);
                SaveOneNVecRPL(hset, bundle, curFN, curRPLInfo->saveBinary);
                strcpy(curRPLInfo->cacheInSpkr, curRPLInfo->curInSpkr);
            }
            curRPLInfo = curRPLInfo->nextInfo;
        }
    }
}

/* cz277 - xform */
void InstallOneUttNMatRPLs(UttElem *uttElem) {
    int i;
    RPLInfo *curRPLInfo;
    char curSpkr[MAXSTRLEN], curFN[MAXSTRLEN];

    i = 0;
    curRPLInfo = GetHeadNMatRPLInfo();
    while (curRPLInfo != NULL) {
        MaskMatch(curRPLInfo->inRPLMask, curSpkr, uttElem->uttName);
        MakeFN(curSpkr, curRPLInfo->inRPLDir, curRPLInfo->inRPLExt, curFN);
        if (trace & T_RPL)
            printf("Using replaceable matrix for %s from %s\n", curRPLInfo->id->name, curFN);
        SetNMatBundleByNMatBundle(uttElem->curUttNMatRPLs[i], curRPLInfo->curNMat);
        curRPLInfo = curRPLInfo->nextInfo;
        ++i;
    }
}

void InstallOneUttNVecRPLs(UttElem *uttElem) {
    int i;
    RPLInfo *curRPLInfo;
    char curSpkr[MAXSTRLEN], curFN[MAXSTRLEN];

    i = 0;
    curRPLInfo = GetHeadNVecRPLInfo();
    while (curRPLInfo != NULL) {
        MaskMatch(curRPLInfo->inRPLMask, curSpkr, uttElem->uttName);
        MakeFN(curSpkr, curRPLInfo->inRPLDir, curRPLInfo->inRPLExt, curFN);
        if (trace & T_RPL)
            printf("Using replaceable vector for %s from %s\n", curRPLInfo->id->name, curFN);
        SetNVecBundleByNVecBundle(uttElem->curUttNVecRPLs[i], curRPLInfo->curNVec);
        curRPLInfo = curRPLInfo->nextInfo;
        ++i;
    }
}

void ResetNMatRPL() {
    RILink curRPLInfo;

    curRPLInfo = GetHeadNMatRPLInfo();
    while (curRPLInfo != NULL) {
        if (curRPLInfo->curNMat->variables->matElems != curRPLInfo->savNMat.variables->matElems) 
            SetNMatBundleByNMatBundle(&curRPLInfo->savNMat, curRPLInfo->curNMat);
        curRPLInfo = curRPLInfo->nextInfo;
    }
}

void ResetNVecRPL() {
    RILink curRPLInfo;

    curRPLInfo = GetHeadNVecRPLInfo();
    while (curRPLInfo != NULL) {
        if (curRPLInfo->curNVec->variables->vecElems != curRPLInfo->savNVec.variables->vecElems)
            SetNVecBundleByNVecBundle(&curRPLInfo->savNVec, curRPLInfo->curNVec);
        curRPLInfo = curRPLInfo->nextInfo;
    }
}

/* load one utterance into cache, at dstPos (usually nxtUttPos) */
static ReturnStatus LoadOneUtt(DataCache *cache, int dstPos) {
    int i, j, len, dim, sIdx, lsIdx = 0, transcnt;	/* cz277 - trans */
    long stIdx, edIdx, curIdx;
    char feaBuf[MAXSTRLEN], spkBuf[MAXSTRLEN], labBuf[MAXSTRLEN], hmmBuf[MAXSTRLEN];
    char fnBuf[MAXSTRLEN], pathBuf[MAXSTRLEN], latBuf[MAXSTRLEN];
    FILE *filePtr;
    Boolean isPipe, isPhoneLab;	/* cz277 - trans */
    Observation *obs;
    UttElem *uttElem;
    float *dstPtr;
    Vector x;
    Transcription *trans;
    LLink llink;
    MLink macDef;
    StreamElem *streamElem;
    HLink hlink;
    ParmBuf parmBuf;
    BufferInfo pbInfo;
    LabId hmmId = NULL, lhmmId;	/* cz277 - trans */
    Vector tmpVec;
    /* cz277 - trans */
    TrAcc *ta;

    /* check the destinate position */
    if (dstPos >= cache->tUttNum) 
        return FAIL;
    /* if no word to read in the script */
    if (GetNextScpWord(cache->scpFile, feaBuf) == NULL) 
        HError(8913, "LoadOneUtt: Fail to read the next word in the script");
    /* monitor speaker change */
    if (cache->xfInfo != NULL) 
        UpdateSpkrStats(cache->hmmSet, cache->xfInfo, feaBuf);
    /* reset the heap for the next utterance */
    ResetHeap(&pbufStack);
    /* open the next utterance */
    parmBuf = OpenBuffer(&pbufStack, feaBuf, 0, UNDEFF, TRI_UNDEF, TRI_UNDEF);
    if (!parmBuf) 
        HError(8910, "LoadOneUtt: Open input data failed");
    /* prepare to load the data */
    uttElem = &cache->uttElems[dstPos];
    if (cache->saveUttName) 
        uttElem->uttName = CopyString(cache->cmem, feaBuf);
    uttElem->uttLen = ObsInBuffer(parmBuf);
    uttElem->frmUsed = 0; /* the usage counter of the frames */
    /* load all frames into cache */
    len = uttElem->uttLen;
    dim = cache->frmDim;
    obs = cache->obs;
    uttElem->frmMat = (float *) New(cache->cmem, len * dim * sizeof(float));
    dstPtr = uttElem->frmMat;
    for (i = 0; i < len; ++i) {
        ReadAsTable(parmBuf, i, obs);
        x = obs->fv[cache->streamIdx];
        /* TODO: apply compFX form */
        /* ApplyCompXForm ??? */
        for (j = 1; j <= dim; ++j, ++dstPtr) {   /* just copy the data */
            if (isnan(x[j]))
                HError(8923, "LoadOneUtt: %s, frame no. %d, dim %d has nan value", uttElem->uttName, i, j);
            if (isinf(x[j]))
                HError(8923, "LoadOneUtt: %s, frame no. %d, dim %d has inf value", uttElem->uttName, i, j);
            *dstPtr = x[j];
        }
    }
    /* cz277 - aug */
    /* load the augmented feature vectors */
    for (i = 1; i <= MAXAUGFEAS; ++i) {
        uttElem->augFeaVec[i] = NULL;
        if (cache->streamIdx == 1) {
            tmpVec = GetAugFeaVector(parmBuf, i);
            if (tmpVec != NULL) {
                uttElem->augFeaVec[i] = CreateVector(cache->cmem, VectorSize(tmpVec));
                CopyVector(tmpVec, uttElem->augFeaVec[i]);
            }
        }
    }
    
    /* close current buffer */
    CloseBuffer(parmBuf);
    /* set frmOrder */
    if (cache->visitKind == PLUTTFRMVK || cache->visitKind == UTTFRMVK) {
        /* initialise */
        uttElem->frmOrder = (int *) New(cache->cmem, len * sizeof(int));
        for (i = 0; i < len; ++i) 
            uttElem->frmOrder[i] = i;
        /* shuffle the order array */
        ShuffleSegment(uttElem->frmOrder, 0, len, sizeof(int));
    }
    else 
        uttElem->frmOrder = NULL;
    /* update the frame count in cache */
    cache->frmNum += len;
    cache->tFrmNum += len;
    /* save the xforms */
    /*if (cache->xfInfo != NULL && cache->xfInfo->inXForm != NULL) {
        uttElem->inXForm = (AdaptXForm *) New(cache->cmem, sizeof(AdaptXForm));
        memcpy(uttElem->inXForm, cache->xfInfo->inXForm, sizeof(AdaptXForm));
    }
    else {
        uttElem->inXForm = NULL;
    }
    if (cache->xfInfo != NULL && cache->xfInfo->paXForm != NULL) {
        uttElem->paXForm = (AdaptXForm *) New(cache->cmem, sizeof(AdaptXForm));
        memcpy(uttElem->paXForm, cache->xfInfo->paXForm, sizeof(AdaptXForm));
    }
    else {
        uttElem->paXForm = NULL;
    }*/
    /* cz277 - xform */
    uttElem->inXForm = cache->xfInfo->inXForm;
    uttElem->paXForm = cache->xfInfo->paXForm;
    /* cz277 - 150811 */
    uttElem->curUttNMatRPLs = NULL;
    uttElem->curUttNVecRPLs = NULL;
    i = GetNumNMatRPLInfo();
    j = GetNumNVecRPLInfo();
    if (i > 0) {
        uttElem->curUttNMatRPLs = (NMatBundle **) New(cache->cmem, i * sizeof(NMatBundle *));
        memset(uttElem->curUttNMatRPLs, 0, i * sizeof(NMatBundle *));
        LoadOneUttNMatRPLs(cache->hmmSet, uttElem); 
    }
    if (j > 0) {
        uttElem->curUttNVecRPLs = (NVecBundle **) New(cache->cmem, j * sizeof(NVecBundle *));
        memset(uttElem->curUttNVecRPLs, 0, j * sizeof(NVecBundle *));
        LoadOneUttNVecRPLs(cache->hmmSet, uttElem);
    }

    /* set the labels */
    if (cache->labelInfo != NULL) {
        /* process feature files */
        uttElem->flabMat = NULL;
        if ((cache->labelInfo->labelKind & FEALK) != 0) {	/* load feature files */
            if (GetNextScpWord(cache->labelInfo->scpFLab, labBuf) == NULL)
                HError(8913, "LoadOneUtt: Fail to acquire next feature type label path"); 
            parmBuf = OpenBuffer(&pbufStack, labBuf, 0, UNDEFF, TRI_UNDEF, TRI_UNDEF);
            if (!parmBuf)
                HError(8910, "LoadOneUtt: Open feature type label data failed");
            if (uttElem->uttLen != ObsInBuffer(parmBuf))
                HError(8924, "LoadOneUtt: Inconsistent utterance lengths %s vs %s", feaBuf, labBuf);
            dim = cache->labelInfo->dimFLab;
            uttElem->flabMat = (float *) New(cache->cmem, len * dim * sizeof(float));
            dstPtr = uttElem->flabMat;
            obs = cache->labelInfo->obsFLab;
            for (i = 0; i < len; ++i) {
                ReadAsTable(parmBuf, i, obs);
                x = obs->fv[cache->streamIdx]; 
                for (j = 1; j <= dim; ++j, ++dstPtr) {
                    if (isnan(x[j]))
                        HError(8923, "LoadOneUtt: %s, frame no. %d, dim %d has nan value", labBuf, i, j);
                    if (isinf(x[j]))
                        HError(8923, "LoadOneUtt: %s, frame no. %d, dim %d has inf value", labBuf, i, j);
                    *dstPtr = x[j];
                }
            }
            CloseBuffer(parmBuf);
        }
        /* process lab files */
        uttElem->labIdxes = NULL;
        if ((cache->labelInfo->labelKind & LABLK) != 0) { /* load lab files */
            if (cache->labelInfo->labFileMask != NULL) {
                if (!MaskMatch(cache->labelInfo->labFileMask, spkBuf, feaBuf)) {
                    HError(8919, "LoadOneUtt: Mask %s has no match with segment %s", cache->labelInfo->labFileMask, feaBuf);
                }
                MakeFN(spkBuf, cache->labelInfo->labDir, cache->labelInfo->labExt, labBuf);
            }
            else {
                MakeFN(feaBuf, cache->labelInfo->labDir, cache->labelInfo->labExt, labBuf);
            }
            ResetHeap(&transStack);
            trans = LOpen(&transStack, labBuf, UNDEFF);
            /* convert trans to labIdxes */
            uttElem->labIdxes = (int *) New(cache->cmem, len * sizeof(int));
            i = 0;
            for (llink = trans->head->head->succ; llink->succ != NULL; llink = llink->succ) {
                /* cz277 - trans */
		lhmmId = hmmId;
                lsIdx = sIdx;
                
                ExtractState(llink->labid->name, hmmBuf, &sIdx);    /* support hmm state by hacking labid */
                if (sIdx == 0) {
                    hmmId = llink->labid;
                    /* cz277 - trans */
                    isPhoneLab = TRUE;
                }
                else {
                    hmmId = GetLabId(hmmBuf, FALSE);
                    /* cz277 - trans */
                    isPhoneLab = FALSE;
                }
                if (hmmId == NULL) {
                    HError(8925, "LoadOneUtt: Failed to find model for label \"%s\" given in the input MLF file", hmmBuf);
                }
                if ((macDef = FindMacroName(cache->hmmSet, 'l', hmmId)) == NULL) {
                    HError(8925, "LoadOneUtt: Unknown label %s", hmmId->name);
                }
                hlink = (HLink) macDef->structure;
                if (sIdx == 0) {    /* if it is a phone label */
                    /* check whether all states of that hmm share the same ANN target */
                    for (j = 3; j < hlink->numStates; ++j) {
                        if ((hlink->svec[2].info->pdf[cache->streamIdx].targetSrc != hlink->svec[j].info->pdf[cache->streamIdx].targetSrc) ||
                            (hlink->svec[2].info->pdf[cache->streamIdx].targetIdx != hlink->svec[j].info->pdf[cache->streamIdx].targetIdx)) {
                            HError(8925, "LoadOneUtt: Phone label in the label file does not match the state level definition");
                        }
                    }
                    sIdx = 2;
                }
                else if (sIdx <= 1 || sIdx >= hlink->numStates) {
                    HError(8925, "LoadOneUtt: Illegal state index in the label file");
                }
                /* get the stream element pointer */
                streamElem = &hlink->svec[sIdx].info->pdf[cache->streamIdx]; 
                /* check the output layer source */
                if (streamElem->targetSrc != cache->hmmSet->annSet->outLayers[cache->streamIdx]) {
                    HError(8926, "LoadOneUtt: Only one output layer is allowed in one stream");
                }
                /* get the frame indexes */
                GetBufferInfo(parmBuf, &pbInfo);
                stIdx = (long) (llink->start / pbInfo.tgtSampRate + 0.5);
                edIdx = (long) (llink->end / pbInfo.tgtSampRate + 0.5);
                if (stIdx > edIdx) {
                    HError(8927, "LoadOneUtt: Empty segment");
                }
                /* cz277 - trans */
                if ((cache->labelInfo->uFlags & UPTRANS) != 0) {
                    ta = (TrAcc *) GetHook(hlink->transP);
                    if (isPhoneLab) {
                        transcnt = (int) ((edIdx - stIdx) / (hlink->numStates - 2.0) + 0.5);
                        transcnt -= 1;
                        lsIdx = 1;
                        for (j = 2; j <= hlink->numStates; ++j) {
                            ta->occ[lsIdx] += 1;
                            ta->tran[lsIdx][j] += 1;
                            if (transcnt > 0) {
                                ta->occ[j] += transcnt;
                                ta->tran[j][j] += transcnt;
                            }
                            lsIdx = j;
                        }
                    }
                    else {
                        if (lhmmId != hmmId) {
                            lsIdx = 1;
                        }
                        ta->occ[lsIdx] += 1;
                        ta->tran[lsIdx][sIdx] += 1;
			transcnt = edIdx - stIdx - 1;
                        ta->occ[sIdx] += transcnt;
                        if (transcnt > 0) {
                            ta->tran[sIdx][sIdx] += transcnt;
                        }
                    }
                }

                if ((cache->labelInfo->uFlags & UPTARGETPEN) != 0) {
                    streamElem->occAcc += edIdx - stIdx;
                }
                for (curIdx = stIdx; curIdx < edIdx; ++curIdx, ++i) {
                    if (curIdx != i) {
                        HError(8927, "LoadOneUtt: Discontinuous Utterance");
                    }
                    uttElem->labIdxes[i] = streamElem->targetIdx - 1;   /* targetIdx >= 1, labIdxes >= 0 */
                }
            }
            if (i != uttElem->uttLen) {
                HError(8924, "LoadOneUtt: %s Feature and Utterance lengths (%i, %i) do not match", uttElem->uttName, i, uttElem->uttLen);
            }
        }
        /* process lattice files */
        if (((cache->labelInfo->labelKind & LATLK) != 0) && (cache->streamIdx == 1)) { /* load lattice files */
            /*CreateHeap(&uttElem->latStack, "latStore", MSTAK, 1, 1.0, 5000, 500000);*/
            uttElem->numInDen = NULL;
            for (i = 0; i < MAXLATSUTT; ++i) {
                uttElem->denLats[i] = NULL;
                uttElem->numLats[i] = NULL;
            }
            /* set basic lattice file mask */
            if (cache->labelInfo->latFileMask != NULL) {
                if (!MaskMatch(cache->labelInfo->latFileMask, latBuf, feaBuf)) {
                    HError(8928, "LoadOneUtt: Mask %s has no match with segment %s", cache->labelInfo->latFileMask, feaBuf);
                }
            }
            else {
                strcpy(latBuf, feaBuf);
            }
            /* load denorminator lattices */
            if (cache->labelInfo->nDenLats > 0) {
                for (i = 0; i < cache->labelInfo->nDenLats; ++i) {
                    if (cache->labelInfo->denLatSubDirPat[0]) {
                        if (!MaskMatch(cache->labelInfo->denLatSubDirPat, spkBuf, latBuf)) {
                            HError(8928, "LoadOneUtt: Mask %s has not match with segment %s", cache->labelInfo->denLatSubDirPat, latBuf);
                        }
                        MakeFN(spkBuf, cache->labelInfo->denLatDir[i], NULL, pathBuf);
                    }
                    else {
                        strcpy(pathBuf, cache->labelInfo->denLatDir[i]);
                    }
                    if (cache->labelInfo->latMaskDen != NULL) {
                        if (!MaskMatch(cache->labelInfo->latMaskDen, spkBuf, latBuf)) {
                            HError(8928, "LoadOneUtt: Mask %s has not match with segment %s", cache->labelInfo->latMaskDen, latBuf);
                        }
                        MakeFN(spkBuf, pathBuf, NULL, fnBuf);
                        strcpy(pathBuf, fnBuf); 
                    }
                    if (cache->labelInfo->useLLF) {
                        uttElem->denLats[i] = GetLattice(latBuf, pathBuf, cache->labelInfo->latExt, cache->cmem, cache->labelInfo->vocab, FALSE, TRUE);    
                        /*GetLatticeNoRet(latBuf, pathBuf, cache->labelInfo->latExt, cache->cmem, cache->labelInfo->vocab, FALSE, TRUE, &uttElem->denLats[i]);*/
                    }
                    else {
                        MakeFN(latBuf, pathBuf, cache->labelInfo->latExt, fnBuf);
                        filePtr = FOpen(fnBuf, NetFilter, &isPipe);
                        if (!filePtr) {
                            HError(8910, "LoadOneUtt: Could not open file %s", fnBuf);
                        }
                        /* printf("Reading lattice from file: %s\n", fnBuf); fflush(stdout); */
                        uttElem->denLats[i] = ReadLattice(filePtr, cache->cmem, cache->labelInfo->vocab, FALSE, TRUE);
                        /*uttElem->denLats[i] = ReadLattice(filePtr, &uttElem->latStack, cache->labelInfo->vocab, FALSE, TRUE);*/
                        FClose(filePtr, isPipe);
                    }
                }
            }
            /* load numerator lattices */
            if (cache->labelInfo->nNumLats > 0) {
                uttElem->numInDen = (Boolean *) New(cache->cmem, cache->labelInfo->nNumLats * sizeof(Boolean));
                /*uttElem->numInDen = (Boolean *) New(&uttElem->latStack, cache->labelInfo->nNumLats * sizeof(Boolean));*/
                /*memset(uttElem->numInDen, 1, cache->labelInfo->nNumLats * sizeof(Boolean));*/
                for (i = 0; i < cache->labelInfo->nNumLats; ++i) {
                    if (cache->labelInfo->numLatSubDirPat[0]) {
                        if (!MaskMatch(cache->labelInfo->numLatSubDirPat, spkBuf, latBuf)) {
                            HError(8928, "LoadOneUtt: Mask %s has no match with segment %s", cache->labelInfo->numLatSubDirPat, latBuf);
                        }
                        MakeFN(spkBuf, cache->labelInfo->numLatDir[i], NULL, pathBuf);
                    }
                    else {
                        strcpy(pathBuf, cache->labelInfo->numLatDir[i]);
                    }
                    if (cache->labelInfo->latMaskNum != NULL) {
                        if (!MaskMatch(cache->labelInfo->latMaskNum, spkBuf, latBuf)) {
                            HError(8928, "LoadOneUtt: Mask %s has not match with segment %s", cache->labelInfo->latMaskNum, latBuf);
                        }
                        MakeFN(spkBuf, pathBuf, NULL, fnBuf);
                        strcpy(pathBuf, fnBuf);
                    }
                    if (cache->labelInfo->useLLF) {
                        uttElem->numLats[i] = GetLattice(latBuf, pathBuf, cache->labelInfo->latExt, cache->cmem, cache->labelInfo->vocab, FALSE, TRUE);
                        /*GetLatticeNoRet(latBuf, pathBuf, cache->labelInfo->latExt, cache->cmem, cache->labelInfo->vocab, FALSE, TRUE, &uttElem->numLats[i]);*/
                    }
                    else {
                        MakeFN(latBuf, pathBuf, cache->labelInfo->latExt, fnBuf);
                        filePtr = FOpen(fnBuf, NetFilter, &isPipe);
                        if (!filePtr) {
                            HError(8910, "LoadOneUtt: Could not open file %s", fnBuf);
                        }
                        uttElem->numLats[i] = ReadLattice(filePtr, cache->cmem, cache->labelInfo->vocab, FALSE, TRUE);
                        FClose(filePtr, isPipe);
                    }
                    /* to include this num lattice as a den lattice or not */
                    uttElem->numInDen[i] = TRUE;
                    if (cache->labelInfo->incNumInDen == TRUE) {
                        for (j = 0; j < cache->labelInfo->nDenLats; ++j) {
                            if (LatInLat(uttElem->numLats[i], uttElem->denLats[j])) {
                                uttElem->numInDen[i] = FALSE;
                                break;
                            }
                        }
                    }
                    else {
                        uttElem->numInDen[i] = FALSE;
                    }
                }
            }
        }
    }

    return SUCCESS;
}

/* cz277 - mtload */
/* fill the entire cache in one go */
static int FillCacheSGT(DataCache *cache) {
    int newUttCnt = 0;

    if (cache->revisit == TRUE && need2Unload == FALSE) {
        return newUttCnt;
    }

    /* load utterance by utterance until the cache is full */
    while (cache->nxtUttPos < cache->tUttNum && cache->frmNum < cache->cacheSamples) {
        /* load the next utterance */
        if (LoadOneUtt(cache, cache->nxtUttPos++) < SUCCESS) {
            HError(8913, "FillCacheSGT: Load next utterance failed");
        }
        else {
            ++newUttCnt;
        }
    }
    return newUttCnt;
}

/* cz277 - mtload */
/*static void *FillCacheEXT(void *arg) {
    int newUttCnt = 0;
    DataCache *cache;

    cache = (DataCache *) arg;
    if (cache->revisit == TRUE && need2Unload == FALSE) 
        pthread_exit(NULL);

    while (cache->nxtUttPos < cache->tUttNum && cache->frmNum < cache->cacheSamples) {
        if (LoadOneUtt(cache, cache->nxtUttPos++) < SUCCESS) {
            HError(8913, "FillCacheEXT: Load next utterance failed");
        }
        else {
            ++newUttCnt;
        }
    }
    pthread_exit(NULL);
}*/

/* cz277 - mtload */
void LoadCacheData(DataCache *cache) {
    /*int ret;

    if (extThreadLoad == TRUE) {
        cache->firstLoad = TRUE;
        ret = pthread_create(&cache->extThread, NULL, FillCacheEXT, (void *) cache);
        if (ret != 0) {
            HError(8929, "LoadCacheData: Fail to create the extra thread for cache loading");
        }
    }
    else {
        ret = FillCacheSGT(cache);
    }*/
    FillCacheSGT(cache);

}

/* need to make ensure that all frames have been used */
static void UnloadOneUtt(DataCache *cache, int dstPos) {
    UttElem *uttElem;
    int i;

    uttElem = &cache->uttElems[dstPos];
    /* check if all frames have been used */
    if (uttElem->frmUsed != uttElem->uttLen) 
        HError(8993, "UnloadOneUtt: all frames should be used once before unload");
    /* if has been unloaded, it has no need to unload */
    if (uttElem->frmMat == NULL)
        return;
    /* update the total number of frame cached */
    cache->frmNum -= uttElem->uttLen;
    /* release the space for frame matrix */
    Dispose(cache->cmem, uttElem->frmMat);
    /*free(uttElem->frmMat);*/
    uttElem->frmMat = NULL;
    /* cz277 - aug */
    /* release the space for augmented feature vectors */
    for (i = 1; i <= MAXAUGFEAS; ++i) 
        if (uttElem->augFeaVec[i] != NULL) 
            Dispose(cache->cmem, uttElem->augFeaVec[i]);
    /* release the space for utterance name */
    if (uttElem->uttName != NULL) 
        Dispose(cache->cmem, uttElem->uttName);
    /* release the space for frame order array */
    if (uttElem->frmOrder != NULL) {
        Dispose(cache->cmem, uttElem->frmOrder);
        uttElem->frmOrder = NULL;
    }
    /* cz277 - 150811 */
    if (uttElem->curUttNMatRPLs != NULL)
        Dispose(cache->cmem, uttElem->curUttNMatRPLs);
    if (uttElem->curUttNVecRPLs != NULL)
        Dispose(cache->cmem, uttElem->curUttNVecRPLs);
    /* unload the label */
    if (cache->labelInfo != NULL) {
        if ((cache->labelInfo->labelKind & FEALK) != 0) 
            Dispose(cache->cmem, uttElem->flabMat);
        if ((cache->labelInfo->labelKind & LABLK) != 0) 
            Dispose(cache->cmem, uttElem->labIdxes);
        /* dispose the xforms */
        /*if (uttElem->inXForm != NULL) {
            Dispose(cache->cmem, uttElem->inXForm);
        }
        if (uttElem->paXForm != NULL) {
            Dispose(cache->cmem, uttElem->paXForm);
        }*/
        if ((cache->labelInfo->labelKind & LATLK) != 0) {
            /* dispose the lattices */
            for (i = 0; i < cache->labelInfo->nNumLats; ++i) 
                FreeLattice(uttElem->numLats[i]);
            for (i = 0; i < cache->labelInfo->nDenLats; ++i) 
                FreeLattice(uttElem->denLats[i]);
            if (uttElem->numInDen != NULL) 
                Dispose(cache->cmem, uttElem->numInDen);
        }
    }
}

static ReturnStatus UpdateUttOrder(DataCache *cache) {
    /* if no newly loaded utterances to be indexed */
    if (cache->edUttPos == cache->nxtUttPos) {  /* nxtUttPos stops at tUttNum */
        return FAIL;
    }
    /* update cache->stUttPos */
    cache->stUttPos = cache->edUttPos;
    /* update cache->edUttPos */
    cache->edUttPos = cache->nxtUttPos;

    /* shuffle the indexes if needed */
    if (cache->visitKind == UTTFRMVK || cache->visitKind == UTTVK || cache->visitKind == PLUTTVK || cache->visitKind == PLUTTFRMVK) {
        ShuffleSegment(cache->uttOrder, cache->stUttPos, cache->edUttPos, sizeof(int));        
    }

    return SUCCESS;
}

/* useful for FRMVK only */
/* when usinf this, the rest frames in previous cache are assumed to
   have been loaded into the mini batch, so no need to take care of 
   previous frames
*/
static ReturnStatus UpdateFrmOrder(DataCache *cache) {
    size_t i, j, k;

    /* first, make sure frame visit index is needed */
    if (cache->visitKind != FRMVK) {
        return FAIL;
    }
    /* second, get the frame visit length */
    cache->fvLen = 0;
    for (i = cache->stUttPos; i < cache->edUttPos; ++i) {
        cache->fvLen += cache->uttElems[i].uttLen;
    }
    /* third, malloc the data */
    if (cache->frmOrder != NULL) {
        Dispose(cache->cmem, cache->frmOrder);
    }
    cache->frmOrder = (FrmIndex *) New(cache->cmem, cache->fvLen * sizeof(FrmIndex));
    /* fourth, initialise frmVisit */
    for (i = cache->stUttPos, k = 0; i < cache->edUttPos; ++i) {
        for (j = 0; j < cache->uttElems[i].uttLen; ++j, ++k) {
            cache->frmOrder[k].uttIdx = i;
            cache->frmOrder[k].frmIdx = j;
        }
    }
    /* shuffle the frames in the entire index */
    ShuffleSegment(cache->frmOrder, 0, cache->fvLen, sizeof(FrmIndex));
    /* reset orderPtr */
    cache->orderPtr = 0;

    return SUCCESS;
}

/* initialise the cache, is used only once */
void InitCache(DataCache *cache) {
    int i;

    /* init cache order */
    for (i = 0; i < cache->tUttNum; ++i) {
        cache->uttOrder[i] = i;
    }
    /* cz277 - mtload */
    /* fill the buffer first */
    FillCacheSGT(cache);
    /* initialise the uttOrder */
    UpdateUttOrder(cache);
    /* init orderPtr */
    cache->orderPtr = 0;
    /* initialise frmOrder (optional) and the pointers */
    switch (cache->visitKind) {
        case FRMVK:
            UpdateFrmOrder(cache);
            break;
        case NONEVK:
        case UTTFRMVK:
        case UTTVK:
            cache->frmPtrs[0].uttIdx = cache->uttOrder[cache->orderPtr++];
            cache->frmPtrs[0].frmIdx = 0;
            break;
        case PLNONEVK:
        case PLUTTFRMVK:
        case PLUTTVK:
            for (i = 0; i < cache->ptrNum; ++i) {
                /* in case there are not enough utterances in the cache */
                if (cache->orderPtr < cache->edUttPos) {
                    cache->frmPtrs[i].uttIdx = cache->uttOrder[cache->orderPtr++];
                    cache->frmPtrs[i].frmIdx = 0;
                }
                else {
                    cache->frmPtrs[i].uttIdx = -1;
                    cache->frmPtrs[i].frmIdx = -1;
                }
            }
            break;
        default:
            HError(8991, "InitCache: Unknown visiting order");
            break;
    }
    /* cz277 - mtload */
    /*if (extThreadLoad == TRUE) {
        memset(&cache->extThread, 0, sizeof(pthread_t));
        cache->firstLoad = FALSE;
    }*/
}

/* only useful for visitKinds other than FRMVK */
/* frmIdx == cache->uttElems[uttIdx].uttLen will never happen */
/* if fail to acquire the next utterance, uttIdx == -1 */
static inline void UpdateFrmPtr(DataCache *cache, FrmIndex *frmPtr) {
    int uttIdx;

    /* set each frmPtr */
    uttIdx = frmPtr->uttIdx;   
    if (uttIdx >= 0 && (frmPtr->frmIdx + 1 < cache->uttElems[uttIdx].uttLen)) {    /* get the next frame from the same utterance */
        ++frmPtr->frmIdx;
    }
    else {  /* uttIdx < 0 || frmPtr->frmIdx + 1 == cache->uttElems[uttIdx].uttLen, need to load a new utterance */
        if (cache->orderPtr == cache->tUttNum) {    /* no available utterance */
            frmPtr->uttIdx = -1;
            frmPtr->frmIdx = -1;
        }
        else {
            if (cache->orderPtr == cache->edUttPos) {   /* need to rebuild the order */
                UpdateUttOrder(cache); 
            }
            if (cache->orderPtr < cache->edUttPos) {    /* if  UpdateUttOrder() succeeded */
                frmPtr->uttIdx = cache->uttOrder[cache->orderPtr++];
                frmPtr->frmIdx = 0;
            }
            else {  /* if no new utt in the cache, disable this frmPtr */
                frmPtr->uttIdx = -1;
                frmPtr->frmIdx = -1;
            }
        }
    }
}


/* cz277 - split */
/* copy a frame with its context expansion to form a extended frame */
static inline void CopyExtFrame2Batch(UttElem *uttElem, int curIdx, FELink feaElem, NFloat *dstPtr) {
    int i, srcIdx;
    float *srcPtr;
#ifdef DOUBLEANN
    int j;
#endif

    /* do context expansion */
    for (i = 1; i <= feaElem->ctxMap[0]; ++i) {
        /* cz277 - xform */
        /* no frame could exceed the boundary */
        /*srcIdx = curIdx + feaElem->ctxMap[i];
        if (srcIdx < 0) {
            srcIdx = 0;
        }
        else if (srcIdx >= uttElem->uttLen) {
            srcIdx = uttElem->uttLen - 1;
        }*/
        srcIdx = ClipInt(0, uttElem->uttLen - 1, curIdx + feaElem->ctxMap[i]);

        /* compute the source address */
        srcPtr = uttElem->frmMat + srcIdx * feaElem->srcDim + feaElem->dimOff;
#ifdef DOUBLEANN
        /* copy the frame vector */
        for (j = 0; j < feaElem->feaDim; ++j, ++srcPtr, ++dstPtr) {
            /*if (isnan(*srcPtr))
                HError(9999, "CopyExtFrame2Batch: %s, frame no. %d, dim %d has NaN value", uttElem->uttName, curIdx, j); 
            if (isinf(*srcPtr))
                HError(9999, "CopyExtFrame2Batch: %s, frame no. %d, dim %d has inf value", uttElem->uttName, curIdx, j);*/
            *dstPtr = *srcPtr;
            /* TODO: GPU support */
        }
#else
        memcpy(dstPtr, srcPtr, feaElem->feaDim * sizeof(float));
        dstPtr += feaElem->feaDim;
#endif
    }
}


/* fill the batch in the FRM series way */
static inline Boolean FillBatchFRM(DataCache *cache, int nSamples) {
    int i;
    Boolean finish = FALSE;

    for (i = 0; i < nSamples; ++i) {
        if (cache->orderPtr == cache->fvLen) {  /* if need to update frmOrder */
            if (cache->edUttPos == cache->tUttNum) {    /* if no more utterance available */
                break;
            }
            else {  /* load more utterance and build new frmOrder */
                if (UpdateUttOrder(cache) == SUCCESS)
                    UpdateFrmOrder(cache);
                else 
                    break;  /* will return FALSE, but cache->batLen < batchSamples  */
            }
        }
        /* copy the frame */
        memcpy(&cache->frmBatch[cache->batLen++], &cache->frmOrder[cache->orderPtr++], sizeof(FrmIndex));
        /* update the pointers */
    }
    /* if all data are finished */
    if (cache->edUttPos == cache->tUttNum && cache->orderPtr == cache->fvLen) {
        finish = TRUE;
    }

    return finish;
}

/* fill the batch in the UTT series way */
/* load a maximum *uttCnt + 1 utterances into the batch */
static inline Boolean FillBatchUTT(DataCache *cache, int nSamples, int *uttCnt) {
    int i, uttIdx, frmIdx;
    UttElem *uttElem;
    Boolean finish = FALSE;

    for (i = 0; i < nSamples; ++i) {
        /* get current uttIdx and frmIdx */
        uttIdx = cache->frmPtrs[0].uttIdx;
        frmIdx = cache->frmPtrs[0].frmIdx;
        uttElem = &cache->uttElems[uttIdx];
        /* if need to get the true frame index */
        if (uttElem->frmOrder != NULL) {
            frmIdx = uttElem->frmOrder[frmIdx];
        }
        /* cz277 - semi */
        if (cache->frmPtrs[0].frmIdx == 0) {
            cache->CMDVecPL[i] = 0;
        }
        else {
            cache->CMDVecPL[i] = -1;
        }
        /* copy the frame */
        memcpy(&cache->frmBatch[cache->batLen++], &cache->frmPtrs[0], sizeof(FrmIndex));
        /* get the FrmIndex for the next frame */
        UpdateFrmPtr(cache, &cache->frmPtrs[0]);
        /* indicate fail to acquire the next utterance or the end of the file */
        if (cache->frmPtrs[0].uttIdx < 0) {
            --(*uttCnt);
            if (cache->orderPtr == cache->tUttNum && frmIdx == uttElem->uttLen - 1) {   /* frmIdx before UpdateFrmPtr() */
                finish = TRUE;
                break;
            }
            else {
                HError(8913, "FillBatchUTT: Fail to acquire the next frame");
            }
        }
        else if (cache->frmPtrs[0].uttIdx != uttIdx) { /* switch to a new utterance */
            if (*uttCnt > 0) {
                --(*uttCnt);
            }
            if (*uttCnt == 0) {    /* if underfill is allowed and the specified number of utterances have been loaded */
                break;
            }
        }
    }

    return finish;
}

/* fill the batch in the PLUTT series way */
static inline Boolean FillBatchPLUTT(DataCache *cache, int nSamples) {
    int i, uttIdx, frmIdx, sampCnt, off;
    UttElem *uttElem;
    Boolean finish = TRUE;

    /* update the (-1, -1) pointers and generate CMDVecPL */
    for (i = cache->batLen, off = 0; i < cache->ptrNum; ++i) {
        /* try to update frmPtr[i] */
        while ((i + off < cache->ptrNum) && (cache->frmPtrs[i + off].uttIdx < 0)) {
            /* try to assign a new utterance to previous empty frmPtr */
            UpdateFrmPtr(cache, &cache->frmPtrs[i + off]);
            if (cache->frmPtrs[i + off].uttIdx < 0)
                ++off;
            else 
                break;
        }
        /* if need to shift frmPtrs */
        if (off > 0) {
            if (i + off < cache->ptrNum) {	/* cache->frmPtrs[i + off].uttIdx >= 0 */
                cache->frmPtrs[i] = cache->frmPtrs[i + off];
                cache->CMDVecPL[i] = i + off;
            }
            else {	/* if no rest pointer has uttIdx > 0 */
                for (; i < cache->ptrNum; ++i) {
                    cache->frmPtrs[i].uttIdx = -1;
                    cache->frmPtrs[i].frmIdx = -1;
                    cache->CMDVecPL[i] = -1;        /* do nothing */
                }
                break;
            }
        }
        else {	/* off == 0 */
            if (cache->frmPtrs[i].frmIdx == 0) 
                cache->CMDVecPL[i] = 0;	/* clean the context */
            else
                cache->CMDVecPL[i] = -1;	/* do nothing */
        }
    }
    /* copy to the batch */
    for (i = cache->batLen, sampCnt = 0; (i < cache->ptrNum) && (cache->frmPtrs[i].uttIdx >= 0); ++i, ++sampCnt) {
        /* get current uttIdx and frmIdx */
        uttIdx = cache->frmPtrs[i].uttIdx;
        frmIdx = cache->frmPtrs[i].frmIdx;
        /* if there's available frame by this frmPtr */
        uttElem = &cache->uttElems[uttIdx];
        /* get the true frame index */
        if (uttElem->frmOrder != NULL) {
            frmIdx = uttElem->frmOrder[frmIdx];
        }
        /* copy the frame */
        memcpy(&cache->frmBatch[cache->batLen++], &cache->frmPtrs[i], sizeof(FrmIndex));
        /* get the FrmIndex for the next frame */
        UpdateFrmPtr(cache, &cache->frmPtrs[i]);
        /* check if there's more frame in the utterance when no more utterance is available */
        if (finish && cache->frmPtrs[i].uttIdx >= 0)   /* means need one more batch (uttIdx updated in UpdateFrmPtr())*/
            finish = FALSE;
        if (sampCnt == nSamples)
            break;
    }
    if (finish) {
        for (; i < cache->ptrNum; ++i) {
            if (cache->frmPtrs[i].uttIdx >= 0) {
                finish = FALSE;
                break;
            }
        }
    }

    return finish;
}

/* cz277 - mtload */
/* update the cache status, release useless utterances and fill the cache */
int UnloadCacheData(DataCache *cache) {
    int i, rmvUttCnt = 0;
    FrmIndex *frmPtr;
    UttElem *uttElem;

    /* update the cache content */
    if (cache->batLen > 0) {
        frmPtr = &cache->frmBatch[0];
        for (i = 0; need2Unload && i < cache->batLen; ++i, ++frmPtr) {
            /* get the right UttElem */
            uttElem = &cache->uttElems[frmPtr->uttIdx];
            /* update the reference */
            ++uttElem->frmUsed;
            /* to check if the space could be released */
            if (uttElem->frmUsed == uttElem->uttLen) {
                /* could unload current utterance */
                UnloadOneUtt(cache, frmPtr->uttIdx);
                ++ rmvUttCnt;
            }
        }
        /* reset the batch length */
        cache->batLen = 0;
    }

    return rmvUttCnt;
}


/* cz277 - split */
/* fill the feature batch according to specified context map and frmBatch */
/* return the number of frame in the batch */
/*static inline int GetDataFromCacheOLD(DataCache *cache, FELink feaElem, int offset) {
    int i, uttIdx, frmIdx, extDim;
    NFloat *feaPtr;
    UttElem *uttElem;

    feaPtr = feaBat + offset;
    extDim = feaElem->extDim;
    for (i = offset; i < cache->batLen; ++i, feaPtr += extDim) {
        uttIdx = cache->frmBatch[i].uttIdx;
        frmIdx = cache->frmBatch[i].frmIdx;
        uttElem = &cache->uttElems[uttIdx];
        CopyExtFrame2Batch(uttElem, frmIdx, feaElem, feaPtr);
    }

    return cache->batLen;
}*/

/* cz277 - many */
static inline int GetDataFromCache(DataCache *cache, FELink feaElem, int offset) {
    int i, j, n, uttIdx, frmIdx, extDim;
    NFloat *feaPtr;
    UttElem *uttElem;

    n = IntVecSize(feaElem->ctxPool);
    /* set the pointers */
    extDim = feaElem->extDim;
    /* fetch each extended frame */
    for (i = 1; i <= n; ++i) {
        feaPtr = feaElem->feaMats[i]->matElems + offset;
        for (j = offset; j < cache->batLen; ++j, feaPtr += extDim) {
            uttIdx = cache->frmBatch[j].uttIdx;
            frmIdx = cache->frmBatch[j].frmIdx + feaElem->ctxPool[i];	/* cz277 - many */
            /* get the right UttElem */
            uttElem = &cache->uttElems[uttIdx];
            /* make and copy the extended frame */
            CopyExtFrame2Batch(uttElem, frmIdx, feaElem, feaPtr);
        }
    }

    return cache->batLen;
}


/* cz277 - aug */
/* TODO:? copy feature elements to the batch?? */
/* fill the feature batch according to frmBatch */
/*static inline int GetAugFeaFromCacheOLD(DataCache *cache, FELink feaElem, NFloat *feaBat) {
    int i, j, uttIdx;
    NFloat *feaPtr;
    UttElem *uttElem;

    feaPtr = feaBat;
    for (i = 0; i < cache->batLen; ++i) {
        uttIdx = cache->frmBatch[i].uttIdx;
        uttElem = &cache->uttElems[uttIdx];
        if (VectorSize(uttElem->augFeaVec[feaElem->augFeaIdx]) != feaElem->feaDim) {
            HError(9999, "GetAugFeaFromCache: Augmented feature %d dimension does not match the feature element dimension", feaElem->augFeaIdx);
        }
        for (j = 1; j < feaElem->ctxMap[0]; ++j, feaPtr += feaElem->feaDim) {
            memcpy(feaPtr, &uttElem->augFeaVec[feaElem->augFeaIdx][1], feaElem->feaDim * sizeof(float));
        }
    }

    return cache->batLen;
}*/

/* cz277 - aug */
/* cz277 - many */
/* fill the feature batch according to frmBatch */
static inline int GetAugFeaFromCache(DataCache *cache, FELink feaElem, int offset) {
    int i, j, uttIdx;
    NFloat *feaPtr;
    UttElem *uttElem;

    feaPtr = feaElem->feaMats[1]->matElems + offset;
    for (i = 0; i < cache->batLen; ++i) {
        uttIdx = cache->frmBatch[i].uttIdx;
        /* get the right UttElem */
        uttElem = &cache->uttElems[uttIdx];
        if (VectorSize(uttElem->augFeaVec[feaElem->augFeaIdx]) != feaElem->feaDim) {
            HError(8922, "GetAugFeaFromCache: Augmented feature %d dim does't match the feature element dim", feaElem->augFeaIdx);
        }
        for (j = 1; j <= feaElem->ctxMap[0]; ++j, feaPtr += feaElem->feaDim) {
            memcpy(feaPtr, &uttElem->augFeaVec[feaElem->augFeaIdx][1], feaElem->feaDim * sizeof(float));
        }
    }

    return cache->batLen;
}

/* fill the label batch according to frmBatch and labelKind */
static inline void GetHardLabelFromCache(DataCache *cache, int tgtDim, NFloat *labBat, int offset) {
    int i, uttIdx, frmIdx, tgtIdx;
    NFloat *labPtr;
    UttElem *uttElem;

    if ((cache->labelInfo->labelKind & LABLK) == 0) {
        HError(8901, "GetHardLabelFromCache: Function does not support current label kind");
    }

    labPtr = labBat + offset;
    for (i = offset; i < cache->batLen; ++i, labPtr += tgtDim) {
        uttIdx = cache->frmBatch[i].uttIdx;
        frmIdx = cache->frmBatch[i].frmIdx;
        /* get the right UttElem */
        uttElem = &cache->uttElems[uttIdx];
        tgtIdx = uttElem->labIdxes[frmIdx];
        cache->labVec[i + 1] = tgtIdx;   /* fill labVec */
        if (tgtIdx < 0 || tgtIdx >= tgtDim)
            HError(8993, "GetHardLabelFromCache: Label index out of range");
        memset(labPtr, 0.0, tgtDim * sizeof(NFloat));
        labPtr[tgtIdx] = 1.0; 
    }
}

/* fill the label batch according to frmBatch and labelKind */
/* return the number of frame in the batch */
static inline int GetFeatureLabelFromCache(DataCache *cache, int tgtDim, NFloat *labBat, int offset) {
    int i, uttIdx, frmIdx;
    NFloat *labPtr;
    UttElem *uttElem;
    float *flabPtr;
#ifdef DOUBLEANN
    int j;
#endif

    if ((cache->labelInfo->labelKind & FEALK) == 0) {
        HError(8901, "GetFeatureLabelFromCache: Function does not support current label kind");
        return -1;
    }
    if (cache->labelInfo->dimFLab != tgtDim) {
        HError(8930, "GetFeatureLabelFromCache: Inconsistent dimensions between feature type label and output layer");
        return -1;
    }

    labPtr = labBat + offset;
    for (i = offset; i < cache->batLen; ++i, labPtr += tgtDim) {
        uttIdx = cache->frmBatch[i].uttIdx;
        frmIdx = cache->frmBatch[i].frmIdx;
        /* get the right UttElem */
        uttElem = &cache->uttElems[uttIdx];
        flabPtr = &uttElem->flabMat[frmIdx * tgtDim];
#ifdef DOUBLEANN
        for (j = 0; j < tgtDim; ++j)
            labPtr[j] = flabPtr[j];
#else
        memcpy(labPtr, flabPtr, tgtDim * sizeof(float));
#endif
    }

    return cache->batLen;
}


/* return the name of the utterance that the frame in frmPtrs[0] is associated with */
char *GetCurUttName(DataCache *cache) {
    if (cache->visitKind != NONEVK && cache->visitKind != UTTFRMVK && cache->visitKind != UTTVK) {
        return NULL;
    }
    return cache->uttElems[cache->frmPtrs[0].uttIdx].uttName;
}

/* return the length of the utterance that the frame in frmPtrs[0] is associated with */
int GetCurUttLen(DataCache *cache) {
    if (cache->visitKind != NONEVK && cache->visitKind != UTTFRMVK && cache->visitKind != UTTVK) {
        return -1;
    }
    return cache->uttElems[cache->frmPtrs[0].uttIdx].uttLen;
}

UttElem *GetCurUttElem(DataCache *cache) {
    if (cache->visitKind != NONEVK && cache->visitKind != UTTFRMVK && cache->visitKind != UTTVK) {
        HError(8901, "GetCurUttElem: Function only support utterance based visiting");
    }
    if (cache->streamIdx != 1) {
        HError(8992, "GetCurUttElem: Lattices can only be visited through the cache of 1st stream");
    }

    return &cache->uttElems[cache->frmPtrs[0].uttIdx];
}

/* get the inXForm and paXForm associated with current utterance */
FBLatInfo *LoadXFormsFromUttElem(UttElem *uttElem, FBLatInfo *fbInfo) {

    fbInfo->inXForm = uttElem->inXForm;
    fbInfo->paXForm = uttElem->paXForm; 
    return fbInfo;
}

/* get the numerator lattices associated with current utterance */
FBLatInfo *LoadNumLatsFromUttElem(UttElem *uttElem, FBLatInfo *fbInfo) {
    int i;

    for (i = 0; (i < MAXLATSUTT) && (uttElem->numLats[i] != NULL); ++i) {
        FBLatAddLattice(fbInfo, uttElem->numLats[i]);
    }
    return fbInfo;    
}

/* get the denorminator lattices associated with current utterance */
FBLatInfo *LoadDenLatsFromUttElem(UttElem *uttElem, FBLatInfo *fbInfo) {
    int i;

    for (i = 0; (i < MAXLATSUTT) && (uttElem->denLats[i] != NULL); ++i) {
        FBLatAddLattice(fbInfo, uttElem->denLats[i]);
    }
    for (i = 0; (i < MAXLATSUTT) && (uttElem->numLats[i] != NULL); ++i) {
        if (uttElem->numInDen[i]) {
            FBLatAddLattice(fbInfo, uttElem->numLats[i]);
        }
    }
    return fbInfo;
}

/* fill all cache related batches as well as the label batch (if needed) */
/* underfill is only useful for UTT series and PLUTT series VisitKind */
/*     for UTT, unfilled batch happens at the end of a utterance */
/*     for PLUTT, unfilled batch happens at the end of all data (insufficient utts for the frmPtrs) */
/* return TRUE if no more data available; nSamples returns the number of samples loaded */
Boolean FillAllInpBatch(DataCache *cache, int *nSamples, int *uttCnt) {
    int i, nInp, tgtDim;
    Boolean finish = FALSE;
    FELink *inpElem;
    NFloat *batPtr;
#ifdef CUDA
    int j,n;
#endif

    /* init nSamples */
    *nSamples = 0;
    nInp = cache->hmmSet->nInp[cache->streamIdx];
    inpElem = cache->hmmSet->inpElem[cache->streamIdx];
    tgtDim = cache->outLayer->nodeNum;

    /* cz277 - mtload */
    /*if (extThreadLoad == TRUE && cache->firstLoad == TRUE) 
        if (pthread_join(cache->extThread, NULL) != 0) 
            HError(8929, "FillAllInpBatch: Error when joining the extra thread for loading cache");*/

    /* check whether need to release the data and reload again */
    /* update the cache status if needed */
    if (cache->batLen > 0) {
        /* cz277 - mtload */
        UnloadCacheData(cache);
        FillCacheSGT(cache); 

        UpdateUttOrder(cache);
        if (cache->visitKind == FRMVK) 
            UpdateFrmOrder(cache);
    }
    /* fill the internal batch */
    switch (cache->visitKind) {
        case FRMVK:
            finish = FillBatchFRM(cache, cache->batchSamples - (*nSamples));
            break;
        case NONEVK:
        case UTTFRMVK:
        case UTTVK:
            finish = FillBatchUTT(cache, cache->batchSamples - (*nSamples), uttCnt);
            break;
        case PLNONEVK:
        case PLUTTFRMVK:
        case PLUTTVK:
            finish = FillBatchPLUTT(cache, cache->batchSamples - (*nSamples));
            break;
        default:
            HError(8991, "FillAllInpBatch: Unknown visiting order");
    }
    /* fill each input batches */
    for (i = 0; i < nInp; ++i) {
        if (inpElem[i]->inputKind == INPFEAIK) 
            GetDataFromCache(cache, inpElem[i], (*nSamples) * inpElem[i]->extDim);
        else if (inpElem[i]->inputKind == AUGFEAIK) 	/* cz277 - aug */
            GetAugFeaFromCache(cache, inpElem[i], (*nSamples) * inpElem[i]->extDim);
        else 
            HError(8992, "FillAllInpBatch: Can only have INPFEAIK and AUGFEAIK from host memory");
#ifdef CUDA
        n = IntVecSize(inpElem[i]->ctxPool);
        for (j = 1; j <= n; ++j) 	/* cz277 - many */
            SyncNMatrixHost2Dev(inpElem[i]->feaMats[j]);
#endif
    }
    /* fill the label batch if needed */
    if (cache->labMat != NULL) {
        batPtr = cache->labMat->matElems + (*nSamples) * tgtDim;
        if (cache->labelInfo->labelKind & FEALK)
            GetFeatureLabelFromCache(cache, tgtDim, batPtr, 0);
        else if (cache->labelInfo->labelKind & LABLK)
            GetHardLabelFromCache(cache, tgtDim, batPtr, 0);
        else
            HError(8992, "FillAllInpBatch: Label kind does not support cache label matrix");
#ifdef CUDA
        SyncNMatrixHost2Dev(cache->labMat);
#endif
    }
    /* update nSamples */
    *nSamples += cache->batLen;
    /* only applicable to UTT and PLUTT series */
    if (*uttCnt == 0)
        *uttCnt = -1;

    return finish;
}

/* update the log prior list  */
void UpdateTargetLogPrior(DataCache *cache, float offset) {
    /*HMMSet *hset;*/
    MLink m, mret;
    HLink hmm;
    StreamElem *streamElem;
    int h, s;
    double tOcc = 0.0;

    /*if (!cache->labelInfo->countOcc) {*/
    if ((cache->labelInfo->uFlags & UPTARGETPEN) == 0) {
        return;
    }

    /*hset = (HMMSet *) cache->hmmSet;*/
    /* first, check each state */
    for (h = 0; h < MACHASHSIZE; ++h) {
        for (m = cache->hmmSet->mtab[h]; m != NULL; m = m->next) {
            if (m->type == 'h') {
                hmm = (HLink) m->structure;
                for (s = 2; s < hmm->numStates; ++s) {
                    streamElem = &hmm->svec[s].info->pdf[cache->streamIdx];
                    if (streamElem->occAcc <= 0.0) {
                        mret = FindMacroStruct(cache->hmmSet, 's', hmm->svec[s].info);
                        if (mret == NULL) 
                            HError(-8931, "UpdateTargetPen: HMM State %s[%d] has no training samples", m->id->name, s);
                        else 
                            HError(-8931, "UpdateTargetPen: HMM State %s has no training samples", mret->id->name);
                    }
                }
            }
        }
    }
    /* then accumulate the total occupancies */
    tOcc = 0.0;
    for (h = 0; h < MACHASHSIZE; ++h) {
        for (m = cache->hmmSet->mtab[h]; m != NULL; m = m->next) {
            if (m->type == 'h') {
                hmm = (HLink) m->structure;
                for (s = 2; s < hmm->numStates; ++s) {
                    streamElem = &hmm->svec[s].info->pdf[cache->streamIdx];
                    if (streamElem->occAcc > 0.0) {
                        tOcc += streamElem->occAcc;
                        streamElem->targetPen = streamElem->occAcc;
                        streamElem->occAcc = -1.0; 
                    }
                }
            }
        }
    }
    /* at last, compute log(P(s)) + offset and clear occAcc counters */
    for (h = 0; h < MACHASHSIZE; ++h) {
        for (m = cache->hmmSet->mtab[h]; m != NULL; m = m->next) {
            if (m->type == 'h') {
                hmm = (HLink) m->structure;
                for (s = 2; s < hmm->numStates; ++s) {
                    streamElem = &hmm->svec[s].info->pdf[cache->streamIdx];
                    if (streamElem->occAcc < 0.0) {
                        streamElem->occAcc = streamElem->targetPen;
                        streamElem->targetPen = (float) ((-1.0) * log(streamElem->targetPen / tOcc) + offset);
                    }
                }
            }
        }
    }
}



