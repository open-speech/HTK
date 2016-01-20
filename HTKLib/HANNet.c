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
/*         File: HANNet.c  ANN model definition data type      */
/* ----------------------------------------------------------- */

char *hannet_version = "!HVER!HANNet:   3.5.0 [CUED 12/10/15]";
char *hannet_vc_id = "$Id: HANNet.c,v 1.0 2015/10/12 12:07:24 cz277 Exp $";

#include "config.h"
#include <time.h>
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
#include "HNet.h"
#include "HArc.h"
#include "HFBLat.h"
#include "HDict.h"
#include "HAdapt.h"
#include <math.h>

/* ------------------------------ Trace Flags ------------------------------ */

static int trace = 0;

#define T_TOP 0001
#define T_CCH 0002

/* --------------------------- Memory Management --------------------------- */


/* ----------------------------- Configuration ------------------------------*/

static ConfParam *cParm[MAXGLOBS];      /* config parameters */
static int nParm = 0;
static size_t batchSamples = 1;                 /* the number of samples in batch; 1 sample by default */
static char ANNUpdateFlagStr[MAXSTRLEN];          /* the string pointer indicating the layers to update */
/*static char layerUpdateFlagStr[MAXSTRLEN];*/	/* cz277 - 150811 */
/*static char actfunUpdateFlagStr[MAXSTRLEN];*/	/* cz277 - 150811 */
static char matrixUpdateFlagStr[MAXSTRLEN];	/* cz277 - 151020 */
static char vectorUpdateFlagStr[MAXSTRLEN];     /* cz277 - 151020 */

/*static Boolean hasShownUpdtFlag = FALSE;*/
/* cz277 - 1007 */
static int batchIndex = 0;
/* cz277 - 150811 */
static RILink headNMatRPLInfo = NULL;
static int numNMatRPLInfo = 0;
static RILink headNVecRPLInfo = NULL;
static int numNVecRPLInfo = 0;
static char maskStrNMatRPLInfo[MAXSTRLEN];
static char inDirStrNMatRPLInfo[MAXSTRLEN];
static char extStrNMatRPLInfo[MAXSTRLEN];
static char outDirStrNMatRPLInfo[MAXSTRLEN];
static char maskStrNVecRPLInfo[MAXSTRLEN];
static char inDirStrNVecRPLInfo[MAXSTRLEN];
static char extStrNVecRPLInfo[MAXSTRLEN];
static char outDirStrNVecRPLInfo[MAXSTRLEN];

/* get the batch size */
int GetNBatchSamples(void) {
    return batchSamples;
}

/* set the batch size */
void SetNBatchSamples(int userBatchSamples) {
    batchSamples = userBatchSamples;
#ifdef CUDA
    RegisterTmpNMat(1, batchSamples);
#endif
}

/* cz277 - xform */
/*void InitRPLInfo(RPLInfo *rplInfo) {
    rplInfo->nSpkr = 0;
    rplInfo->inRPLMask = NULL;
    rplInfo->curOutSpkr = NewString(&gcheap, MAXSTRLEN);
    rplInfo->curInSpkr = NewString(&gcheap, MAXSTRLEN);
    rplInfo->cacheInSpkr = NewString(&gcheap, MAXSTRLEN);
    rplInfo->inRPLDir = NULL;
    rplInfo->inRPLExt = NULL;
    rplInfo->outRPLDir = NULL;
    rplInfo->outRPLExt = NULL;
    rplInfo->saveBinary = FALSE;
    rplInfo->rplNMat = NULL;
    memset(&rplInfo->saveRplNMatHost, 0, sizeof(NMatHost));
    rplInfo->rplNVec = NULL;
    memset(&rplInfo->saveRplNVecHost, 0, sizeof(NVecHost));
}*/

/*  */
void InitANNet(void)
{
    int intVal;
    ConfParam *cpVal;

    /* cz277 - 150811 */
    strcpy(ANNUpdateFlagStr, "");
    /*strcpy(layerUpdateFlagStr, "");
    strcpy(actfunUpdateFlagStr, "");*/
    /* cz277 - 1501020 */
    strcpy(matrixUpdateFlagStr, "");
    strcpy(vectorUpdateFlagStr, "");

    /* cz277 - 150811 */
    strcpy(maskStrNMatRPLInfo, "");
    strcpy(inDirStrNMatRPLInfo, "");
    strcpy(extStrNMatRPLInfo, "");
    strcpy(outDirStrNMatRPLInfo, "");
    strcpy(maskStrNVecRPLInfo, "");
    strcpy(inDirStrNVecRPLInfo, "");
    strcpy(extStrNVecRPLInfo, "");
    strcpy(outDirStrNVecRPLInfo, "");

    Register(hannet_version, hannet_vc_id);
    nParm = GetConfig("HANNET", TRUE, cParm, MAXGLOBS);

    if (nParm > 0) {
        if (GetConfInt(cParm, nParm, "TRACE", &intVal)) { 
            trace = intVal;
        }
        if (GetConfInt(cParm, nParm, "MINIBATCHSIZE", &intVal)) {
            if (intVal <= 0) 
                HError(8720, "InitANNet: Negative or zero batch size");
            SetNBatchSamples(intVal);
        }
        /* cz277 - 150811 */
        if (GetConfAny(cParm, nParm, "ANNUPDATEFLAG", &cpVal)) {
            while (cpVal != NULL) {
                if (cpVal->kind != StrCKind)
                    HError(8720, "InitANNet: ANNUPDATEFLAG has to be string kind");
                strcat(ANNUpdateFlagStr, cpVal->val.s);
                if (cpVal->val.s[strlen(cpVal->val.s) - 1] != ';')
                    strcat(ANNUpdateFlagStr, ";");
                cpVal = cpVal->append;
            }
        }
        /* cz277 - 151020 */
        if (GetConfAny(cParm, nParm, "NMATUPDATEFLAG", &cpVal)) {
            while (cpVal != NULL) {
                if (cpVal->kind != StrCKind)
                    HError(8720, "InitANNet: NMATUPDATEFLAG has to be string kind");
                strcat(matrixUpdateFlagStr, cpVal->val.s);
                if (cpVal->val.s[strlen(cpVal->val.s) - 1] != ';')
                    strcat(matrixUpdateFlagStr, ";");
                cpVal = cpVal->append;
            }
        } 
        if (GetConfAny(cParm, nParm, "NVECUPDATEFLAG", &cpVal)) {
            while (cpVal != NULL) {
                if (cpVal->kind != StrCKind)
                    HError(8720, "InitANNet: NVECUPDATEFLAG has to be string kind");
                strcat(vectorUpdateFlagStr, cpVal->val.s);
                if (cpVal->val.s[strlen(cpVal->val.s) - 1] != ';')
                    strcat(vectorUpdateFlagStr, ";");
                cpVal = cpVal->append;
            }
        }
        /* cz277 - 150811 */
        if (GetConfAny(cParm, nParm, "REPLACEABLENMATMASK", &cpVal)) {
            while (cpVal != NULL) {
                if (cpVal->kind != StrCKind)
                    HError(8720, "InitANNet: REPLACEABLENMATMASK has to be string kind");
                strcat(maskStrNMatRPLInfo, cpVal->val.s);
                if (cpVal->val.s[strlen(cpVal->val.s) - 1] != ';')
                    strcat(maskStrNMatRPLInfo, ";");
                cpVal = cpVal->append;
            }
        }
        if (GetConfAny(cParm, nParm, "REPLACEABLENMATINDIR", &cpVal)) {
            while (cpVal != NULL) {
                if (cpVal->kind != StrCKind)
                    HError(8720, "InitANNet: REPLACEABLENMATINDIR has to be string kind");
                strcat(inDirStrNMatRPLInfo, cpVal->val.s);
                if (cpVal->val.s[strlen(cpVal->val.s) - 1] != ';')
                    strcat(inDirStrNMatRPLInfo, ";");
                cpVal = cpVal->append;
            }
        }
        if (GetConfAny(cParm, nParm, "REPLACEABLENMATEXT", &cpVal)) {
            while (cpVal != NULL) {
                if (cpVal->kind != StrCKind)
                    HError(8720, "InitANNet: REPLACEABLENMATEXT has to be string kind");
                strcat(extStrNMatRPLInfo, cpVal->val.s);
                if (cpVal->val.s[strlen(cpVal->val.s) - 1] != ';')
                    strcat(extStrNMatRPLInfo, ";");
                cpVal = cpVal->append;
            }
        }
        if (GetConfAny(cParm, nParm, "REPLACEABLENMATOUTDIR", &cpVal)) {
            while (cpVal != NULL) {
                if (cpVal->kind != StrCKind)
                    HError(8720, "InitANNet: REPLACEABLENMATOUTDIR has to be string kind");
                strcat(outDirStrNMatRPLInfo, cpVal->val.s);
                if (cpVal->val.s[strlen(cpVal->val.s) - 1] != ';')
                    strcat(outDirStrNMatRPLInfo, ";");
                cpVal = cpVal->append;
            }
        }
        if (GetConfAny(cParm, nParm, "REPLACEABLENVECMASK", &cpVal)) {
            while (cpVal != NULL) {
                if (cpVal->kind != StrCKind)
                    HError(8720, "InitANNet: REPLACEABLENVECMASK has to be string kind");
                strcat(maskStrNVecRPLInfo, cpVal->val.s);
                if (cpVal->val.s[strlen(cpVal->val.s) - 1] != ';')
                    strcat(maskStrNVecRPLInfo, ";");
                cpVal = cpVal->append;
            }
        }
        if (GetConfAny(cParm, nParm, "REPLACEABLENVECINDIR", &cpVal)) {
            while (cpVal != NULL) {
                if (cpVal->kind != StrCKind)
                    HError(8720, "InitANNet: REPLACEABLENVECINDIR has to be string kind");
                strcat(inDirStrNVecRPLInfo, cpVal->val.s);
                if (cpVal->val.s[strlen(cpVal->val.s) - 1] != ';')
                    strcat(inDirStrNVecRPLInfo, ";");
                cpVal = cpVal->append;
            }
        }
        if (GetConfAny(cParm, nParm, "REPLACEABLENVECEXT", &cpVal)) {
            while (cpVal != NULL) {
                if (cpVal->kind != StrCKind)
                    HError(8720, "InitANNet: REPLACEABLENVECEXT has to be string kind");
                strcat(extStrNVecRPLInfo, cpVal->val.s);
                if (cpVal->val.s[strlen(cpVal->val.s) - 1] != ';')
                    strcat(extStrNVecRPLInfo, ";");
                cpVal = cpVal->append;
            }
        }
        if (GetConfAny(cParm, nParm, "REPLACEABLENVECOUTDIR", &cpVal)) {
            while (cpVal != NULL) {
                if (cpVal->kind != StrCKind)
                    HError(8720, "InitANNet: REPLACEABLENVECOUTDIR has to be string kind");
                strcat(outDirStrNVecRPLInfo, cpVal->val.s);
                if (cpVal->val.s[strlen(cpVal->val.s) - 1] != ';')
                    strcat(outDirStrNVecRPLInfo, ";");
                cpVal = cpVal->append;
            }
        }
    }

}


/* cz277 - 150811 */
char *GetANNUpdateFlagStr() {
    return ANNUpdateFlagStr;
}

/* cz277 - 150820 */
char *GetNMatUpdateFlagStr() {
    return matrixUpdateFlagStr;
}

/* cz277 - 150820 */
char *GetNVecUpdateFlagStr() {
    return vectorUpdateFlagStr;
}

/* cz277 - 150811 */
/*char *GetLayerUpdateFlagStr() {
    return layerUpdateFlagStr;
}*/

/* cz277 - 150811 */
/*char *GetActFunUpdateFlagStr() {
    return actfunUpdateFlagStr;
}*/

/* cz277 - 150811 */
char *GetMaskStrNMatRPLInfo() {
    return maskStrNMatRPLInfo;
}

/* cz277 - 150811 */
char *GetInDirStrNMatRPLInfo() {
    return inDirStrNMatRPLInfo;
}

/* cz277 - 150811 */
char *GetExtStrNMatRPLInfo() {
    return extStrNMatRPLInfo;
}

/* cz277 - 150811 */
char *GetOutDirStrNMatRPLInfo() {
    return outDirStrNMatRPLInfo;
}

/* cz277 - 150811 */
char *GetMaskStrNVecRPLInfo() {
    return maskStrNVecRPLInfo;
}

/* cz277 - 150811 */
char *GetInDirStrNVecRPLInfo() {
    return inDirStrNVecRPLInfo;
}

/* cz277 - 150811 */
char *GetExtStrNVecRPLInfo() {
    return extStrNVecRPLInfo;
}

/* cz277 - 150811 */
char *GetOutDirStrNVecRPLInfo() {
    return outDirStrNVecRPLInfo;
}

/* cz277 - 150811 */
RILink GetHeadNMatRPLInfo() {
    return headNMatRPLInfo;
}

/* cz277 - 150811 */
RILink GetHeadNVecRPLInfo() {
    return headNVecRPLInfo;
}

/* cz277 - 150811 */
void SetHeadNMatRPLInfo(RILink info) {
    headNMatRPLInfo = info;
}

/* cz277 - 150811 */
void SetHeadNVecRPLInfo(RILink info) {
    headNVecRPLInfo = info;
}

int GetNumNMatRPLInfo() {
    return numNMatRPLInfo;
}

int GetNumNVecRPLInfo() {
    return numNVecRPLInfo;
}

void SetNumNMatRPLInfo(int n) {
    numNMatRPLInfo = n;
}

void SetNumNVecRPLInfo(int n) {
    numNVecRPLInfo = n;
}

int GetGlobalBatchIndex() {
    return batchIndex;
}

void SetGlobalBatchIndex(int index) {
    batchIndex = index;
}

static inline void FillBatchFromFeaMixOLD(FeaMix *feaMix, int batLen, int *CMDVecPL) {
    int i, j, k, srcOff = 0, curOff = 0, dstOff, hisOff, hisDim;
    FELink feaElem;

    /* if it is the shared */
    if (feaMix->feaList[0]->feaMats[0] == feaMix->mixMats[0]) 
        return;

    /* cz277 - 1007 */
    if (feaMix->batchIndex != batchIndex - 1 && feaMix->batchIndex != batchIndex) 
        HError(8790, "FillBatchFromFeaMix: Wrong batch index");
    else if (feaMix->batchIndex == batchIndex) 
        return;
    else 
        ++feaMix->batchIndex;

    /* otherwise, fill the batch with a mixture of the FeaElem */
    for (i = 0; i < feaMix->elemNum; ++i) {
        feaElem = feaMix->feaList[i];

        if (feaElem->inputKind == INPFEAIK || feaElem->inputKind == AUGFEAIK) {
            for (j = 0, srcOff = 0, dstOff = curOff; j < batLen; ++j, srcOff += feaElem->extDim, dstOff += feaMix->mixDim) {
                CopyNSegment(feaElem->feaMats[0], srcOff, feaElem->extDim, feaMix->mixMats[0], dstOff);
            }
        }
        else if (feaElem->inputKind == ANNFEAIK) {  /* ANNFEAIK, left context is consecutive */
            for (j = 0; j < batLen; ++j) {
                /* cz277 - gap */
                hisDim = feaElem->hisLen * feaElem->feaDim;
                hisOff = j * hisDim;
                if (CMDVecPL != NULL && feaElem->hisMat != NULL) {
                    if (CMDVecPL[j] == 0) {	/* reset the history */
			ClearNMatrixSegment(feaElem->hisMat, hisOff, hisDim);
                    }
                    else if (CMDVecPL[j] > 0) {	/* shift the history */
                        CopyNSegment(feaElem->hisMat, CMDVecPL[j] * hisDim, hisDim, feaElem->hisMat, hisOff);
                    }
                }
                /* standard operations */
                dstOff = j * feaMix->mixDim + curOff;
                for (k = 1; k <= feaElem->ctxMap[0]; ++k, dstOff += feaElem->feaDim) { 
                    if (feaElem->ctxMap[k] < 0) {
                        /* first, previous segments from hisMat to feaMix->mixMat */
                        srcOff = ((j + 1) * feaElem->hisLen + feaElem->ctxMap[k]) * feaElem->feaDim;
                        CopyNSegment(feaElem->hisMat, srcOff, feaElem->feaDim, feaMix->mixMats[0], dstOff);
                    }
                    else if (feaElem->ctxMap[k] == 0) {
                        /* second, copy current segment from feaMat to feaMix->mixMat */
                        srcOff = j * feaElem->srcDim + feaElem->dimOff;
                        CopyNSegment(feaElem->feaMats[0], srcOff, feaElem->feaDim, feaMix->mixMats[0], dstOff);
                    }
                    else {
                        HError(9999, "FillBatchFromFeaMix: The future of ANN features are not applicable");
                    }
                }
                /* shift history info in hisMat and copy current segment from feaMat to hisMat */
                if (feaElem->hisMat != NULL) {
                    dstOff = hisOff;
                    srcOff = dstOff + feaElem->feaDim;
                    for (k = 0; k < feaElem->hisLen - 1; ++k, srcOff += feaElem->feaDim, dstOff += feaElem->feaDim) {
                        CopyNSegment(feaElem->hisMat, srcOff, feaElem->feaDim, feaElem->hisMat, dstOff);    
                    }
                    srcOff = j * feaElem->srcDim + feaElem->dimOff;
                    CopyNSegment(feaElem->feaMats[0], srcOff, feaElem->feaDim, feaElem->hisMat, dstOff);
                }
            }
        }
        curOff += feaElem->extDim;
    }
}

/* cz277 - xform */
static inline void FillBatchFromFeaMix(LELink layerElem, int batLen) {
    int i, j, k, l, n, m, srcOff, curOff, dstOff, t, c, curCtx;
    FELink feaElem;
    FeaMix *feaMix;
    NMatrix *mixMat, *feaMat;

    feaMix = layerElem->feaMix;
    if (feaMix->batchIndex != batchIndex - 1 && feaMix->batchIndex != batchIndex)
        HError(8790, "FillBatchFromFeaMix: Problematic batch index of a feaMix");
    if (feaMix->batchIndex == batchIndex)
        return;
    /* update the batchIndex */
    ++feaMix->batchIndex;

    /* if mixMats are shared with feaMats -- no need to reload the features */
    if (feaMix->elemNum == 1) {
        feaElem = feaMix->feaList[0];
        if (!(feaElem->inputKind == ANNFEAIK && IntVecSize(feaElem->ctxMap) > 1))
            return;
    }
    /* if feaMat is shared and is processed for current batch */
    /*if (feaMix->batchIndex != batchIndex || feaMix->batchIndex != batchIndex + 1)
        HError(9999, "FillBatchFromFeaMix: Wrong batch index");
    else if (feaMix->batchIndex == batchIndex + 1)
        return;
    ++feaMix->batchIndex;*/
    /* otherwise */
    n = IntVecSize(layerElem->drvCtx);
    for (i = 1, j = 1; i <= n; ++i) {
        while (layerElem->drvCtx[i] != feaMix->ctxPool[j])
            ++j;
        mixMat = feaMix->mixMats[j];
        for (k = 0, curOff = 0; k < feaMix->elemNum; ++k) {
            feaElem = feaMix->feaList[k];
            l = 1;
            /* 1. extended input features */
            if (feaElem->inputKind == INPFEAIK || feaElem->inputKind == AUGFEAIK) {
                if (feaElem->inputKind == INPFEAIK) {
                    while (layerElem->drvCtx[i] != feaElem->ctxPool[l])
                        ++l;
                }
                feaMat = feaElem->feaMats[l];
                for (t = 0, srcOff = 0, dstOff = curOff; t < batLen; ++t, srcOff += feaElem->extDim, dstOff += feaMix->mixDim) {
                    CopyNSegment(feaMat, srcOff, feaElem->extDim, mixMat, dstOff);
                }
            }
            else if (feaElem->inputKind == ANNFEAIK) {	/* 2. ANN features */
                m = IntVecSize(feaElem->ctxMap);
                for (c = 1; c <= m; ++c) {
                    curCtx = feaElem->ctxMap[c] + layerElem->drvCtx[i];
                    while (curCtx != feaElem->ctxPool[l])
                        ++l;
                    feaMat = feaElem->feaMats[l];
                    for (t = 0; t < batLen; ++t) {
                        srcOff = t * feaElem->srcDim + feaElem->dimOff;
                        dstOff = t * feaMix->mixDim + curOff + (c - 1) * feaElem->feaDim;
                        CopyNSegment(feaMat, srcOff, feaElem->feaDim, mixMat, dstOff);
                    }
                }
            }
            curOff += feaElem->extDim;
        }
    }
}


/* fill a batch with error signal */
static inline void FillBatchFromErrMixOLD(FeaMix *errMix, int batLen, NMatrix *mixMat) {
    int i, j, srcOff, dstOff;
    FELink errElem;

    /* if it is the shared */
    if (errMix->feaList[0]->feaMats[1] == mixMat) {
        return;
    }

    /* otherwise, fill the batch with a mixture of the FeaElem */
    dstOff = 0;
    /* reset mixMat to 0 */
    /*SetNMatrix(0.0, mixMat, batLen);*/
    ClearNMatrix(mixMat, batLen);
    /* accumulate the error signals from each source */
    for (i = 0; i < batLen; ++i) {
        for (j = 0; j < errMix->elemNum; ++j) {
            errElem = errMix->feaList[j];
            srcOff = i * errElem->srcDim + errElem->dimOff;
            AddNSegment(errElem->feaMats[1], srcOff, errElem->extDim, mixMat, dstOff);
            dstOff += errElem->extDim;
        }
    }
}

/* cz277 - many */
static inline void FillBatchFromErrMix(LELink layerElem, int batLen) {
    int c, i, j, k, l, m, n, t, srcOff, dstOff;
    FELink errElem;
    FeaMix *errMix;

    errMix = layerElem->errMix;
    if (errMix == NULL)
        return;
    if (errMix->batchIndex != batchIndex - 1 && errMix->batchIndex != batchIndex)
        HError(8790, "FillBatchFromErrMix: Wrong batch index of an errMix");
    if (errMix->batchIndex == batchIndex)
        return;
    /* update the batchIndex */
    ++errMix->batchIndex;

    if (errMix->elemNum == 1) {
        errElem = errMix->feaList[0];
        if (IntVecSize(errElem->ctxMap) == 1)
            if (errElem->srcDim == errElem->feaDim)
                return;
    }
    
    n = IntVecSize(layerElem->drvCtx);
    for (i = 1; i <= n; ++i)
        ClearNMatrix(errMix->mixMats[i], batLen);

    for (i = 0; i < errMix->elemNum; ++i) {
        errElem = errMix->feaList[i];
        m = IntVecSize(errElem->ctxPool);
        n = IntVecSize(errElem->ctxMap);
        for (j = 1; j <= m; ++j) {
            for (k = 1; k <= n; ++k) {
                srcOff = errElem->dimOff + (k - 1) * errElem->feaDim;
                c = errElem->ctxPool[j] + errElem->ctxMap[k];
                l = 1;
                while (errMix->ctxPool[l] != c)
                    ++l;
                for (t = 0, dstOff = 0; t < batLen; ++t, srcOff += errElem->srcDim, dstOff += errElem->feaDim) {
                    AddNSegment(errElem->feaMats[j], srcOff, errElem->feaDim, errMix->mixMats[l], dstOff);
                }
            }
        }
    }

    /* scale the mixMats */
    /*n = IntVecSize(layerElem->drvCtx);
    for (i = 1; i <= n; ++i) {
        if (layerElem->trainInfo->drvCnt[i] > 1)
            ScaleNMatrix(1.0 / (float) layerElem->trainInfo->drvCnt[i], batLen, errMix->mixDim, errMix->mixMats[i]);
    }*/

}


/* temp function */
void ShowAddress(ANNSet *annSet) {
  /*int i;*/
    AILink curAI;
    ADLink annDef;
    /*LELink layerElem;*/

    curAI = annSet->defsHead;
    while (curAI != NULL) {
        annDef = curAI->annDef;
        printf("ANNInfo = %p. ANNDef = %p: \n", curAI, annDef);
        /*for (i = 0; i < annDef->layerNum; ++i) {
	  layerElem = annDef->layerList[i];*/
            /*printf("layerElem = %p, feaMix[0]->feaMat = %p, xFeaMat = %p, yFeaMat = %p, trainInfo = %p, dxFeaMat = %p, dyFeaMat = %p, labMat = %p\n", layerElem, layerElem->feaMix->feaList[0]->feaMat, layerElem->xFeaMat, layerElem->yFeaMat, layerElem->trainInfo, layerElem->trainInfo->dxFeaMat, layerElem->trainInfo->dyFeaMat, layerElem->trainInfo->labMat);*/
        /*}
	  printf("\n");*/
        curAI = curAI->next;
    }
}

/* update the map sum matrix for outputs */
void UpdateOutMatMapSum(ANNSet *annSet, int batLen, int streamIdx) {

    /* cz277 - many */
    HNBlasTNgemm(annSet->mapStruct->mappedTargetNum, batLen, annSet->outLayers[streamIdx]->nodeNum, 1.0, annSet->mapStruct->maskMatMapSum[streamIdx], annSet->outLayers[streamIdx]->yFeaMats[1], 0.0, annSet->mapStruct->outMatMapSum[streamIdx]);
}

/* update the map sum matrix for labels */
void UpdateLabMatMapSum(ANNSet *annSet, int batLen, int streamIdx) {

    HNBlasTNgemm(annSet->mapStruct->mappedTargetNum, batLen, annSet->outLayers[streamIdx]->nodeNum, 1.0, annSet->mapStruct->maskMatMapSum[streamIdx], annSet->outLayers[streamIdx]->trainInfo->labMat, 0.0, annSet->mapStruct->labMatMapSum[streamIdx]);
}


/* the batch with input features are assumed to be filled */
void CheckANNBatchIndex(ANNSet *annSet, int index) {
    int i, j;
    AILink curAI;
    ADLink annDef;
    LELink layerElem;
    Boolean first = TRUE;

    /* init the ANNInfo pointer */
    curAI = annSet->defsHead;
    /* proceed in the forward fashion */
    while (curAI != NULL) {
        /* fetch current ANNDef */
        annDef = curAI->annDef;
        /* proceed layer by layer */
        for (i = 0; i < annDef->layerNum; ++i) {
            layerElem = annDef->layerList[i];
            switch (layerElem->layerKind) {
            case ACTIVATIONONLYLAK: HError(8701, "CheckANNBatchIndex: Function not implemented"); break;
            case CONVOLUTIONLAK: HError(8701, "CheckANNBatchIndex: Function not implemented"); break;
            case PERCEPTRONLAK: 
                if (index < 0 && first == TRUE) {
                    index = layerElem->wghtMat->batchIndex;
                    first = FALSE;
                }
                if (layerElem->wghtMat->batchIndex != index)
                    HError(8790, "CheckANNBatchIndex: Wrong batch index of ~M \"%s\"", layerElem->wghtMat->id->name);
                if (layerElem->biasVec->batchIndex != index)
                    HError(8790, "CheckANNBatchIndex: Wrong batch index of ~V \"%s\"", layerElem->biasVec->id->name); 
                if (layerElem->actfunVecs != NULL)
                    for (j = 1; j <= layerElem->actfunParmNum; ++j)
                        if (layerElem->actfunVecs[j]->batchIndex != index)
                            HError(8790, "CheckANNBatchIndex: Wrong batch index of ~V \"%s\"", layerElem->actfunVecs[j]->id->name);
                break;
            case SUBSAMPLINGLAK: HError(8701, "CheckANNBatchIndex: Function not implemented"); break;
            default:
                HError(8791, "CheckANNBatchIndex: Unknown layer kind");
            }
        }

        /* get the next ANNDef */
        curAI = curAI->next;
    }
}

/* cz277 - batch sync */
/* the batch with input features are assumed to be filled */
void SetNBundleBatchIndex(ANNSet *annSet, int index) {
  int i, j;
  AILink curAI;
  ADLink annDef;
  LELink layerElem;
 
  /* init the ANNInfo pointer */
  curAI = annSet->defsHead;
  /* proceed in the forward fashion */
  while (curAI != NULL) {
    /* fetch current ANNDef */
    annDef = curAI->annDef;
    /* proceed layer by layer */
    for (i = 0; i < annDef->layerNum; ++i) {
      layerElem = annDef->layerList[i];
      switch (layerElem->layerKind) {
      case ACTIVATIONONLYLAK: HError(8701, "SetANNBatchIndex: Function not implemented"); break;
      case CONVOLUTIONLAK: HError(8701, "SetANNBatchIndex: Function not implemented"); break;
      case PERCEPTRONLAK:
	layerElem->wghtMat->batchIndex = index;
	layerElem->biasVec->batchIndex = index;
	if (layerElem->actfunVecs != NULL)
	  for (j = 1; j <= layerElem->actfunParmNum; ++j)
	    layerElem->actfunVecs[j]->batchIndex = index;
	break;
      case SUBSAMPLINGLAK: HError(8701, "SetANNBatchIndex: Function not implemented"); break;
      default:
	HError(8791, "SetANNBatchIndex: Unknown layer kind");
      }
    }
    /* get the next ANNDef */
    curAI = curAI->next;
  }
}


/* the batch with input features are assumed to be filled */
void SetFeaMixBatchIndex(ANNSet *annSet, int index) {
    int i, j;
    AILink curAI;
    ADLink annDef;
    LELink layerElem;

    /*batchIndex = index;*/
    /* init the ANNInfo pointer */
    curAI = annSet->defsHead;
    /* proceed in the forward fashion */
    while (curAI != NULL) {
        /* fetch current ANNDef */
        annDef = curAI->annDef;
        /* proceed layer by layer */
        for (i = 0; i < annDef->layerNum; ++i) {
            layerElem = annDef->layerList[i];
            if (layerElem->feaMix != NULL)
                layerElem->feaMix->batchIndex = index;
            if (layerElem->errMix != NULL)
                layerElem->errMix->batchIndex = index;
        }

        /* get the next ANNDef */
        curAI = curAI->next;
    }

}


/* cz277 - pact */
/* y = 1 / sqrt(var) * x + (- mean / sqrt(var)) */
static void InitAffineScaleByVar(int vecLen, NVector *varVec) {
    int i;
  
    if (!(vecLen <= varVec->vecLen)) 
        HError(8721, "InitAffineScaleByVar: Wrong vector length");
#ifdef CUDA
    SyncNVectorDev2Host(varVec);
#endif
    /* convert variance */
    for (i = 0; i < vecLen; ++i) {
        if (varVec->vecElems[i] <= 0.0) 
            HError(8721, "InitAffineScaleByVar: variance should be > 0.0");
        varVec->vecElems[i] = 1.0 / sqrt(varVec->vecElems[i]);
    }
#ifdef CUDA
    SyncNVectorHost2Dev(varVec);
#endif
}

/* cz277 - pact */
/* y = 1 / sqrt(var) * x + (- mean / sqrt(var)) */
static void InitAffineShiftByMean(int vecLen, NVector *scaleVec, NVector *meanVec) {
    int i;
    
    if (!(vecLen <= meanVec->vecLen && vecLen <= scaleVec->vecLen)) 
        HError(8721, "InitAffineShiftByMean: Wrong vector length");
#ifdef CUDA
    SyncNVectorDev2Host(meanVec);
    SyncNVectorDev2Host(scaleVec);
#endif
    /* convert mean */
    for (i = 0; i < vecLen; ++i) 
        meanVec->vecElems[i] = (-1.0) * meanVec->vecElems[i] * scaleVec->vecElems[i];
#ifdef CUDA
    SyncNVectorHost2Dev(meanVec);
    SyncNVectorHost2Dev(scaleVec);
#endif
}

/* cz277 - pact */
void DoStaticUpdateOperation(int status, int drvIdx, LELink layerElem, int batLen) {
    double nSamples;
    size_t *cnt1, *cnt2;

    if (layerElem->trainInfo == NULL)
        return;
    if (layerElem->trainInfo->initFlag == FALSE)
        return;

    switch (layerElem->actfunKind) {
    case AFFINEAF:
        if (layerElem->drvCtx[drvIdx] == 0) {
            cnt1 = (size_t *) layerElem->actfunVecs[1]->accptr;
            cnt2 = (size_t *) layerElem->actfunVecs[2]->accptr;
            switch (status) {
	    case 0:
	        *cnt1 += batLen;
                *cnt2 += batLen;	
		break;
            case 1:
		nSamples = *cnt2;
                if (nSamples == 0)
                    HError(-8721, "DoStaticUpdateOperation: nSamples = 0, inf will generate");
                AccMeanNVector(layerElem->yFeaMats[drvIdx], batLen, layerElem->nodeNum, (NFloat) nSamples, layerElem->actfunVecs[2]->variables);
                break;
            case 2:
                nSamples = *cnt1;
                if (nSamples == 0)
                    HError(-8721, "DoStaticUpdateOperation: nSamples = 0, inf will generate");
                AccVarianceNVector(layerElem->yFeaMats[drvIdx], batLen, layerElem->nodeNum, (NFloat) nSamples, layerElem->actfunVecs[2]->variables, layerElem->actfunVecs[1]->variables);
                break;
            case 3:
                nSamples = *cnt1;
                if (nSamples > 0) {
                    InitAffineScaleByVar(layerElem->nodeNum, layerElem->actfunVecs[1]->variables);
                    *cnt1 = 0;
                }
                break;
            case 4:
                nSamples = *cnt2;
                if (nSamples > 0) {
                    InitAffineShiftByMean(layerElem->nodeNum, layerElem->actfunVecs[1]->variables, layerElem->actfunVecs[2]->variables);
                    *cnt2 = 0;
                }
                break;
            default:
                break;
            }
        }
        break;
    default:
        break;
    }
}

/* cz277 - pact */
/*void ForwardPropBlank(ANNSet *annSet, int batLen) {
    int i, j, n;
    AILink curAI;
    ADLink annDef;
    LELink layerElem;

    curAI = annSet->defsHead;
    while (curAI != NULL) {
        annDef = curAI->annDef;
        for (i = 0; i < annDef->layerNum; ++i) {
            layerElem = annDef->layerList[i];
            n = IntVecSize(layerElem->drvCtx);
            for (j = 1; j <= n; ++j) {
                DoStaticUpdateOperation(layerElem->status, j, layerElem, batLen);
            }
        }
        curAI = curAI->next;
    }
}*/

void ComputeForwardPropActivation(int batLen, LELink layerElem, int ctxIdx) {
    NMatrix *yNMat;
                
    yNMat = layerElem->yFeaMats[ctxIdx];
    switch (layerElem->actfunKind) {
    case AFFINEAF:
        ApplyAffineAct(yNMat, batLen, layerElem->nodeNum, layerElem->actfunVecs[1]->variables, layerElem->actfunVecs[2]->variables, yNMat);
        break;
    case HERMITEAF:
        ApplyHermiteAct(yNMat, batLen, layerElem->nodeNum, layerElem->actfunVecs[1]->variables, yNMat);
        break;
    case LINEARAF:
        CopyNSegment(yNMat, 0, batLen * layerElem->nodeNum, yNMat, 0);
        break;
    case RELUAF:
        ApplyReLUAct(yNMat, batLen, layerElem->nodeNum, 0.0, yNMat);
        break;
    case PRELUAF:
        ApplyPReLUAct(yNMat, batLen, layerElem->nodeNum, layerElem->actfunVecs[1]->variables, yNMat);
        break;
    case PARMRELUAF:
        if (layerElem->trainInfo != NULL)
            CopyNSegment(yNMat, 0, batLen * layerElem->nodeNum, layerElem->trainInfo->cacheMats[ctxIdx], 0);
        ApplyParmReLUAct(yNMat, batLen, layerElem->nodeNum, layerElem->actfunVecs[1]->variables, layerElem->actfunVecs[2]->variables, yNMat);
        break;
    case SIGMOIDAF:
        ApplySigmoidAct(yNMat, batLen, layerElem->nodeNum, yNMat);
        break;
    case LHUCSIGMOIDAF:
        ApplyLHUCSigmoidAct(yNMat, batLen, layerElem->nodeNum, layerElem->actfunVecs[1]->variables, yNMat);
        break;
    case PSIGMOIDAF:
        ApplyPSigmoidAct(yNMat, batLen, layerElem->nodeNum, layerElem->actfunVecs[1]->variables, yNMat);
        break;
    case PARMSIGMOIDAF:
        if (layerElem->trainInfo != NULL)
            CopyNSegment(yNMat, 0, batLen * layerElem->nodeNum, layerElem->trainInfo->cacheMats[ctxIdx], 0);
        ApplyParmSigmoidAct(yNMat, batLen, layerElem->nodeNum, layerElem->actfunVecs[1]->variables, layerElem->actfunVecs[2]->variables, layerElem->actfunVecs[3]->variables, yNMat);
        break;
    case SOFTRELUAF:
        ApplySoftReLAct(yNMat, batLen, layerElem->nodeNum, yNMat);
        break;
    case SOFTMAXAF:
        ApplySoftmaxAct(yNMat, batLen, layerElem->nodeNum, yNMat);
        break;
    case TANHAF:
        ApplyTanHAct(yNMat, batLen, layerElem->nodeNum, yNMat);
        break;
    default:
        HError(8791, "ComputeActivationForwardPropBatch: Unknown activation function type");
    }
}

void ForwardPropActivationOnlyLayer(int batLen, LELink layerElem) {

    HError(8701, "ForwardPropActivationOnlyLayer: Function not implemented");
    return;
}

void ForwardPropConvolutionLayer(int batLen, LELink layerElem) {

    HError(8701, "ForwardPropConvolutionLayer: Function not implemented");
    return;
}

void ForwardPropPerceptronLayer(int batLen, LELink layerElem) {
    int i, n;

    if (layerElem->layerKind != PERCEPTRONLAK)
        HError(8792, "ForwardPropPerceptronLayer: Function can only process a PERCEPTRON layer");

    n = IntVecSize(layerElem->drvCtx);
    for (i = 1; i <= n; ++i) {
        /* y = b, B^T should be row major matrix, duplicate the bias vectors */ 
        DupNVector(layerElem->biasVec->variables, layerElem->yFeaMats[i], batLen);
        /* y += w * b, X^T is row major, W^T is column major, Y^T = X^T * W^T + B^T */
        HNBlasTNgemm(layerElem->nodeNum, batLen, layerElem->inputDim, 1.0, layerElem->wghtMat->variables, layerElem->xFeaMats[i], 1.0, layerElem->yFeaMats[i]);
        /* cz277 - pact */
        DoStaticUpdateOperation(layerElem->status, i, layerElem, batLen);
        /* apply activation transformation */
        ComputeForwardPropActivation(batLen, layerElem, i);
    }

    return;
}

void ForwardPropSubsamplingLayer(int batLen, LELink layerElem) {

    HError(8701, "ForwardPropSubsamplingLayer: Function not implemented");
    return;
}


/* the batch with input features are assumed to be filled */
void ForwardProp(ANNSet *annSet, int batLen, int *CMDVecPL) {
    /*int i, j, n;*/
    int i;
    AILink curAI;
    ADLink annDef;
    LELink layerElem;

    /* update the batch index */
    ++batchIndex;

    /* init the ANNInfo pointer */
    curAI = annSet->defsHead;
    /* proceed in the forward fashion */
    while (curAI != NULL) {
        /* fetch current ANNDef */
        annDef = curAI->annDef;
        /* proceed layer by layer */
        for (i = 0; i < annDef->layerNum; ++i) {
            layerElem = annDef->layerList[i];
            FillBatchFromFeaMix(layerElem, batLen);
            switch (layerElem->layerKind) {
            case ACTIVATIONONLYLAK: ForwardPropActivationOnlyLayer(batLen, layerElem); break;
            case CONVOLUTIONLAK: ForwardPropConvolutionLayer(batLen, layerElem); break;
            case PERCEPTRONLAK: ForwardPropPerceptronLayer(batLen, layerElem); break;
            case SUBSAMPLINGLAK: ForwardPropSubsamplingLayer(batLen, layerElem); break;
            default:
                HError(8791, "ForwardProp: Unknown layer kind");
            }
        }
        /* get the next ANNDef */
        curAI = curAI->next;
    }
}


/* cz277 - gradprobe */
#ifdef GRADPROBE
void AccGradProbeWeight(LayerElem *layerElem) {
    int i, j, k, size;
    NFloat *wghtMat;

#ifdef CUDA
    SyncNMatrixDev2Host(layerElem->wghtMat->gradients);
#endif
    wghtMat = layerElem->wghtMat->gradients->matElems;
    /* weights */
    size = layerElem->nodeNum * layerElem->inputDim;
    j = DVectorSize(layerElem->wghtGradInfoVec);
    for (i = 0; i < size; ++i) {
        if (wghtMat[i] > layerElem->maxWghtGrad)
            layerElem->maxWghtGrad = wghtMat[i];
        if (wghtMat[i] < layerElem->minWghtGrad)
            layerElem->minWghtGrad = wghtMat[i];
        layerElem->meanWghtGrad += wghtMat[i];
        k = wghtMat[i] / PROBERESOLUTE + j / 2;
        layerElem->wghtGradInfoVec[k + 1] += 1;
    }
}
#endif

/* cz277 - gradprobe */
#ifdef GRADPROBE
void AccGradProbeBias(LayerElem *layerElem) {
    int i, j, k, size;
    NFloat *biasVec;

#ifdef CUDA
    SyncNVectorDev2Host(layerElem->biasVec->gradients);
#endif
    biasVec = layerElem->biasVec->gradients->vecElems;
    /* biases */
    size = layerElem->nodeNum;
    j = DVectorSize(layerElem->biasGradInfoVec);
    for (i = 0; i < size; ++i) {
        if (biasVec[i] > layerElem->maxBiasGrad)
            layerElem->maxBiasGrad = biasVec[i];
        if (biasVec[i] < layerElem->minBiasGrad)
            layerElem->minBiasGrad = biasVec[i];
        layerElem->meanBiasGrad += biasVec[i];
        k = biasVec[i] / PROBERESOLUTE + j / 2;
        layerElem->biasGradInfoVec[k + 1] += 1;
    }
}
#endif


/* function to compute the error signal for frame level criteria (for sequence level, do nothing) */
/*void CalcOutLayerBackwardSignal(LELink layerElem, int batLen, ObjFunKind objfunKind, int ctxIdx) {*/
void ComputeBackwardPropOutActivation(ObjFunKind objfunKind, int batLen, LELink layerElem, int ctxIdx) {

    if (layerElem->isFinalLayer == FALSE) 
        HError(8792, "ComputeBackwardPropInputSignal: Function only valid for output layers");
    if (ctxIdx != 1 || layerElem->drvCtx[ctxIdx] != 0) 
      HError(8701, "ComputeBackwardPropInputSignal: Out layer can only have single current frame now");

    switch (objfunKind) {
    case MMSEOF:
        /* proceed for MMSE objective function */
        switch (layerElem->actfunKind) {
        case LINEARAF:
            SubNMatrix(layerElem->yFeaMats[ctxIdx], layerElem->trainInfo->labMat, batLen, layerElem->nodeNum, layerElem->yFeaMats[ctxIdx]);
            break; 
        default:
            HError(8701, "ComputeBackwardPropInputSignal: Unsupported output activation function for MMSE");
        }
        break;
    case XENTOF:
        /* proceed for XENT objective function */
        switch (layerElem->actfunKind) {
        case SOFTMAXAF:
            SubNMatrix(layerElem->yFeaMats[ctxIdx], layerElem->trainInfo->labMat, batLen, layerElem->nodeNum, layerElem->yFeaMats[ctxIdx]);
            break;
        default:
            HError(8701, "ComputeBackwardPropInputSignal: Unsupported output activation function for XENT");
        }
        break;
    case MLOF:
    case MMIOF:
    case MPEOF:
    case MWEOF:
    case SMBROF:
        break;
    default:
        HError(8791, "ComputeBackwardPropInputSignal: Unknown objective function kind");
    }
}

void ComputeBackwardPropHiddenActivation(int batLen, Boolean accFlag, LELink layerElem, int ctxIdx) {
    NMatrix *yNMat, *dyNMat;

    yNMat = layerElem->yFeaMats[ctxIdx];
    dyNMat = layerElem->trainInfo->dyFeaMats[ctxIdx];      /* sum_k w_{k,j} * delta_k */
    switch (layerElem->actfunKind) {
    case AFFINEAF:
        ApplyTrAffineAct(dyNMat, yNMat, batLen, layerElem->nodeNum, layerElem->actfunVecs[1]->variables, layerElem->actfunVecs[2]->variables, accFlag, layerElem->actfunVecs[1]->gradients, layerElem->actfunVecs[2]->gradients);
        ApplyDAffineAct(yNMat, batLen, layerElem->nodeNum, layerElem->actfunVecs[1]->variables, layerElem->actfunVecs[2]->variables, yNMat); 
        break;
    case HERMITEAF:
        HError(8701, "ComputeBackwardPropHiddenActivation: HERMITE Not implemented yet");
        break;
    case LINEARAF:
        ApplyDLinearAct(yNMat, batLen, layerElem->nodeNum, yNMat);
        break;
    case RELUAF:
        ApplyDReLUAct(yNMat, batLen, layerElem->nodeNum, 0.0, yNMat);
        break;
    case PRELUAF:
        ApplyTrPReLUAct(dyNMat, yNMat, batLen, layerElem->nodeNum, layerElem->actfunVecs[1]->variables, accFlag, layerElem->actfunVecs[1]->gradients);
        ApplyDPReLUAct(yNMat, batLen, layerElem->nodeNum, layerElem->actfunVecs[1]->variables, yNMat);
        break;
    case PARMRELUAF:
        ApplyTrParmReLUAct(dyNMat, layerElem->trainInfo->cacheMats[ctxIdx], batLen, layerElem->nodeNum, accFlag, layerElem->actfunVecs[1]->gradients, layerElem->actfunVecs[2]->gradients);
        ApplyDParmReLUAct(layerElem->trainInfo->cacheMats[ctxIdx], batLen, layerElem->nodeNum, layerElem->actfunVecs[1]->variables, layerElem->actfunVecs[2]->variables, yNMat);
        break;
    case SIGMOIDAF:
        ApplyDSigmoidAct(yNMat, batLen, layerElem->nodeNum, yNMat);
        break;
    case LHUCSIGMOIDAF:
        ApplyTrLHUCSigmoidAct(dyNMat, yNMat, batLen, layerElem->nodeNum, layerElem->actfunVecs[1]->variables, accFlag, layerElem->actfunVecs[1]->gradients);
        ApplyDLHUCSigmoidAct(yNMat, batLen, layerElem->nodeNum, layerElem->actfunVecs[1]->variables, yNMat);
        break;
    case PSIGMOIDAF:
        ApplyTrPSigmoidAct(dyNMat, yNMat, layerElem->actfunVecs[1]->variables, batLen, layerElem->nodeNum, accFlag, layerElem->actfunVecs[1]->gradients);
        ApplyDPSigmoidAct(yNMat, batLen, layerElem->nodeNum, layerElem->actfunVecs[1]->variables, yNMat);
        break;
    case PARMSIGMOIDAF:
        ApplyTrParmSigmoidAct(dyNMat, layerElem->trainInfo->cacheMats[ctxIdx], batLen, layerElem->nodeNum, layerElem->actfunVecs[1]->variables, layerElem->actfunVecs[2]->variables, layerElem->actfunVecs[3]->variables, accFlag, layerElem->actfunVecs[1]->gradients, layerElem->actfunVecs[2]->gradients, layerElem->actfunVecs[3]->gradients);
        ApplyDParmSigmoidAct(yNMat, batLen, layerElem->nodeNum, layerElem->actfunVecs[1]->variables, layerElem->actfunVecs[2]->variables, layerElem->actfunVecs[3]->variables, yNMat);
        break;
    case SOFTRELUAF:
        ApplyDSoftReLAct(yNMat, batLen, layerElem->nodeNum, yNMat);
        break;
    case SOFTMAXAF:
        HError(8701, "ComputeBackwardPropHiddenActivation: SOFTMAX as hidden activation function not implemented yet!");
        break;
    case TANHAF:
        ApplyDTanHAct(yNMat, batLen, layerElem->nodeNum, yNMat);
        break;
    default:
        HError(8791, "ComputeBackwardPropHiddenActivation: Unsupported hidden activation function kind");
    }
}


/* attention: these two operations are gonna to change dyFeaMat elements to their square */
void ComputeBackwardPropSumSquaredGradients(int batLen, LELink layerElem, NMatrix *dyFeaMat) {
    int i, n;
    
    n = IntVecSize(layerElem->drvCtx);
    for (i = 1; i <= n; ++i) {
        SquaredNMatrix(layerElem->xFeaMats[i], batLen, layerElem->inputDim, GetTmpNMat());
        SquaredNMatrix(dyFeaMat, batLen, layerElem->nodeNum, dyFeaMat);
        if (layerElem->wghtMat->updateflag == TRUE)
            HNBlasNTgemm(layerElem->inputDim, layerElem->nodeNum, batLen, 1.0, GetTmpNMat(), dyFeaMat, 1.0, layerElem->wghtMat->sumsquaredgrad);
        if (layerElem->biasVec->updateflag == TRUE)
            SumNMatrixByCol(dyFeaMat, batLen, layerElem->nodeNum, TRUE, layerElem->biasVec->sumsquaredgrad);
    }
}

void BackwardPropActivationOnlyLayer(ObjFunKind objfunKind, int batLen, Boolean accFlag, LELink layerElem) {

    HError(8701, "BackwardPropActivationOnlyLayer: Function not implemented yet!");
    return;
}

void BackwardPropConvolutionLayer(ObjFunKind objfunKind, int batLen, Boolean accFlag, LELink layerElem) {

    HError(8701, "BackwardPropConvolutionLayer: Function not implemented yet!");
    return;
}

void BackwardPropPerceptronLayer(ObjFunKind objfunKind, int batLen, Boolean accFlag, LELink layerElem) {
    int i, n;
    Boolean acc;
    NMatrix *dyNMat;

    n = IntVecSize(layerElem->drvCtx);
    for (i = 1, acc = accFlag; i <= n; ++i, acc = TRUE) {
        if (layerElem->isFinalLayer) {
            /* delta_k */
            dyNMat = layerElem->yFeaMats[i];
            ComputeBackwardPropOutActivation(objfunKind, batLen, layerElem, i);
        }
        else {
            /* sum_k w_{k,j} * delta_k */
            dyNMat = layerElem->trainInfo->dyFeaMats[i];
            ComputeBackwardPropHiddenActivation(batLen, acc, layerElem, i);
            /* times sigma_k (dyFeaMat, from the next layer) */
            /* dyFeaMat: sum_k w_{k,j} * delta_k -> delta_j */
            /* delta_j = h'(a_j) * (sum_k w_{k,j} * delta_k) */
            MulNMatrix(layerElem->yFeaMats[i], dyNMat, batLen, layerElem->nodeNum, dyNMat);
        }
        /* Y^T is row major, W^T is column major, X^T = Y^T * W^T */
        /* sum_k w_{k,j} * delta_k */
        HNBlasNNgemm(layerElem->inputDim, batLen, layerElem->nodeNum, 1.0, layerElem->wghtMat->variables, dyNMat, 0.0, layerElem->trainInfo->dxFeaMats[i]);
        /* compute and accumulate the updates */
        /* {layerElem->xFeaMat[n_frames * inputDim]}^T * dyFeaMat[n_frames * nodeNum] = deltaWeights[inputDim * nodeNum] */
        if (layerElem->wghtMat->updateflag == TRUE) {
            HNBlasNTgemm(layerElem->inputDim, layerElem->nodeNum, batLen, 1.0, layerElem->xFeaMats[i], dyNMat, acc, layerElem->wghtMat->gradients);
#ifdef GRADPROBE
            AccGradProbeWeight(layerElem);
#endif
        }
        /* graidents for biases */
        if (layerElem->biasVec->updateflag == TRUE) {
            SumNMatrixByCol(dyNMat, batLen, layerElem->nodeNum, acc, layerElem->biasVec->gradients);
#ifdef GRADPROBE
            AccGradProbeBias(layerElem);
#endif
        }
        /* cz277 - ssginfo*/
        if (layerElem->wghtMat->sumsquaredgrad != NULL && layerElem->biasVec->sumsquaredgrad != NULL)
            ComputeBackwardPropSumSquaredGradients(batLen, layerElem, dyNMat);
    }
    
    return;
}

void BackwardPropSubsamplingLayer(ObjFunKind objfunKind, int batLen, Boolean accFlag, LELink layerElem) {

    HError(8701, "BackwardPropSubsamplingLayer: Function not implemented yet!");
    return;
}


/* delta_j = h'(a_j) * sum_k [w_k,j * delta_k] */
/*   dtl_j = sum_k [w_k,j * dtl_k * h'(a_k)] */
/*   dtl_j = delta_j / h'(a_j) */
/* backward propagation algorithm */
void BackwardProp(ObjFunKind objfunKind, ANNSet *annSet, int batLen, Boolean accFlag) {
    int i;
    AILink curAI;
    ADLink annDef;
    LELink layerElem;

    /* init the ANNInfo pointer */
    curAI = annSet->defsTail;
    /* proceed in the backward fashion */
    while (curAI != NULL) {
        /* fetch current ANNDef */
        annDef = curAI->annDef;
        /* proceed layer by layer */
        for (i = annDef->layerNum - 1; i >= 0; --i) {
            /* get current LayerElem */
            layerElem = annDef->layerList[i];
            FillBatchFromErrMix(layerElem, batLen);	/* cz277 - many */
            switch (layerElem->layerKind) {
            case ACTIVATIONONLYLAK: BackwardPropActivationOnlyLayer(objfunKind, batLen, accFlag, layerElem); break;
            case CONVOLUTIONLAK: BackwardPropConvolutionLayer(objfunKind, batLen, accFlag, layerElem); break;
            case PERCEPTRONLAK: BackwardPropPerceptronLayer(objfunKind, batLen, accFlag, layerElem); break;
            case SUBSAMPLINGLAK: BackwardPropSubsamplingLayer(objfunKind, batLen, accFlag, layerElem); break;
            default:
                HError(8791, "BackwardProp: Unknown layer kind");
            }
        }
        /* get the previous ANNDef */
        curAI = curAI->prev;
    }
}

void ResetAllBundleProcessedFields(char *invoker, ANNSet *annSet) {
    int i, j;
    AILink curAI;
    ADLink annDef;
    LELink layerElem;

    /* reset all processed fields to FALSE */
    curAI = annSet->defsHead;
    while (curAI != NULL) {
        annDef = curAI->annDef;
        for (i = 0; i < annDef->layerNum; ++i) {
            layerElem = annDef->layerList[i];
            switch (layerElem->layerKind) {
            case ACTIVATIONONLYLAK: HError(8701, "%s: Function not implemented", invoker); break;
            case CONVOLUTIONLAK: HError(8701, "%s: Function not implemented", invoker); break;
            case PERCEPTRONLAK:
                layerElem->wghtMat->processed = FALSE;
                layerElem->biasVec->processed = FALSE;
                if (layerElem->actfunVecs != NULL)
                    for (j = 1; j <= layerElem->actfunParmNum; ++j)
                        layerElem->actfunVecs[j]->processed = FALSE;
                break;
            case SUBSAMPLINGLAK: HError(8701, "%s: Function not implemented", invoker); break;
            default:
                HError(8791, "%s: Unknown layer kind", invoker);
            }
        }
        /* get the next ANNDef */
        curAI = curAI->next;
    }
}

static void NormNVecBundleGradient(NVecBundle *bundle, float scale) {
    BTLink curLink;
    LELink layerElem;
    int drvCnt = 0; 

    if (bundle->kind != SIBK)
        HError(8792, "NormNVecBundleGradient: Only SIBK bundle is allowed");
    if (bundle->hook == NULL)
        HError(8793, "NormNVecBundleGradient: SI bundle should have the trace field set");
    /*if (bundle->batchIndex != batchIndex && bundle->batchIndex != batchIndex + 1)
        HError(9999, "NormNVecBundleGradient: Wrong bundle batch index");*/
    if (scale == 0.0)
        HError(8793, "NormNVecBundleGradient: Input scaling factor can not be 0, try 1.0");
    if (bundle->processed == TRUE)
        return;
    bundle->processed = TRUE;

    curLink = (BTLink) bundle->hook;
    while (curLink != NULL) {
        layerElem = curLink->layerElem;        
        if (layerElem->trainInfo == NULL)
            HError(8700, "NormNVecBundleGradient: All trainInfo should be initialised");
        drvCnt += layerElem->trainInfo->tDrvCnt;
        curLink = curLink->nextTrace;
    }
    if (scale * drvCnt != 1.0)
        ScaleNVector(1.0 / (scale * drvCnt), bundle->gradients->vecLen, bundle->gradients);
    /* cz277 - ssgInfo */

}

static void NormNMatBundleGradient(NMatBundle *bundle, float scale) {
    BTLink curLink;
    LELink layerElem;
    int drvCnt = 0; 

    if (bundle->kind != SIBK)
        HError(8792, "NormNMatBundleGradient: Only SIBK bundle is allowed");
    if (bundle->hook == NULL)
        HError(8793, "NormNMatBundleGradient: SI bundle should have the trace field set");
    if (scale == 0.0)
        HError(8793, "NormNMatBundleGradient: Input scaling factor can not be 0, try 1.0");
    if (bundle->processed == TRUE)
        return;
    bundle->processed = TRUE;

    curLink = (BTLink) bundle->hook;
    while (curLink != NULL) {
        layerElem = curLink->layerElem;
        if (layerElem->trainInfo == NULL)
            HError(8700, "NormNMatBundleGradient: All trainInfo should be initialised");
        drvCnt += layerElem->trainInfo->tDrvCnt;
        curLink = curLink->nextTrace;
    }
    if (scale * drvCnt != 1.0)
        ScaleNMatrix(1.0 / (scale * drvCnt), bundle->gradients->rowNum, bundle->gradients->colNum, bundle->gradients);
    /* cz277 - ssgInfo */

}

/* the batch with input features are assumed to be filled */
void NormBackwardPropGradients(ANNSet *annSet, float scale) {
    int i, j;
    AILink curAI;
    ADLink annDef;
    LELink layerElem;

    /* reset all processed fields to FALSE */
    ResetAllBundleProcessedFields("NormBackwardPropGradients", annSet);

    /* init the ANNInfo pointer */
    curAI = annSet->defsHead;
    /* proceed in the forward fashion */
    while (curAI != NULL) {
        /* fetch current ANNDef */
        annDef = curAI->annDef;
        /* proceed layer by layer */
        for (i = 0; i < annDef->layerNum; ++i) {
            layerElem = annDef->layerList[i];
            switch (layerElem->layerKind) {
            case ACTIVATIONONLYLAK: HError(8701, "NormBackwardPropGradients: Function not implemented"); break;
            case CONVOLUTIONLAK: HError(8701, "NormBackwardPropGradients: Function not implemented"); break;
            case PERCEPTRONLAK:
                NormNMatBundleGradient(layerElem->wghtMat, scale);
                NormNVecBundleGradient(layerElem->biasVec, scale);
                if (layerElem->actfunVecs != NULL)
                    for (j = 1; j <= layerElem->actfunParmNum; ++j)
                        NormNVecBundleGradient(layerElem->actfunVecs[j], scale);
                break;
            case SUBSAMPLINGLAK: HError(8701, "NormBackwardPropGradients: Function not implemented"); break;
            default:
                HError(8791, "NormBackwardPropGradients: Unknown layer kind");
            }
        }
        /* get the next ANNDef */
        curAI = curAI->next;
    }
}

/* randomise an ANN layer */
void RandANNLayer(LELink layerElem, int seed, float scale) {
    float r;

    switch (layerElem->actfunKind) {
    case AFFINEAF:
    case LINEARAF:
    case RELUAF:
    case PRELUAF:
    case PARMRELUAF:
    case SOFTRELUAF:
        r = 16.0 / ((float) (layerElem->nodeNum + layerElem->inputDim));
	/* 0.004 for a (2000, 2000) layer; r = 0.001 for a (12000, 2000) layer */	
        r *= scale;
        RandInit(seed);
        RandNSegmentUniform(-1.0 * r, r, layerElem->nodeNum * layerElem->inputDim, layerElem->wghtMat->variables->matElems);
        break;
        /*r = sqrt(2.0 / ((1.0 + pow(PLRELUNEGSCALE, 2.0)) * layerElem->nodeNum));     
        RandNSegmentGaussian(0.0, r, layerElem->wghtMat->rowNum * layerElem->wghtMat->colNum, layerElem->wghtMat->matElems);*/
    default:
        r = 4 * sqrt(6.0 / (float) (layerElem->nodeNum + layerElem->inputDim));
        r *= scale;
        RandInit(seed);
        RandNSegmentUniform(-1.0 * r, r, layerElem->nodeNum * layerElem->inputDim, layerElem->wghtMat->variables->matElems);
	/* r = 0.22 for a (1000, 1000) layer; r = 0.083 for a (12000, 2000) layer */
        break;
    }

    /*if (layerElem->actfunKind == RELAF || layerElem->actfunKind == SOFTRELAF) {
        RandMaskNSegment(0.25, 0.0, layerElem->wghtMat->rowNum * layerElem->wghtMat->colNum, layerElem->wghtMat->matElems);
    }*/

    ClearNVector(layerElem->biasVec->variables);
    /* TODO: if HERMITEAF */
#ifdef CUDA
    SyncNMatrixHost2Dev(layerElem->wghtMat->variables);
    SyncNVectorDev2Host(layerElem->biasVec->variables);
#endif

}

/* generate a new ANN layer and randomise it */
/*LELink GenNewPerceptronLayer(HMMSet *hset, int nodeNum, int inputDim, char *wghtName, char *biasName) {
    LELink layerElem;

    layerElem = GenBlankLayer(hset->heap);
    layerElem->nodeNum = nodeNum;
    layerElem->inputDim = inputDim;
    layerElem->wghtMat = FetchNMatBundle(hset, wghtName);
    if (layerElem->wghtMat->variables == NULL) {
        layerElem->wghtMat->variables = CreateNMatrix(heap, nodeNum, inputDim);
        layerElem->wghtMat->kind = SIBK;
    }
    else {
        if (layerElem->wghtMat->variables->rowNum != nodeNum)
            HError(9999, "GenNewPerceptronLayer: Wrong weight matrix row");
        if (layerElem->wghtMat->variables->colNum != inputDim)
            HError(9999, "GenNewPerceptronLayer: Wrong weight matrix column");
    }
    CreateBundleTrace(heap, layerElem, &layerElem->wghtMat->hook);
    layerElem->biasVec = FetchNVecBundle(hset, biasName);
    if (layerElem->biasVec->variables == NULL) {
        layerElem->biasVec->variables = CreateNVector(heap, nodeNum);
        layerElem->biasVec->kind = SIBK;
    }
    else if (layerElem->biasVec->variables->vecLen != nodeNum)
        HError(9999, "GenNewPerceptronLayer: Wrong bias vector length");
    CreateBundleTrace(heap, layerElem, &layerElem->biasVec->hook);

    return layerElem;     
}*/

/*void SetFeaMixBatchIdxes(ANNSet *annSet, int newIdx) {
    int i;
    AILink curAI;
    ADLink annDef;
    LELink layerElem;
    NMatrix *dyFeaMat;

    curAI = annSet->defsTail;
    while (curAI != NULL) {
        annDef = curAI->annDef;
        for (i = annDef->layerNum - 1; i >= 0; --i) {
            layerElem = annDef->layerList[i];
	    if (layerElem->feaMix->batIdx == 0) {
                layerElem->feaMix->batIdx = newIdx; 
            }
        }
        curAI = curAI->next;
    }
}*/

/* cz277 - max norm2 */
Boolean IsLinearInvariant(ActFunKind actfunKind) {
    switch (actfunKind) {
        case LINEARAF:
        case RELUAF:
            return TRUE;
        default:
            return FALSE;
    }
}

/* cz277 - pact */
Boolean CacheActMatrixOrNot(ActFunKind actfunKind) {
    switch (actfunKind) {
    case PARMRELUAF:
    case PARMSIGMOIDAF:
        return TRUE;
    default:
        return FALSE;
    }
}

/* cz277 - 150824 */
void CreateBundleTrace(MemHeap *heap, LELink layerElem, BTLink *head) {
    BTLink *trace;

    /* create a node at the end of the tracing list */
    trace = head;
    while (*trace != NULL)
        trace = &(*trace)->nextTrace;
    /* allocate a new BundleTrace */
    *trace = (BTLink) New(heap, sizeof(BundleTrace));
    memset(*trace, 0, sizeof(BundleTrace));
    (*trace)->layerElem = layerElem;
}

void CancelBundleTrace(MemHeap *heap, LELink layerElem, BTLink *head) {
    BTLink *trace, *prev = NULL;
    
    trace = head;
    while (*trace != NULL && (*trace)->layerElem != layerElem) {
        prev = trace;
        trace = &(*trace)->nextTrace;
    }
    if (*trace == NULL)
        HError(8700, "CancelBundleTrace: Fail to get the target layer from the trace list");
    if (prev == NULL) 
        *head = (*trace)->nextTrace;
    else 
       (*prev)->nextTrace = (*trace)->nextTrace; 
    if (heap->type != MSTAK)
        Dispose(heap, *trace);
}

/* ------------------------- End of HANNet.c ------------------------- */

