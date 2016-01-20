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
/*         File: HANNet.h  ANN model definition data type      */
/* ----------------------------------------------------------- */

/* !HVER!HANNet:   3.5.0 [CUED 12/10/15] */


#ifndef _HANNET_H_
#define _HANNET_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include "HMem.h"

/* ------------------------- Trace Flags ------------------------- */

/*
 The following types define the in-memory representation of a ANN
*/

/* ------------------------- Predefined Types ------------------------- */

#define MAPTARGETPREF "_MAPPED"	/* the prefix to convert the mapped target */
/* cz277 - 1007 */
#define MAXFEAMIXOWNER 20	/* the maximum number of layers or hset streams that could own a feature mixture */

/* cz277 - gradprobe */
#ifdef GRADPROBE
#define PROBERESOLUTE 50
#define PROBEBOUNDARY 1000
#endif

#define AUTOMACRONAMEPREFIX "AUTOMACRONAME_"

/* ------------------------- ANN Definition ------------------------- */

enum _InputKind {INPFEAIK, ANNFEAIK, ERRFEAIK, AUGFEAIK}; /* cz277 - aug */
typedef enum _InputKind InputKind;

enum _ActFunKind {AFFINEAF, HERMITEAF, LINEARAF, RELUAF, LHUCRELUAF, PRELUAF, PARMRELUAF, SIGMOIDAF, LHUCSIGMOIDAF, PSIGMOIDAF, PARMSIGMOIDAF, SOFTRELUAF, LHUCSOFTRELUAF, PSOFTRELUAF, PARMSOFTRELUAF, SOFTMAXAF, TANHAF};
typedef enum _ActFunKind ActFunKind;

enum _LayerKind {ACTIVATIONONLYLAK, CONVOLUTIONLAK, PERCEPTRONLAK, SUBSAMPLINGLAK};
typedef enum _LayerKind LayerKind;

enum _ObjFunKind {UNKOF = 0, MLOF = 1, MMIOF = 2, MMSEOF = 4, MPEOF = 8, MWEOF = 16, SMBROF = 32, XENTOF = 64};
typedef enum _ObjFunKind ObjFunKind;

/*enum _ANNUpdtKind {WEIGHTUK = 1, BIASUK = 2, ACTFUNUK = 4};*/     /* cz277 - 150811 */
/*typedef enum _ANNUpdtKind ANNUpdtKind;*/

enum _BundleKind {SDBK, SIBK};
typedef enum _BundleKind BundleKind;


typedef struct _LayerElem *LELink;
typedef struct _FeaElem *FELink;
typedef struct _ANNDef *ADLink;
typedef struct _ANNInfo *AILink;
typedef struct _LayerInfo *LILink;
typedef struct _RPLInfo *RILink;
typedef struct _BundleTrace *BTLink;

/* cz277 - 150811 */
typedef struct _BundleTrace {
    LELink layerElem;
    size_t x;
    size_t y;
    BTLink nextTrace;
} BundleTrace;

/* cz277 - 150811 */
typedef struct _NVecBundle {
    LabId id;
    BundleKind kind;
    NVector *variables;
    NVector *gradients;
    NVector *updates;
    NVector *neglearnrates;
    NVector *sumsquaredgrad;
    int batchIndex;
    Boolean processed;
    Boolean updateflag;		/* cz277 - 151020 */
    size_t accum;
    size_t *accptr;
    int nUse;
    Ptr hook;
} NVecBundle;

/* cz277 - 150811 */
typedef struct _NMatBundle {
    LabId id;
    BundleKind kind;
    NMatrix *variables;
    NMatrix *gradients;
    NMatrix *updates;
    NMatrix *neglearnrates;
    NMatrix *sumsquaredgrad;
    int batchIndex;
    Boolean processed;
    Boolean updateflag;		/* cz277 - 151020 */
    size_t accum;
    size_t *accptr;
    int nUse;
    Ptr hook;
} NMatBundle;

/* cz277 - xform */
typedef struct _RPLInfo {
    int nSpkr;
    char *inRPLMask;
    char curOutSpkr[MAXSTRLEN];
    char curInSpkr[MAXSTRLEN];
    char cacheInSpkr[MAXSTRLEN];
    char *inRPLDir;
    char *inRPLExt;
    char *outRPLDir;
    char *outRPLExt;
    /*char *macroName;*/
    Boolean saveBinary;
    LabId id;
    union {
        NMatBundle *curNMat;
        NVecBundle *curNVec;
    };
    union {
        NMatBundle savNMat;
        NVecBundle savNVec;
    };
    RILink nextInfo;
} RPLInfo;

typedef struct _FeaElem {
    int feaDim;                 /* the dimension of this kind of feature (without context expansion) */
    int extDim;                 /* the dimension of this kind of feature (with context expansion and transforms) */
    IntVec ctxMap;              /* the array contains the offset to current for context expansion */
    InputKind inputKind;        /* the kind of the feature */
    LELink feaSrc;              /* the layer pointer to the source of current feature element */
    NMatrix **feaMats;		/* cz277 - many */
    int dimOff;                 /* the offset of the start dimension in feaMat of this FeaElem; useful for backprop */
    int srcDim;                 /* the dimension of the feature in feaMat */
    int augFeaIdx;		/* the index of this (if it was) augmented feature index; default: 0 */
    int streamIdx;		/* the index of the associated stream; default: 0 */
    char mName[MAXSTRLEN];      /* the ANN feature source macro name */
    char mType;                 /* the ANN feature source macro type */
    Boolean doBackProp;		/* cz277 - semi */
    int hisLen;			/* the length of the history */
    NMatrix *hisMat;		/* the matrix for ANN feature history */
    int nUse;                   /* the usage counter */
    IntVec ctxPool;		/* cz277 - many */ /* tells what each feaMat is associated with (only valid for INPFEAIK) */
} FeaElem;

typedef struct _FeaMix {
    /* cz277 - 1007 */
    int batchIndex;			/* the number of batches been processed */
    int ownerNum;		/* the number of owners of this feature mixture  */
    LELink ownerList[MAXFEAMIXOWNER];		/* the layers which employs this feature mxiture */
    int elemNum;                /* number of different feature components */
    int mixDim;                 /* the total dimension of the input (a mixture of different features) */
    FELink *feaList;            /* the feature information structure */
    NMatrix **mixMats;		/* cz277 - many */
    IntVec ctxPool;		/* cz277 - many */
    int nUse;                   /* usage counter */
} FeaMix;


typedef struct _TrainInfo {
    NMatrix *labMat;            /* the batches for the output targets for all streams */ 
    NMatrix **dxFeaMats;        /* cz277 - many */  /* de/dx */
    NMatrix **dyFeaMats;        /* cz277 - many */  /* de/dy */
    NMatrix **cacheMats;	/* cz277 - 150811 */
    IntVec drvCnt;		/* cz277 - many */
    int tDrvCnt;		/* cz277 - many */
    /*ANNUpdtKind updateFlag;*/     /* whether update this layer or not */
    /*long actfunUpdateFlag;*/	/* cz277 - 150811 */
    Boolean initFlag;		/* cz277 - 150811 */
} TrainInfo;

typedef struct _ANNInfo {
    AILink next;                /* pointer to the next item of the chain */
    AILink prev;                /* pointer to the previous item of the chain */
    ADLink annDef;              /* one owner of this layer */
    int index;                  /* for LayerElem, the index of this layer in that owner ANNDef */
    int fidx;			/* the file index of the associated ANNDef */
} ANNInfo;                      /* for ANNSet, the index of this of this ANNDef in that ANNSet */

typedef struct _LayerElem {
    int ownerCnt;               /* the number of owners in the owner chain */
    ANNInfo *ownerHead;         /* the head of the chain contains all owners */
    ANNInfo *ownerTail;         /* the tail of the chain contains all owners */
    /*ActFunInfo actfunInfo;*/	/* cz277 - pact */
    FeaMix *feaMix;             /* a list of different features for forward propagation */
    FeaMix *errMix;             /* a list of different error signals for back propagation */
    int inputDim;               /* the number of inputs to each node in current layer (column number of wgthMat) */
    int nodeNum;                /* the number of nodes in current layer (row number of wghtMat) */
    NMatBundle *wghtMat;        /* the weight matrix of current layer (a transposed matrix) */
    NVecBundle *biasVec;        /* the bias vector of current layer */
    ActFunKind actfunKind;
    int actfunParmNum;
    NVecBundle **actfunVecs;
    NMatrix **xFeaMats;         /* cz277 - many */  /* the feature batch for the input signal, could point to another yFeaMat in a different LayerElem */
    NMatrix **yFeaMats;         /* cz277 - many */  /* the feature batch for the output signal */
    TrainInfo *trainInfo;       /* the structure for training info, could be NULL (if not training) */
    LayerKind layerKind;     	/* the type of current layer */
    Boolean isFinalLayer;	/* cz277 - 150811 */
    int nUse;                   /* usage counter */
    int nDrv;                   /* feature derived counter */
    IntVec drvCtx;		/* cz277 - many */ 
    int status;			/* cz277 - many */
    /* cz277 - gradprobe */
#ifdef GRADPROBE
    DVector wghtGradInfoVec;
    DVector biasGradInfoVec;
    NFloat maxWghtGrad;
    NFloat minWghtGrad;
    NFloat meanWghtGrad;
    NFloat maxBiasGrad;
    NFloat minBiasGrad;
    NFloat meanBiasGrad;
#endif    
} LayerElem;

typedef struct _ANNDef {
    int layerNum;               /* the number of layers */
    LELink *layerList;          /* a list of layers */
    int targetNum;              /* number of targets in this ANN */
    char *annDefId;             /* identifier for the ANNDef */ 
    int nUse;                   /* usage counter */
    int nDrv;
} ANNDef;

typedef struct _TargetMap {
    char *name;                 /* the input name of this target */
    char *mappedName;           /* the modified name of this mapped target (with MAPTARGETPREF) */
    int index;                  /* the index of this mapped target, in MappedList, maxResults, and sumResults */
    IntVec maxResults;          /* mapped confusion list */
    IntVec sumResults;          /* summed confusion list */
    int sampNum;		/* the total number of samples */
    float mappedTargetPen[SMAX];/* the penalty of the mapped target */
} TargetMap;

typedef struct _TargetMapStruct {
    TargetMap *targetMapList;   /* used to convert the index of a mapped target to its structure */
    int mappedTargetNum;        /* the total number of mapped targets */
    IntVec mapVectors[SMAX];    /* the mapping vectors for target map */
    NMatrix *maskMatMapSum[SMAX]; /* the mat matrix generated by extending mapVec to get outMatMapSum*/
    NMatrix *labMatMapSum[SMAX];
    NMatrix *outMatMapSum[SMAX];/* the mapping matrix for the yFeaMat of the output layers */
    NMatrix *llhMatMapSum[SMAX];/* the mapping matrix with llh values */
    NVector *penVecMapSum[SMAX];
} TargetMapStruct;

typedef struct _ANNSet {
    int annNum;                 /* an ANN is a mixture of a set of sub ANNs */
    AILink defsHead;            /* the head of the chain contains all sub ANNs (ANNDefs) */
    AILink defsTail;            /* the tail of the chain contains all sub ANNs (ANNDefs) */
    LELink outLayers[SMAX];     /* pointers to the output layer in each stream */ 
    TargetMapStruct *mapStruct;	/* the structure for target mapping */
    NMatrix *llhMat[SMAX];	/* the llr matrix of the yFeaMat of the output layers */
    NVector *penVec[SMAX];
} ANNSet;


/* ------------------------ Global Settings ------------------------- */

int GetNBatchSamples(void);
void SetNBatchSamples(int userBatchSamples);
void InitANNet(void);
int GetGlobalBatchIndex(void);
void SetGlobalBatchIndex(int index);

void UpdateOutMatMapSum(ANNSet *annSet, int batLen, int streamIdx);
void UpdateLabMatMapSum(ANNSet *annSet, int batLen, int streamIdx);
void ForwardProp(ANNSet *annSet, int batLen, int *CMDVecPL);
void ComputeBackwardPropOutActivation(ObjFunKind objfunKind, int batLen, LELink layerElem, int ctxIdx);
void BackwardProp(ObjFunKind objfunKind, ANNSet *annSet, int batLen, Boolean accFlag);

void RandANNLayer(LELink layerElem, int seed, float scale);
/*void SetFeaMixBatchIdxes(ANNSet *annSet, int newIdx);*/

/* cz277 - max norm2 */
Boolean IsLinearInvariant(ActFunKind actfunKind);


/* cz277 - pact */
Boolean CacheActMatrixOrNot(ActFunKind actfunKind);

char *GetANNUpdateFlagStr(void);
/*char *GetLayerUpdateFlagStr(void);
char *GetActFunUpdateFlagStr(void);*/
char *GetNMatUpdateFlagStr(void);
char *GetNVecUpdateFlagStr(void);

char *GetMaskStrNMatRPLInfo(void);
char *GetInDirStrNMatRPLInfo(void);
char *GetExtStrNMatRPLInfo(void);
char *GetOutDirStrNMatRPLInfo(void);
char *GetMaskStrNVecRPLInfo(void);
char *GetInDirStrNVecRPLInfo(void);
char *GetExtStrNVecRPLInfo(void);
char *GetOutDirStrNVecRPLInfo(void);
RILink GetHeadNMatRPLInfo(void);
RILink GetHeadNVecRPLInfo(void);
void SetHeadNMatRPLInfo(RILink info);
void SetHeadNVecRPLInfo(RILink info);
int GetNumNMatRPLInfo(void);
int GetNumNVecRPLInfo(void);
void SetNumNMatRPLInfo(int n);
void SetNumNVecRPLInfo(int n);

/* cz277 - 150824 */
void ResetAllBundleProcessedFields(char *invoker, ANNSet *annSet); 
void CheckANNBatchIndex(ANNSet *annSet, int index);
void NormBackwardPropGradients(ANNSet *annSet, float scale);
/* cz277 - batch sync */
void SetNBundleBatchIndex(ANNSet *annSet, int index);
void SetFeaMixBatchIndex(ANNSet *annSet, int index);

void CreateBundleTrace(MemHeap *heap, LELink layerElem, BTLink *head);
void CancelBundleTrace(MemHeap *heap, LELink layerElem, BTLink *head);


#ifdef __cplusplus
}
#endif

#endif  /* _HANNET_H_ */

/* ------------------------- End of HANNet.h ------------------------- */ 
