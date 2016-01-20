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
/*   File: HNTrainSGD.c  SGD based ANN model training program  */
/* ----------------------------------------------------------- */

char *hntrainsgd_version = "!HVER!HNTrainSGD:   3.5.0 [CUED 12/10/15]";
char *hntrainsgd_vc_id = "$Id: HNTrainSGD.c,v 1.0 2015/10/12 12:07:24 cz277 Exp $";

/* 
    This program is used to train various ANN models based on 
    SGD using a single machine.
*/

#include "config.h"
#ifdef IMKL
#include "mkl.h"
#endif
#include "HShell.h"
#include "HMem.h"
#include "HMath.h"
#include "HSigP.h"
#include "HWave.h"
#include "HLabel.h"
#include "HAudio.h"
#include "HParm.h"
#include "HDict.h"
#include "HANNet.h"
#include "HModel.h"
#include "HTrain.h"
#include "HUtil.h"
#include "HAdapt.h"
#include "HFB.h"
#include "HNet.h"       /* for Lattice */
#include "HLM.h"
#include "HLat.h"       /* for Lattice */
#include "HArc.h"
#include "HFBLat.h"
#include "HExactMPE.h"
#include "HNCache.h"

#include <time.h>
#include <math.h>
#include <float.h>
#include <strings.h>

/* -------------------------- Trace Flags & Vars ------------------------ */

/* Trace Flags */
#define T_TOP   0001    /* Top level tracing */
#define T_TIM   0002    /* Output timings */
#define T_SCH   0004    /* Scheduler behavior tracing */
static int trace = 0;
#define EXIT_STATUS 0	/* Exit status */

/* -------------------------- Global Variables etc ---------------------- */

#define MAX(a, b) ((a)>(b)?(a):(b))
#define MIN(a, b) ((a)<(b)?(a):(b))
#define ABS(a) ((a)>0?(a):-(a))
#define FINITE(x) (!isnan(x) && x<1.0e+30 && x>-1.0e+30)

enum _TrainMode {FRAMETM, SEQTM};
typedef enum _TrainMode TrainMode;

enum _LRSchdKind {ADAGRADSK, EXPSK, LISTSK, NEWBOBSK};
typedef enum _LRSchdKind LRSchdKind;

enum _NewBobCrt {ACCNBC, MAPACCNBC, LLHVALNBC, MAPLLHVALNBC};
typedef enum _NewBobCrt NewBobCrt;

enum _BatchUpdateKind {BATLEVEL, UTTLEVEL};
typedef enum _BatchUpdateKind BatchUpdateKind;

typedef struct _CriteriaInfo {
    double cSampAcc;
    double MMSEAcc;
    double XENTAcc;
    double MPEAcc;
    double tUttAcc;
    double tNWordAcc;
    double tSampAcc;
    double LLHAcc;
    double LLHVal;
    double NumLLHAcc;
    double DenLLHAcc;
    int MMIFRAcc;
    double cSampAccMapMax;
    double cSampAccMapSum;
    double MMSEAccMapSum;
    double XENTAccMapSum;
    double LLHAccMapSum;
    double LLHValMapSum;
} CriteriaInfo;

typedef struct _ModelSetInfo *MSILink;

typedef struct _ModelSetInfo {
    char *baseDir;
    char *hmmExt;
    char *macExt;
    char **macFN;
    int macCnt;
    float crtVal;
    MSILink next;
    MSILink prev;
    int epochIdx;
    int updtIdx;
} ModelSetInfo;

/* for AdaGrad */
static int AdaGrad_K = 1;                       /* the K value for AdaGrad learning rate scheduler */
/* for Exponential */
static float Exp_Gamma = 0.0;                   /* the gamma parameter for Exponential learning rate scheduler */
static float Exp_TrSampIdx = 0.0;               /* the number of samples proceeded for Exponential learning rate scheduler */
static float Exp_Base = 10.0;			/* eta * Exp_Base^{-1.0 * Exp_TrSampIdx / Exp_Gamma}*/
/* for List */
static Vector List_LRs;                         /* the learning rate(s) */
/* for NewBob */
static float NewBob_RampStart = 0.005;          /* the min_derror_ramp_start parameter for NewBob learning rate scheduler */
static float NewBob_Stop = 0.005;               /* the min_derror_stop parameter for NewBob learning rate scheduler */
static int NewBob_Status = 0;                   /* the status of NewBob (0: initia, 1: ramping) */
static float NewBob_DecayFactor = 0.5;
static NewBobCrt NewBob_Crt = ACCNBC;           /*  */

static LRSchdKind schdKind = NEWBOBSK;          /* the kind of the learning rate scheduler */
static float initLR = 0.01;                     /* the initial learning rates */
static float curNegLR;
static float minLR = 0.0;                       /* the lower learning rate threshold */
static int minEpochNum = -1;                    /* the minimum number of epoch to run */
static int maxEpochNum = -1;                    /* the maximum number of epoch to run */

/*static float momentum = 0.0;*/                    /* the factor for momentum */
/* cz277 - mmt */
static Vector List_MMTs = NULL;
static Vector List_WeightDecay = NULL;

/*static float weightDecay = 0.0; */                /* the factor for weight decay */
static int epochOff = 1;                        /* current epoch offset */
static float logObsvPrior = 0.0;		/* logP(O) */

static char *hmmListFn = NULL;                  /* model list filename (optional) */  
static char *hmmDir = NULL;                     /* directory to look for HMM def files */
static char *hmmExt = NULL;                     /* HMM def file extension */
static char *mappingFn = NULL;                  /* the name of target mapping file */
static XFInfo xfInfo;                           /* transforms/adaptations */
static char *newDir = NULL;                     /* directory to store new HMM def files */
static char *newExt = NULL;                     /* extension of new retrained HMM files */
static UPDSet uFlags = UPTARGETPEN | UPANNPARAM;/* update flags */
static Boolean saveBinary = FALSE;              /* save output in binary */
static HMMSet hset;                             /* the HMM set */
static char *epcBaseDir = NULL;                 /* the base directory for epoch models (default: newDir) */
static char *epcDirPref = "epoch";              /* the prefix of the subdirectory to store the model per epoch */
static char *fnUpdate = "UPDATE";               /* the file name for update files */
static char *fncurNegLR = "NEGLEARNRATE";       /* the file name for negative learning rates */
static char *fnSquareGrad = "SQUAREGRAD";       /* the file name for sum of squared gradients */
static char *fnLRSchd = "SCHEDULER";            /* the file name for learning rate scheduler */
static int updateIndex = 0;

static IntVec recVec = NULL;                    /* the vector contains hypothesis labels */
static IntVec recVecLLH = NULL;
static IntVec recVecMapSum = NULL;              /* the vector contains the mapped hypothesis labels */
static IntVec recVecLLHMapSum = NULL;
static char *labDir = NULL;                     /* label (transcription) file directory */
static char *labExt = "lab";                    /* label file extension */
static FileFormat dff = UNDEFF;                 /* data file format */
static FileFormat lff = UNDEFF;                 /* label file format */
static char *labFileMask = NULL;                /* mask for reading labels */
static Boolean useLLF = FALSE;                  /* use directory based LLF files instead of individual lattices */

static char *denLatDir[MAXLATS];                /* denominator lattices */
static int nDenLats = 0;                        /* number of denominator lattices */
static char *numLatDir[MAXLATS];                /* numerator-alignment lattices */
static int nNumLats = 0;                        /* number of numerator lattices */
static char *latExt = "lat";                    /* lattice file extension */
static char *latFileMask = NULL;                /* mask for reading lattices */
static char *latMask_Num = NULL;                /* mask for reading numerator lattices */
static char *latMask_Den = NULL;                /* mask for reading denominator lattices */
static char numLatSubDirPat[MAXSTRLEN] = "\0";  /* path mask for numerator lattices */
static char denLatSubDirPat[MAXSTRLEN] = "\0";  /* path mask for denominator lattices */
static Vocab vocab;                             /* fake vocabulary */
static int corrIdx = 0;
static int recogIdx1 = 1;
static int recogIdx2 = 999;
static Boolean procNumLats;
static Boolean procDenLats;
static float probScale = 1.0;

static ObjFunKind objfunKind = XENTOF;          /* the objective function for the ANN models */
static ObjFunKind showObjFunKind = UNKOF;
static ObjFunKind objfunFSmooth = XENTOF;
static float FSmoothH = 1.0;			/* F-smoothing is used when FSmoothH != 1.0 */
static ObjFunKind objfunISmooth = MLOF;
static float ISmoothTau = 0;
static float minOccFrameReject = 0.0;

static Boolean initialHV = FALSE;		/* wheter to do the initial held-out validation */
static FILE *scriptTr = NULL;                   /* script file for train set */
static int scriptCntTr = 0;                     /* number of words in scriptTr */
static int tSampCntTr = 0;                      /* number of samples in scriptTr */
static FILE *scriptHV = NULL;                   /* script file for held-out validation set if any */
static int scriptCntHV = 0;                     /* number of words in scriptHo */
static int tSampCntHV = 0;                      /* number of samples in scriptHV number of samples in scriptHV  */

static FILE *scriptFLabTr = NULL;
static int scriptCntFLabTr = 0;
static int tSampCntFLabTr = 0;
static FILE *scriptFLabHV = NULL;
static int scriptCntFLabHV = 0;
static int tSampCntFLabHV = 0;

static LabelKind labelKind = NULLLK;            /* the kind of the labels */
static LabelInfo *labelInfo = NULL;             /* the structure for the labels */
static BatchUpdateKind updtKind = BATLEVEL;     /* the update kind */  
static int numPerUpdt = 1;                      /* update the parameters once per updtUnitNum batches/utterances */
static DataCache *cacheTr[SMAX];                /* the cache structures for the train set */
static DataCache *cacheHV[SMAX];                /* the cache structures for the held-out validation set */
static Observation obs;                         /* array of Observations */

static ModelSetInfo inputMSI;                   /* the MSI with the input model set */
static MSILink headMSI;                         /* the head pointer for the MSI list */
static MSILink tailMSI;                         /* the tail pointer for the MSI list */
static float newMSI_CrtVal;               	/* the temporary global variable for ModelSetInfo.crtVal*/

static char **macroFN;                          /* the macro file name list */
int macroCnt = 0;                               /* the macro number */
static FBLatInfo fbInfo;
static int NumAccs;
static float MinOccTrans = 10.0;		/* Minimum numerator (ML) occupancy for a transition row */
static float CTrans = 1.0;

/* cz277 - semi */
static int bgWaitNBatchPL = 0;			/* the number of batches be held */
static int edAccBatchLenPL = 0;			/* the length of batch start to be held */
static VisitKind visitKindHV = NONEVK;		/* default visit kind for held-out validation */

/* cz277 - gradlim */
static float gradientClip = 0.0;	/* 0.32 */
static float updateClip = 0.0;		/* 0.32 */
static float gradientL2Scale = 0.0;
static float updateL2Scale = 0.0;
/*static float actfunUpdtPosClip = 0.32;		
static float actfunUpdtNegClip = -0.32;          
static float wghtUpdtPosClip = 0.32;		// suggested values: 1.6, 0.08, 0.02, 0.002 -> 800, 40, 10, 1 
static float wghtUpdtNegClip = -0.32;
static float biasUpdtPosClip = 0.32;
static float biasUpdtNegClip = -0.32;*/
/* cz277 - max norm */
static float weightL2NormBound = 0.0;		/* suggested value: 1.0, 0.8, 0.6... */
/* cz277 - pact */
static int staticUpdtCode = 0;
static float clipScaleFactor = 0.0;

/* ------------------------- Global Options ----------------------------- */

static Boolean optHasSSG = FALSE;               /* whether the trainInfo has SSG structure */
static Boolean optHasNLR = FALSE;               /* whether the trainInfo has NLR structure */
static Boolean optHasMMT = FALSE;               /* whether has momentum (should save the update file) */
static Boolean optHasLabMat = FALSE;            /* whether do supervised learning or not (associated with NULLLK) */
/*static Boolean optSeqTrain = FALSE;*/             /* whether sequence training or not */
static TrainMode optTrainMode = FRAMETM;	/* frame level training, by default */
static Boolean optSavEpcMod = FALSE;            /* whether saves the HMM files for each epoch */
static Boolean optSavSchd = FALSE;              /* whether saves the learning rate scheduler */
static Boolean optMapTarget = FALSE;            /* do target mapping or not */
static Boolean optHasFSmooth = FALSE;
static Boolean optHasISmooth = FALSE;
static Boolean optFrameReject = FALSE;
static Boolean optIncNumInDen = TRUE;
/*static Boolean optActfunUpdtClip = TRUE;
static Boolean optWghtUpdtClip = FALSE;
static Boolean optBiasUpdtClip = FALSE;*/
/*static Boolean optScaleClipByDepth = FALSE;*/
/* cz277 - max norm */
static Boolean optWeightL2Norm = FALSE;
/* cz277 - pact */
static Boolean optDoStaticUpdt = FALSE;
/* general parameters */
static Boolean optNormLearnRate = FALSE;                  /* whether to normalise the learning rates according to the updated samples */
static Boolean optNormWeightDecay = FALSE;
static Boolean optNormMomentum = FALSE;
static Boolean optNormGradientClip = FALSE;
static Boolean optNormUpdateClip = FALSE;
static Boolean optNormL2GradientScale = FALSE;
static Boolean optNormL2UpdateScale = FALSE;

/* ------------------------------ Heaps --------------------------------- */

static MemHeap modelHeap;                       /* the memory heap for models */
static MemHeap cacheHeap;                       /* the memory heap for data caches */
static MemHeap transHeap;                       /* the memory heap for transcriptions */
static MemHeap latHeap;                         /* the memory heap for lattices */
static MemHeap accHeap;				/* the accumulated memory heap */

/* -------------------- Configuration Parameters ------------------------ */

static ConfParam *cParm[MAXGLOBS];              /* configuration parameters */
static int nParm = 0;                           /* total num params */
static ParmKind flabPK;
static short vecSizeFLab = 0;
static Observation obsFLab;                         /* array of Observations */

/* -------------------------- Prototypes -------------------------------- */


/* ----------------------- Process Command Line ------------------------- */

static int ParseFloatConfStr(char *parmName, ConfParam *cpVal, Vector *resVec, float minVal, float maxVal) {
    Vector confVec=NULL;
    int numParm=0, intVal;
    char buf[MAXSTRLEN], *charPtr;
    
    if (cpVal->kind == FltCKind) {
        confVec = CreateVector(&gcheap, 1);
        confVec[1] = cpVal->val.f;
        if (confVec[1] < minVal || confVec[1] >= maxVal)
            HError(4321, "ParseFloatConfStr: float %e of %s out of range", confVec[1], parmName);
        numParm = 1;
    }
    else if (cpVal->kind == IntCKind) {
        confVec = CreateVector(&gcheap, 1);
        confVec[1] = cpVal->val.i;
        if (confVec[1] < minVal || confVec[1] >= maxVal)
            HError(4321, "ParseFloatConfStr: float %e of %s out of range", confVec[1], parmName);
        numParm = 1;
    }
    else if (cpVal->kind == StrCKind) {
        /* count for the number of tokens */
        numParm = 0;
        strcpy(buf, cpVal->val.s);
        charPtr = strtok(buf, ",");
        while (charPtr != NULL) {
            ++numParm;
            charPtr = strtok(NULL, ",");
        }
        if (numParm == 0)
            HError(4320, "ParseFloatConfStr: No token available for %s", parmName);
        /* malloc a new vector */
        confVec = CreateVector(&gcheap, numParm);
        intVal = 1;
        /* convert each token */
        strcpy(buf, cpVal->val.s);
        charPtr = strtok(buf, ",");
        while (charPtr != NULL) {
            confVec[intVal] = (float) atof(charPtr);
            if (confVec[intVal] < minVal || confVec[intVal] >= maxVal)
                HError(4321, "ParseFloatConfStr: float %e of %s out of range", confVec[intVal], parmName);
            ++intVal;
            charPtr = strtok(NULL, ",");
        }
    }
    else
        HError(4322, "ParseFloatConfStr: Wrong parameter type for %s", parmName);

    *resVec = confVec;
    return numParm;
}

void SetConfParms(void)
{
    int intVal;
    double doubleVal;
    Boolean boolVal;
    char buf[MAXSTRLEN], buf2[MAXSTRLEN];
    char *charPtr, *charPtr2;
    ConfParam *cpVal;

    /* initialise HNTrainSGD parameters */
    nParm = GetConfig("HNTRAINSGD", TRUE, cParm, MAXGLOBS);
    if (nParm > 0) {
        if (GetConfInt(cParm, nParm, "TRACE", &intVal)) 
            trace = intVal;
        if (GetConfBool(cParm, nParm, "EPOCHSAVE", &boolVal)) 
            optSavEpcMod = boolVal;
        /* set learning rate scheduler kind */
        if (GetConfStr(cParm, nParm, "LRSCHEDULER", buf)) {
            if (strcmp(buf, "ADAGRAD") == 0) {
                schdKind = ADAGRADSK;
                optHasSSG = TRUE;
                optHasNLR = TRUE;
            }
            else if (strcmp(buf, "EXPONENTIAL") == 0) {
                schdKind = EXPSK;
                optHasSSG = FALSE;
                optHasNLR = FALSE;
                Exp_TrSampIdx = 0;
            }
            else if (strcmp(buf, "LIST") == 0) {
                schdKind = LISTSK;
                optHasSSG = FALSE;
                optHasNLR = FALSE;
            }
            else if (strcmp(buf, "NEWBOB") == 0) {
                schdKind = NEWBOBSK;
                optHasSSG = FALSE;
                optHasNLR = FALSE;
                optSavEpcMod = TRUE;
                optSavSchd = FALSE;
            }
            else 
                HError(4322, "SetConfParms: Unknown learning rate scheduler kind");
        }
        /* set parameters associated with the learning rate scheduler */
        switch (schdKind) {
        case ADAGRADSK:
            if (GetConfInt(cParm, nParm, "K", &intVal)) {
                if (intVal < 1) 
                    HError(4321, "SetConfParms: K for AdaGrad scheduler out of range");
                AdaGrad_K = intVal;
            }
            if (GetConfFlt(cParm, nParm, "LEARNRATE", &doubleVal)) {
                if (doubleVal < 0.0) 
                    HError(4321, "SetConfParms: Initial learning rate for AdaGrad scheduler out of range");
                initLR = (float) doubleVal;
            }
            break;
        case EXPSK:
            if (GetConfFlt(cParm, nParm, "GAMMA", &doubleVal)) {
                if (doubleVal <= 0.0) 
                    HError(4321, "SetConfParms: Gamma for Exponential scheduler out of range");
                Exp_Gamma = (float) doubleVal;
            }
            if (GetConfFlt(cParm, nParm, "BASE", &doubleVal)) {
                if (doubleVal <= 1.0) 
                    HError(4321, "SetConfParms: Base for Exponential scheduler out of range");
                Exp_Base = (float) doubleVal;
            }
            if (GetConfFlt(cParm, nParm, "LEARNRATE", &doubleVal)) {
                if (doubleVal < 0.0) 
                    HError(4321, "SetConfParms: Initial learning rate for Exponential scheduler out of range");
                initLR = (float) doubleVal;
            }
            break;
        case LISTSK:
            if (GetConfAny(cParm, nParm, "LEARNRATE", &cpVal)) 
                maxEpochNum = ParseFloatConfStr("LEARNRATE", cpVal, &List_LRs, 0.0, FLT_MAX);
            break;
        case NEWBOBSK:
            if (GetConfFlt(cParm, nParm, "LEARNRATE", &doubleVal)) {
                if (doubleVal <= 0.0) 
                    HError(4321, "SetConfParms: Initial learning rate for NewBob scheduler out of range");
                initLR = (float) doubleVal;
            }
            if (GetConfStr(cParm, nParm, "NEWBOBCRT", buf)) {
                if (strcmp(buf, "ACC") == 0) 
                    NewBob_Crt = ACCNBC;
                else if (strcmp(buf, "MAPACC") == 0) 
                    NewBob_Crt = MAPACCNBC;
                else if (strcmp(buf, "LLHVAL") == 0) {
                    NewBob_Crt = LLHVALNBC;
                    showObjFunKind = showObjFunKind | MLOF;
                }
                else if (strcmp(buf, "MAPLLHVAL") == 0) {
                    NewBob_Crt = MAPLLHVALNBC;
                    showObjFunKind = showObjFunKind | MLOF;
                }
                else 
                    HError(4322, "SetConfParms: Unknown criterion for NewBob");
            }
            if (GetConfFlt(cParm, nParm, "DECAYFACTOR", &doubleVal)) 
                NewBob_DecayFactor = (float) doubleVal;
            if (GetConfFlt(cParm, nParm, "RAMPSTART", &doubleVal)) 
                NewBob_RampStart = (float) doubleVal;
            if (GetConfFlt(cParm, nParm, "STOPDIFF", &doubleVal)) 
                NewBob_Stop = (float) doubleVal;
            break;
        default:
            break;
        }
        if (schdKind != LISTSK && GetConfInt(cParm, nParm, "MINEPOCHNUM", &intVal)) 
            minEpochNum = intVal;
        if (schdKind != LISTSK && GetConfFlt(cParm, nParm, "MINLEARNRATE", &doubleVal)) 
            minLR = (float) doubleVal;
        if (schdKind != LISTSK && GetConfInt(cParm, nParm, "MAXEPOCHNUM", &intVal)) {
            if (intVal <= 0 || intVal < minEpochNum) 
                HError(4321, "SetConfParms: Maximum number of epoch out of range");
            maxEpochNum = intVal;
        }
        if (GetConfBool(cParm, nParm, "NORMLEARNRATE", &boolVal)) 
            optNormLearnRate = boolVal;
        /* cz277 - varlen */
        if (GetConfBool(cParm, nParm, "NORMWEIGHTDECAY", &boolVal)) 
            optNormWeightDecay = boolVal;
        if (GetConfBool(cParm, nParm, "NORMMOMENTUM", &boolVal)) 
            optNormMomentum = boolVal;
        if (GetConfBool(cParm, nParm, "NORMGRADIENTCLIP", &boolVal))
            optNormGradientClip = boolVal;
        if (GetConfBool(cParm, nParm, "NORMUPDATECLIP", &boolVal))
            optNormUpdateClip = boolVal;
        if (GetConfBool(cParm, nParm, "NORML2GRADIENTSCALE", &boolVal))
            optNormL2GradientScale = boolVal;
        if (GetConfBool(cParm, nParm, "NORML2UPDATESCALE", &boolVal))
            optNormL2UpdateScale = boolVal;

        if (GetConfInt(cParm, nParm, "EPOCHOFFSET", &intVal)) {
            if (intVal <= 0) 
                HError(4321, "SetConfParms: The offset of epoch index out of range");
            epochOff = intVal; 
        }
        /* cz277 - mmt */
        if (GetConfAny(cParm, nParm, "MOMENTUM", &cpVal)) {
            ParseFloatConfStr("MOMENTUM", cpVal, &List_MMTs, 0.0, FLT_MAX);
            optHasMMT = FALSE;
            for (intVal = 1; intVal <= VectorSize(List_MMTs); ++intVal)
                if (List_MMTs[intVal] != 0.0)
                    optHasMMT = TRUE;
        }
        if (GetConfAny(cParm, nParm, "WEIGHTDECAY", &cpVal)) 
            ParseFloatConfStr("WEIGHTDECAY", cpVal, &List_WeightDecay, 0.0, FLT_MAX); 
        if (GetConfStr(cParm, nParm, "UPDATEMODE", buf)) {
            if (strcmp(buf, "BATCHLEVEL") == 0) 
                updtKind = BATLEVEL;
            else if (strcmp(buf, "UTTERANCELEVEL") == 0) 
                updtKind = UTTLEVEL;
            else 
                HError(4322, "SetConfParms: Unknown batch based parameter update mode");
        }
        if (GetConfInt(cParm, nParm, "NUMPERUPDATE", &intVal)) {
            if (intVal <= 0) 
                HError(4321, "SetConfParms: NUMPERUPDATE should be positive integer");
            numPerUpdt = intVal;
        }
        /* set training criterion */
        if (GetConfStr(cParm, nParm, "CRITERION", buf)) {
            if (strcmp(buf, "ML") == 0) {
                objfunKind = MLOF; 
                optTrainMode = SEQTM;           
                showObjFunKind = showObjFunKind | MLOF;
            }
            else if (strcmp(buf, "MMI") == 0) {
                objfunKind = MMIOF;
                optTrainMode = SEQTM;
                showObjFunKind = showObjFunKind | MMIOF;
            }
            else if (strcmp(buf, "MMSE") == 0) {
                objfunKind = MMSEOF;
                optTrainMode = FRAMETM;
            }
            else if (strcmp(buf, "MPE") == 0) {
                objfunKind = MPEOF;
                optTrainMode = SEQTM;
                showObjFunKind = showObjFunKind | MPEOF;
            }
            else if (strcmp(buf, "MWE") == 0) {
                objfunKind = MWEOF;
                optTrainMode = SEQTM;
                showObjFunKind = showObjFunKind | MWEOF;
            }
            else if (strcmp(buf, "SMBR") == 0) {
                objfunKind = SMBROF;
                optTrainMode = SEQTM;
            }
            else if (strcmp(buf, "XENT") == 0) {
                objfunKind = XENTOF;
                optTrainMode = FRAMETM;
            }
            else 
                HError(4322, "SetConfParms: Unknown objective function kind");
        }
        /* set evaluation criteria */
        if (GetConfStr(cParm, nParm, "EVALCRITERIA", buf)) {
            charPtr = buf;
            while (charPtr != NULL) {
                charPtr2 = strchr(charPtr, '|');
                if (charPtr2 != NULL) {
                    *charPtr2 = '\0';
                }
                strcpy(buf2, charPtr);
                if (charPtr2 != NULL) {
                    *charPtr2 = '|';
                    ++charPtr2;
                }
                charPtr = charPtr2;
                if (strcmp(buf2, "ML") == 0) 
                    showObjFunKind = showObjFunKind | MLOF;
                else if ((strcmp(buf2, "MMI") == 0) && (optTrainMode == SEQTM)) 
                    showObjFunKind = showObjFunKind | MMIOF;
                else if (strcmp(buf2, "MMSE") == 0) 
                    showObjFunKind = showObjFunKind | MMSEOF;
                else if ((strcmp(buf2, "MPE") == 0) && (optTrainMode == SEQTM)) 
                    showObjFunKind = showObjFunKind | MPEOF;
                else if (strcmp(buf2, "XENT") == 0) 
                    showObjFunKind = showObjFunKind | XENTOF;
            }
        }
        /* set the updating flag for transitions */
        if (GetConfBool(cParm, nParm, "UPDATETRANS", &boolVal)) {
            if (boolVal) 
                uFlags = uFlags | UPTRANS;
            else 
                uFlags = uFlags & (~UPTRANS);
        }
        if ((uFlags & UPTRANS) != 0 && GetConfFlt(cParm, nParm, "CTRANS", &doubleVal)) 
            CTrans = (float) doubleVal;
        if ((uFlags & UPTRANS) != 0 && GetConfFlt(cParm, nParm, "MINOCCTRANS", &doubleVal)) 
            MinOccTrans = (float) doubleVal;
        /* set the frame criterion for F-smoothing */
        if (GetConfStr(cParm, nParm, "FSMOOTHCRITERION", buf)) {
            if (strcmp(buf, "MMSE") == 0) 
                objfunFSmooth = MMSEOF;
            else if (strcmp(buf, "XENT") == 0) 
                objfunFSmooth = XENTOF;
            else 
                HError(4322, "SetConfParms: Unknown objective function kind for F-smoothing");
        }
        /* set the H value for F-smoothing */ 
        if (GetConfFlt(cParm, nParm, "FSMOOTHHVALUE", &doubleVal)) {
            FSmoothH = (float) doubleVal;
            if (FSmoothH != 1.0) 
                optHasFSmooth = TRUE;
        }        
        /* set the sequence criterion for I-smoothing */
        if (GetConfStr(cParm, nParm, "ISMOOTHCRITERION", buf)) {
            if (strcmp(buf, "ML") == 0) {
                if (!(objfunKind == MMIOF && objfunKind == MPEOF && objfunKind == MWEOF && objfunKind == SMBROF)) 
                    HError(4322, "SetConfParms: ML prior is only valid for MMI and MBR training objective functions");
                objfunISmooth = MLOF;
            }
            else if (strcmp(buf, "MMI") == 0) {
                if (!(objfunKind == MPEOF && objfunKind == MWEOF && objfunKind == SMBROF)) 
                    HError(4322, "SetConfParms: ML prior is only valid for MBR training objective functions");
                objfunISmooth = MMIOF;
            }
            else 
                HError(4322, "SetConfParms: Unknown objective function kind for I-smoothing");
        }
        /* set the tau for I-smoothing */
        if (GetConfFlt(cParm, nParm, "ISMOOTHTAU", &doubleVal)) {
            ISmoothTau = (float) doubleVal;
            if (ISmoothTau != 0) 
                optHasISmooth = TRUE;
        } 
        /* set whether do frame rejection */
        if (GetConfFlt(cParm, nParm, "MMIFRMINOCC", &doubleVal)) {
            if (doubleVal > 0.0 && objfunKind == MMIOF) {
                minOccFrameReject = doubleVal;
                optFrameReject = TRUE; 
            }
        }
        /* set the updating flag for target penalties */
        if (GetConfBool(cParm, nParm, "UPDATETARGETPEN", &boolVal)) {
            if (boolVal) 
                uFlags = uFlags | UPTARGETPEN; 
            else 
                uFlags = uFlags & (~UPTARGETPEN);
        }
        /* update the log observation prior */
        if (GetConfFlt(cParm, nParm, "LOGPRIOROBSV", &doubleVal)) 
            logObsvPrior = (float) doubleVal;
        /* cz277 - semi */
        if (GetConfInt(cParm, nParm, "BGNPLBATCHWAIT", &intVal)) {
            if (intVal < 0) 
                HError(4321, "SetConfParms: BGNPLBATCHWAIT should be non-negative integer");
            bgWaitNBatchPL = intVal;
        }
        if (GetConfInt(cParm, nParm, "EDPLBATCHLENACC", &intVal)) {
            if (intVal < 0) 
                HError(4321, "SetConfParms: EDPLBATCHLENACC should be non-negative integer");
            edAccBatchLenPL = intVal;
        }
        if (GetConfFlt(cParm, nParm, "L2GRADIENTSCALE", &doubleVal)) 
            gradientL2Scale = (float) doubleVal;
        if (GetConfFlt(cParm, nParm, "GRADIENTCLIP", &doubleVal)) 
            gradientClip = (float) fabs(doubleVal);
        if (GetConfFlt(cParm, nParm, "L2UPDATESCALE", &doubleVal)) 
            updateL2Scale = (float) doubleVal;
        if (GetConfFlt(cParm, nParm, "UPDATECLIP", &doubleVal)) 
            updateClip = (float) fabs(doubleVal);
        if (GetConfFlt(cParm, nParm, "CLIPSCALEFACTOR", &doubleVal)) 
            clipScaleFactor = (float) doubleVal; 
        /*if (GetConfFlt(cParm, nParm, "ACTIVATIONUPDATECLIP", &doubleVal)) {
            if (doubleVal <= 0.0) 
                optActfunUpdtClip = FALSE;
            else {
                optActfunUpdtClip = TRUE;
                actfunUpdtPosClip = (float) doubleVal;
                actfunUpdtNegClip = -1.0 * actfunUpdtPosClip;
            }
        }
        if (GetConfFlt(cParm, nParm, "WEIGHTUPDATECLIP", &doubleVal)) {
            if (doubleVal <= 0.0) 
                optWghtUpdtClip = FALSE;
            else {
                optWghtUpdtClip = TRUE;
                wghtUpdtPosClip = (float) doubleVal;
                wghtUpdtNegClip = -1.0 * wghtUpdtPosClip;
            }
        }
        if (GetConfFlt(cParm, nParm, "BIASUPDATECLIP", &doubleVal)) {
            if (doubleVal <= 0.0) 
                optBiasUpdtClip = FALSE;
            else {
                optBiasUpdtClip = TRUE;
                biasUpdtPosClip = (float) doubleVal;
                biasUpdtNegClip = -1.0 * biasUpdtPosClip;
            }
        }*/
        /* cz277 - max norm */
        if (GetConfFlt(cParm, nParm, "WEIGHTL2MAXNORMBOUND", &doubleVal)) {
            if (doubleVal <= 0.0) 
	        optWeightL2Norm = FALSE;
	    else {
	        optWeightL2Norm = TRUE;
	        weightL2NormBound = (float) doubleVal;
	    }
	}
        if (GetConfInt(cParm, nParm, "STATICUPDATECODE", &intVal)) 
            staticUpdtCode = intVal;

        /* label file mask */
        if (GetConfStr(cParm, nParm, "LABFILEMASK", buf)) 
            labFileMask = CopyString(&gcheap, buf);
        /* lattice file mask */
        if (GetConfStr(cParm, nParm, "LATFILEMASK", buf)) 
            latFileMask = CopyString(&gcheap, buf);
        if (GetConfStr(cParm, nParm, "LATMASKNUM", buf)) 
            latMask_Num = CopyString(&gcheap, buf);
        if (GetConfStr(cParm, nParm, "LATMASKDEN", buf)) 
            latMask_Den = CopyString(&gcheap, buf);
        /* speaker adaptation mask */
        if (GetConfStr(cParm, nParm, "INXFORMMASK", buf)) 
            xfInfo.inSpkrPat = CopyString(&gcheap, buf);
        if (GetConfBool(cParm, nParm, "USELLF", &boolVal)) 
            useLLF = boolVal;
        if (GetConfBool(cParm, nParm, "INCNUMLATINDENLAT", &boolVal)) 
            optIncNumInDen = boolVal;
    }
    /* HParm: HNTrainSGD */
    SetChannel("HPARMLABEL");
    nParm = GetConfig("HPARMLABEL", TRUE, cParm, MAXGLOBS);
    if (nParm > 0) {
        if (GetConfStr(cParm, nParm, "TARGETKIND", buf))
            flabPK = Str2ParmKind(buf); 
        if (GetConfInt(cParm, nParm, "VECSIZE", &intVal))
            vecSizeFLab = (short) intVal;
    }

    /* fetch parameters associated with current training criterion */
    switch (objfunKind) {
    case MMIOF:
        procNumLats = TRUE;
        procDenLats = TRUE;
        recogIdx1 = 1;
        recogIdx2 = 999;
        corrIdx = 0;
        break;
    case MLOF:
        procNumLats = TRUE;
        procDenLats = FALSE;
        recogIdx1 = 999;    /* useless */
        recogIdx2 = 999;    /* useless */
        corrIdx = 0;
        break;
    case MMSEOF:
        break;
    case MPEOF:
        procNumLats = FALSE;
        procDenLats = TRUE;
        recogIdx1 = 0;
        recogIdx2 = 1;
        corrIdx = 2;
        break;
    case MWEOF:
        HError(4301, "MWE not implemented yet");
        break;
    case SMBROF:
        HError(4301, "SMBR not implemented yet");
        break;
    case XENTOF:
        break;
    default:
        break;
    }
}

void ReportUsage (void) {
    printf("\nUSAGE: HNTrainSGD [options] [HMMList]\n\n");
    printf(" Option                                       Default\n\n");
    printf(" -a      Use input transformation             off\n");
    printf(" -c      Do initial held-out validation       off\n");
    printf(" -d s    Dir to find HMM definitions          current\n");
    printf(" -e s    Subdir prefix for epoch models       epoch\n");
    printf(" -eb s   Base dir to save intermediate HMMs   -d dir\n");
    printf(" -fl s   Path of input learning rate file     none\n");
    printf(" -h s    Speaker name pattern                 none\n");
    printf(" -l s    Label kind [fea, lab, lat, null]     null\n");
    printf(" -m s    Target mapping file s                off\n");
    printf(" -nu s   Name of the update file              UPDATE\n");
    printf(" -nn s   Name of the neg learning rate file   curNegLR\n");
    printf(" -ng s   Name of the sum of squared grad file SQUAREGRAD\n");
    printf(" -ns s   Name of the scheduler file           SCHEDULER\n");
    printf(" -o s    Extensions for new HMM files         as src\n");
    printf(" -q s    Directory for numerator lats         [needed. May use >1 -q option]\n");
    printf(" -qp s   Subdir pattern for numerator lats    none\n");
    printf(" -r s    Directory for denominator lats       [needed. May use >1 -r option]\n");
    printf(" -rp s   Subdir pattern for denominator lats  none\n");
    printf(" -u tbw  Update t)rans b)ias w)eight for ANN  tbw\n");
    printf(" -x s    Extension for HMM files              none\n");
    printf(" -N s    Input script file for validation set none\n");
    PrintStdOpts("BFGHIJLMSTX");    /* E, K removed */
    printf("\n\n");
}


float CalApproxAvgBatchLikelhood(DataCache *cache, int batLen) {
    LELink layerElem;
    ANNSet *annSet;
    int i, labTgt;
    float llh = 0.0;

    annSet = cache->hmmSet->annSet;
    layerElem = annSet->outLayers[cache->streamIdx];
    for (i = 0; i < batLen; ++i) {
        labTgt = cache->labVec[i];
        llh += cache->hmmSet->annSet->llhMat[cache->streamIdx]->matElems[(i - 1) * layerElem->nodeNum + labTgt];
    }
    llh /= batLen;
    
    return llh;
}

void AccCriteriaPerB(DataCache *cache, int batLen, CriteriaInfo *criteria) {
    LELink layerElem;
    int i, labTgt, recTgt, recTgtMapSum, recLLHTgt, recTgtLLHMapSum;
    IntVec mapVec;
    ANNSet *annSet;

    annSet = cache->hmmSet->annSet;
    /* do accumulateion */
    layerElem = annSet->outLayers[cache->streamIdx];
    /* for tSamp */
    criteria->tSampAcc += batLen;
    /* for accuracy */
    for (i = 1; i <= batLen; ++i) {
        labTgt = cache->labVec[i];
        recTgt = recVec[i];
        if (labTgt == recTgt) {
            ++criteria->cSampAcc;
        }
        /* for LLH values */
        if (showObjFunKind & MLOF) {
            recLLHTgt = recVecLLH[i];
            if (labTgt == recLLHTgt) {
                criteria->LLHVal += cache->hmmSet->annSet->llhMat[cache->streamIdx]->matElems[(i - 1) * layerElem->nodeNum + recLLHTgt];
            }
        }
    }

    /* for mapped accuracy by max and sum*/
    if (optMapTarget) {
        mapVec = hset.annSet->mapStruct->mapVectors[cache->streamIdx];
        for (i = 1; i <= batLen; ++i) {
            labTgt = mapVec[cache->labVec[i] + 1];
            recTgtMapSum = recVecMapSum[i];
            if (labTgt == recTgtMapSum) {
                ++criteria->cSampAccMapSum;
            }
            UpdateTargetMapStats(hset.annSet, labTgt, recTgtMapSum);
            /* for LLH  values */
            if (showObjFunKind & MLOF) {
                recTgtLLHMapSum = recVecLLHMapSum[i];
                if (labTgt == recTgtLLHMapSum) {
                    criteria->LLHValMapSum += cache->hmmSet->annSet->mapStruct->llhMatMapSum[cache->streamIdx]->matElems[(i - 1) * hset.annSet->mapStruct->mappedTargetNum + recTgtMapSum];
                }
            }
        }
    }

    /* MMSE */
    if (showObjFunKind & MMSEOF) {
        criteria->MMSEAcc += CalMMSECriterion(cache->labMat, layerElem->yFeaMats[1], batLen);	/* cz277 - many */
        if (optMapTarget) {
            criteria->MMSEAccMapSum += CalMMSECriterion(annSet->mapStruct->labMatMapSum[cache->streamIdx], annSet->mapStruct->outMatMapSum[cache->streamIdx], batLen);
        }
    }

    /* XENT */
    if (showObjFunKind & XENTOF) {
        criteria->XENTAcc += CalXENTCriterion(cache->labMat, layerElem->yFeaMats[1], batLen);	/* cz277 - many */
        if (optMapTarget) {
            criteria->XENTAccMapSum += CalXENTCriterion(annSet->mapStruct->labMatMapSum[cache->streamIdx], annSet->mapStruct->outMatMapSum[cache->streamIdx], batLen);
        }
    }
}

void AccCriteriaPerU(DataCache *cache, int batLen, CriteriaInfo *criteria) {
    LELink layerElem;
    ANNSet *annSet;

    annSet = cache->hmmSet->annSet;
    /* do accumulateion */
    layerElem = annSet->outLayers[cache->streamIdx];
    /* for tSamp */
    criteria->tSampAcc += batLen;
    /* for tUtt */
    ++criteria->tUttAcc;

    if ((showObjFunKind & MLOF) && (cache->streamIdx == 1)) {
        criteria->LLHVal += fbInfo.pr;
    }
    if ((showObjFunKind & MPEOF) && (cache->streamIdx == 1)) {
        criteria->MPEAcc += fbInfo.AvgCorr;
        criteria->tNWordAcc += fbInfo.MPEFileLength;
    }
    if ((showObjFunKind & MMIOF) && (cache->streamIdx == 1)) {
        criteria->NumLLHAcc += fbInfo.latPr[corrIdx];
        criteria->DenLLHAcc += fbInfo.latPr[recogIdx1];
    }
    /* MMSE */
    if (showObjFunKind & MMSEOF) {
        criteria->MMSEAcc += CalMMSECriterion(cache->labMat, layerElem->yFeaMats[1], batLen);	/* cz277 - many */
    }
    /* XENT */
    if (showObjFunKind & XENTOF) {
        criteria->XENTAcc += CalXENTCriterion(cache->labMat, layerElem->yFeaMats[1], batLen);	/* cz277 - many */
    }
}



void PrintCriteria(CriteriaInfo *criteria, char *setid) {
    float accVal;
    int cSampInt, tSampInt;

    if (optTrainMode == FRAMETM) {
        cSampInt = (int) criteria->cSampAcc;
        tSampInt = (int) criteria->tSampAcc;
        accVal = criteria->cSampAcc / criteria->tSampAcc;
        printf("\t\t%s Accuracy = %.2f%% [%d right out of %d samples]\n", setid, accVal * 100.0, cSampInt, tSampInt);
        if (showObjFunKind & XENTOF) {
            printf("\t\tCross Entropy/Frame = %.2f\n", criteria->XENTAcc / criteria->tSampAcc);
        }
        if (showObjFunKind & MMSEOF) {
            printf("\t\tMean Square Error/Frame = %.2f\n", criteria->MMSEAcc / criteria->tSampAcc);
        }
        if (showObjFunKind & MLOF) {
            printf("\t\tLog-Likelihood/Frame = %e\n", criteria->LLHVal / criteria->tSampAcc);
        }

        if (optMapTarget) {
            cSampInt = (int) criteria->cSampAccMapSum;
            accVal = criteria->cSampAccMapSum / criteria->tSampAcc;
            printf("\t\tMapped Accuracy/Frame by Sum = %.2f%% [%d right out of %d samples]\n", accVal * 100.0, cSampInt, tSampInt);
            if (showObjFunKind & XENTOF) {
                printf("\t\tMapped Cross Entropy/Frame by Sum = %.2f\n", criteria->XENTAccMapSum / criteria->tSampAcc);
            }
            if (showObjFunKind & MMSEOF) {
                printf("\t\tMapped Mean Square Error/Frame by Sum = %.2f\n", criteria->MMSEAccMapSum / criteria->tSampAcc);
            }
            if (showObjFunKind & MLOF) {
                printf("\t\tLog-Likelihood/Frame by Sum = %e\n", criteria->LLHValMapSum / criteria->tSampAcc);
            }
        }
    }

    if (optTrainMode == SEQTM) {
        if (showObjFunKind & MMIOF) {
            if (optFrameReject == TRUE) {
                printf("\t\t%d frames rejected in training\n", criteria->MMIFRAcc);
            }
            printf("\t\tMutual Information/Frame = %f [Num/Frame = %f, Den/Frame = %f]\n", (criteria->NumLLHAcc - criteria->DenLLHAcc) / criteria->tSampAcc, criteria->NumLLHAcc / criteria->tSampAcc, criteria->DenLLHAcc / criteria->tSampAcc);
        }
        if (showObjFunKind & MPEOF) {
            printf("\t\tMPE/MWE criterion is: %f [%d words in total]\n", criteria->MPEAcc / criteria->tNWordAcc, (int) criteria->tNWordAcc);
        }
        if (showObjFunKind & MMSEOF) {
            printf("\t\tMean Square Error/Frame = %.2f\n", criteria->MMSEAcc / criteria->tSampAcc);
        }
        if (showObjFunKind & MLOF) {
            printf("\t\tLog-Likelihood/Frame = %e\n", criteria->LLHVal / criteria->tSampAcc);
        }
    }

    fflush(stdout);

    /* set performance criterion for NewBob */
    switch (NewBob_Crt) {
        case ACCNBC:
            newMSI_CrtVal = criteria->cSampAcc / criteria->tSampAcc;
            break;
        case MAPACCNBC:
            newMSI_CrtVal = criteria->cSampAccMapSum / criteria->tSampAcc;
            break;
        case LLHVALNBC:
            newMSI_CrtVal = criteria->LLHVal / criteria->tSampAcc;
            break;
        case MAPLLHVALNBC:
            newMSI_CrtVal = criteria->LLHValMapSum / criteria->tSampAcc;
            break;
    }

}

ReturnStatus InitScriptFile(char *fn, FILE **script, int *scriptCnt) {
    char buf[MAXSTRLEN];
    int cnt = 0;

    /*CheckFn(fn);*/
    if (*script != NULL) 
        HError(4300, "InitScriptFile: Script file has been initialised");
    if ((*script = fopen(fn, "r")) == NULL) 
        HError(4310, "InitScriptFile: Cannot open script file %s for validation set", fn);
    while (GetNextScpWord(*script, buf) != NULL) 
        ++cnt;
    rewind(*script);
    *scriptCnt = cnt;
    return SUCCESS;
}

char *ScpWord(FILE *script, char *wordbuf) {
    int ch, qch, i;

    i = 0;
    ch = ' ';
    while (isspace(ch))
        ch = fgetc(script);
    if (ch == EOF)
        return NULL;
    if (ch == '\'' || ch == '"') {
        qch = ch;
        ch = fgetc(script);
        while (ch != qch && ch != EOF) {
            wordbuf[i ++] = ch;
            ch = fgetc(script);
        }
        if (ch == EOF)
            HError(4319, "Closing quote missing in the extended scp file");
    }
    else {
        do {
            wordbuf[i ++] = ch;
            ch = fgetc(script);
        } while (!isspace(ch) && ch != EOF);
    }
    wordbuf[i] = '\0';
   
    return wordbuf;
}

int GetExtScpWordDur(char *str) {
    char *lb, *rb, *co;
    char buf[MAXSTRLEN];
    int stidx, edidx;

    strcpy(buf, str);
    /*eq = strchr(buf, '=');*/
    lb = strchr(buf, '[');
    if (lb == NULL) {
        return -1;
    }
    if ((co = strchr(buf, ',')) == NULL)
        HError(4319, "GetExtScpWordDur: , missing in index spec");
    if ((rb = strchr(buf, ']')) == NULL)
        HError(4319, "GetExtScpWordDur: ] missing in index spec");
    *rb = '\0';
    edidx = atol(co + 1);
    *co = '\0';
    stidx = atol(lb + 1);
    if (stidx < 0 || edidx < 0 || edidx < stidx)
        HError(4319, "GetExtScpWordDur: Illegal start or end index spec");
    return edidx - stidx + 1;
}

/* Get the total sample count in the script file */
int GetScpSampCnt(FILE *script) {
    int sampCnt = 0, curDur;
    char buf[MAXSTRLEN];
    ParmBuf parmBuf;

    if (script == NULL)
        HError(4300, "GetScpSampCnt: Uninitialised input script file");
    /* get the first scp word to test */
    ScpWord(script, buf);
    curDur = GetExtScpWordDur(buf);
    if (curDur < 0) {
        HError(-4323, "GetScpSampCnt: Using non-ext scp causes an extra pass of data loading");
        rewind(script);
        while (GetNextScpWord(script, buf) != NULL) {
            parmBuf = OpenBuffer(&gstack, buf, 0, UNDEFF, TRI_UNDEF, TRI_UNDEF);
            if (!parmBuf) {
                HError(4313, "GetScpSampCnt: Open input data failed");
            }   
            sampCnt += ObsInBuffer(parmBuf);
            CloseBuffer(parmBuf);
        }
    }
    else {
        sampCnt += curDur;
        while (ScpWord(script, buf) != NULL) {
            curDur = GetExtScpWordDur(buf);
            if (curDur < 0)
                HError(4319, "GetScpSampCnt: Non-ext scp line in ext scp file");
            sampCnt += curDur;
        }
    }
    rewind(script);

    return sampCnt;
}

void SetModelSetInfo(char *baseDir, char *hmmExt, char *macExt, MSILink msi, int epochIdx) {
    char buf[MAXSTRLEN];
    int i;
    MILink mmf;

    /* baseDir */
    msi->baseDir = NULL;
    if (baseDir != NULL) {
        msi->baseDir = CopyString(&gcheap, baseDir);
    }
    /* hmmExt */
    msi->hmmExt = NULL;
    if (hmmExt != NULL) {
        msi->hmmExt = CopyString(&gcheap, hmmExt);
    }
    /* macExt */
    msi->macExt = NULL;
    if (macExt != NULL) {
        msi->macExt = CopyString(&gcheap, macExt);
    }
    /* set macro paths */
    msi->macFN = NULL;
    if (epochIdx <= 0) {  /* only in the first invoke */
        msi->macCnt = macroCnt;
        if (msi->macCnt > 0) {
            msi->macFN = (char **) New(&gcheap, sizeof(char *) * msi->macCnt);
            macroFN = (char **) New(&gcheap, sizeof(char *) * macroCnt);
            macroCnt = 0;
            for (i = 0, mmf = hset.mmfNames; mmf != NULL; ++i, mmf = mmf->next) {
                /* store initial macro names */
                msi->macFN[i] = CopyString(&gcheap, mmf->fName);
                /* exclude UPDATE, LEARNRATE, and SQUAREGRAD */
                NameOf(mmf->fName, buf);
                if (strcmp(buf, fnUpdate) != 0 && strcmp(buf, fncurNegLR) != 0 && strcmp(buf, fnSquareGrad) != 0) {
                    macroFN[macroCnt] = CopyString(&gcheap, buf);
                    ++macroCnt;
                }
            }
        }
    }
    else {
        msi->macCnt = macroCnt;
        if (optHasSSG)
            ++msi->macCnt;
        if (optHasNLR)
            ++msi->macCnt;
        if (optHasMMT)
            ++msi->macCnt;
        if (msi->macCnt > 0) {
            msi->macFN = (char **) New(&gcheap, sizeof(char *) * msi->macCnt);
            for (i = 0; i < macroCnt; ++i) {
                MakeFN(macroFN[i], baseDir, macExt, buf);
                msi->macFN[i] = CopyString(&gcheap, buf);
            }
            if (optHasSSG) {
                MakeFN(fnSquareGrad, baseDir, macExt, buf);
                msi->macFN[i++] = CopyString(&gcheap, buf);
            }
            if (optHasNLR) {
                MakeFN(fncurNegLR, baseDir, macExt, buf);
                msi->macFN[i++] = CopyString(&gcheap, buf);
            }
            if (optHasMMT) {
                MakeFN(fnUpdate, baseDir, macExt, buf);
                msi->macFN[i++] = CopyString(&gcheap, buf);
            }
        }
    }
    msi->next = NULL;
    msi->prev = NULL;
    msi->epochIdx = epochIdx;
    msi->updtIdx = updateIndex;
}

void AppendModelSetInfo(MSILink newMSI) {
    tailMSI->next = newMSI;
    newMSI->prev = tailMSI;
    tailMSI = tailMSI->next;
}

void PopModelSetInfo() {
    if (tailMSI->baseDir != NULL) {
        Dispose(&gcheap, tailMSI->baseDir);
    }
    if (tailMSI->hmmExt != NULL) {
        Dispose(&gcheap, tailMSI->hmmExt);
    }
    if (tailMSI->macExt != NULL) {
        Dispose(&gcheap, tailMSI->macExt);
    }
    tailMSI = tailMSI->prev;
    Dispose(&gcheap, tailMSI->next);
}

void CheckFitMaxNorm(ANNSet *annSet) {
    int i;
    AILink curAI;
    ADLink annDef;
    LELink layerElem;
    Boolean fitMaxNorm = TRUE;

    /* proceed each ANNDef */
    curAI = annSet->defsTail;
    while (curAI != NULL) {
        annDef = curAI->annDef;
        for (i = annDef->layerNum - 2; i >= 0; --i) {
            layerElem = annDef->layerList[i];
            /* if no parameter to update in this layer */
            switch (layerElem->layerKind) {
            case ACTIVATIONONLYLAK: HError(4301, "CheckFitMaxNorm: Not Implemented yet"); break;
            case CONVOLUTIONLAK: HError(4301, "CheckFitMaxNorm: Not Implemented yet"); break;
            case PERCEPTRONLAK: fitMaxNorm &= IsLinearInvariant(layerElem->actfunKind); break;
            case SUBSAMPLINGLAK: HError(4301, "CheckFitMaxNorm: Not Implemented yet"); break;
            default:
                HError(4399, "CheckFitMaxNorm: Unknown layer type");
            }
        }
        /* fetch next ANNDef */
        curAI = curAI->prev;
    }

    if (optWeightL2Norm == TRUE && fitMaxNorm == FALSE) 
        HError(-4324, "CheckFitMaxNorm: Max norm does not fit this model, suggest to turn off");
}


void Initialise(void) {
    Boolean eSep;
    /*int s, tSampCntTr, tSampCntHV;*/
    int s;
    VisitKind visitKindTr=GetDefaultVisitKind();
    short swidth[SMAX];

    /* initialise epcBaseDir */
    if (epcBaseDir == NULL) {
        epcBaseDir = newDir;
    }
    /* initialise the MSI list */
    SetModelSetInfo(hmmDir, hmmExt, NULL, &inputMSI, 0);
    headMSI = &inputMSI;
    tailMSI = &inputMSI;

    /* initialise the memory heaps */
    CreateHeap(&cacheHeap, "cache heap", CHEAP, 1, 0, 100000000, ULONG_MAX);
    CreateHeap(&transHeap, "transcription heap", MSTAK, 1, 0, 8000, 80000);
    CreateHeap(&latHeap, "lattice heap", MSTAK, 1, 1.0, 50000, 500000);
    CreateHeap(&accHeap, "acc heap", MSTAK, 1, 1.0, 50000, 500000);

    /* load HMMs and HMMSet related global variables */
    if (trace & T_TOP) {
        printf("Reading ANN models...\n");
        fflush(stdout);
    }
    if (hmmListFn != NULL && MakeHMMSet(&hset, hmmListFn) < SUCCESS) 
        HError(4300, "Initialise: MakeHMMSet failed");
    if (LoadHMMSet(&hset, hmmDir, hmmExt) < SUCCESS)
        HError(4300, "Initialise: LoadHMMSet failed");
    if (hset.annSet == NULL) 
        HError(4300, "Initialise: No ANN model available"); 
    /* init train struct */
    InitTrainInfo(&hset, optHasLabMat, optHasNLR, optHasSSG, TRUE);
    InitErrMix(&hset);
    /* setup the mappings */
    if (optMapTarget) {
        SetupStateInfoList(&hset);
        if (SetupTargetMapList(&hset, mappingFn, 0) < SUCCESS) 
            HError(4300, "Initialise: Failed to load the target mapping file");
        InitMapStruct(&hset);
        recVecMapSum = CreateIntVec(&gcheap, GetNBatchSamples());
        /*ClearMappedTargetCounters(hset.annSet);*/
    }
    CreateTmpNMat(hset.hmem);

    SetStreamWidths(hset.pkind, hset.vecSize, hset.swidth, &eSep);
    SetANNUpdateFlag(&hset);
    SetNMatUpdateFlag(&hset);
    SetNVecUpdateFlag(&hset);
    if (trace & T_TOP) {
        printf("ANN model structure:\n");
        ShowANNSet(&hset);
        fflush(stdout);
    }
    SetupNMatRPLInfo(&hset);
    SetupNVecRPLInfo(&hset);
    /* for training set */
    obs = MakeObservation(&gcheap, hset.swidth, hset.pkind, FALSE, eSep);
    scriptTr = GetTrainScript(&scriptCntTr);
    tSampCntTr = GetScpSampCnt(scriptTr);
    AccAllCacheSamples(tSampCntTr);
    if (trace & T_TOP) 
        printf("%d utterances (%d samples) in the training set\n", scriptCntTr, tSampCntTr);
    if (scriptHV != NULL) {
        tSampCntHV = GetScpSampCnt(scriptHV);
        if (trace & T_TOP) 
            printf("%d utterances (%d samples) in the validation set\n", scriptCntHV, tSampCntHV);
        AccAllCacheSamples(tSampCntHV);
    }

    /* initialise adaptation */
    if (xfInfo.inSpkrPat == NULL)
        xfInfo.inSpkrPat = xfInfo.outSpkrPat;
    if (xfInfo.paSpkrPat == NULL)
        xfInfo.paSpkrPat = xfInfo.outSpkrPat;

    /* initialise labels */
    if (labelKind != NULLLK) {
        labelInfo = (LabelInfo *) New(&gcheap, sizeof(LabelInfo));
        memset(labelInfo, 0, sizeof(LabelInfo));
        labelInfo->labelKind = labelKind;
        if ((labelKind & FEALK) != 0) {
            SetStreamWidths(flabPK, vecSizeFLab, swidth, &eSep);
            obsFLab = MakeObservation(&gcheap, swidth, flabPK, FALSE, eSep); 
            if (scriptTr != NULL) {
                if (scriptFLabTr == NULL)
                    HError(4315, "HNTrainSGD: -s expected when -l = FEATURE");
                if (scriptCntTr != scriptCntFLabTr)
                    HError(4325, "HNTrainSGD: Inconsistent utterance number between -S and -s");
                tSampCntFLabTr = GetScpSampCnt(scriptFLabTr);
                if (tSampCntTr != tSampCntFLabTr)
                    HError(4325, "HNTrainSGD: Inconsistent total sample number between -S and -s");
            }
            if (scriptHV != NULL) {
                if (scriptFLabHV == NULL)
                    HError(4315, "HNTrainSGD: -n expected when -l = FEATURE and -N given");
                if (scriptCntHV != scriptCntFLabHV)
                    HError(4325, "HNTrainSGD: Inconsistent utterance number between -N and -n");
                tSampCntFLabHV = GetScpSampCnt(scriptFLabHV);
                if (tSampCntHV != tSampCntFLabHV)
                    HError(4325, "HNTrainSGD: Inconsistent total sample number between -N and -n");
            }
            labelInfo->obsFLab = &obsFLab;
        }
        if ((labelKind & LABLK) != 0) {
            labelInfo->labFileMask = labFileMask;
            labelInfo->labDir = labDir;
            labelInfo->labExt = labExt;
        }
        if ((labelKind & LATLK) != 0) {
            /* The actual dict is not needed, only the structure; this relates to HNet and reading lattices. */
            InitVocab(&vocab);
            labelInfo->latFileMask = latFileMask;
            labelInfo->latMaskNum = latMask_Num;
            labelInfo->numLatDir = numLatDir;
            labelInfo->nNumLats = nNumLats;
            labelInfo->numLatSubDirPat = numLatSubDirPat;
            labelInfo->latMaskDen = latMask_Den;
            labelInfo->denLatDir = denLatDir;
            labelInfo->nDenLats = nDenLats;
            labelInfo->denLatSubDirPat = denLatSubDirPat;
            labelInfo->latExt = latExt;
            labelInfo->vocab = (Ptr) &vocab;
            labelInfo->useLLF = useLLF;
            probScale = GetProbScale();
            labelInfo->incNumInDen = optIncNumInDen;
        }
        labelInfo->uFlags = uFlags;
    }
    recVec = CreateIntVec(&gcheap, GetNBatchSamples());
    recVecLLH = CreateIntVec(&gcheap, GetNBatchSamples());

    /* initialise the cache structures */
    /*obs = MakeObservation(&gcheap, hset.swidth, hset.pkind, FALSE, eSep);
    scriptTr = GetTrainScript(&scriptCntTr);
    tSampCntTr = GetScpSampCnt(scriptTr);
    if (trace & T_TOP) {
        printf("%d utterances (%d samples) in the training set\n", scriptCntTr, tSampCntTr);
    }*/
    labelInfo->scpFLab = scriptFLabTr;
    for (s = 1; s <= hset.swidth[0]; ++s) {
        /*AccAllCacheSamples(tSampCntTr);*/
        visitKindTr = GetDefaultVisitKind();
        labelInfo->dimFLab = swidth[s];
        cacheTr[s] = CreateCache(&cacheHeap, scriptTr, scriptCntTr, &hset, &obs, s, GetDefaultNCacheSamples(), visitKindTr, &xfInfo, labelInfo, TRUE);
    }
    if (scriptHV != NULL) {
        /*tSampCntHV = GetScpSampCnt(scriptHV);
        if (trace & T_TOP) {
            printf("%d utterances (%d samples) in the validation set\n", scriptCntHV, tSampCntHV);
        }*/
        labelInfo->uFlags = labelInfo->uFlags & (~UPTARGETPEN);
        labelInfo->uFlags = labelInfo->uFlags & (~UPTRANS);	/* cz277 - trans */
        labelInfo->scpFLab = scriptFLabHV;
        for (s = 1; s <= hset.swidth[0]; ++s) {
            /*AccAllCacheSamples(tSampCntHV);*/
            /* cz277 - semi */
            if (visitKindTr == PLUTTVK || visitKindTr == PLNONEVK || visitKindTr == PLUTTFRMVK) 
                visitKindHV = PLNONEVK;
            labelInfo->dimFLab = swidth[s];
            cacheHV[s] = CreateCache(&cacheHeap, scriptHV, scriptCntHV, (Ptr) &hset, &obs, s, GetDefaultNCacheSamples(), visitKindHV, &xfInfo, labelInfo, TRUE);
        }
    }
    /* set need2Unload flag */
    SetNeed2UnloadFlag();
    /* cz277 - trans */
    /* initialise Acc structures for TransP update */
    if ((uFlags & UPTRANS) != 0) {
        switch (objfunKind) {
            case XENTOF:
            case MMSEOF:
            case MLOF:
                NumAccs = 1;
                break;
            case MMIOF:
	        NumAccs = 2;
                break;
            case SMBROF:
            case MPEOF:
            case MWEOF:
                NumAccs = 3;
                break;
            default:
                HError(4399, "Initialise: Unkown objective function for transition probability update");
        }
        AttachAccsParallel(&hset, &accHeap, uFlags, NumAccs);
        ZeroAccsParallel(&hset, uFlags, NumAccs);
    }

    if ((labelKind & LATLK) != 0) {
        InitialiseFBInfo(&fbInfo, &hset, cacheTr[1]->labelInfo->uFlags, FALSE);
        for (s = 1; s <= hset.swidth[0]; ++s) {
            fbInfo.llhMat[s] = hset.annSet->llhMat[s];
            fbInfo.occMat[s] = hset.annSet->outLayers[s]->yFeaMats[1];	/* cz277 - many */
        }
        fbInfo.FSmoothH = FSmoothH;
        /* cz277 - frame rejection */
        for (s = 1; s <= hset.swidth[0]; ++s) {
            if ((labelKind & LABLK) == 0) {	/* ref target comes from comparing the num occs */
                fbInfo.refVec[s] = CreateIntVec(&latHeap, GetNBatchSamples());
                fbInfo.findRef = TRUE;
            }
            else {	/* ref target comes from hard labels */
                fbInfo.refVec[s] = cacheTr[s]->labVec;
                fbInfo.findRef = FALSE;
            }
            fbInfo.occVec[s] = CreateDVector(&latHeap, GetNBatchSamples());
        }
    }
    /* set update flags */
    /*SetANNUpdateFlag(&hset);
    SetNMatUpdateFlag(&hset);
    SetNVecUpdateFlag(&hset);*/
    if (optWeightL2Norm == TRUE)
        CheckFitMaxNorm(hset.annSet);
}

/* cz277 - gradprobe */
#ifdef GRADPROBE
void ClearGradProbe(ANNSet *annSet) {
    int i;
    AILink curAI;
    ADLink annDef;
    LELink layerElem;

    curAI = annSet->defsTail;
    while (curAI != NULL) {
        annDef = curAI->annDef;
        for (i = annDef->layerNum - 1; i >= 0; --i) {
            layerElem = annDef->layerList[i];
            memset(layerElem->wghtGradInfoVec + 1, 0, DVectorSize(layerElem->wghtGradInfoVec) * sizeof(double));
            memset(layerElem->biasGradInfoVec + 1, 0, DVectorSize(layerElem->biasGradInfoVec) * sizeof(double));
            layerElem->maxWghtGrad = -1.0E30;
            layerElem->minWghtGrad = 1.0E30;
            layerElem->meanWghtGrad = 0.0;
            layerElem->maxBiasGrad = -1.0E30;
            layerElem->minBiasGrad = 1.0E30;
            layerElem->meanBiasGrad = 0.0;
        }
        curAI = curAI->prev;
    }
}
#endif

/* cz277 - gradprobe */
#ifdef GRADPROBE
void ShowGradProbe(ANNSet *annSet, int batCnt) {
    int i, j, k, size;
    AILink curAI;
    ADLink annDef;
    LELink layerElem;
    double totalWght, totalBias;

    curAI = annSet->defsTail;
    while (curAI != NULL) {
        annDef = curAI->annDef;
        for (i = annDef->layerNum - 1; i >= 0; --i) {
            layerElem = annDef->layerList[i];
            totalWght = layerElem->nodeNum * layerElem->inputDim;
            totalWght *= batCnt;
            totalBias = layerElem->nodeNum;
            totalBias *= batCnt;
            printf("Gradients of Layer %d:\n", i + 1);
            size = DVectorSize(layerElem->wghtGradInfoVec);
            printf("\tWeights:\n");
            printf("\t\tmaxWghtGrad = %e\n", layerElem->maxWghtGrad);
            printf("\t\tminWghtGrad = %e\n", layerElem->minWghtGrad);
            printf("\t\tmeanWghtGrad = %e\n", layerElem->meanWghtGrad / totalWght);
            printf("\t\tValue Buckets:\n");
            for (j = 1; j <= size; ++j) {
                k = (j - 1 - size / 2) * PROBERESOLUTE;
                if (layerElem->wghtGradInfoVec[j] != 0)
                    printf("\t\t\t%d ~ %d ==> %f%% [%e]\n", k, k + PROBERESOLUTE, 100.0 * layerElem->wghtGradInfoVec[j] / totalWght, layerElem->wghtGradInfoVec[j]);
            }
            size = DVectorSize(layerElem->biasGradInfoVec);
            printf("\tBiases:\n");
            printf("\t\tmaxBiasGrad = %e\n", layerElem->maxBiasGrad);
            printf("\t\tminBiasGrad = %e\n", layerElem->minBiasGrad);
            printf("\t\tmeanBiasGrad = %e\n", layerElem->meanBiasGrad / totalWght);
            printf("\t\tValue Buckets:\n");
            for (j = 1; j <= size; ++j) {
                k = (j - 1 - size / 2) * PROBERESOLUTE;
                if (layerElem->biasGradInfoVec[j] != 0)
                    printf("\t\t\t%d ~ %d ==> %f%% [%e]\n", k, k + PROBERESOLUTE, 100.0 * layerElem->biasGradInfoVec[j] / totalBias, layerElem->biasGradInfoVec[j]);
            }
            printf("\n\n");
        }
        curAI = curAI->prev;
    }
}
#endif

/* cz277 - gradlim */
float GetCurClipScalingFactor(int layerIdx, float scale) {
    return (float) pow(2.0, scale * layerIdx);
}


void NMatBundleSGDUpdate(NMatBundle *bundle, float learnRate, float momentum, float weightDecay, float gradClip, float updtClip, float gradL2Scale, float updtL2Scale) {
    int nrows, ncols, batchIndex;

    if (bundle == NULL)
        HError(4390, "NMatBundleSGDUpdate: NULL bundle is given");
    if (bundle->kind != SIBK)
        HError(4391, "NMatBundleSGDUpdate: Only SI bundle kind is allowed");
    batchIndex = GetGlobalBatchIndex();
    if (bundle->batchIndex != batchIndex - 1 && bundle->batchIndex != batchIndex)
        HError(4392, "NMatBundleSGDUpdate: Wrong bundle batch index");
    if (bundle->batchIndex == batchIndex)
        return;
    ++bundle->batchIndex;

    nrows = (int) bundle->variables->rowNum;
    ncols = (int) bundle->variables->colNum;
    /* gradient clipping */
    if (gradClip > 0.0)
        ClipNMatrixVals(bundle->gradients, nrows, ncols, gradClip, (-1.0) * gradClip, bundle->gradients);
    /* gradient l2 norm */
    if (gradL2Scale != 0.0) {
        gradL2Scale *= ncols;
        CalNMatrixL2NormByRow(bundle->gradients, GetTmpNVec());
        ScaleNVector(1.0 / fabs(gradL2Scale), bundle->gradients->rowNum, GetTmpNVec());
        if (gradL2Scale > 0.0)
            ClipNVectorVals(GetTmpNVec(), bundle->gradients->rowNum, FLT_MAX, 1.0, GetTmpNVec());
        DivideNMatrixByRow(bundle->gradients, GetTmpNVec(), bundle->gradients);
    }
    /* weight decay */
    if (weightDecay != 0.0)
        AddScaledNMatrix(bundle->variables, nrows, ncols, weightDecay, bundle->gradients);
    /* learning rates */
    if (learnRate != 0.0)
        ScaleNMatrix(learnRate, nrows, ncols, bundle->gradients);
    else if (bundle->neglearnrates != NULL)
        MulNMatrix(bundle->gradients, bundle->neglearnrates, nrows, ncols, bundle->gradients);
    /* update clipping */
    if (updtClip > 0.0)
        ClipNMatrixVals(bundle->gradients, nrows, ncols, updtClip, (-1.0) * updtClip, bundle->gradients);
    /* update l2 norm */
    if (updtL2Scale != 0.0) {
        updtL2Scale *= ncols;
        CalNMatrixL2NormByRow(bundle->gradients, GetTmpNVec());
        ScaleNVector(1.0 / fabs(updtL2Scale), bundle->gradients->rowNum, GetTmpNVec());
        if (updtL2Scale > 0.0)
            ClipNVectorVals(GetTmpNVec(), bundle->gradients->rowNum, FLT_MAX, 1.0, GetTmpNVec());
        DivideNMatrixByRow(bundle->gradients, GetTmpNVec(), bundle->gradients);
    }
    /* momentum */
    ScaledSelfAddNMatrix(bundle->gradients, nrows, ncols, momentum, bundle->updates);
    /* update the parameters */
    AddNMatrix(bundle->updates, nrows, ncols, bundle->variables);
    /* parameter max norm */
    if (optWeightL2Norm == TRUE) {
        CalNMatrixL2NormByRow(bundle->variables, GetTmpNVec());
        ScaleNVector(1.0 / weightL2NormBound, bundle->variables->rowNum, GetTmpNVec());
        ClipNVectorVals(GetTmpNVec(), bundle->variables->rowNum, FLT_MAX, 1.0, GetTmpNVec());
        DivideNMatrixByRow(bundle->variables, GetTmpNVec(), bundle->variables);
    }
}

void NVecBundleSGDUpdate(NVecBundle *bundle, float learnRate, float momentum, float weightDecay, float gradClip, float updtClip, float gradL2Scale, float updtL2Scale) {
    int nlen;
    int batchIndex;
    float scale;

    if (bundle == NULL)
        HError(4390, "NVecBundleSGDUpdate: NULL bundle is given");
    if (bundle->kind != SIBK)
        HError(4391, "NVecBundleSGDUpdate: Only SI bundle kind is allowed");
    batchIndex = GetGlobalBatchIndex();
    if (bundle->batchIndex != batchIndex - 1 && bundle->batchIndex != batchIndex)
        HError(4392, "NVecBundleSGDUpdate: Wrong bundle batch index");
    if (bundle->batchIndex == batchIndex)
        return;
    ++bundle->batchIndex;

    nlen = (int) bundle->variables->vecLen; 
    /* gradient clipping */
    if (gradClip > 0.0)
        ClipNVectorVals(bundle->gradients, nlen, gradClip, (-1.0) * gradClip, bundle->gradients);
    /* gradient l2 norm */
    if (gradL2Scale != 0.0) {
        gradL2Scale *= nlen;
        CalNVectorL2Norm(bundle->gradients, &scale);
        scale = fabs(gradL2Scale) / scale;
        if (gradL2Scale > 0.0 && scale > 1.0)
            scale = 1.0;
        ScaleNVector(scale, nlen, bundle->gradients);
    }
    /* weight decay */
    if (weightDecay != 0.0)
        AddScaledNVector(bundle->variables, nlen, weightDecay, bundle->gradients);
    /* learning rates */
    if (learnRate != 0.0)
        ScaleNVector(learnRate, nlen, bundle->gradients);
    else if (bundle->neglearnrates != NULL)
        MulNVector(bundle->gradients, bundle->neglearnrates, nlen, bundle->gradients);
    /* update clipping */
    if (updtClip > 0.0)
        ClipNVectorVals(bundle->gradients, nlen, updtClip, (-1.0) * updtClip, bundle->gradients); 
    /* update l2 norm */
    if (updtL2Scale != 0.0) {
        updtL2Scale *= nlen;
        CalNVectorL2Norm(bundle->gradients, &scale);
        scale = fabs(updtL2Scale) / scale;
        if (updtL2Scale > 0.0 && scale > 1.0)
            scale = 1.0;
        ScaleNVector(scale, nlen, bundle->gradients);
    }
    /* momentum */
    ScaledSelfAddNVector(bundle->gradients, nlen, momentum, bundle->updates);
    /* update the parameters */
    AddNVector(bundle->updates, nlen, bundle->variables);

}

void SGDUpdateActivationOnlyLayer(LELink layerElem, float learnRate, float momentum, float weightDecay, float gradClip, float updtClip, float gradL2Scale, float updtL2Scale) {
    HError(4301, " SGDUpdateActivationOnlyLayer: Function not implemented");
}

void SGDUpdateConvolutionLayer(LELink layerElem, float learnRate, float momentum, float weightDecay, float gradClip, float updtClip, float gradL2Scale, float updtL2Scale) {
    HError(4301, "SGDUpdateConvolutionLayer: Function not implemented");
}

void SGDUpdatePerceptronLayer(LELink layerElem, float learnRate, float momentum, float weightDecay, float gradClip, float updtClip, float gradL2Scale, float updtL2Scale) {
    int i;

    if (layerElem->layerKind != PERCEPTRONLAK)
        HError(4392, "SGDUpdatePerceptronLayer: Function only applicable to PERCEPTRON layer");
    
    if (layerElem->wghtMat->updateflag == TRUE)
        NMatBundleSGDUpdate(layerElem->wghtMat, learnRate, momentum, weightDecay, gradClip, updtClip, gradL2Scale, updtL2Scale);
        /*NMatBundleSGDUpdate(layerElem->wghtMat, learnRate, momentum, weightDecay, optWghtUpdtClip, wghtUpdtPosClip / clipScale, wghtUpdtNegClip / clipScale);*/
    if (layerElem->biasVec->updateflag == TRUE)
        NVecBundleSGDUpdate(layerElem->biasVec, learnRate, momentum, weightDecay, gradClip, updtClip, gradL2Scale, updtL2Scale);
        /*NVecBundleSGDUpdate(layerElem->biasVec, learnRate, momentum, weightDecay, optBiasUpdtClip, biasUpdtPosClip / clipScale, biasUpdtNegClip / clipScale); */
    if (layerElem->actfunVecs != NULL)
        for (i = 1; i <= layerElem->actfunParmNum; ++i)
            if (layerElem->actfunVecs[i]->updateflag == TRUE)
                NVecBundleSGDUpdate(layerElem->actfunVecs[i], learnRate, momentum, weightDecay, gradClip, updtClip, gradL2Scale, updtL2Scale);
                /*NVecBundleSGDUpdate(layerElem->actfunVecs[i], learnRate, momentum, weightDecay, optActfunUpdtClip, actfunUpdtPosClip / clipScale, actfunUpdtNegClip / clipScale);*/

}

void SGDUpdateSubsamplingLayer(LELink layerElem, float learnRate, float momentum, float weightDecay, float gradClip, float updtClip, float gradL2Scale, float updtL2Scale) {
    HError(4301, "SGDUpdateSubsamplingLayer: Function not implemented");
}

void SGDUpdateANNSet(ANNSet *annSet, float scale, float learnRate, float momentum, float weightDecay, float gradClip, float updtClip, float gradL2Scale, float updtL2Scale) {
    int i;
    AILink curAI;
    ADLink annDef;
    LELink layerElem;
    float curGradClip, curUpdtClip;

    /* do the sample number based normalisations */
    if (optNormWeightDecay == TRUE)
        weightDecay *= scale;
    if (optNormMomentum == TRUE)
        momentum *= scale;
    if (optNormGradientClip == TRUE)
        gradClip *= scale;
    if (optNormUpdateClip == TRUE)
        updtClip *= scale;
    if (optNormL2GradientScale == TRUE)
        gradL2Scale *= scale;
    if (optNormL2UpdateScale == TRUE)
        updtL2Scale *= scale;
    /* proceed each ANNDef */
    curAI = annSet->defsTail;
    while (curAI != NULL) {
        annDef = curAI->annDef;
        for (i = annDef->layerNum - 1; i >= 0; --i) {
            layerElem = annDef->layerList[i];
            curGradClip = gradClip;
            curUpdtClip = updtClip;
            if (clipScaleFactor != 0.0) {
                curGradClip /= GetCurClipScalingFactor(i, clipScaleFactor);
                curUpdtClip /= GetCurClipScalingFactor(i, clipScaleFactor);
            }
            /* if no parameter to update in this layer */
            /*if (layerElem->trainInfo->updateFlag == 0)
                continue;*/
            switch (layerElem->layerKind) {
            case ACTIVATIONONLYLAK: 
                SGDUpdateActivationOnlyLayer(layerElem, learnRate, momentum, weightDecay, curGradClip, curUpdtClip, gradL2Scale, updtL2Scale); 
                break;
            case CONVOLUTIONLAK: 
                SGDUpdateConvolutionLayer(layerElem, learnRate, momentum, weightDecay, curGradClip, curUpdtClip, gradL2Scale, updtL2Scale); 
                break;
            case PERCEPTRONLAK: 
                SGDUpdatePerceptronLayer(layerElem, learnRate, momentum, weightDecay, curGradClip, curUpdtClip, gradL2Scale, updtL2Scale); 
                break;
            case SUBSAMPLINGLAK: 
                SGDUpdateSubsamplingLayer(layerElem, learnRate, momentum, weightDecay, curGradClip, curUpdtClip, gradL2Scale, updtL2Scale); 
                break;
            default:
                HError(4399, "SGDUpdateANNSet: Unknown layer type");
            }
        }
        /* fetch next ANNDef */
        curAI = curAI->prev;
    }

    ++updateIndex;
}

ReturnStatus ReloadHMMSet(MSILink MSIPtr) {
    int i, mappedTargetNum=-1;

    if (trace & T_TOP) {
        if (MSIPtr->epochIdx <= 0)
            printf("\tReload initial HMMSet\n");
        else
            printf("\tReload HMMSet generated by epoch %d\n", MSIPtr->epochIdx);
    }
    if (optMapTarget) 
        mappedTargetNum = hset.annSet->mapStruct->mappedTargetNum;
    /* reset current hset */
    ResetHMMSet(&hset);
    ResetHeap(&modelHeap);
    /* reset update index*/
    updateIndex = MSIPtr->updtIdx;
    /* reset batch size (actually to regist a tmpNMat for FindMaxElement) */
    SetNBatchSamples(GetNBatchSamples());
    /* create HMMSet */
    CreateHMMSet(&hset, &modelHeap, TRUE);
    /* AddMMFs */
    for (i = 0; i < MSIPtr->macCnt; ++i) 
        AddMMF(&hset, MSIPtr->macFN[i]);
    /* Load HMM list */
    if (hmmListFn != NULL && MakeHMMSet(&hset, hmmListFn) < SUCCESS) 
        HError(4300, "ReloadHMMSet: MakeHMMSet failed");
    if (LoadHMMSet(&hset, MSIPtr->baseDir, MSIPtr->hmmExt) < SUCCESS) 
        HError(4300, "ReloadHMMSet: LoadHMMSet failed");
    if (hset.annSet == NULL) 
        HError(4300, "ReloadHMMSet: No ANN model available");
    /* init train struct */
    InitTrainInfo(&hset, optHasLabMat, optHasNLR, optHasSSG, TRUE);
    InitErrMix(&hset);
    if (optMapTarget) {
        SetupStateInfoList(&hset);
        if (SetupTargetMapList(&hset, mappingFn, mappedTargetNum) < SUCCESS) 
            HError(4300, "Initialise: Failed to load the target mapping file");
        InitMapStruct(&hset);
        recVecMapSum = CreateIntVec(&gcheap, GetNBatchSamples());
    }
    CreateTmpNMat(hset.hmem);
    SetANNUpdateFlag(&hset);	/* cz277 - 150811 */
    /*SetLayerUpdateFlag(&hset);*/	/* cz277 - 150811 */
    /*SetActFunUpdateFlag(&hset);*/	/* cz277 - 150811 */
    SetNMatUpdateFlag(&hset);   /* cz277 - 151020 */
    SetNVecUpdateFlag(&hset);   /* cz277 - 151020 */
    /* cz277 - 1015 */
    CheckANNBatchIndex(hset.annSet, 0);
    /* cz277 - batch sync */
    SetFeaMixBatchIndex(hset.annSet, GetGlobalBatchIndex());
    SetNBundleBatchIndex(hset.annSet, GetGlobalBatchIndex());

    /* update cache associated configs */
    for (i = 1; i <= hset.swidth[0]; ++i) 
        ResetCacheHMMSetCfg(cacheTr[i], &hset);    
    if (scriptHV != NULL) 
        for (i = 1; i <= hset.swidth[0]; ++i) 
            ResetCacheHMMSetCfg(cacheHV[i], &hset);

    return SUCCESS;
}

void UpdateTransValues(int stateNum, float *acc1, float *acc2, float *newWeights, float *oldWeights, float C) {
    int i, iter;
    float sum=0.0, objVal[3]={0.0,0.0,0.0}, accVal, maxF;
    Vector fValues = CreateVector(&gstack, stateNum);

    if (C == 0) {
        return;
    }
    for (i = 1; i <= stateNum; ++i) {
        sum += acc1[i];
    }
    if (sum < 1.0) {
        return;
    }
    for (iter = 1; iter <= 100; ++iter) {	/* 100 is copied from HMMIRest */
        /* calculate objective */
        objVal[2] = objVal[1];
        objVal[1] = objVal[0];
        objVal[0] = 0.0;
        for (i = 1; i <= stateNum; ++i) {
            if (newWeights[i] > 0.0) {
                if (acc2 != NULL)
                    accVal = acc2[i];
                else
                    accVal = 0.0;
                objVal[0] = acc1[i] * log(newWeights[i]) - accVal / C * exp(log(newWeights[i] / oldWeights[i]) * C) + objVal[0];
            }
        }
        if (objVal[0] < objVal[1] && objVal[0] < objVal[2] && objVal[2] != 0 && fabs(objVal[0] - objVal[1]) > fabs(objVal[0]) * 0.0001)
            HError(-1, "UpdateTransValues: Objective not increasing, %e < %e < %e", objVal[0], objVal[1], objVal[2]);
        /* find max f_m */
        maxF = 0.0;
        for (i = 1; i <= stateNum; ++i) {
            if (newWeights[i] > 0.0) {
                if (acc2 != NULL)
                    accVal = acc2[i];
                else
                    accVal = 0.0;
                fValues[i] = accVal / oldWeights[i] * exp(log(newWeights[i] / oldWeights[i]) * (C - 1));
                if (C > 1.0)
                    fValues[i] *= C;
                maxF = MAX(maxF, fValues[i]);
                /*if (fValues[i] > maxF)
                    maxF = fValues[i];*/
            }
        }
        sum = 0.0;
        for (i = 1; i <= stateNum; ++i) {
            if (newWeights[i] > 0.0) {
                newWeights[i] = newWeights[i] * (maxF - fValues[i]) + acc1[i];
                sum += newWeights[i];
            }
        }
        for (i = 1; i <= stateNum; ++i)
            newWeights[i] /= sum;
    }
    Dispose(&gstack, fValues);
}

/* cz277 - trans */
void UpdateHMMTransLab(int px, HLink hmm) {
    int i, j, N;
    float x, occi, sum;
    TrAcc *ta;

    ta = (TrAcc *) GetHook(hmm->transP);
    if (ta == NULL)
        return;
    N = hmm->numStates;
    for (i = 1; i < N; ++i) {
        occi = ta->occ[i];
        sum = 0.0;
        if (occi > 0.0) {
            for (j = 2; j <= N; ++j) {
                x = ta->tran[i][j] / occi;
                hmm->transP[i][j] = x;
                sum += x;
            }
        }
        else {
            HError(-1, "UpdateHMMTransLab: Model %d[%s]: no transition out of state %d", px, HMMPhysName(&hset, hmm), i);
        }
        for (j = 2; j <= N; ++j) {
            x = hmm->transP[i][j] / sum;
            if (x < MINLARG) {
                hmm->transP[i][j] = LZERO;
            }
            else {
                hmm->transP[i][j] = log(x);
            }
        }
    }
    SetHook(hmm->transP, NULL);
}

void UpdateHMMTransLat(int px, HLink hmm) {
    int i, j, stateNum=hmm->numStates;
    TrAcc *trAcc1, *trAcc2, *trAcc3;
    Vector newWeights = CreateVector(&gstack, stateNum);
    Vector oldWeights = CreateVector(&gstack, stateNum);
    float transP, occ;

    trAcc1 = GetHook(hmm->transP);
    trAcc2 = (NumAccs == 1? NULL: trAcc1 + 1);
    trAcc3 = ((NumAccs == 3 && optHasISmooth)? trAcc1 + 2: NULL);
    if (trAcc1 == NULL) {
        return;
    }

    for (i = 1; i < stateNum; ++i) {
        if (trAcc3 != NULL) 
            occ = trAcc3->occ[i];
        else 
            occ = trAcc1->occ[i];
        if (occ > MinOccTrans) {
            for (j = 1; j <= stateNum; ++j) {
                if (hmm->transP[i][j] > MINEARG)
                    transP = exp(hmm->transP[i][j]);
                else
                    transP = 0.0;
                newWeights[j] = transP;
                oldWeights[j] = transP;
            }
            if ((uFlags & UPTRANS) && (trAcc2 != NULL))
                for (j = 1; j < stateNum; ++j)
                    trAcc2->tran[i][j] = 0.0;
            UpdateTransValues(stateNum, trAcc1->tran[i], trAcc2? trAcc2->tran[i]: NULL, newWeights, oldWeights, CTrans);
            for (j = 1; j < stateNum; ++j)
                if (newWeights[i] == 0 && oldWeights[i] != 0)
                    HError(-1, "UpdateTrans: Transitions going to zero: advise setting e.g. ISMOOTHTAUT = 10");
            for (j = 1; j <= stateNum; ++j) {
                if (newWeights[j] > 0.0)
                    hmm->transP[i][j] = log(newWeights[j]);
                else
                    hmm->transP[i][j] = LZERO;
            }
        }
    }
    SetHook(hmm->transP, NULL);
    /*Dispose(&gstack, oldWeights);*/
    Dispose(&gstack, newWeights);
}

/* cz277 - trans */
void UpdateAllTrans(void) {
    HMMScanState hss;
    int px;
    
    px = 1;
    NewHMMScan(&hset, &hss);
    do {
        switch (updtKind) {
            case BATLEVEL:
                UpdateHMMTransLab(px++, hss.hmm);    
                break;
            case UTTLEVEL:
                UpdateHMMTransLat(px++, hss.hmm);
                break;
            default:
                HError(-1, "UpdateAllTrans: Unknown update mode for transitions");
        }
    } while (GoNextHMM(&hss));
    EndHMMScan(&hss);

    /* reset the accs */
    ResetHeap(&accHeap);
    AttachAccsParallel(&hset, &accHeap, uFlags, NumAccs);
    ZeroAccsParallel(&hset, uFlags, NumAccs);
}

Boolean TermLRSchdOrNot(int curEpochNum) {
    float floatVal;
    float curPosLR = (-1.0) * curNegLR;

    if (curEpochNum == 0) {
        return TRUE;
    }
    if (minEpochNum > 0 && curEpochNum < minEpochNum) {
        return TRUE;
    }
    switch (schdKind) {
        case ADAGRADSK:
            if (maxEpochNum > 0 && curEpochNum >= maxEpochNum) {
                return FALSE;
            }
            break;
        case EXPSK:
            if ((maxEpochNum > 0 && curEpochNum >= maxEpochNum) || (curPosLR < minLR)) {
                return FALSE;
            }
            break;
        case LISTSK:
            if (curEpochNum >= maxEpochNum) {
                return FALSE;
            }
            break;
        case NEWBOBSK:
            if (maxEpochNum > 0 && curEpochNum >= maxEpochNum) {
                if (trace & T_SCH) 
                    printf("NewBob: Stop since the maximum allowed epoch number is reached\n");

                return FALSE;
            }
            if (curPosLR < minLR) {
                if (trace & T_SCH) 
                    printf("NewBob: Stop since the criterion value change is smaller than the allowed minimum\n"); 
   
                return FALSE;
            }
            floatVal = tailMSI->crtVal - tailMSI->prev->crtVal;
            if (NewBob_Status == 1 && floatVal < NewBob_Stop) {
                if (trace & T_SCH)
                    printf("NewBob: Stop since the criterion value change is smaller than the stopping threshold\n");

                /* need to reload previous model  */
                if (floatVal < 0.0) {
                    /* have model to detach */
                    if (tailMSI != &inputMSI) {
                        PopModelSetInfo();
                    }
                    ReloadHMMSet(tailMSI);
                }
                return FALSE;
            }
            break;
        default:
            HError(4399, "Unknown learning rate scheduler");
    }

    return TRUE;
}

void SaveLRSchd(char *fname) {
    FILE *fp;

    if ((fp = fopen(fname, "w")) == NULL) 
        HError(4311, "SaveLRSchd: Fail to save the auxiliary file");

    switch(schdKind) {
        case NEWBOBSK:
            /* <LRSCHEDULER> NEWBOB */
            fprintf(fp, "<LRSCHEDULER> ");
            fprintf(fp, "NEWBOB");
            fprintf(fp, "\n");
            /* <NEWBOBCRT> ? */
            fprintf(fp, "<NEWBOBCRT> ");
            switch (NewBob_Crt) {
                case ACCNBC:
                    fprintf(fp, "ACC");
                case MAPACCNBC:
                    fprintf(fp, "MAPACC");
                case LLHVALNBC:
                    fprintf(fp, "LLHVAL");
                case MAPLLHVALNBC:
                    fprintf(fp, "MAPLLHVAL");
                default:
                    HError(4399, "SaveLRSchd: Unsupported NewBob criterion");
            }
            fprintf(fp, "\n");
            /* <STATUS> INITIAL or RAMPING */
            fprintf(fp, "<STATUS> ");
            if (NewBob_Status == 0) 
                fprintf(fp, "INITIAL");
            else if (NewBob_Status == 1)
                fprintf(fp, "RAMPING");
            else
                fprintf(fp, "UNKNOWN");
            fprintf(fp, "\n");
            /* <RAMPSTART> x.y */
            fprintf(fp, "<RAMPSTART> ");
            fprintf(fp, "%e", NewBob_RampStart);
            fprintf(fp, "\n");
            /* <RAMPSTART> x.y */
            fprintf(fp, "<STOPDIFF> ");
            fprintf(fp, "%e", NewBob_Stop);
            fprintf(fp, "\n");
        default:
            HError(4399, "SaveLRSchd: Unsupported learning rate scheduler for the auxiliary file");
    }
    /* <CURLRVAL> x.y */
    fprintf(fp, "<LEARNRATE> ");
    fprintf(fp, "%e", (-1.0) * curNegLR);
    fprintf(fp, "\n");
    /* <EPOCHOFFSET> */
    fprintf(fp, "<EPOCHOFFSET> ");
    fprintf(fp, "%d", epochOff);
    fprintf(fp, "\n");
}

void LoadLRSchd(char *fname) {
    Source auxSrc;
    char buf[MAXSTRLEN];
    float floatVal;

    if (InitSource(fname, &auxSrc, NoFilter) < SUCCESS) 
        HError(4310, "LoadLRSchd: cannot open the auxiliary file");

    SkipComment(&auxSrc);
    /* <LRSCHEDULER> ? */
    ReadString(&auxSrc, buf);
    if (strcmp(buf, "<LRSCHEDULER>") == 0) {
        if (strcmp(buf, "NEWBOB") == 0) 
            schdKind = NEWBOBSK;
        else 
            HError(4326, "LoadLRSchd: Unsupported learning rate kind");
        /* proceed each learning rate related options */
        while (ReadString(&auxSrc, buf)) {
            /* <NEWBOBCRT> ? */
            if (schdKind == NEWBOBSK && strcmp(buf, "<NEWBOBCRT>") == 0) {
                ReadString(&auxSrc, buf);
                if (strcmp(buf, "ACC") == 0) 
                    NewBob_Crt = ACCNBC;
                else if (strcmp(buf, "MAPACC") == 0) 
                    NewBob_Crt = MAPACCNBC;
                else if (strcmp(buf, "LLHVAL") == 0) {
                    NewBob_Crt = LLHVALNBC;
                    showObjFunKind = showObjFunKind | MLOF;
                }
                else if (strcmp(buf, "MAPLLHVAL") == 0) {
                    NewBob_Crt = MAPLLHVALNBC;
                    showObjFunKind = showObjFunKind | MLOF;
                }
                else  
                    HError(4326, "LoadLRSchd: Unknown criterion for NewBob");
            }
            else if (schdKind == NEWBOBSK && strcmp(buf, "<STATUS>") == 0) {    /* <STATUS> ? */
                ReadString(&auxSrc, buf);
                if (strcmp(buf, "INITIAL") == 0)
                    NewBob_Status = 0;
                else if (strcmp(buf, "RAMPING") == 0)
                    NewBob_Status = 1;
                else
                    HError(4326, "LoadLRSchd: Unknown NewBob scheduler status");
            }
            else if (schdKind == NEWBOBSK && strcmp(buf, "<DECAYFACTOR>") == 0)  /* <DECAYFACTOR> x.y */
                ReadFloat(&auxSrc, &NewBob_DecayFactor, 1, FALSE);
            else if (schdKind == NEWBOBSK && strcmp(buf, "<RAMPSTART>") == 0)  /* <RAMPSTART> x.y */
                ReadFloat(&auxSrc, &NewBob_RampStart, 1, FALSE);
            else if (schdKind == NEWBOBSK && strcmp(buf, "<STOPDIFF>") == 0)   /* <STOPDIFF> x.y */
                ReadFloat(&auxSrc, &NewBob_Stop, 1, FALSE);
            else if (strcmp(buf, "<LEARNRATE>") == 0) {  /* <LEARNRATE> x.y */
                ReadFloat(&auxSrc, &floatVal, 1, FALSE);
                curNegLR = (-1.0) * floatVal;
            }
            else if (strcmp(buf, "<EPOCHOFFSET>") == 0)  /* <EPOCHOFFSET> x */
                ReadInt(&auxSrc, &epochOff, 1, FALSE);
            else 
                HError(4326, "LoadLRSchd: Unknown option in the auxiliary file");
        }
    }

    CloseSource(&auxSrc);
}

void UpdateLRSchdAdaGrad(ANNSet *annSet, float eta, int K) {
    int i, j;
    LELink layerElem;
    AILink annInfo;
    ADLink annDef;

    /* reset all processed fields */
    ResetAllBundleProcessedFields("UpdateLRSchdAdaGrad", annSet);

    /* initialise ANN info */
    annInfo = annSet->defsHead;
    while (annInfo != NULL) {
        /* get current ANN def */
        annDef = annInfo->annDef;
        /* proceed each layer */
        for (i = 0; i < annDef->layerNum; ++i) {
            /* get current layer */
            layerElem = annDef->layerList[i];
            switch (layerElem->layerKind) {
            case ACTIVATIONONLYLAK: HError(4301, "UpdateLRSchdAdaGrad: Function not implemented"); break;
            case CONVOLUTIONLAK: HError(4301, "UpdateLRSchdAdaGrad: Function not implemented"); break;
            case PERCEPTRONLAK:
                if (layerElem->wghtMat->processed == FALSE) {
                    CompAdaGradNMatrix(eta, K, layerElem->wghtMat->sumsquaredgrad, layerElem->wghtMat->neglearnrates);
                    layerElem->wghtMat->processed = TRUE;
                }
                if (layerElem->biasVec->processed == FALSE) {
                    CompAdaGradNVector(eta, K, layerElem->biasVec->sumsquaredgrad, layerElem->biasVec->neglearnrates);
                    layerElem->biasVec->processed = TRUE;
                }
                if (layerElem->actfunVecs != NULL)
                    for (j = 1; j <= layerElem->actfunParmNum; ++j)
                        if (layerElem->actfunVecs[j]->processed == FALSE) {
                            CompAdaGradNVector(eta, K, layerElem->actfunVecs[j]->sumsquaredgrad, layerElem->actfunVecs[j]->neglearnrates);
                            layerElem->actfunVecs[j]->processed = TRUE;
                        }
                break;
            case SUBSAMPLINGLAK: HError(4301, "UpdateLRSchdAdaGrad: Function not implemented"); break;
            default:
                HError(4399, "UpdateLRSchdAdaGrad: Unknown layer kind");
            }
        }
        /* fetch next ANN info */
        annInfo = annInfo->next;
    }
}

/* per update */
float UpdateLRSchdPerU(int curEpochNum, int updtSamp) {
    float retNegLR = 0.0;

    switch (schdKind) {
    case ADAGRADSK:
        retNegLR = (-1.0) * initLR;
        if (optNormLearnRate) 
            retNegLR /= updtSamp;
        UpdateLRSchdAdaGrad(hset.annSet, retNegLR, AdaGrad_K);
        retNegLR = 0.0;
        break;
    case EXPSK:
        if (Exp_Gamma <= 0.0) {
            if (minEpochNum <= 0) 
                Exp_Gamma = 2.0 * minEpochNum * tSampCntTr;
            else if (maxEpochNum <= 0) 
                Exp_Gamma = 2.0 * maxEpochNum * tSampCntTr;
            else 
                Exp_Gamma = 2.0 * Exp_Base * tSampCntTr;
            printf("UpdateLRSchdPerU: GAMMA for exponential learning rate scheduler not set, reset to %e\n", Exp_Gamma);
        }
        retNegLR = (-1.0) * Exp_TrSampIdx / Exp_Gamma;
        curNegLR = (-1.0) * initLR * pow(Exp_Base, retNegLR);
        retNegLR = curNegLR;
        if (optNormLearnRate) 
            retNegLR /= updtSamp;
        Exp_TrSampIdx += updtSamp;
        break;
    case LISTSK:
    case NEWBOBSK:
        retNegLR = curNegLR;
        if (optNormLearnRate) 
            retNegLR /= updtSamp;
        break;
    default:
        break;
    }
    return retNegLR;
}

/* per epoch */
void UpdateLRSchdPerE(int curEpochNum) {
    float floatVal;

    switch (schdKind) {
    case LISTSK:
        curNegLR = (-1.0) * List_LRs[curEpochNum + 1];
        break;
    case NEWBOBSK:
        if (curEpochNum == 0) 
            curNegLR = (-1.0) * initLR;
        else if (NewBob_Status == 0) {   /* in initia status */
            if (tailMSI->prev != NULL) 
                floatVal = tailMSI->crtVal - tailMSI->prev->crtVal;
            else 
                floatVal = 1.0;
            if (floatVal < NewBob_RampStart) {
                if (curEpochNum < minEpochNum) {    /* to restore the model and reduce the learning rate by half */
                    if (trace & T_SCH)
                        printf("NewBob: Val diff %e < RAMPSTART = %e, learning rate *= DECAYFACTOR = %e\n", floatVal, NewBob_RampStart, NewBob_DecayFactor);
                    curNegLR *= NewBob_DecayFactor;
                }
                else {  /* go into the ramping status */
                    if (trace & T_SCH) 
                        printf("NewBob: Val diff %e < RAMPSTART = %e and cur epoch %d is > min %d, learning rate *= DECAYFACTOR = %e, start ramping\n", floatVal, NewBob_RampStart, curEpochNum, minEpochNum, NewBob_DecayFactor);
                    curNegLR *= NewBob_DecayFactor;
                    NewBob_Status = 1;
                }
                /* need to reload the model? */
                if (floatVal < 0.0) {
                    /* have model to detach */
                    if (tailMSI != &inputMSI) 
                        PopModelSetInfo();
                    if (trace & T_SCH) 
                        printf("NewBob: Value diff %e < 0, restore previous model set\n", floatVal);
                    ReloadHMMSet(tailMSI);
                }
            }
        }
        else if (NewBob_Status == 1)   /* in ramping status */
            curNegLR *= NewBob_DecayFactor;
        break;
    default:
        break;
    }
}

float GetConfParmPerEpoch(int curEpochNum, Vector confVec) {
    float value = 0.0;

    if (confVec != NULL) {
        if (curEpochNum > VectorSize(confVec) - 1)
            value = confVec[VectorSize(confVec)];
        else
            value = confVec[curEpochNum + 1];
    }

    return value;
}

void ComputeLogLikelihoods(int nLoaded) {
    int i, S;
    LELink layerElem;
    
    S = hset.swidth[0];
    /* synchronise the data */
    for (i = 1; i <= S; ++i) {
        layerElem = hset.annSet->outLayers[i];
        /* download likelihoods */
        if ((optTrainMode == SEQTM) || (showObjFunKind & MLOF)) {
            ApplyLogTrans(layerElem->yFeaMats[1], nLoaded, layerElem->nodeNum, hset.annSet->llhMat[i]);
            AddNVectorTargetPen(hset.annSet->llhMat[i], hset.annSet->penVec[i], nLoaded, hset.annSet->llhMat[i]);
#ifdef CUDA
            SyncNMatrixDev2Host(hset.annSet->llhMat[i]);
#endif
        }
        /* download likelihoods (for mapped targets) */
        if (optMapTarget) {
            UpdateOutMatMapSum(hset.annSet, nLoaded, i);
            /* convert posteriors to llr */
            if (showObjFunKind & MLOF) {
                ApplyLogTrans(hset.annSet->mapStruct->outMatMapSum[i], nLoaded, hset.annSet->mapStruct->mappedTargetNum, hset.annSet->mapStruct->llhMatMapSum[i]);
                AddNVectorTargetPen(hset.annSet->mapStruct->llhMatMapSum[i], hset.annSet->mapStruct->penVecMapSum[i], nLoaded, hset.annSet->mapStruct->llhMatMapSum[i]);
#ifdef CUDA
                SyncNMatrixDev2Host(hset.annSet->mapStruct->llhMatMapSum[i]);
#endif
            }
#ifdef CUDA
            SyncNMatrixDev2Host(hset.annSet->mapStruct->outMatMapSum[i]);
#endif
        }
        /* download posteriors */
#ifdef CUDA
        SyncNMatrixDev2Host(layerElem->yFeaMats[1]);
#endif
    }
}

void AccCriteria(CriteriaInfo *criteria, DataCache **cache, int nLoaded, Boolean accBatch, Boolean accUtter) {
    int i, S;
    LELink layerElem;

    S = hset.swidth[0];
    /* accumulate for the criteria */
    if (accBatch == TRUE) {
        for (i = 1; i <= S; ++i) {
            layerElem = hset.annSet->outLayers[i];
            FindMaxElement(layerElem->yFeaMats[1], nLoaded, layerElem->nodeNum, recVec);
            if (showObjFunKind & MLOF)
                FindMaxElement(hset.annSet->llhMat[i], nLoaded, layerElem->nodeNum, recVecLLH);
            if (optMapTarget) {
                FindMaxElement(hset.annSet->mapStruct->outMatMapSum[i], nLoaded, hset.annSet->mapStruct->mappedTargetNum, recVecMapSum);
                if (showObjFunKind & MLOF)
                    FindMaxElement(hset.annSet->mapStruct->llhMatMapSum[i], nLoaded, hset.annSet->mapStruct->mappedTargetNum, recVecLLHMapSum);
                UpdateLabMatMapSum(hset.annSet, nLoaded, i);
#ifdef CUDA
                SyncNMatrixDev2Host(hset.annSet->mapStruct->labMatMapSum[i]);
#endif
            }
            AccCriteriaPerB(cache[i], nLoaded, &criteria[i]);
        }
    }
    /* accumulate at utterance level */
    if (accUtter == TRUE) 
        for (i = 1; i <= S; ++i)
            AccCriteriaPerU(cache[i], nLoaded, &criteria[i]);
}

Boolean BatchLevelUpdateOrNot(VisitKind visitKind, Boolean finish, int batchCnt, int nLoaded) {

    if (finish == TRUE)
        return TRUE;

    if (batchCnt % numPerUpdt == 0) {
        switch (visitKind) {
        case PLNONEVK:
        case PLUTTVK:
        case PLUTTFRMVK:
            if (batchCnt >= bgWaitNBatchPL && nLoaded >= edAccBatchLenPL)
                return TRUE;
            return FALSE;
        default:
            break;
        }
        return TRUE;
    }
    return FALSE;
}

void BatchLevelTrainProcess(int curEpochNum) {
    int i, S, nLoaded;
    int updtCnt, sampCnt, batchCnt, tSampCnt, uttCnt;
    Boolean finish = FALSE;
    Boolean accGrad;
    CriteriaInfo criteria[SMAX];
    float retNegLR;
    /* cz277 - mmt */
    float scale, momentum, weightDecay, posGradClip, negGradClip;

    S = hset.swidth[0];
    updtCnt = 0;
    sampCnt = 0;
    tSampCnt = 0;
    /* process the first batch */
    batchCnt = 0;
    uttCnt = -1;

    /* initialise the criteriaInfo */
    memset(&criteria, 0, sizeof(CriteriaInfo) * SMAX);
    SetGlobalBatchIndex(0);
    /* cz277 - batch sync */
    SetNBundleBatchIndex(hset.annSet, 0);
    SetFeaMixBatchIndex(hset.annSet, 0);
    /* initialise the cache until all data finished */
    for (i = 1; i <= S; ++i) 
        InitCache(cacheTr[i]);

    /* cz277 - gradprobe */
#ifdef GRADPROBE
    ClearGradProbe(hset.annSet);
#endif
    weightDecay = GetConfParmPerEpoch(curEpochNum, List_WeightDecay);
    momentum = GetConfParmPerEpoch(curEpochNum, List_MMTs);
    accGrad = FALSE;
    while (!finish) {
        /* load data */
        for (i = 1; i <= S; ++i) {
            finish |= FillAllInpBatch(cacheTr[i], &nLoaded, &uttCnt);
            UnloadCacheData(cacheTr[i]);
            LoadCacheData(cacheTr[i]);
        }
        ForwardProp(hset.annSet, nLoaded, cacheTr[1]->CMDVecPL); /* do forwarding */
        ComputeLogLikelihoods(nLoaded); /* compute sequence level criteria */
        AccCriteria(criteria, cacheTr, nLoaded, TRUE, FALSE); /* accumulate classification criteria */
        BackwardProp(objfunKind, hset.annSet, nLoaded, accGrad); /* do back-propagation */
        accGrad = TRUE;
        /* accumulate the statistics */
        batchCnt += 1;
        sampCnt += nLoaded;
        tSampCnt += nLoaded;
        if (BatchLevelUpdateOrNot(cacheTr[1]->visitKind, finish, batchCnt, nLoaded)) {
	    /* cz277 - batch sync */
	    SetNBundleBatchIndex(hset.annSet, GetGlobalBatchIndex() - 1);
            retNegLR = UpdateLRSchdPerU(curEpochNum, sampCnt);
            NormBackwardPropGradients(hset.annSet, 1.0);
            scale = ((float) sampCnt) / (GetNBatchSamples() * numPerUpdt);
            SGDUpdateANNSet(hset.annSet, scale, retNegLR, momentum, weightDecay, gradientClip, updateClip, gradientL2Scale, updateL2Scale);
            accGrad = FALSE;
            /* update updtCnt */
            ++updtCnt;
            sampCnt = 0;
        }
    }
    /* cz277 - trans */
    /* update the transition probabilities */
    if ((uFlags & UPTRANS) != 0) 
        UpdateAllTrans();

    /* update the target penalties */
    if ((curEpochNum == 0) && ((uFlags & UPTARGETPEN) != 0) && (hset.hsKind == HYBRIDHS))
        for (i = 1; i <= S; ++i) 
            UpdateTargetLogPrior(cacheTr[i], logObsvPrior);
    /* show criteria */
    for (i = 1; i <= S; ++i) {
        if (S > 1) 
            printf("\t\tStream %d: ", i);
        PrintCriteria(&criteria[i], "Train");
    }
    /* cz277 - gradprobe */
#ifdef GRADPROBE
    ShowGradProbe(hset.annSet, batchCnt);
#endif

    /* reset all cache */
    for (i = 1; i <= S; ++i) 
        ResetCache(cacheTr[i]);
    /* output the timming */
    printf("\t\tBatch count = %d, Update count = %d\n", batchCnt, updtCnt);
}

void ForceNMatrixValueByRow(NMatrix *mat, int rowIdx, float val) {
    int colNum = mat->colNum;

#ifdef CUDA
    SetNSegmentCUDA(val, &mat->devElems[rowIdx * colNum], colNum);
#else
    SetNSegmentCPU(val, &mat->matElems[rowIdx * colNum], colNum);
#endif
}

/* cz277 - frame rejection */
void MMIFrameRejection(ANNSet *annSet, int batLen, FBLatInfo *fbInfoPtr, CriteriaInfo *crtList) {
    double denOcc;
    int i, s;
    /*IntVec labVec;*/
    CriteriaInfo *criteria;

    if (optFrameReject == FALSE) {
        return;
    }
    if (objfunKind != MMIOF) {
        HError(-4324, "MMIFrameRejection: Frame rejection is only valid for MMI training");
        return;
    }
    /*if ((labelKind & LABLK) == 0) {
        HError(9999, "MMIFrameRejection: Hard labels are needed for frame rejection");
        return;
    }*/
    if (optHasFSmooth) {
        HError(-4324, "MMIFrameRejection: Frame rejection conflicts with F-smoothing");
        return;
    }

    for (s = 1; s <= fbInfoPtr->S; ++s) {
        /*labVec = fbInfoPtr->refVec[s];*/
        criteria = &crtList[s];
        for (i = 0; i < batLen; ++i) {
            denOcc = fbInfoPtr->occVec[s][i + 1];
            if (denOcc < minOccFrameReject) {
                ForceNMatrixValueByRow(hset.annSet->outLayers[s]->yFeaMats[1], i, 0.0);	/* cz277 - many */
                ++criteria->MMIFRAcc;
            }
        }
    }
}

Boolean LatticeBasedSequenceTraining(UttElem *uttElem, int nLoaded, Boolean hasFSmooth, Boolean hasFrameReject, CriteriaInfo *criteria) {
    int i, S;
    LELink layerElem;
    Boolean sentFail = FALSE;
    Lattice *MPECorrLat = NULL;

    S = hset.swidth[0];
    /* reset the occ matrices */
    for (i = 1; i <= S; ++i) {
        layerElem = hset.annSet->outLayers[i];
        if (hasFSmooth == TRUE) {
	    ComputeBackwardPropOutActivation(objfunFSmooth, nLoaded, layerElem, 1);
            ScaleNMatrix(1.0 - FSmoothH, nLoaded, layerElem->nodeNum, layerElem->yFeaMats[1]);
        }
        else
            SetNMatrix(0.0, layerElem->yFeaMats[1], nLoaded);
#ifdef CUDA
        SyncNMatrixDev2Host(layerElem->yFeaMats[1]);
#endif
    }
    if (procNumLats) {
        LoadNumLatsFromUttElem(uttElem, &fbInfo);
        if (FBLatFirstPass(&fbInfo, UNDEFF, uttElem->uttName, NULL, NULL)) 
            FBLatSecondPass(&fbInfo, corrIdx, 999);
        else
            sentFail = TRUE;
    }
    if (procDenLats && (!sentFail)) {
        switch (objfunKind) {
        case SMBROF:
        case MPEOF:
        case MWEOF:
            MPECorrLat = uttElem->numLats[0];
            break;
        default:
            MPECorrLat = NULL;
        }
        LoadDenLatsFromUttElem(uttElem, &fbInfo);
        if (FBLatFirstPass(&fbInfo, UNDEFF, uttElem->uttName, NULL, MPECorrLat))
            FBLatSecondPass(&fbInfo, recogIdx1, recogIdx2);
        else
            sentFail = TRUE;
    }
#ifdef CUDA
    for (i = 1; i <= S; ++i) {
        layerElem = hset.annSet->outLayers[i];
        SyncNMatrixHost2Dev(layerElem->yFeaMats[1]);
    }
#endif
    if (hasFrameReject == TRUE)
        MMIFrameRejection(hset.annSet, nLoaded, &fbInfo, criteria);
    
    return !sentFail;
}

void UtterLevelTrainProcess(int curEpochNum) {
    int i, S, nLoaded;
    int updtCnt, sampCnt, batchCnt, tSampCnt, tUttCnt, uttCnt, updtUttCnt;
    Boolean finish = FALSE;
    Boolean accGrad, sentSuccess;
    CriteriaInfo criteria[SMAX];
    float retNegLR;
    Boolean skipOneUtt = FALSE;
    UttElem *uttElem;
    /* cz277 - mmt */
    float scale, momentum, weightDecay, posGradClip, negGradClip;

    S = hset.swidth[0];
    updtCnt = 0;
    tSampCnt = 0;
    /* process the first batch */
    batchCnt = 0;
    tUttCnt = 0;
    SetGlobalBatchIndex(0);
    /* cz277 - batch sync */
    SetNBundleBatchIndex(hset.annSet, 0);
    SetFeaMixBatchIndex(hset.annSet, 0);
    /* initialise the cache */
    for (i = 1; i <= S; ++i) 
        InitCache(cacheTr[i]);

    /* initialise the criteriaInfo */
    memset(&criteria, 0, sizeof(CriteriaInfo) * SMAX);
    weightDecay = GetConfParmPerEpoch(curEpochNum, List_WeightDecay);
    momentum = GetConfParmPerEpoch(curEpochNum, List_MMTs);
    /* process until all data are finished */
    while (!finish) {
        /* set uttCnt */
        accGrad = FALSE;
        sampCnt = 0;
        updtUttCnt = 0; 
        while ((!finish) && (updtUttCnt < numPerUpdt)) {
            uttCnt = 1;
            uttElem = GetCurUttElem(cacheTr[1]);	/* cz277 - xform */
            /* install the current replaceable parts */
            InstallOneUttNMatRPLs(uttElem);
            InstallOneUttNVecRPLs(uttElem); 
            /* get current utterance info */
            if (optTrainMode == SEQTM) {
                if (uttElem->uttLen > GetNBatchSamples()) {
                    HError(-4327, "UtterLevelTrainProcess: %s has %d samples > batch size %d", uttElem->uttName, uttElem->uttLen, GetNBatchSamples());
                    skipOneUtt = TRUE;
                }
                else 
                    skipOneUtt = FALSE;
                /* init fbInfo */
                fbInfo.T = uttElem->uttLen;
                fbInfo.uFlags = cacheTr[1]->labelInfo->uFlags;
                if (optFrameReject)
                    fbInfo.rejFrame = optFrameReject;
                LoadXFormsFromUttElem(uttElem, &fbInfo);
            }
            while ((!finish) && uttCnt > 0) {
                /* load data */
                for (i = 1; i <= S; ++i) {
                    finish |= FillAllInpBatch(cacheTr[i], &nLoaded, &uttCnt);
                    LoadCacheData(cacheTr[i]);
                }
                /* whether skip this utterance or not */
                if (skipOneUtt)
                    continue;
                /* forward propagation */
                ForwardProp(hset.annSet, nLoaded, cacheTr[1]->CMDVecPL);
                sentSuccess = TRUE;
                /* synchronise the data */
                ComputeLogLikelihoods(nLoaded);
                /* for sequence training */
                if (optTrainMode == SEQTM) 
                    sentSuccess = LatticeBasedSequenceTraining(uttElem, nLoaded, optHasFSmooth, optFrameReject, criteria);
                /* accumulate for the criteria */
                AccCriteria(criteria, cacheTr, nLoaded, optTrainMode == FRAMETM, (optTrainMode == SEQTM) && sentSuccess);
                /* backward propagation */
                if (sentSuccess) {
		  BackwardProp(objfunKind, hset.annSet, nLoaded, accGrad);
                } else {
		  SetFeaMixBatchIndex(hset.annSet, GetGlobalBatchIndex());/*cz277-batch sync*/
		}
                /* update statistics */
                accGrad = TRUE;
                batchCnt += 1;
                sampCnt += nLoaded;
            }
            ++updtUttCnt;
            /* update cache status */
            for (i = 1; i <= S; ++i) 
                UnloadCacheData(cacheTr[i]);
        }
        tUttCnt += updtUttCnt;
        /* do the update */
        if (sampCnt != 0) {
	    /* cz277 - batch sync */
	    SetNBundleBatchIndex(hset.annSet, GetGlobalBatchIndex() - 1);
            /* update the learning rates*/
            retNegLR = UpdateLRSchdPerU(curEpochNum, sampCnt);
            /* update the parameters */
            NormBackwardPropGradients(hset.annSet, 1.0);
            scale = ((float) sampCnt) / (GetNBatchSamples() * numPerUpdt);
            SGDUpdateANNSet(hset.annSet, scale, retNegLR, momentum, weightDecay, gradientClip, updateClip, gradientL2Scale, updateL2Scale);
            /* update updtCnt */
            ++updtCnt;
        }
        /* update statistics */
        tSampCnt += sampCnt;
    }
    /* reset the replaceable parts */
    ResetNMatRPL();
    ResetNVecRPL();
    /* update the target penalties */
    if (((uFlags & UPTARGETPEN) != 0) && (hset.hsKind == HYBRIDHS)) 
        for (i = 1; i <= S; ++i) 
            UpdateTargetLogPrior(cacheTr[i], logObsvPrior);
    /* update the transition probabilities */
    if ((uFlags & UPTRANS) != 0) 
        UpdateAllTrans();
    /* show criteria */
    for (i = 1; i <= S; ++i) {
        if (S > 1) 
            printf("\t\tStream %d: ", i);
        PrintCriteria(&criteria[i], "Train");
    }
    /* reset all cache */
    for (i = 1; i <= S; ++i) 
        ResetCache(cacheTr[i]);
    /* output the timming */
    printf("\t\tTotal batch count = %d, update count = %d\n", batchCnt, updtCnt);
}

void UtterLevelHVProcess(void) {
    int i, S, nLoaded;
    int sampCnt, batchCnt, tSampCnt, tUttCnt, uttCnt;
    Boolean finish = FALSE, sentSuccess;
    CriteriaInfo criteria[SMAX];
    Boolean skipOneUtt = FALSE;
    UttElem *uttElem;

    S = hset.swidth[0];
    batchCnt = 0;
    tSampCnt = 0;
    tUttCnt = 0;
    SetGlobalBatchIndex(0);
    /* cz277 - batch sync */
    SetNBundleBatchIndex(hset.annSet, 0);
    SetFeaMixBatchIndex(hset.annSet, 0);
    /* initialise the cache */
    for (i = 1; i <= S; ++i) 
        InitCache(cacheHV[i]);

    /* initialise the criteriaInfo */
    memset(&criteria, 0, sizeof(CriteriaInfo) * SMAX);
    /* process until all data are finished */
    while (!finish) {
        sampCnt = 0;
        uttCnt = 1;
        uttElem = GetCurUttElem(cacheHV[1]);
        /* install the current replaceable parts */
        InstallOneUttNMatRPLs(uttElem);
        InstallOneUttNVecRPLs(uttElem);
        /* get current utterance info */
	if (optTrainMode == SEQTM) {
	    if (uttElem->uttLen > GetNBatchSamples()) {
                HError(-4327, "UtterLevelHVProcess: %s has %d samples > batch size %d", uttElem->uttLen, uttElem->uttName, GetNBatchSamples());
	        skipOneUtt = TRUE;
	    }
	    else 
	        skipOneUtt = FALSE;
	    /* init fbInfo */
	    fbInfo.T = uttElem->uttLen;
            fbInfo.uFlags = cacheHV[1]->labelInfo->uFlags;
            fbInfo.rejFrame = FALSE;
	    LoadXFormsFromUttElem(uttElem, &fbInfo);
	}
        while ((!finish) && uttCnt > 0) {
            /* load data */
            for (i = 1; i <= S; ++i) {
                finish |= FillAllInpBatch(cacheHV[i], &nLoaded, &uttCnt);
                LoadCacheData(cacheHV[i]);
            }
            /* whether skip this utterance or not */
            if (skipOneUtt) 
                continue;
            ForwardProp(hset.annSet, nLoaded, cacheHV[1]->CMDVecPL); /* forward propagation */ 
            sentSuccess = TRUE;
            ComputeLogLikelihoods(nLoaded); /* compute sequence level show criteria */
            /* for sequence training */
            if (optTrainMode == SEQTM) 
                sentSuccess = LatticeBasedSequenceTraining(uttElem, nLoaded, FALSE, FALSE, criteria);
            /* accumulate the statistics for criteria */
            AccCriteria(criteria, cacheHV, nLoaded, optTrainMode == FRAMETM, (optTrainMode == SEQTM) && sentSuccess);
            batchCnt += 1;
            sampCnt += nLoaded;
        }
        for (i = 1; i <= S; ++i) 
            UnloadCacheData(cacheHV[i]);
        /* update the statistics */
        tUttCnt += 1;
        tSampCnt += sampCnt;
    }
    /* reset the replaceable parts */
    ResetNMatRPL();
    ResetNVecRPL();
    /* show criteria */
    for (i = 1; i <= S; ++i) {
        if (S > 1) 
            printf("\t\tStream %d: ", i);
        PrintCriteria(&criteria[i], "Validation");
    }
    /* reset all cache */
    for (i = 1; i <= S; ++i) 
        ResetCache(cacheHV[i]);
    /* output statistics */
    printf("\t\tBatch count = %d\n", batchCnt);
}

void SafetyCheck() {
    if (optTrainMode == SEQTM) {
        if (hset.hsKind != HYBRIDHS) 
            HError(4324, "SafetyCheck: Unsupported HMM set kind for sequence training");
    } else if (optTrainMode == FRAMETM) {
        if (hset.hsKind != HYBRIDHS && hset.hsKind != ANNHS) 
            HError(4324, "SafetyCheck: Unsupported HMM set kind for frame level training");
    }
}

/* cz277 - pact */
void SetStaticUpdtCode(int statusCode) {
    int i;
    AILink curAI;
    ADLink annDef;
    LELink layerElem;

    printf("SetStaticUpdtCode: Executing oper code %d\n", statusCode);
    curAI = hset.annSet->defsHead;
    while (curAI != NULL) {
        annDef = curAI->annDef;
        for (i = 0; i < annDef->layerNum; ++i) {
            layerElem = annDef->layerList[i];
            layerElem->status = statusCode;
        }
        curAI = curAI->next;
    }
}

/* cz277 - pact */
void UtterLevelStaticUpdate(int curUpdtCode) {
    int i, S, uttCnt, nLoaded;
    Boolean finish = FALSE;
    UttElem *uttElem;

    S = hset.swidth[0];
    /* initialise the cache */
    for (i = 1; i <= S; ++i) 
        InitCache(cacheTr[i]);
    /* set status code */
    SetStaticUpdtCode(curUpdtCode);
    /* process until all data are finished */
    while (!finish) {
        uttCnt = 1;
        uttElem = GetCurUttElem(cacheTr[1]);
        /* install the current replaceable parts */
        InstallOneUttNMatRPLs(uttElem);
        InstallOneUttNVecRPLs(uttElem);
        /* process current utterance */
        while ((!finish) && uttCnt > 0) { 
            /* load data */
            for (i = 1; i <= S; ++i) {
                finish |= FillAllInpBatch(cacheTr[i], &nLoaded, &uttCnt);
                LoadCacheData(cacheTr[i]);
            }
            /* forward propagation */
            ForwardProp(hset.annSet, nLoaded, cacheTr[1]->CMDVecPL);
            /*switch (curUpdtCode) {
            case 0:
            case 3:
            case 4:
                ForwardPropBlank(hset.annSet, nLoaded);
                break;
            case 1:
            case 2:
                ForwardProp(hset.annSet, nLoaded, cacheTr[1]->CMDVecPL);
                break;
            default:
                break;
            }*/
        }
        for (i = 1; i <= S; ++i) 
            UnloadCacheData(cacheTr[i]);
    }
    /* reset replaceable parts */
    ResetNMatRPL();
    ResetNVecRPL();
    /* update the target penalties */
    if ((curUpdtCode == 0) && ((uFlags & UPTARGETPEN) != 0) && (hset.hsKind == HYBRIDHS)) 
        for (i = 1; i <= S; ++i) 
            UpdateTargetLogPrior(cacheTr[i], logObsvPrior);
    /* reset all cache */
    for (i = 1; i <= S; ++i) 
        ResetCache(cacheTr[i]);
}

void SaveModelSet(char *baseDir) {
    char buf[MAXSTRLEN];

    /* saves current model */
    SaveHMMSet(&hset, baseDir, newExt, NULL, saveBinary);
    /* saves update */
    if (optHasMMT) {
        MakeFN(fnUpdate, baseDir, NULL, buf);
        SaveANNUpdate(&hset, buf, saveBinary);
    }
    /* saves nlr */
    if (optHasNLR) {
        MakeFN(fncurNegLR, baseDir, NULL, buf);
        SaveANNNegLR(&hset, buf, saveBinary);
    }
    /* saves ssg */
    if (optHasSSG) {
        MakeFN(fnSquareGrad, baseDir, NULL, buf);
        SaveANNStore(&hset, buf, saveBinary);
    }
    /* saves schd */
    if (optSavSchd) {
        MakeFN(fnLRSchd, baseDir, NULL, buf);
        SaveLRSchd(buf);
    }
}

int main(int argc, char *argv[]) {
    int i, curEpochNum, curEpochIdx;
    char *str;
    char buf[MAXSTRLEN], curEpcDir[MAXSTRLEN], absEpcDir[MAXSTRLEN];
    clock_t stClock, edClock;
    MSILink MSIPtr;

    if (InitShell(argc, argv, hntrainsgd_version, hntrainsgd_vc_id) < SUCCESS) 
        HError(4300, "HNTrainSGD: InitShell failed");
    InitMem();
    InitMath();
    InitSigP();
    InitWave();
    InitLabel();
    InitAudio();
#ifdef CUDA
    InitCUDA();
#endif
    InitANNet();
    InitModel();
    if (InitParm() < SUCCESS) 
        HError(4300, "HNTrainSGD: InitParm failed");
    InitUtil();
    InitFBLat();
    InitExactMPE();
    InitArc();
    InitDict();
    InitLat();
    InitNet();
    InitAdapt();
    InitXFInfo(&xfInfo);
    InitNCache();
    if (!InfoPrinted() && NumArgs() == 0) 
        ReportUsage();
    if (NumArgs() == 0) 
        Exit(0);

    CreateHeap(&modelHeap, "model heap",  MSTAK, 1, 0.0, 300000000, ULONG_MAX);
    CreateHMMSet(&hset, &modelHeap, TRUE);
    SetConfParms();
    /* load each command */
    while (NextArg() == SWITCHARG) {
        str = GetSwtArg();
        /* set each option */
        switch (str[0]) {
            case 'a':
                xfInfo.useInXForm = TRUE;
                break;
            case 'c':
                initialHV = TRUE;
                break;
            case 'd':
                if (NextArg() != STRINGARG) 
                    HError(4319, "HNTrainSGD: HMM definition directory expected");
                hmmDir = GetStrArg();
                break;
            case 'e':
                if (NextArg() != STRINGARG) 
                    HError(4319, "HNTrainSGD: Directory to store temporary HMM definitions expected");
                if (strcmp(str, "e") == 0) 
                    epcDirPref = GetStrArg();
                else if (strcmp(str, "eb") == 0) 
                    epcBaseDir = GetStrArg();
                break;
            /*case 'f':
                if (NextArg() != STRINGARG) 
                    HError(4319, "HNTrainSGD: Auxiliary file path expected");
                if (strcmp(str, "fl") == 0) 
                    LoadLRSchd(GetStrArg());
                else 
                    HError(4319, "HNTrainSGD: Unknown label kind");    
                break;*/
            case 'h':
                if (NextArg() != STRINGARG) 
                    HError(4319, "HNTrainSGD: Speaker name pattern expected");
                xfInfo.outSpkrPat = GetStrArg();
                if (NextArg() == STRINGARG) {
                    xfInfo.inSpkrPat = GetStrArg();
                    if (NextArg() == STRINGARG) 
                        xfInfo.paSpkrPat = GetStrArg();
                }
                if (NextArg() != SWITCHARG) 
                    HError(4319, "HNTrainSGD: Cannot have -h as the last option");
                break;
            case 'j':
                if (strcmp(str, "ju") == 0)
                    fnUpdate = GetStrArg();
                else if (strcmp(str, "jl") == 0)
                    fncurNegLR = GetStrArg();
                else if (strcmp(str, "jg") == 0)
                    fnSquareGrad = GetStrArg();
                else if (strcmp(str, "js") == 0)
                    fnLRSchd = GetStrArg();
                else
                    HError(4319, "HNTrainSGD: Unknown option %s", str);
                break;
            case 'k':
                if (NextArg() != STRINGARG)
                    HError(4319, "HNTrainSGD: Auxiliary file path expected");
                if (strcmp(str, "kl") == 0)
                    LoadLRSchd(GetStrArg());
                else
                    HError(4319, "HNTrainSGD: Unknown training status kind");
                break;
            case 'l':
                if (NextArg() != STRINGARG) 
                    HError(4319, "HNTrainSGD: Label kind expected");
                str = GetStrArg();
                if (strcasecmp(str, "LAB") == 0 || strcasecmp(str, "LABEL") == 0) {
                    labelKind = LABLK;   
                    optHasLabMat = TRUE;
                }
                else if (strcasecmp(str, "LAT") == 0 || strcasecmp(str, "LATTICE") == 0) {
                    labelKind = LATLK;
                    optHasLabMat = FALSE;
                }
                else if (strcasecmp(str, "FEA") == 0 || strcasecmp(str, "FEATURE") == 0) {
                    labelKind = FEALK;
                    optHasLabMat = TRUE;
                }
                else if (strcasecmp(str, "LABLAT") == 0 || strcasecmp(str, "LATLAB") == 0) {
                    labelKind = LABLK | LATLK;
                    optHasLabMat = TRUE;
                }
                else if (strcasecmp(str, "NULL") == 0) {
                    labelKind = NULLLK;
                    optHasLabMat = FALSE;
                }
                else 
                    HError(4319, "HNTrainSGD: Unknown label kind");
                break;
            case 'm':
                if (NextArg() != STRINGARG) 
                    HError(4319, "HNTrainSGD: Target mapping file path expected");
                optMapTarget = TRUE;
                mappingFn = GetStrArg();
                break;
            /*case 'n':
                if (strcmp(str, "nu") == 0) 
                    fnUpdate = GetStrArg();
                else if (strcmp(str, "nl") == 0) 
                    fncurNegLR = GetStrArg();
                else if (strcmp(str, "ng") == 0) 
                    fnSquareGrad = GetStrArg();
                else if (strcmp(str, "ns") == 0) 
                    fnLRSchd = GetStrArg();
                else 
                    HError(4319, "HNTrainSGD: Unknown option %s", str);
                break;*/
            case 'n':
                if (NextArg() != STRINGARG)
                    HError(4319, "HNTrainSGD: Feature type label for validation set file name expected");
                InitScriptFile(GetStrArg(), &scriptFLabHV, &scriptCntFLabHV);
                break;
            case 'o':
                if (NextArg() != STRINGARG) 
                    HError(4319, "HNTrainSGD: HMM file extension expected");
                newExt = GetStrArg();
                break;
            case 'q':
                if (strcmp(str, "q") == 0) 
                    numLatDir[nNumLats++] = GetStrArg();
                else if (strcmp(str, "qp") == 0) {
                    strcpy(numLatSubDirPat, GetStrArg());
                    if (strchr(numLatSubDirPat, '%') == NULL) 
                        HError(4319, "HNTrainSGD: Numerator lattice path mask invalid");
                }
                else 
                    HError(4319, "HNTrainSGD: Unknown option %s", str);
                break;
            case 'r':
                if (strcmp(str, "r") == 0) 
                    denLatDir[nDenLats++] = GetStrArg();
                else if (strcmp(str, "rp") == 0) {
                    strcpy(denLatSubDirPat, GetStrArg());
                    if (strchr(denLatSubDirPat, '%') == NULL) 
                        HError(4319, "HNTrainSGD: Denominator lattice path mask invalid");
                }
                else 
                    HError(4319, "HNTrainSGD: Unknown option %s", str);
                break;
            case 's':
                if (NextArg() != STRINGARG)
                    HError(4319, "HNTrainSGD: Feature type label for training set file name expected");
                InitScriptFile(GetStrArg(), &scriptFLabTr, &scriptCntFLabTr);
                break;
            case 'u':
                optDoStaticUpdt = TRUE;
                break;
            case 'x':
                if (NextArg() != STRINGARG) 
                    HError(4319, "HNTrainSGD: HMM file extension expected");
                hmmExt = GetStrArg();
                break;
            case 'B':
                saveBinary = TRUE;
                break;
            case 'F':
                if (NextArg() != STRINGARG) 
                    HError(4319, "HNTrainSGD: Data file format expected");
                if ((dff = Str2Format(GetStrArg())) == ALIEN) 
                    HError(4319, "HNTrainSGD: Warnings ALIEN data file format set");
                break;
            case 'G':
                if (NextArg() != STRINGARG) 
                    HError(4319, "HNTrainSGD: Label file format expected");
                if ((lff = Str2Format(GetStrArg())) == ALIEN) 
                    HError(4319, "HNTrainSGD: Warnings ALIEN label file format set");
                break;
            case 'H':
                if (NextArg() != STRINGARG) 
                    HError(4319, "HNTrainSGD: HMM macro file name expected");
                ++macroCnt;
                AddMMF(&hset, GetStrArg());
                break;
            case 'I':
                if (NextArg() != STRINGARG) 
                    HError(4319, "HNTrainSGD: MLF file name expected");
                LoadMasterFile(GetStrArg());
                break;
            case 'J':
                if (NextArg() != STRINGARG) 
                    HError(4319, "HNTrainSGD: Input transform directory expected");
                AddInXFormDir(&hset, GetStrArg());
                if (NextArg() == STRINGARG) {
                    if (xfInfo.inXFormExt == NULL) 
                        xfInfo.inXFormExt = GetStrArg();
                    else 
                        HError(4319, "HNTrainSGD: Only one input transform extension may be specified");
                }
                if (NextArg() != SWITCHARG) 
                    HError(4319, "HNTrainSGD: Cannot have -J as the last option");
                break;
            case 'L':
                if (NextArg() != STRINGARG) 
                    HError(4319, "HNTrainSGD: Label file directory expected");
                labDir = GetStrArg();
                break;
            case 'M':
                if (NextArg() != STRINGARG) 
                    HError(4319, "HNTrainSGD: Output macro file directory expected");
                newDir = GetStrArg();
                break;
            case 'N':
                if (NextArg() != STRINGARG) 
                    HError(4319, "HNTrainSGD: Validation set file name expected");
                InitScriptFile(GetStrArg(), &scriptHV, &scriptCntHV);                
                break;
            case 'Q':
                if (NextArg() != STRINGARG) 
                    HError(4319, "HNTrainSGD: Lattice extension expected");
                latExt = GetStrArg();
                break;
            case 'T':
                trace = GetChkedInt(0, 0100000, str);
                break;
            case 'X':
                if (NextArg() != STRINGARG) 
                    HError(4319, "HNTrainSGD: Label file extension expected");
                labExt = GetStrArg();
                break;
            default:
                HError(4319, "HNTrainSGD: Unknown switch %s", str);
        }
    }
    /* load hmmListFn (optional) */
    if (NextArg() == STRINGARG) 
        hmmListFn = GetStrArg();
    else if (NextArg() != NOARG) 
        HError(4319, "HNTrainSGD: Only an optional HMM list file is possible at the end of the command line");

    /* command check */
    if (NumArgs() == 0 && hset.numFiles == 0) 
        HError(4319, "HNTrainSGD: At least one input should be given");

#ifdef CUDA
    StartCUDA();
#endif
    /* cz277 - 151020 */
#ifdef MKL
    StartMKL();
#endif
    /* initialise */
    Initialise();
#ifdef CUDA
    ShowGPUMemUsage();
#endif
    /* do safety check */
    SafetyCheck();

    stClock = 0;
    edClock = 0;
    curEpochNum = 0;
    printf("\n");
    /* cz277 - pact */
    if (optDoStaticUpdt == FALSE) {
        /* process the held-out validation set first */
        if (initialHV == TRUE && scriptHV != NULL) {
            printf("Init Training ************************\n");
            printf("\tProcessing validation set...\n");
            stClock = clock();
            UtterLevelHVProcess();
            edClock = clock();
            if (trace & T_TIM)
                printf("\t\tValidation time cost = %.2fs\n", (edClock - stClock) / (double) CLOCKS_PER_SEC);
            tailMSI->crtVal = newMSI_CrtVal;
            printf("\n\n");
        }
        
        /* process training */
        while (TermLRSchdOrNot(curEpochNum)) {
            /* compute current epoch index */
            curEpochIdx = epochOff + curEpochNum;
            SetEpochIndex(curEpochIdx);
            printf("Epoch %d ******************************\n", curEpochIdx);
            /* update the learning rate by epoch (4 list and newbob) */
            UpdateLRSchdPerE(curEpochNum);
            /* process the train set */
            printf("\tProcessing training set...\n");
            stClock = clock();
            switch (updtKind) {
            case BATLEVEL: BatchLevelTrainProcess(curEpochNum); break;
            case UTTLEVEL: UtterLevelTrainProcess(curEpochNum); break;
            default: 
                HError(4399, "Unknown update kind");
            }
            /* show current learning rate */
            switch (schdKind) {
            case LISTSK:
            case NEWBOBSK:
                if (trace & T_TOP)
                    printf("\t\tLearning rate = %e\n", (-1.0) * curNegLR);
                break;
            case ADAGRADSK:
            case EXPSK:
                break;
            }
            /* compute the time cost */
            edClock = clock();
            if (trace & T_TIM)
                printf("\t\tTraining time cost = %.2fs\n", (edClock - stClock) / (double) CLOCKS_PER_SEC);
            /* process held-out set */
            if (scriptHV != NULL) {
                printf("\tProcessing validation set...\n");
                stClock = clock();
                UtterLevelHVProcess();
                edClock = clock();
                if (trace & T_TIM)
                    printf("\t\tValidation time cost = %.2fs\n", (edClock - stClock) / (double) CLOCKS_PER_SEC);
            }
            /* save the intermediate models */
            if (optSavEpcMod) {
                /* generate current epoch directory first */
                sprintf(buf, "%d", curEpochIdx);
                strcpy(curEpcDir, epcDirPref);
                strcat(curEpcDir, buf);
                /* setup the absolute epc directory */
                CatDirs(epcBaseDir, curEpcDir, absEpcDir);
                SetupDir(absEpcDir);
                /* saves current models */
                SaveModelSet(absEpcDir);
                /* attach current MSI */
                MSIPtr = (MSILink) New(&gcheap, sizeof(ModelSetInfo));
                SetModelSetInfo(absEpcDir, hmmExt, NULL, MSIPtr, curEpochIdx);
                AppendModelSetInfo(MSIPtr);
                /* saves crtVal */
                tailMSI->crtVal = newMSI_CrtVal;
            }
            /* update curEpochNum */
            ++curEpochNum;
            fflush (stdout);
            printf("\n\n");
        }
        printf("Finish Training ***********************\n");
        /* show statistics */
        if (trace & T_TOP) 
            printf("\t%d updates processed in total\n\n", updateIndex);
            printf("\n\n");
    }
    else {
        /* cz277 - pact */
        if (scriptHV != NULL)
            HError(-4324, "HNTrainSGD: Static update will not use the validation scp");
        if (!(GetDefaultVisitKind() == NONEVK || GetDefaultVisitKind() == UTTVK))
            HError(4324, "HNTrainSGD: Only utterance level cache visiting order is allowed for static update");
        while (curEpochNum <= staticUpdtCode) 
            UtterLevelStaticUpdate(curEpochNum++);
    }

    /* write the output model */
    SaveModelSet(newDir);
    /* free ANNSet */
    FreeANNSet(&hset);
    for (i = 1; i <= hset.swidth[0]; ++i) 
        FreeCache(cacheTr[i]);
    if (scriptHV != NULL) 
        for (i = 1; i <= hset.swidth[0]; ++i) 
            FreeCache(cacheHV[i]);

#ifdef CUDA
    StopCUDA();
#endif

    Exit(EXIT_STATUS);
    return 0;
}

/* ----------------------------------------------------------- */
/*                      END:  HNTrainSGD.c                     */
/* ----------------------------------------------------------- */

