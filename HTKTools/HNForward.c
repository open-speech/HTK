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
/*  File: HNForward.c  ANN forwarding and evalutation program  */
/* ----------------------------------------------------------- */

char *hnforward_version = "!HVER!HNForward:   3.5.0 [cz277 12/10/15]";
char *hnforward_vc_id = "$Id: HNForward.c,v 1.0 2015/10/12 12:07:24 cz277 Exp $";

/*
  This program is designed for evaluating or forwarding ANNs
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
#include "HNCache.h"

#include <time.h>
#include <math.h>
#include <strings.h>
/* -------------------------- Trace Flags & Vars ------------------------ */

/* Trace Flags */
#define T_TOP   0001    /* Top level tracing */
#define T_TIM   0002    /* Output timings */
static int trace = 0;
#define EXIT_STATUS 0   /* Exit status */

/* -------------------------- Global Variables etc ---------------------- */

#define MAX(a, b) ((a)>(b)?(a):(b))
#define MIN(a, b) ((a)<(b)?(a):(b))
#define ABS(a) ((a)>0?(a):-(a))
#define FINITE(x) (!isnan(x) && x<1.0e+30 && x>-1.0e+30)

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

typedef struct _SpeakerInfo{           		/* list of spkr records */
   int uttCnt;					/* utterance count associated with this speaker */
   CriteriaInfo criteria[SMAX];			/* criteriion values associated with this speaker */
   struct _SpeakerInfo *next;			/* to the next speaker */
   char *name;					/* current speaker name */
} SpeakerInfo;

static char *hmmListFn = NULL;                  /* model list filename (optional) */  
static char *hmmDir = NULL;                     /* directory to look for HMM def files */
static char *hmmExt = NULL;                     /* HMM def file extension */
static char *mappingFn = NULL;                  /* the name of target mapping file */
static XFInfo xfInfo;                           /* transforms/adaptations */
static HMMSet hset;                             /* the HMM set */
static UPDSet uFlags = 0;			/* */

static IntVec recVec = NULL;                    /* the vector contains hypothesis labels */
static IntVec recVecLLH = NULL;
static IntVec recVecMapSum = NULL;		/* the vector contains the mapped hypothesis labels */
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

static ObjFunKind showObjFunKind = UNKOF;
static float minFrameConfMat = 0.0;
static FBLatInfo fbInfo;                        /* structure for forward-backward */

static CriteriaInfo criteriaAll[SMAX];		/* the value of the criteria functions for all data */
static CriteriaInfo criteriaUtt[SMAX];		/* the value of the criteria functions for current utterance */
static SpeakerInfo *speakerInfoHead = NULL;	/* the header of the speaker info list */
static int speakerCnt = 0;			/* the total number of speakers */
static int failedSpeakers = 0;			/* num time speaker pattern failed to match */

static FILE *scriptIn = NULL;                   /* script file for input */
static int scriptCntIn = 0;                     /* number of words in scriptIn */
static FILE *scriptOut = NULL;                  /* script file for output */
static int scriptCntOut = 0;                    /* number of words in scriptOut */

static LabelKind labelKind = NULLLK;            /* the kind of the labels */
static LabelInfo *labelInfo = NULL;             /* the structure for the labels */
static DataCache *cacheIn[SMAX];                  /* the cache structures for the train set */
static Observation obsIn;                       /* array of input observations */
static Observation obsOut;			/* array of output observation */
static FileFormat tgtFF = HTK;			/* the file format of the output data */
static ParmKind tgtPK = USER;	/*ANN;*/			/* ANN parmkind */
/*static ParmBuf parmBuf;*/				/* parm buffer */
static ParmBuf parmBuf;
/*static ParmBuf dstPBuf;*/
static BufferInfo bufInfo;			/* buffer info */

/* ------------------------- Global Options ----------------------------- */

static Boolean optHasLabMat = FALSE;            /* whether do supervised learning or not (associated with NULLLK) */
static Boolean optShowUttStats = FALSE;		/* whether show criteria value for each utterance */
static Boolean optGenANNFeas = FALSE;		/* whether generate the ANN features or not */
static Boolean optMapTarget = FALSE;		/* do target mapping or not */
static Boolean optIncNumInDen = TRUE;
static Boolean optShowSeqObjVal = FALSE;
static Boolean optShowFrameConfMat = FALSE;

/* ------------------------------ Heaps --------------------------------- */

static MemHeap modelHeap;                       /* the memory heap for models */
static MemHeap cacheHeap;                       /* the memory heap for data caches */
static MemHeap transHeap;                       /* the memory heap for transcriptions */
static MemHeap latHeap;                         /* the memory heap for lattices */
static MemHeap feaHeap;

/* -------------------- Configuration Parameters ------------------------ */

static ConfParam *cParm[MAXGLOBS];              /* configuration parameters */
static int nParm = 0;                           /* total num params */

/* -------------------------- Prototypes -------------------------------- */



/* ----------------------- Process Command Line ------------------------- */

void SetConfParms(void)
{
    int intVal;
    double doubleVal;
    Boolean boolVal;
    char buf[MAXSTRLEN], buf2[MAXSTRLEN];
    char *charPtr, *charPtr2;

    nParm = GetConfig("HNFORWARD", TRUE, cParm, MAXGLOBS);
    if (nParm > 0) {
        if (GetConfInt(cParm, nParm, "TRACE", &intVal)) {
            trace = intVal;
        }
        /* set evaluation criteria */
        if (GetConfStr(cParm, nParm, "EVALCRITERION", buf)) {
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
                if (strcmp(buf2, "ML") == 0) {	/* log-likelihood */
                    showObjFunKind = showObjFunKind | MLOF;
                }
                else if ((strcmp(buf2, "MMI") == 0) && (labelKind == LABLK)) {
                    showObjFunKind = showObjFunKind | MMIOF;
                    optShowSeqObjVal = TRUE;
                }
                else if (strcmp(buf2, "MMSE") == 0) {
                    showObjFunKind = showObjFunKind | MMSEOF;
                }
                else if ((strcmp(buf2, "MPE") == 0) && (labelKind == LABLK)) {
                    showObjFunKind = showObjFunKind | MPEOF;
                    optShowSeqObjVal = TRUE;
                }
                /*else if ((strcmp(buf2, "MWE") == 0) && (labelKind == LABLK)) {
                    showObjFunKind = showObjFunKind | MWEOF;
                }
                else if ((strcmp(buf2, "SMBR") == 0) && (labelKind == LABLK)) {
                    showObjFunKind = showObjFunKind | SMBROF;
                }*/
                else if (strcmp(buf2, "XENT") == 0) {
                    showObjFunKind = showObjFunKind | XENTOF;
                }
            }
        }
        /* set target file format */
        if (GetConfStr(cParm, nParm, "TARGETFORMAT", buf)) {
            tgtFF = Str2Format(buf);
        }
        /* */
        if (GetConfBool(cParm, nParm, "SHOWFRAMECM", &boolVal)) {
            optShowFrameConfMat = TRUE;
        }
        /* */
        if (GetConfFlt(cParm, nParm, "FRAMECMTHRES", &doubleVal)) {
            minFrameConfMat = (float) doubleVal;
        }
        /* label file mask */
        if (GetConfStr(cParm, nParm, "LABFILEMASK", buf)) {
            labFileMask = (char *) malloc(strlen(buf) + 1);
            strcpy(labFileMask, buf);
        }
        /* lattice file mask */
        if (GetConfStr(cParm, nParm, "LATFILEMASK", buf)) {
            latFileMask = (char *) malloc(strlen(buf) + 1);
            strcpy(latFileMask, buf);
        }
        if (GetConfStr(cParm, nParm, "LATMASKNUM", buf)) {
            latMask_Num = (char *) malloc(strlen(buf) + 1);
            strcpy(latMask_Num, buf);
        }
        if (GetConfStr(cParm, nParm, "LATMASKDEN", buf)) {
            latMask_Den = (char *) malloc(strlen(buf) + 1);
            strcpy(latMask_Den, buf);
        }
        /* speaker adaptation mask */
        if (GetConfStr(cParm, nParm, "INXFORMMASK", buf)) {
            xfInfo.inSpkrPat = (char *) malloc(strlen(buf) + 1);
            strcpy(xfInfo.inSpkrPat, buf);
        }
        /*if (GetConfStr(cParm, nParm, "PAXFORMMASK", buf)) {
            xfInfo.paSpkrPat = (char *) malloc(strlen(buf) + 1);
            strcpy(xfInfo.paSpkrPat, buf);
        }*/
        if (GetConfBool(cParm, nParm, "USELLF", &boolVal)) {
            useLLF = boolVal;
        }
        if (GetConfBool(cParm, nParm, "INCNUMLATINDENLAT", &boolVal)) {
            optIncNumInDen = boolVal;
        }
    }

}

void ReportUsage (void)
{
    printf("\nUSAGE: HNForward [options] [HMMList]\n\n");
    printf(" Option                                       Default\n\n");
    printf(" -a      Use input transformation             off\n");
    printf(" -d s    Dir to find HMM definitions          current\n");
    printf(" -f      Show utterance statistics            off\n");
    printf(" -h s    Speaker name pattern                 none\n");
    printf(" -l s    Label kind [fea, lab, lat, null]     null\n");
    printf(" -m s    Target mapping file s                off\n");
    printf(" -o s    Extensions for new HMM files         as src\n");
    printf(" -q s    Directory for numerator lats         [needed. May use >1 -q option]\n");
    printf(" -qp s   Subdir pattern for numerator lats    none\n");
    printf(" -r s    Directory for denominator lats       [needed. May use >1 -r option]\n");
    printf(" -rp s   Subdir pattern for denominator lats  none\n");
    printf(" -x s    Extension for HMM files              none\n");
    printf(" -N s    Script file for forwarded data       none\n");
    PrintStdOpts("BFGHIJLMSTX");    /* E, K removed */
    printf("\n\n");
}

/*  */
SpeakerInfo *GetSpeakerInfo(char *uttFN) {
    char name[MAXSTRLEN];
    LabId id;
    SpeakerInfo *sPtr, *tPtr;
    Boolean found;

    found = MaskMatch(xfInfo.inSpkrPat, name, uttFN);
    if (!found) {
        ++failedSpeakers;
        return NULL;
    }
    id = GetLabId(name, TRUE);
    if (id->aux == 0) {
        sPtr = (SpeakerInfo *) New(&gcheap, sizeof(SpeakerInfo)); 
        memset(&sPtr->criteria, 0, sizeof(CriteriaInfo) * SMAX);    
        sPtr->name = id->name;
        sPtr->next = NULL;
        id->aux = (Ptr) sPtr;
        ++speakerCnt;
        if (speakerInfoHead == NULL || strcmp(sPtr->name, speakerInfoHead->name) < 0) {
            sPtr->next = speakerInfoHead;
            speakerInfoHead = sPtr;
        }
        else {
            for (tPtr = speakerInfoHead; tPtr != NULL; tPtr = tPtr->next) {
                if (tPtr->next == NULL || strcmp(sPtr->name, tPtr->next->name) < 0) {
                    sPtr->next = tPtr->next;
                    tPtr->next = sPtr;
                    break;
                }
            }
        }
    }
    else 
        sPtr = (SpeakerInfo *) id->aux;

    return sPtr;
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

/*  */
void AccCriteria(DataCache *cache, int batLen, CriteriaInfo *criteria) {
    LELink layerElem;
    int i, labTgt, recTgt, recTgtMapSum, recLLHTgt, recTgtLLHMapSum;
    IntVec mapVec;
    ANNSet *annSet;

    annSet = ((HMMSet *) cache->hmmSet)->annSet;
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
    /* for MMSE */
    if (showObjFunKind & MMSEOF) {
        criteria->MMSEAcc += CalMMSECriterion(cache->labMat, layerElem->yFeaMats[1], batLen);	/* cz277 - many */
        if (optMapTarget) {
            criteria->MMSEAccMapSum += CalMMSECriterion(annSet->mapStruct->labMatMapSum[cache->streamIdx], annSet->mapStruct->outMatMapSum[cache->streamIdx], batLen);
        }
    }
    /* for XENT */
    if (showObjFunKind & XENTOF) {
        criteria->XENTAcc += CalXENTCriterion(cache->labMat, layerElem->yFeaMats[1], batLen);	/* cz277 - many */
        if (optMapTarget) {
            criteria->XENTAccMapSum += CalXENTCriterion(annSet->mapStruct->labMatMapSum[cache->streamIdx], annSet->mapStruct->outMatMapSum[cache->streamIdx], batLen);
        }
    }
    /* for ML? */
    if (FALSE && (showObjFunKind & MLOF) && (cache->streamIdx == 1)) {
        criteria->LLHVal += fbInfo.pr;
    }

    /* for MPE */
    if ((showObjFunKind & MPEOF) && (cache->streamIdx == 1)) {
        criteria->MPEAcc += fbInfo.AvgCorr;
        criteria->tNWordAcc += fbInfo.MPEFileLength;
    }
    /* for MMI */
    if ((showObjFunKind & MMIOF) && (cache->streamIdx == 1)) {
        criteria->NumLLHAcc += fbInfo.latPr[corrIdx];
        criteria->DenLLHAcc += fbInfo.latPr[recogIdx1];
    }
}

void PrintCriteria(CriteriaInfo *criteria) {
    float accVal;
    int cSampInt, tSampInt;

    cSampInt = (int) criteria->cSampAcc;
    tSampInt = (int) criteria->tSampAcc;
    accVal = criteria->cSampAcc / criteria->tSampAcc;
    printf("\t\tAccuracy = %.2f%% [%d right out of %d samples]\n", accVal * 100.0, cSampInt, tSampInt);

    if (showObjFunKind & XENTOF) {
        printf("\t\tCross Entropy = %.2f\n", criteria->XENTAcc / criteria->tSampAcc);
    }
    if (showObjFunKind & MMSEOF) {
        printf("\t\tMean Square Error = %.2f\n", criteria->MMSEAcc / criteria->tSampAcc);
    }
    if (showObjFunKind & MLOF) {
        printf("\t\tLog-Likelihood/frame = %e\n", criteria->LLHVal / criteria->tSampAcc);
    }

    if (optMapTarget) {
        printf("\n");
        /* for sum */
        cSampInt = (int) criteria->cSampAccMapSum;
        accVal = criteria->cSampAccMapSum / criteria->tSampAcc;
        printf("\t\tMapped Accuracy by Sum = %.2f%% [%d right out of %d samples]\n", accVal * 100.0, cSampInt, tSampInt);
        if (showObjFunKind & XENTOF) {
            printf("\t\tMapped Cross Entropy by Sum = %.2f\n", criteria->XENTAccMapSum / criteria->tSampAcc);
        }
        if (showObjFunKind & MMSEOF) {
            printf("\t\tMapped Mean Square Error by Sum = %.2f\n", criteria->MMSEAccMapSum / criteria->tSampAcc);
        }
        if (showObjFunKind & MLOF) {
            printf("\t\tLog-Likelihood/frame by Sum = %e\n", criteria->LLHValMapSum / criteria->tSampAcc);
        }
        printf("\n");
        /* show confusion matrices */
        if (optShowFrameConfMat) {
            ShowMapConfusionMatrices(hset.annSet, 0.0);
            /* rest counters */
            ClearMappedTargetCounters(hset.annSet);
        }
    }
    fflush(stdout);
}

ReturnStatus InitScriptOutFile(char *fn) {
    char buf[MAXSTRLEN];

    /*CheckFn(fn);*/
    if (optGenANNFeas) 
        HError(4220, "InitScriptOutFile: Only one output script file is allowed");
    optGenANNFeas = TRUE;
    if ((scriptOut = fopen(fn, "r")) == NULL) 
        HError(4210, "InitScriptOutFile: Cannot open output script file %s", fn);
    while (GetNextScpWord(scriptOut, buf) != NULL) 
        ++scriptCntOut;
    rewind(scriptOut);
    return SUCCESS;
}

void Initialise(void) {
    Boolean eSep;
    int s, tgtSize;
    short tgtSwidth[SMAX];

    /* initialise the memory heaps */
    CreateHeap(&cacheHeap, "cache heap", CHEAP, 1, 0, 100000000, ULONG_MAX);
    CreateHeap(&transHeap, "transcription heap", MSTAK, 1, 0, 8000, 80000);
    CreateHeap(&latHeap, "lattice heap", MSTAK, 1, 1.0, 50000, 500000);
    CreateHeap(&feaHeap, "feature heap", MSTAK, 1, 0, 100000, ULONG_MAX);

    /* load HMMs and HMMSet related global variables */
    if (trace & T_TOP) {
        printf("Reading ANN models...\n");
        fflush(stdout);
    }
    if (hmmListFn != NULL && MakeHMMSet(&hset, hmmListFn) < SUCCESS) 
        HError(4200, "Initialise: MakeHMMSet failed");
    if (LoadHMMSet(&hset, hmmDir, hmmExt) < SUCCESS) 
        HError(4200, "Initialise: LoadHMMSet failed");
    if (hset.annSet == NULL) 
        HError(4200, "Initialise: No ANN model available"); 
    /* init train struct */
    if (optHasLabMat) 
        InitTrainInfo(&hset, TRUE, FALSE, FALSE, TRUE);
    /* setup the mappings */
    if (optMapTarget) {
        SetupStateInfoList(&hset);
        if (SetupTargetMapList(&hset, mappingFn, 0) < SUCCESS) 
            HError(4200, "Initialise: Failed to load the target mapping file");
        InitMapStruct(&hset);
        recVecMapSum = CreateIntVec(&gcheap, GetNBatchSamples());
        recVecLLHMapSum = CreateIntVec(&gcheap, GetNBatchSamples());
        /*ClearMappedTargetCounters(hset.annSet);*/
    }
    CreateTmpNMat(hset.hmem);

    SetStreamWidths(hset.pkind, hset.vecSize, hset.swidth, &eSep);
    if (trace & T_TOP) {
        printf("ANN model structure:\n");
        ShowANNSet(&hset);
        fflush(stdout);
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
            /* MakeObservation(&gcheap, ...); */
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
    if (optHasLabMat) {
        recVec = CreateIntVec(&gcheap, GetNBatchSamples());
        recVecLLH = CreateIntVec(&gcheap, GetNBatchSamples());
        /* initialise the criteria */
        memset(&criteriaAll, 0, sizeof(CriteriaInfo) * SMAX);
    }

    /* initialise the cache structures */
    obsIn = MakeObservation(&gcheap, hset.swidth, hset.pkind, FALSE, eSep);
    scriptIn = GetTrainScript(&scriptCntIn);
    if (trace & T_TOP) {
        printf("%d utterances to process\n", scriptCntIn);
    }
    for (s = 1; s <= hset.swidth[0]; ++s) {
        cacheIn[s] = CreateCache(&cacheHeap, scriptIn, scriptCntIn, &hset, &obsIn, s, GetDefaultNCacheSamples(), NONEVK, &xfInfo, labelInfo, TRUE);
    }

    if ((labelKind & LATLK) != 0) {
        InitialiseFBInfo(&fbInfo, &hset, cacheIn[1]->labelInfo->uFlags, FALSE);
        for (s = 1; s <= hset.swidth[0]; ++s) {
            fbInfo.llhMat[s] = hset.annSet->llhMat[s];
            fbInfo.occMat[s] = hset.annSet->outLayers[s]->yFeaMats[1];	/* cz277 - many */
        }
    }

    /* set the output observation */
    if (optGenANNFeas) {
        tgtSize = 0;
        ZeroStreamWidths(hset.swidth[0], tgtSwidth);
        for (s = 1; s <= hset.swidth[0]; ++s) {
            tgtSwidth[s] = hset.annSet->outLayers[s]->nodeNum;
            tgtSize += tgtSwidth[s];
        }
        SetStreamWidths(tgtPK, tgtSize, tgtSwidth, &eSep);
        obsOut = MakeObservation(&gcheap, tgtSwidth, tgtPK, FALSE, eSep);
    }

}

void ShowCriteriaInfo(CriteriaInfo *criteria) {
    int i, S;

    S = hset.swidth[0];
    for (i = 1; i <= S; ++i) {
        if (S > 1) {
            printf("Stream %d: ", i);
        }
        PrintCriteria(&criteria[i]);
    }
}

void LoadFeaMatToParmBuf(int nFrame) {
    int s, i;
    LELink layerElem;

    for (s = 1; s <= hset.swidth[0]; ++s) {
        layerElem = hset.annSet->outLayers[s];
        for (i = 0; i < nFrame; ++i) {
            /*for (j = 0; j < layerElem->nodeNum; ++j) {
                obsOut.fv[s][j + 1] = (float) layerElem->yFeaMat->matElems[i * layerElem->nodeNum + j];
            }*/
            CopyNFloatSeg2FloatSeg(&layerElem->yFeaMats[1]->matElems[i * layerElem->nodeNum], layerElem->nodeNum, &obsOut.fv[s][1]);	/* cz277 - many */
            AddToBuffer(parmBuf, obsOut);
        } 
    }
}

int main(int argc, char *argv[]) {
    char *str;
    char buf[MAXSTRLEN], uttName[MAXSTRLEN], fnbuf[MAXSTRLEN];
    clock_t stClock, edClock;
    int i, S, nLoaded, sampCnt, batchCnt, tSampCnt, tUttCnt, uttCnt, uttLen;
    Boolean finish = FALSE, skipOneUtt=FALSE, sentFail;
    LELink layerElem;
    SpeakerInfo *speakerInfo=NULL;
    UttElem *uttElem;
    Lattice *MPECorrLat = NULL;

    if (InitShell(argc, argv, hnforward_version, hnforward_vc_id) < SUCCESS) 
        HError(4200, "HNForward: InitShell failed");
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
        HError(4200, "HNForward: InitParm failed");
    InitUtil();
    /* cz277 - xform */
    /*InitAdapt(&xfInfo);*/
    InitAdapt();
    InitXFInfo(&xfInfo);
    
    InitLat();
    InitNCache();

    if (!InfoPrinted() && NumArgs() == 0) {
        ReportUsage();
    }
    if (NumArgs() == 0) {
        Exit(0);
    }

    CreateHeap(&modelHeap, "model heap",  MSTAK, 1, 0.0, 100000000, ULONG_MAX);
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
            case 'd':
                if (NextArg() != STRINGARG) 
                    HError(4219, "HNForward: HMM definition directory expected");
                hmmDir = GetStrArg();
                break;
            case 'f':
                optShowUttStats = TRUE;
                break;
            case 'h':
                if (NextArg() != STRINGARG) 
                    HError(4219, "HNForward: Speaker name pattern expected");
                xfInfo.outSpkrPat = GetStrArg();
                if (NextArg() == STRINGARG) {
                    xfInfo.inSpkrPat = GetStrArg();
                    if (NextArg() == STRINGARG) 
                        xfInfo.paSpkrPat = GetStrArg();
                }
                if (NextArg() != SWITCHARG) 
                    HError(4219, "HNForward: Cannot have -h as the last option");
                break;
            case 'l':
                if (NextArg() != STRINGARG) 
                    HError(4219, "HNForward: Label kind expected");
                str = GetStrArg();
                if (strcasecmp(str, "LAB") == 0 || strcasecmp(str, "LABEL") == 0) {
                    labelKind = LABLK;   
                    optHasLabMat = TRUE;
                }
                else if (strcasecmp(str, "LAT") == 0 || strcasecmp(str, "LATTICE") == 0) {
                    labelKind = LATLK;
                    optHasLabMat = TRUE;
                }
                else if (strcasecmp(str, "FEA") == 0 || strcasecmp(str, "FEATURE") == 0) {
                    labelKind = FEALK;
                    optHasLabMat = TRUE;
                }
                else if (strcasecmp(str, "NULL") == 0) {
                    labelKind = NULLLK;
                    optHasLabMat = FALSE;
                }
                else {
                    HError(4219, "HNForward: Unknown label kind");
                }
                break;
            case 'm':
                if (NextArg() != STRINGARG) 
                    HError(4219, "HNForward: Target mapping file expected");
                optMapTarget = TRUE;
                mappingFn = GetStrArg();
                break;
            case 'q':
                if (strcmp(str, "q") == 0) {
                    numLatDir[nNumLats++] = GetStrArg();
                }
                else if (strcmp(str, "qp") == 0) {
                    strcpy(numLatSubDirPat, GetStrArg());
                    if (strchr(numLatSubDirPat, '%') == NULL) 
                        HError(4219, "HNForward: Numerator lattice path mask invalid");
                }
                else {
                    HError(4219, "HNForward: Unknown option %s", str);
                }
                break;
            case 'r':
                if (strcmp(str, "r") == 0) {
                    denLatDir[nDenLats++] = GetStrArg();
                }
                else if (strcmp(str, "rp") == 0) {
                    strcpy(denLatSubDirPat, GetStrArg());
                    if (strchr(denLatSubDirPat, '%') == NULL) 
                        HError(4219, "HNForward: Denominator lattice path mask invalid");
                }
                else {
                    HError(4219, "HNForward: Unknown option %s", str);
                }
                break;
            case 'x':
                if (NextArg() != STRINGARG) 
                    HError(4219, "HNForward: HMM file extension expected");
                hmmExt = GetStrArg();
                break;
            /*case 'E':
                if (NextArg() != STRINGARG) {
                    HError(4219, "HNForward: Parent transform directory expected");
                }
                xfInfo.usePaXForm = TRUE;
                xfInfo.paXFormDir = GetSrArg();
                if (NextArg() != STRINGARG) {
                    xfInfo.paXFormExt = GetStrArg();
                }
                if (NextArg() != SWITCHARG) {
                    HError(4219, "HNForward: Cannot have -E as the last option");
                }
                break;*/
            case 'F':
                if (NextArg() != STRINGARG) 
                    HError(4219, "HNForward: Data file format expected");
                if ((dff = Str2Format(GetStrArg())) == ALIEN)
                    HError(4219, "HNForward: Warnings ALIEN data file format set");
                break;
            case 'G':
                if (NextArg() != STRINGARG) 
                    HError(4219, "HNForward: Label file format expected");
                if ((lff = Str2Format(GetStrArg())) == ALIEN) 
                    HError(4219, "HNForward: Warnings ALIEN label file format set");
                break;
            case 'H':
                if (NextArg() != STRINGARG) 
                    HError(4219, "HNForward: HMM macro file name expected");
                AddMMF(&hset, GetStrArg());
                break;
            case 'I':
                if (NextArg() != STRINGARG) 
                    HError(4219, "HNForward: MLF file name expected");
                LoadMasterFile(GetStrArg());
                break;
            case 'J':
                if (NextArg() != STRINGARG)
                    HError(4219, "HNForward: Input transform directory expected");
                AddInXFormDir(&hset, GetStrArg());
                if (NextArg() != STRINGARG) {
                    if (xfInfo.inXFormExt == NULL) 
                        xfInfo.inXFormExt = GetStrArg();
                    else 
                        HError(4219, "HNForward: Only one input transform extension may be specified");
                }
                if (NextArg() != SWITCHARG) 
                    HError(4219, "HNForward: Cannot have -J as the last option");
                break;
            case 'L':
                if (NextArg() != STRINGARG) 
                    HError(4219, "HNForward: Label file directory expected");
                labDir = GetStrArg();
                break;
            case 'N':
                if (NextArg() != STRINGARG) 
                    HError(4219, "HNForward: Output script file name expected");
                InitScriptOutFile(GetStrArg());                
                break;
            case 'Q':
                if (NextArg() != STRINGARG)
                    HError(4219, "HNForward: Lattice extension expected");
                latExt = GetStrArg();
                break;
            case 'T':
                trace = GetChkedInt(0, 0100000, str);
                break;
            case 'X':
                if (NextArg() != STRINGARG) 
                    HError(4219, "HNForward: Label file extension expected");
                labExt = GetStrArg();
                break;
            default:
                HError(4219, "HNForward: Unknown switch %s", str);
        }
    }
    /* load hmmListFn (optional) */
    if (NextArg() == STRINGARG) 
        hmmListFn = GetStrArg();
    else if (NextArg() != NOARG) 
        HError(4219, "HNForward: Only an optional HMM list file is possible at the end of the command line");
    /* command check */
    if (NumArgs() == 0 || hset.numFiles == 0) 
        HError(4219, "HNForward: At least one input HMM file should be given");

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

    stClock = 0;
    edClock = 0;
    stClock = clock();

    /* forward each utterance */        
    S = hset.swidth[0];
    batchCnt = 0;
    tSampCnt = 0;
    tUttCnt = 0;
    /* init the cache */
    for (i = 1; i <= S; ++i) 
        InitCache(cacheIn[i]);

    /* process until all data are finished */
    printf("\n");
    printf("Evaluating ************************\n");
    printf("\tProcessing the evaluation set...\n");
    while (!finish) {
        sampCnt = 0;
        uttCnt = 1;
        strcpy(uttName, GetCurUttName(cacheIn[1]));
        uttLen = GetCurUttLen(cacheIn[1]);
        uttElem = GetCurUttElem(cacheIn[1]);
        /* install the current replaceable parts */
        InstallOneUttNMatRPLs(uttElem);
        InstallOneUttNVecRPLs(uttElem);
        if (optShowSeqObjVal) {
            /*uttElem = GetCurUttElem(cacheIn[1]);*/
            if (uttElem->uttLen > GetNBatchSamples()) {
                printf("HNForward: %d samples in utterance %s exceeds batch size %d\n", uttElem->uttLen, uttElem->uttName, GetNBatchSamples());
                skipOneUtt = TRUE;
            }
            else {
                skipOneUtt = FALSE;
            }
            /* init fbInfo */
            fbInfo.T = uttElem->uttLen;
            LoadXFormsFromUttElem(uttElem, &fbInfo);
            fbInfo.uFlags = cacheIn[1]->labelInfo->uFlags;
        }
        if (optHasLabMat && optShowUttStats) 
            memset(&criteriaUtt, 0, sizeof(CriteriaInfo) * SMAX);
        /* write the data, if needed */
        if (optGenANNFeas) {
            str = GetStrArg();
            strcpy(fnbuf, str);
            parmBuf = OpenBuffer(&feaHeap, fnbuf, 50, UNDEFF, TRI_UNDEF, TRI_UNDEF);    
            GetBufferInfo(parmBuf, &bufInfo); 
            bufInfo.tgtPK = tgtPK;
            CloseBuffer(parmBuf);
            parmBuf = EmptyBuffer(&feaHeap, uttLen, obsOut, bufInfo);
            /*CopyParmBufInfo(srcPBuf, dstPBuf);*/
        }
        while ((!finish) && uttCnt > 0) {
            /* load data */
            for (i = 1; i <= S; ++i) {
                finish |= FillAllInpBatch(cacheIn[i], &nLoaded, &uttCnt);
                /* cz277 - mtload */
                /*UpdateCacheStatus(cacheIn[i]);*/
                LoadCacheData(cacheIn[i]);
            }
            /* whether skip this utterance or not */
            if (optShowSeqObjVal && skipOneUtt) 
                continue;
            /* forward propagation */
            ForwardProp(hset.annSet, nLoaded, cacheIn[1]->CMDVecPL);
            sentFail = FALSE;
            /* synchronise the data */
            for (i = 1; i <= S; ++i) {
                layerElem = hset.annSet->outLayers[i];
                /* convert posteriors to llh */
                if ((showObjFunKind & MLOF) || optShowSeqObjVal) {
                    ApplyLogTrans(layerElem->yFeaMats[1], nLoaded, layerElem->nodeNum, hset.annSet->llhMat[i]);	/* cz277 - many */
                    AddNVectorTargetPen(hset.annSet->llhMat[i], hset.annSet->penVec[i], nLoaded, hset.annSet->llhMat[i]);
#ifdef CUDA
                    SyncNMatrixDev2Host(hset.annSet->llhMat[i]);
#endif
                }
                /* for mapped targets */
                if (optMapTarget) {
                    UpdateOutMatMapSum(hset.annSet, nLoaded, i);
                    /* convert posteriors to llh */
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
#ifdef CUDA
                SyncNMatrixDev2Host(layerElem->yFeaMats[1]);	/* cz277 - many */
#endif
            }
            /* for sequence processing */
            if (optShowSeqObjVal) {
                if (procNumLats) {
                    LoadNumLatsFromUttElem(uttElem, &fbInfo);
                    if (FBLatFirstPass(&fbInfo, UNDEFF, uttElem->uttName, NULL, NULL)) {
                        FBLatSecondPass(&fbInfo, corrIdx, 999);
                        sentFail = FALSE;
                    }
                    else {
                        sentFail = TRUE;
                    }
                }
                if (procDenLats) {
                    if (showObjFunKind & MPEOF) {
                        /*MPECorrLat = fbInfo.aInfo->lat[0];*/
                        MPECorrLat = uttElem->numLats[0];
                    }
                    else {
                        MPECorrLat = NULL;
                    }
                    LoadDenLatsFromUttElem(uttElem, &fbInfo);
                    if (FBLatFirstPass(&fbInfo, UNDEFF, uttElem->uttName, NULL, MPECorrLat)) {
                        FBLatSecondPass(&fbInfo, recogIdx1, recogIdx2);
                        sentFail = FALSE;
                    }
                    else {
                        sentFail = TRUE;
                    }
                }
            }
            /* compute the criteria */
            if (optHasLabMat) {
                if (xfInfo.inSpkrPat != NULL) {
                    speakerInfo = GetSpeakerInfo(uttName);
                }
                for (i = 1; i <= S; ++i) {
                    layerElem = hset.annSet->outLayers[i];
                    FindMaxElement(layerElem->yFeaMats[1], nLoaded, layerElem->nodeNum, recVec);	/* cz277 - many */
                    if (showObjFunKind & MLOF) {
                        FindMaxElement(hset.annSet->llhMat[i], nLoaded, layerElem->nodeNum, recVecLLH);
                    }
                    if (optMapTarget) {
                        FindMaxElement(hset.annSet->mapStruct->outMatMapSum[i], nLoaded, hset.annSet->mapStruct->mappedTargetNum, recVecMapSum);
                        if (showObjFunKind & MLOF) {
                            FindMaxElement(hset.annSet->mapStruct->llhMatMapSum[i], nLoaded, hset.annSet->mapStruct->mappedTargetNum, recVecLLHMapSum);
                        }
                        UpdateLabMatMapSum(hset.annSet, nLoaded, i);
#ifdef CUDA
                        SyncNMatrixDev2Host(hset.annSet->mapStruct->labMatMapSum[i]);
#endif
                    }
                    if (!sentFail) {
                        AccCriteria(cacheIn[i], nLoaded, &criteriaAll[i]);
                        if (optShowUttStats) {
                            AccCriteria(cacheIn[i], nLoaded, &criteriaUtt[i]);
                        }
                        if (xfInfo.inSpkrPat != NULL) {
			    if(speakerInfo==NULL)
			      HError(4214, "HNForward: Speaker info not set");
                            AccCriteria(cacheIn[i], nLoaded, &speakerInfo->criteria[i]);
                        }
                    }
                }
            }
            /* write the data, if needed */
            if (optGenANNFeas) 
                LoadFeaMatToParmBuf(nLoaded);
            /* update the statistics */
            batchCnt += 1;

            sampCnt += nLoaded;
            tSampCnt += nLoaded;
        }
        if (optHasLabMat && optShowUttStats) {
            printf("\t\tShow criterion values for %s:\n", uttName);
            ShowCriteriaInfo(criteriaUtt);
        }
        /* write the data, if needed */
        if (optGenANNFeas) {
            GetNextScpWord(scriptOut, buf);
            if (SaveBuffer(parmBuf, buf, tgtFF) < SUCCESS) 
                HError(4214, "HNForward: Could not save parm file %s", buf);
            /*CloseBuffer(parmBuf);*/
            /*printf("%s --> %s\n", uttName, buf);*/
            ResetHeap(&feaHeap);
        }
        /* cz277 - mtload */
        for (i = 1; i <= S; ++i) 
            UnloadCacheData(cacheIn[i]);
        tUttCnt += 1;
    }
    /* reset the replaceable parts */
    ResetNMatRPL();
    ResetNVecRPL();
    /* show criteria */
    if (optHasLabMat) {
        printf("\t\tShow criterion values for all data:\n");
        ShowCriteriaInfo(criteriaAll);
    }

    /* forwarding finished */
    edClock = clock();
    if (trace & T_TIM)
        printf("\t\tTime cost = %.2fs\n", (edClock - stClock) / (double) CLOCKS_PER_SEC);

    /* free ANNSet */
    FreeANNSet(&hset);
    for (i = 1; i <= S; ++i) 
        FreeCache(cacheIn[i]);

#ifdef CUDA
    StopCUDA();
#endif

    Exit(EXIT_STATUS);
    return 0;
}


/* ----------------------------------------------------------- */
/*                      END:  HNForward.c                      */
/* ----------------------------------------------------------- */

