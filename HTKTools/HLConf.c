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
/*           Gunnar Evermann <ge204@eng.cam.ac.uk>             */
/* ----------------------------------------------------------- */
/*           Copyright: Cambridge University                   */
/*                      Engineering Department                 */
/*            1999-2015 Cambridge, Cambridgeshire UK           */
/*                      http://www.eng.cam.ac.uk               */
/*                                                             */
/*   Use of this software is governed by a License Agreement   */
/*    ** See the file License for the Conditions of Use  **    */
/*    **     This banner notice must not be removed      **    */
/*                                                             */
/* ----------------------------------------------------------- */
/*          File: HLConf.c  Lattice confusion network          */
/* ----------------------------------------------------------- */

char *hlconf_version = "!HVER!HLConf:   3.5.0 [CUED 12/10/15]";
char *hlconf_sccs_id = "$Id: HLConf.c,v 1.8 2015/10/12 12:07:24 cz277 Exp $";

#include "HShell.h"
#include "HMem.h"
#include "HMath.h"
#include "HWave.h"
#include "HLabel.h"
#include "HAudio.h"
#include "HParm.h"
#include "HANNet.h"
#include "HModel.h"
#include "HUtil.h"
#include "HDict.h"
#include "HLM.h"
#include "HNet.h"
#include "HRec.h"
#include "HLat.h"

#include <assert.h>

/* -------------------------- Trace Flags & Vars ------------------------ */

#define T_TOP  00001      /* Basic progress reporting */
#define T_TRAN 00002      /* Output Transcriptions */
#define T_CN   00010      /* Confusion network clustering */
#define T_LAT  00020      /* Lattice operation */
#define T_MEM  00040      /* Memory usage, start and finish */

static int trace = 0;

/* ---------------- Configuration Parameters --------------------- */

static ConfParam *cParm[MAXGLOBS];
static int nParm = 0;            /* total num params */

/* -------------------------- Global Variables etc ---------------------- */


typedef enum _ConfMethod {
   CM_GEOMEAN, CM_MAX
} ConfMethod;

static char *dictfn;		/* dict filename from commandline */
static char *latInDir = NULL;   /* Lattice input dir, set by -L  */
static char *latInExt = "lat";  /* Lattice Extension, set by -X */

static FileFormat ifmt=UNDEFF;  /* Label input file format */
#if 0
static char *labInDir = NULL;     /* input label file directory */
static char *labInExt = "rec";    /* input label file extension */
#endif

static FileFormat ofmt=UNDEFF;  /* Label output file format */
static char *labOutDir = NULL;     /* output label file directory */
static char *labOutExt = "rec";    /* output label file extension */
static char *labOutForm = NULL;    /* output label format */

static double lmScale = 1.0;    /* LM scale factor */
static double acScale = 1.0;    /* acoustic scale factor */
static LogDouble wordPen = 0.0; /* inter model log penalty */
static double prScale = 1.0;    /* pronunciation scale factor */
static float confNetPrune = -5.0;    /* second pass confusion network pruning */
/* cz277 - scale conf score */
static float latScoreScale  = 1.0;

static Vocab vocab;		/* wordlist or dictionary */

static ConfMethod confMethod = CM_MAX;

static Boolean confnet = FALSE; /* perform confnet clustering */
static Boolean writeConfNet = FALSE; /* write scf files? */

/* weights for Levenshtein alignment */
static int subPen = 4;	  /* NIST values */
static int delPen = 3;
static int insPen = 3;

static Boolean fixPronProb = FALSE;
static Boolean clampACLike = TRUE;
static Boolean addNullWord = TRUE;

static char *labFileMask = NULL;
static char *labOFileMask = NULL;
static char *latFileMask = NULL;
static char *latOFileMask = NULL;

/* -------------------------- Heaps ------------------------------------- */

static MemHeap latHeap;
static MemHeap cnHeap;
static MemHeap transHeap;

/* -------------------------- Prototypes -------------------------------- */
void SetConfParms(void);
void ReportUsage(void);
void CalcConfFile(char *latfn);
void ConfNetClusterFile (char *latfn_in, char *latfn_ou, char *labfn_ou);
void InitSimScore(void);


/* ---------------- Process Command Line ------------------------- */

/* SetConfParms: set conf parms relevant to this tool */
void SetConfParms(void)
{
   int i;
   Boolean b;
   double f;
   char buf[MAXSTRLEN];

   nParm = GetConfig("HLCONF", TRUE, cParm, MAXGLOBS);
   if (nParm>0){
      if (GetConfInt(cParm,nParm,"TRACE",&i)) trace = i;
      if (GetConfStr(cParm,nParm,"CONFMETHOD",buf)) {
         if (strcmp (buf, "GEOMEAN") == 0)
            confMethod = CM_GEOMEAN;
         else if (strcmp (buf, "MAX") == 0)
            confMethod = CM_MAX;
         else
            HError (4121, "HLConf: unknown CONFMETHOD");
      }
      if (GetConfBool (cParm,nParm,"CLAMPACLIKE",&b)) clampACLike = b;
      if (GetConfBool (cParm,nParm,"FIXPRONPROB",&b)) fixPronProb = b;
      if (GetConfBool (cParm,nParm,"ADDNULLWORD",&b)) addNullWord = b;
      if (GetConfFlt (cParm, nParm, "CONFNETPRUNE", &f))confNetPrune  = f;
      /* cz277 - scale conf score */
      if (GetConfFlt(cParm, nParm, "SCALELATSCORE", &f)) {
          latScoreScale = f;
          if (latScoreScale <= 0.0)
              HError(4122, "HLConf: SCALELATSCORE must be positive");
      }

      if (GetConfStr(cParm,nParm,"LABFILEMASK",buf)) {
         labFileMask = CopyString (&gstack, buf);
      }
      if (GetConfStr(cParm,nParm,"LABOFILEMASK",buf)) {
         labOFileMask = CopyString (&gstack, buf);
      }
      if (GetConfStr(cParm,nParm,"LATFILEMASK",buf)) {
         latFileMask = CopyString (&gstack, buf);
      }
      if (GetConfStr(cParm,nParm,"LATOFILEMASK",buf)) {
         latOFileMask = CopyString (&gstack, buf);
      }
   }
}

void ReportUsage(void)
{
   printf("\nUSAGE: HLConf [options] vocabFile Files...\n\n");
   printf(" Option                                   Default\n\n");
   printf(" -i s    Output transcriptions to MLF s      off\n"); 
   printf(" -l s    dir to store label files	    current\n");
   printf(" -o s    output label formating NCSTWMX      none\n");
   printf(" -p f    inter model trans penalty (log)     0.0\n");
   printf(" -r f    pronunciation scale factor          1.0\n");
   printf(" -s f    grammar scale factor                1.0\n");
   printf(" -a f    acoustic scale factor               1.0\n");
   printf(" -y s    output label file extension         rec\n");
   printf(" -z      perform confusion net clustering    off\n");
   printf(" -Z      write confusion networks            off\n");
   PrintStdOpts("ILSXTGP");
   printf("\n\n");
}

int main(int argc, char *argv[])
{
   char *s, *latfn, latfn_in[MAXSTRLEN], latfn_ou[MAXSTRLEN], labfn_ou[MAXSTRLEN];

   /*#### new error code range */
   if(InitShell (argc, argv, hlconf_version, hlconf_sccs_id) < SUCCESS)
      HError (4100, "HLConf: InitShell failed");
  
   InitMem();
   InitMath();
   InitWave();
   InitLabel();
   InitAudio();
   if (InitParm()<SUCCESS) 
      HError(4100,"HLConf: InitParm failed");
   InitModel();
   InitUtil();
   InitDict();
   InitNet(); 
   InitRec();

   if (!InfoPrinted() && NumArgs() == 0)
      ReportUsage();
   if (NumArgs() == 0) Exit(0);
  
   SetConfParms();

   while (NextArg() == SWITCHARG) {
      s = GetSwtArg();
      if (strlen(s) != 1) 
         HError (4119, "HLConf: Bad switch %s; must be single letter",s);
      switch (s[0]){
      case 'L':
         if (NextArg() != STRINGARG)
            HError (4119,"HLConf: Lattice file directory expected");
         latInDir = GetStrArg(); 
         break;
      case 'X':
         if (NextArg() != STRINGARG)
            HError (4119, "HLConf: Lattice filename extension expected");
         latInExt = GetStrArg(); 
         break;
      case 'i':
         if (NextArg() != STRINGARG)
            HError (4119, "HLConf: Output MLF file name expected");
         if (SaveToMasterfile (GetStrArg()) < SUCCESS)
            HError (4114, "HLConf: Cannot write to MLF");
         break;

      case 'I':
         if (NextArg() != STRINGARG)
            HError (4119, "HLConf: Input MLF file name expected");
         LoadMasterFile (GetStrArg ());
         break;


      case 'G':
         if (NextArg() != STRINGARG)
            HError (4119, "HLConf: Input Label File format expected");
         if((ifmt = Str2Format (GetStrArg ())) == ALIEN)
            HError (-4189, "HLConf: Warning ALIEN Input file format set");
         break;
      case 'P':
         if (NextArg() != STRINGARG)
            HError (4119, "HLConf: Output Label File format expected");
         if((ofmt = Str2Format (GetStrArg ())) == ALIEN)
            HError (-4189, "HLConf: Warning ALIEN Label output file format set");
         break;
      
      case 'l':
         if (NextArg() != STRINGARG)
            HError (4119, "HLConf: Output Label file directory expected");
         labOutDir = GetStrArg(); 
         break;
      case 'o':
         if (NextArg() != STRINGARG)
            HError (4119, "HLConf: Output label format expected");
         labOutForm = GetStrArg(); 
         break;
      case 'y':
         if (NextArg() != STRINGARG)
            HError (4119, "HLConf: Output label file extension expected");
         labOutExt = GetStrArg(); break;
      
      case 'p':
         if (NextArg() != FLOATARG)
            HError (4119, "HLConf: word insertion penalty expected");
         wordPen = GetChkedFlt (-1000.0, 1000.0, s); 
         break;
      case 's':
         if (NextArg() != FLOATARG)
            HError (4119, "HLConf:  grammar scale factor expected");
         lmScale = GetChkedFlt (0.0, 1000.0, s); 
         break;
      case 'r':
         if (NextArg() != FLOATARG)
            HError (4119, "HLConf:  pronunciation scale factor expected");
         prScale = GetChkedFlt (0.0, 1000.0, s); 
         break;
      case 'a':
         if (NextArg() != FLOATARG)
            HError (4119, "HLConf:  acoustic scale factor expected");
         acScale = GetChkedFlt (0.0, 1000.0, s); 
         break;

      case 'z':
         confnet = TRUE;
         break;
      case 'Z':
         writeConfNet = TRUE;
         break;
         
      case 'T':
         trace = GetChkedInt(0, 100, s); 
         break;
      default:
         HError (4119, "HLConf: Unknown switch %s",s);
      }
   }


   if (NextArg() != STRINGARG)
      HError(4119, "Vocab file name expected");
   dictfn=GetStrArg();
  
   /* Read dictionary */
   if (trace & T_TOP) 
      printf ("Reading dictionary from %s\n", dictfn);
   InitVocab (&vocab);
   ReadDict (dictfn, &vocab);

   /* init Heaps */
   CreateHeap (&latHeap, "Lattice heap", MSTAK, 1, 0, 8000, 80000);
   CreateHeap (&cnHeap, "ConfNet heap", MSTAK,1, 0, 8000, 80000);
   CreateHeap (&transHeap, "Transcription heap", MSTAK, 1, 0, 8000, 80000);

   if (confnet)
      InitSimScore ();

   while (NumArgs() > 0) {
      if (NextArg() != STRINGARG)
         HError (4119, "HLConf: Transcription file name expected");
      latfn = GetStrArg();
      if (latFileMask) {
         if (!MaskMatch (latFileMask, latfn_in, latfn))
            HError(4119,"HLRescore: LABFILEMASK %s has no match with segemnt %s", latFileMask, latfn);
      }
      else
         strcpy (latfn_in, latfn);

      if (latOFileMask) {
         if (!MaskMatch (latOFileMask, latfn_ou, latfn))
            HError(4119,"HLRescore: LABFILEMASK %s has no match with segemnt %s", latOFileMask, latfn);
      }
      else
         strcpy (latfn_ou, latfn);

      if (labOFileMask) {
         if (!MaskMatch (labOFileMask, labfn_ou, latfn))
            HError(4119,"HLRescore: LABOFILEMASK %s has no match with segemnt %s", labOFileMask, latfn);
      }
      else
         strcpy (labfn_ou, latfn);

      if (trace & T_TOP) {
         printf ("File: %s\n", latfn);  fflush(stdout);
      }
      if (confnet)
         ConfNetClusterFile (latfn_in, latfn_ou, labfn_ou);
      else {
         abort();
#if 0
         CalcConfFile (latfn);
#endif
      }
   }

   if (trace & T_MEM) {
      printf("Memory State on Completion\n");
      PrintAllHeapStats();
   }
  
   return(0); 
}

void ClampACLike (Lattice *lat)
{
   int i;
   LArc *la;

   for (i = 0, la = lat->larcs; i < lat->na; ++i, ++la) {
      if (la->aclike > 0) {
         HError (-4122, "HLattice: aclike of arc is %f  setting to 0.0\n", la->aclike);  /*#### GE fix errorcode */
         la->aclike = 0.0;
      }
   }
}


/* calc (unnormalalised!) posterior of arc */
#define LArcPosterior(lat, la)  (LNodeFw((la)->start) + \
                                 LArcTotLike((lat),(la)) + \
                                 LNodeBw((la)->end))


#if 0           /* time dep conf scores -- stuff for Luis */

/* the main confidence calculation function,
   assumes that forward/backward scores are stored in lattice

   confidence scores are stored in the lab->score fields of the Transcription
*/
void CalcConfScores(Lattice *lat, LogDouble pX, Transcription *trans)
{
   LabList *labList;
   LLink lab;
   Word word;
   LArc *la;
   int i, l, nlab;
   HTime t, framedur = 1e7;  /* 100 frames per second  #### fix this! */
   LogDouble postLA, postCor, postAll;
   LogDouble confCor, maxPostCor;
   int lenCor;
   
   labList = GetLabelList (trans, 1);
   nlab = CountLabs (labList);
   
   /* for each word: */
   for (l = 1; l <= nlab; ++l) {
      confCor = 0;
      lenCor = 0;
      maxPostCor = LZERO;

      lab = GetLabN (labList, l);
      word = GetWord (&vocab, lab->labid, FALSE);
      if (!word)
         HError (9999, "HLConf: word %s not in dict", lab->labid->name);
         
      if (trace & T_TOP) {
         printf ("%8.0f%8.0f", lab->start, lab->end);
         printf (" %8s %5f\n", lab->labid->name, lab->score);
      }
      
      /* for each frame in word: */
      for (t = lab->start / framedur; t <= lab->end / framedur; t += 0.01) { /* #### fix frame<->t ! */
         postCor = LZERO;
         postAll = LZERO;
         for (i = 0, la = lat->larcs; i < lat->na; ++i, ++la) {  /* check all arcs */
            if ((la->start->time <= t) && (la->end->time > t)) { /* arc intersects time t? */
               postLA = LArcPosterior(lat, la) - pX;         /* normalised posterior */
               
               if (la->end->word->wordName == lab->labid) {   /* correct word? */
                  postCor = LAdd (postCor, postLA);
               }
               /* for sanity check */
               postAll = LAdd (postAll, postLA);
            }
         }
         if (postCor != LZERO) {          /* ignore frames with zero prob */
            confCor += postCor;
            ++lenCor;
         }
         if (postCor > maxPostCor)
            maxPostCor = postCor;

         if (trace & T_TOP)
            printf ("t %.2f postCor: log %f lin %f  postAll: log %f\n", t, postCor, L2F (postCor), postAll);
      }
      /*  normalise by length (geom. mean) */
      if (lenCor > 0)
         confCor = confCor / lenCor;
      else
         confCor = LZERO;


      switch (confMethod) {
      case CM_GEOMEAN: 
         lab->score = L2F (confCor);
         break;
      case CM_MAX: 
         lab->score = L2F (maxPostCor);
         break;
      default:
         abort ();
         break;
      }

      if (trace & T_TOP)
         printf ("confCor %f %f\n", lab->score, L2F (lab->score));
   } /* for each label */
}


/* CalcConfFile (char *latfn)

     called once for each Lattice File
     performs calculation of confidence scores based on time dependent posteriors
     cf. [Evermann & Woodland:ICASSP2000]
 */
void CalcConfFile(char *latfn)
{
   Lattice *lat;
   char lfn[MAXSTRLEN];
   FILE *lf;
   Boolean isPipe;
   Transcription *trans;
   LogDouble pX;        /* prob of data;  p(X) = alpha(final) = beta(root)  */

   MakeFN(latfn, latInDir, latInExt, lfn);
  
   if ((lf = FOpen(lfn,NetFilter,&isPipe)) == NULL)
      HError(4010,"HLConf: Cannot open Lattice file %s", lfn);
  
   lat = ReadLattice (lf, &latHeap, &vocab, FALSE, FALSE);
   FClose(lf, isPipe);

   /*#### GE: why doesn't HRec:Readlattice exit as it should? 
     probably more HAPI rubbish... */
   if (!lat)
      HError (9999, "HLConf: can't read lattice");

   lat->lmscale = lmScale;
   lat->wdpenalty = wordPen;
   lat->acscale = acScale;
   /*### add prscale? */

   pX = LatForwBackw (lat, &latHeap);

#if 0
   /* print lattice arcs with posteriors */
   for (i = 0, la = lat->larcs; i < lat->na; ++i, ++la) {
      printf ("S %d E %d  W %s post %lf\n",
              la->start->n, la->end->n,
              la->end->word->wordName->name,
              LArcPosterior (lat,la));
   }
#endif

   /* read transcription */
   MakeFN(latfn, labInDir, labInExt, lfn);
   trans = LOpen(&transHeap, lfn, ifmt);


   /* calc conf scores */
   CalcConfScores (lat, pX, trans);

   /* write transcription */
   PrintTranscription (trans, "Transcription with conf scores");
   if (labOutForm!=NULL)
      FormatTranscription (trans, 
                           1e7, FALSE, FALSE,	/* #### fix frame stuff */
                           strchr(labOutForm,'X')!=NULL,
                           strchr(labOutForm,'N')!=NULL,strchr(labOutForm,'S')!=NULL,
                           strchr(labOutForm,'C')!=NULL,strchr(labOutForm,'T')!=NULL,
                           strchr(labOutForm,'W')!=NULL,strchr(labOutForm,'M')!=NULL);
  
   MakeFN (latfn, labOutDir, labOutExt, lfn);
   if (LSave (lfn, trans, ofmt) < SUCCESS)
      HError (3214, "CalcConfFile: Cannot save file %s", lfn);

   if (trace & T_MEM) {
      printf("Memory State after processing lattice\n");
      PrintAllHeapStats();
   }
   ResetHeap (&transHeap);
   ResetHeap (&latHeap);
}
#endif


/* ---------------------------------------------------------------------- */
/*                    confusion network clustering                        */

/*------------------------------ data structures -------------------------*/

typedef struct _ConfNet ConfNet;
typedef struct _SCluster SCluster;
typedef struct _SCWord SCWord;

struct _SCWord {
   Word word;
   LogDouble post;
   HTime startT;
   HTime endT;
   SCWord *next;
};

struct _SCluster {
   int n;
   
   SCWord *arc;
   int *predBV;		/* precedence relation BitVector */
   
   HTime startT;		/* boundary time of the cluster */
   HTime endT;		/* i.e. 'outer' limits of link times */
#ifdef DEBUG_SANITY
   int status;		/* used only for debugging */
#endif
   SCluster *next;
   SCluster *prev;
};


struct _ConfNet {
   MemHeap *heap;
   int nClusters;
   int bvsize;
   SCluster *head;   /* doubly linked list with sentinels */
   SCluster *tail;
};



/*GE
	entry for a linked list holding a pair of links that are
	candidates for clustering and their distance score
 */

typedef struct _ClusterCand ClusterCand;

struct _ClusterCand {
   int l1;
   int l2;
   SCluster *sc1;
   SCluster *sc2;
   double score;
   ClusterCand *next;
};


/*------------------------------ prototypes ----------------------------*/
double SimScore(ConfNet *cn, SCluster *sc1, SCluster *sc2);


/*--------------------------bit vector code ----------------------------*/


/* ### these should really be defined as macros.
       unless the automatic inlining of the compiler works 
       really well that would be much faster.
*/

/* allocBV
     return pointer to new, EMPTY bitvector 
*/
int *allocBV(MemHeap *heap, int size) 
{
  int *bv;

  bv = (int *) New (heap, size * sizeof(int));
  memset(bv, 0, size * sizeof(int));

  return (bv);
}

/* copyBV
 */
void copyBV(int *src, int*dest, int size)
{
  int i;

  assert(dest);
  for(i=0; i<size; ++i)
    dest[i]=src[i];
}

/* orBV
   dest= src | dest
 */
void orBV(int *src, int *dest, int size)
{
  int i;

  assert(dest);
  for(i=0; i<size; ++i)
    dest[i]|=src[i];
}


/* setBV
   set bit i in bitvector (starting at 0)
 */
void setBV(int i, int *bv, int size)
{
  assert((i / (8*sizeof(int))) <=size);

  bv[i / (8*sizeof(int))] |= (1<<(i % (8*sizeof(int))));
}

/* getBV
   is bit i set in bitvector (starting at 0)?
 */
int getBV(int i, int *bv, int size)
{
  assert((i / (8*sizeof(int))) <=size);

  return (bv[i / (8*sizeof(int))] & (1<<(i % (8*sizeof(int)))));
}



/*
  overlapT
*/
HTime overlapT(HTime start1, HTime end1, HTime start2, HTime end2)
{
  HTime norm;
  /* no overlap? */
  if ((end2 <= start1)
      || (start2 >= end1))
    return 0.0;
  else {
    norm=end1-start1+end2-start2;
    /* 1	|---------|
       2      |--	*/
    if (start2 < start1 ) {
      /* 1	|---------|
	 2    |------|	    */
      if (end2 < end1)
	return ((end2 - start1)/norm);
      /* 1	|---------|
	 2    |-------------| */
      else
	return ((end1 - start1)/norm);
    }
    /* 1	|---------|
       2	    |---     */
    else {
      /*  1	|---------|
	  2	    |----|   */
      if (end2 < end1)
	return ((end2 - start2)/norm);
      /*  1	|---------|
	  2	    |-------| */
      else
	return ((end1 - start2)/norm);
    }
  }
}

/*
  overlapSC
*/
HTime overlapSC(SCluster *sc1, SCluster *sc2)
{
  return overlapT(sc1->startT, sc1->endT,
		  sc2->startT, sc2->endT);
}





/*-------------------------Levenshtein distance ------------------------*/

/*GE  cell in grid for Levenshtein DP alignment -- from OracleLM.c */
typedef struct cell
{
  int score;		/* weighted alignment score */
  int ins;		/* Total number of insertions */
  int del;		/* Total number of deletions */
  int sub;		/* Total number of substitutions */
  int cor;		/* Total number of matches */
} Cell;

/*GE
	- cache entry for phonetic similarity values
	- currently setup as a linked list. 

	### This should really be a hash table
 */
typedef struct simcache {
   Word w1;
   Word w2;
   double dist;
   struct simcache *next;
} SimCacheEntry;

#define MAXNPHONES 100
static Cell **SimGrid=NULL;
static SimCacheEntry *SimCache=NULL; 


/*GE
  InitSimScore()
	- initialise grid for DP alignment and cache for similarity scores

*/
void InitSimScore(void)
{
   int i;
   
   if ((SimGrid = (Cell **)malloc((MAXNPHONES+1)*sizeof(Cell*))) == NULL)
      HError(4123,"InitSimScore: Cannot create grid pointer array");
   for (i=0; i<=MAXNPHONES;i++)
      if ((SimGrid[i] = (Cell*) calloc(MAXNPHONES+1,sizeof(Cell))) == NULL)
         HError(4123,"InitSimScore: Cannot create grid column %d",i);
   
   /* setup error weights */
   subPen = 2;
   delPen = 1;
   insPen = 1;
   
   SimCache=NULL;
}

/*GE
	CalcWER()

	- perform Levenshtein DP alignment of hyp & ref
	- code mostly nicked from OracleLM.c
	- changed to use global grid memory, to avoid tons of 
	  alloc/free for each alignment
*/
Cell CalcWER(Cell **grid, int hypSize, LabId *hyp, int refSize, LabId *ref)
{
   Cell *gridi,*gridi1,ans;
   int h,d,v,i,j;
   
#if 0
   if ((grid = (Cell **)malloc((hypSize+1)*sizeof(Cell*))) == NULL)
      HError(2,"CalcWER: Cannot create grid pointer array");
   for (i=0; i<=hypSize;i++)
      if ((grid[i] = (Cell*) calloc(refSize+1,sizeof(Cell))) == NULL)
         HError(2,"CalcWER: Cannot create grid column %d",i);
#endif
   
   grid[0][0].score = grid[0][0].cor = 0;
   grid[0][0].ins = grid[0][0].del = grid[0][0].sub = 0;
   
   for (i=1;i<=hypSize;i++) {
      grid[i][0] = grid[i-1][0];
      grid[i][0].score += insPen;
      ++grid[i][0].ins;
   }
   for (i=1;i<=refSize;i++) {
      grid[0][i] = grid[0][i-1];
      grid[0][i].score += delPen;
      ++grid[0][i].del;
   }
   
   for (i=1;i<=hypSize;i++){
      gridi = grid[i]; gridi1 = grid[i-1];
      for (j=1;j<=refSize;j++) {
         h = gridi1[j].score +insPen;
         d = gridi1[j-1].score;
         if (ref[j-1] != hyp[i-1])
            d += subPen;
         v = gridi[j-1].score + delPen;
         if (v <= d && v <= h) {
            gridi[j] = gridi[j-1];
            gridi[j].score = v;
            ++gridi[j].del;
         }
         else if (d <= h) {
            gridi[j] = gridi1[j-1];
            gridi[j].score = d;
            if (ref[j-1] == hyp[i-1])
               ++gridi[j].cor;
            else
               ++gridi[j].sub;
         }
         else {
            gridi[j] = gridi1[j];
            gridi[j].score = h;
            ++ gridi[j].ins;
         }
      }
   }
   
   ans=grid[hypSize][refSize];
   
#if 0
   for (i=0;i<=hypSize;i++)
      free(grid[i]);
   free(grid);
#endif
   
   return(ans);
}


/*GE
  PronDist()
  
  - calculate phonetic distance between canonical pronunciations
  of two words.
 */
double PronDist(Pron p1, Pron p2)
{
   LabId *phones1, *phones2;
   int np1, np2;
   Cell align;
   double dist=0.0;
   
   assert(p1);
   assert(p2);
   
   phones1 = p1->phones;
   np1 = p1->nphones;
   phones2 = p2->phones;
   np2 = p2->nphones;
   
#if 0
   if (np2 > np1) {
      npt=np1;  pt=p1;
      np1=np2;  p1=p2;
      np2=npt;  p2=pt;
   }
#endif
   
   assert (SimGrid);
   align = CalcWER (SimGrid, np1, phones1, np2, phones2);
   dist = align.score / ((double) (np1+np2));
   
   assert (dist <= 1.0);
   return (1.0 - dist);
}

/*GE
  WordDist()
*/
double WordDist(Word w1, Word w2)
{
   double dist=0.0;
   SimCacheEntry *sce;
   Pron p1, p2;

   if (w1 == w2)
      return (1.0);
   
   if (w1 < w2) {
      Word temp;
      temp = w1;
      w1 = w2;
      w2 = temp;
   }
   
   /* look in cache */
   /* ### optimise by using hash tables */
   for (sce = SimCache; sce; sce = sce->next)
      if ((sce->w1 == w1) && (sce->w2 == w2))
         break;
   
   if (sce) {
      return (sce->dist);
   }
   
   p1 = w1->pron;
   p2 = w2->pron;
   
   if (p1 && p2) {
      dist = PronDist(p1, p2);
   }
   else { 
#if 1
      printf("NO Pronunciation for %s %s\n", w1->wordName->name, w2->wordName->name);
#endif
      dist=1.0;
   }
   assert (dist >= 0.0);
   
   /* enter w1/w2 dist into cache */
   sce = (SimCacheEntry *) New (&gcheap, sizeof(SimCacheEntry));
   
   sce->w1 = w1;
   sce->w2 = w2;
   sce->dist = dist;
   sce->next = SimCache;
   SimCache = sce;
   
   return (dist);
}

/*GE
	SimScore()
	
	- calculate similarity of two SClusters, handle averaging and stuff
 */
double SimScore(ConfNet *cn, SCluster *sc1, SCluster *sc2)
{
   int nw1, nw2, n;
   SCWord *scw1, *scw2;
   double wd=0, dist=0.0;
   
   n=0;
   for (nw1=0, scw1 = sc1->arc; scw1; ++nw1, scw1 = scw1->next) {
      for (nw2=0, scw2 = sc2->arc; scw2; ++nw2, scw2 = scw2->next) {
         ++n;
         /*
           printf("SimScore: %s (%ld)  %s (%ld)\n", 
           scw1->sym->name, scw1->sym,
           scw2->sym->name, scw2->sym);
         */
         wd = WordDist(scw1->word, scw2->word);
         
         /* #### EXPT:  GE  25.11.99 */
         /*  disabled for now. This should become a command line option */
#if 1
         dist += wd * L2F(scw1->post) * L2F(scw2->post);
#else
         dist += wd;
#endif
#if 0
         printf("SImScore %-9s %-9s %.4f %e    %f %f ", 
                scw1->sym->name, scw2->sym->name,
                wd, dist,
                scw1->posterior, scw2->posterior);
         if (getBV(sc1->n, sc2->predBV, cn->bvsize)
             || getBV(sc2->n, sc1->predBV, cn->bvsize))
            printf(">>>\n");
         else
            printf("\n");
         
#endif	    
      }
   }
   
#if 0
   printf("      %f\n", (dist/n));
#endif
   return (dist/n);
}


/*------------------------------ the code ------------------------------*/

SCWord *NewSCWord(MemHeap *heap)
{
   SCWord *scw;

   scw = (SCWord *) New (heap, sizeof (SCWord));

   scw->word = NULL;
   scw->post = 0.0;
   scw->startT = scw->endT = 0.0;
   scw->next = NULL;

   return scw;
}

SCluster *NewSCluster(MemHeap *heap)
{
   SCluster *sc;

   sc = (SCluster *) New (heap, sizeof (SCluster));
   sc->n = -1;
   sc->arc = NULL;
   sc->predBV = NULL;
   sc->startT = sc->endT = 0.0;
   sc->prev = sc->next = NULL;

   return sc;
}

/* insert b after a */
void InsertSC (SCluster *a, SCluster *b)
{
   
   b->next = a->next ;
   b->prev = a;
   a->next = b;
   b->next->prev = b;
}


void CalcPosteriors (Lattice *lat)
{
   /* store arc posteriors in la->score */
   int i;
   LArc *la;
   LogDouble pX;        /* prob of data;  p(X) = alpha(final) = beta(root)  */

   LatAttachInfo (&latHeap, sizeof (FBinfo), lat);

   pX = LatForwBackw (lat, LATFB_SUM);
   
   for (i = 0; i < lat->na; ++i) {
      la = &lat->larcs[i];
      la->score = LArcPosterior (lat, la) - pX;
   }
}

static int la_cmp(const void *v1,const void *v2)
{
   LArc *la1,*la2;
   HTime t;

   la1 = *((LArc **) v1);
   la2 = *((LArc **) v2);

   t = la1->start->time - la2->start->time;
   if (t != 0) {
      if (t > 0) 
         return 1;
      else
         return -1;
   }
   else {
      t = la1->end->time - la2->end->time;
      if (t != 0) {
         if (t > 0) 
            return 1;
         else
            return -1;
      }
      else
         return ((int) (la1->end->word - la2->end->word));
   }
}

ConfNet *InitConfNet (MemHeap *heap, Lattice *lat)
{
   /* init clustering: combine arcs with same word&times into clusters */
   int i;
   ConfNet *cn;
   SCluster *sc;
   LArc *la, **orderLA;
   
   /* alloc & init ConfNet */
   cn = (ConfNet *) New (heap, sizeof (ConfNet));
   cn->nClusters = 0;
   cn->heap = heap;
   
   /* alloc sentinels */
   cn->head = NewSCluster (heap);
   cn->tail = NewSCluster (heap);
   cn->head->next = cn->tail;
   cn->tail->prev = cn->head;


   /*# sort arcs based on start time, end time & word ==> orderLA */
   orderLA = (LArc **) New (&gstack, lat->na * sizeof(LArc *));
   for (i = 0; i < lat->na; ++i)
      orderLA[i] = &lat->larcs[i];

   qsort (orderLA, lat->na, sizeof (LArc *), la_cmp);

   sc = NULL;
   for (i = 0; i < lat->na; ++i) {
      la = orderLA[i];
      if (sc && sc->arc->startT == la->start->time &&
          sc->arc->endT == la->end->time &&
          sc->arc->word == la->end->word) {       /* combine */
         
         /* add arc la to cluster sc */
         sc->arc->post = LAdd (sc->arc->post, la->score);
         
         la->hook = (Ptr) sc;
      }
      else {
         sc = NewSCluster (heap);
         InsertSC (cn->tail->prev, sc); /* insert at end */
         cn->nClusters++;
         sc->n = cn->nClusters;
         
         /* put arc la into cluster sc */
         sc->arc = NewSCWord (heap);
         sc->arc->word = la->end->word;
         sc->arc->post = la->score;
         sc->startT = sc->arc->startT = la->start->time;
         sc->endT = sc->arc->endT = la->end->time;
         
         la->hook = (Ptr) sc;
      }
   }

   Dispose (&gstack, orderLA);
   orderLA = NULL;

   /* initialise precedence bit vectors */

   cn->bvsize = ceil(cn->nClusters / (8.0 * sizeof (int)));
   i = 0;
   for (sc = cn->head->next; sc != cn->tail; sc = sc->next) {
      sc->predBV = allocBV (heap, cn->bvsize);
      ++i;
   }
   assert (i == cn->nClusters);

   return cn;
}


void CalcPrecedence (ConfNet *cn, Lattice *lat)
{
   /* calculate predecessor list for each SCluster */
   int i;
   LArc *la, *follLA;
   SCluster *sc, *follSC;
   LNode *ln, **topOrder;

   topOrder = (LNode **) New (&gstack, lat->nn * sizeof(LNode *));
   LatTopSort (lat, topOrder);

   for (i = 0; i < lat->nn; ++i) {
      ln = topOrder[i];         /* traverse in topological order */

      for (la = ln->foll; la; la = la->farc) {
         assert (la->hook);

         sc = (SCluster *) la->hook;
         /* for all outgoing arcs */
         for (follLA = la->end->foll; follLA; follLA = follLA->farc) {
            follSC = (SCluster *) follLA->hook;
            assert (sc != follSC);
            
            /* add sc's predecessors to follSC */
            orBV (sc->predBV, follSC->predBV, cn->bvsize);
            /* add sc itself */
            setBV (sc->n, follSC->predBV, cn->bvsize);
         }

         la->hook = NULL;       /* sanity check */
      }
   }
   Dispose (&gstack, topOrder);
}


void PrintConfNet (ConfNet *cn)
{
   int i;
   SCluster *sc;
   SCWord *scw;

   sc = cn->head;
   for (i = 0; i < cn->nClusters; ++i) {
      sc = sc->next;
      printf ("SC %d  %.2f -- %.2f\n", sc->n, sc->startT, sc->endT);
      for (scw = sc->arc; scw; scw = scw->next) {
         printf ("  SCW  %.2f -- %.2f  %f %s\n", scw->startT, scw->endT,
                 scw->post, scw->word->wordName->name);
      }
   }
   assert (sc->next == cn->tail);
}

/*GE
	InsertCC()
	- insert new cluster candidate into sorted list
*/
ClusterCand *InsertCC(ClusterCand *cc1, 
                      ClusterCand *ccList)
{
   ClusterCand *cc2;
   
   /* cc1 will be first */
   if (!ccList || (cc1->score > ccList->score)) {
      cc1->next = ccList;
      ccList = cc1;
   }
   else {
      /* find cc2 after which cc1 is to be inserted */
      for (cc2 = ccList;
           cc2->next && (cc2->next->score > cc1->score);
           cc2 = cc2->next)
         ;
      cc1->next = cc2->next;
      cc2->next = cc1;
   }
   
   return ccList;
}


/*GE
	FindClusterCand()
	- generate list of candidate pairs for clustering 

        pass 2: only consider clusters with the same word
        pass 3: cluster different words
        
*/
ClusterCand *FindClusterCand(ConfNet *cn, int pass)
{
   SCluster *sc1, *sc2;
   ClusterCand *cc, *ccList;
   int i, j, count=0;
   HTime overlap;
   
   /* consider all pairs of clusters (i,j)  with i<j, since metric is symmetric */

   ccList = NULL;
   for (i = 1, sc1 = cn->head->next; i <= cn->nClusters; ++i, sc1 = sc1->next) {
      for (j = i+1, sc2 = sc1->next; j <= cn->nClusters; ++j, sc2 = sc2->next) {

#if 0
         printf("i %d  %.2f -> %.2f  ", sc1->n, sc1->startT, sc1->endT);
         printf("j %d  %.2f -> %.2f\n", sc2->n, sc2->startT, sc2->endT);
#endif
         /* for inter=0 the clusters are still sorted by end time
            optimised by stopping at 
            sc2->end->time <sc1->start->time */
         
         if (pass == 2 && (sc2->startT >= sc1->endT))
            break;
         
         assert(sc1->startT <= sc1->arc->startT);
         assert(sc1->endT >= sc1->arc->endT);
         assert(sc2->startT <= sc2->arc->startT);
         assert(sc2->endT >= sc2->arc->endT);
         
         if ((pass >= 3) || (sc1->arc->word == sc2->arc->word)) {
            overlap = overlapSC (sc1, sc2);
            assert(overlap >= 0.0);
            
            if ((overlap > 0.0 || pass == 4)
                && (!getBV(sc1->n, sc2->predBV, cn->bvsize)
                    && !getBV(sc2->n, sc1->predBV, cn->bvsize))) {
               ++count;
               cc = (ClusterCand *) New (cn->heap, sizeof(ClusterCand));
               
               cc->l1 = i;
               cc->l2 = j;
               cc->sc1 = sc1;
               cc->sc2 = sc2;

               switch (pass) {
               case 2:
                  cc->score = overlap;
                  break;
               case 3:
               case 4:
                  cc->score = SimScore (cn, sc1, sc2);
                  break;
               default:
                  abort();
                  break;
               }
               if (trace&T_CN)
                  printf ("cluster %d %d %f\n", cc->l1, cc->l2, cc->score);
               
               ccList = InsertCC (cc, ccList);
            }
         }
      } /* for j */
   } /* for i */

   if (trace & T_CN) {
      printf("FindClusterCandidate: found %d candidate pairs\n", count);
   }
   
#ifdef DEBUG_SANITY
   for (clust1=clustList; clust1; clust1=clust1->next)
      --count;
   assert(count==0);
#endif	
   
   return ccList;
}


/*GE
	MergeSClusters()

	- merge two SClusters if they are different and not
	  in relation
	- update precedence relation on all SClusters affected
	- do NOT update the foll/pred pointers -- it's a wste of time
	  we rely on the precednce bitvectors instead.
*/
int MergeSClusters (ConfNet *cn, SCluster *sc1, SCluster *sc2, Boolean merge)
{
   int i;
   SCluster *sca;
   /*  SCList *scl1, *scl2; */

   /* clusters equal? */
   if (sc1 == sc2) {
      printf("clusters are equal!\n");
      return 0;  /* not merged */
   }
   
   /* does sc1 preceed sc2 or vice versa? */
   if ((getBV(sc1->n, sc2->predBV, cn->bvsize)
	|| getBV(sc2->n, sc1->predBV, cn->bvsize))) {
      printf("clusters are in order!\n");
      return 0; /* not merged */
   }  
   
   
#if 0
   printf ("MERGE %d %.2f -- %.2f  %d %.2f -- %.2f  %s\n", sc1->n, sc1->startT, sc1->endT,
           sc2->n, sc2->startT, sc2->endT, sc1->arc->word->wordName->name);
#endif

   if (merge) {
      /* merge word entries */
      assert(sc1->arc->word == sc2->arc->word);
      sc1->arc->post = LAdd(sc1->arc->post, sc2->arc->post);
   }
   else {
      /* chain sc2's arcs to the end of sc1's linked list */
      /* ### this should really combine entries for the same word 
         ### can that really still happen at this stage??	 */
      SCWord *t;
      for (t = sc1->arc; t->next; t = t->next)
         ;
      t->next = sc2->arc;
      sc2->arc = NULL;
   }
   
   /* combine BVs */
   orBV (sc2->predBV, sc1->predBV, cn->bvsize);
   /* now sc1->predBV lists all predecessors of sc1 AND sc2 
      we add these to all successors of the old sc1 and sc2 
   */
   
   /* propagate BV forward from sc1:
      :	we don't need to worry about calculating  transitive 
      :	closures or traversing the clusters in the right order 
      :	as we just add the predBV of sc2 to sc1's successors 
      :	and vice versa.	 */
   /* this loop is hopefully faster than a recursive traversal */
   for (i = 0, sca = cn->head->next; i < cn->nClusters; ++i, sca = sca->next) {
      if (sca->n >= 0) {
         if (getBV (sc1->n, sca->predBV, cn->bvsize)){
            /* sca succeeds sc1! */
            orBV (sc1->predBV, sca->predBV, cn->bvsize);
            /*	setBV(sc2->n, sca->predBV, cn->bvsize); */
         }
         if (getBV (sc2->n, sca->predBV, cn->bvsize)){
            /* sca succeeds sc2! */
            orBV (sc1->predBV, sca->predBV, cn->bvsize);
            setBV (sc1->n, sca->predBV, cn->bvsize);
         }
      }
   }
   
   /* deactivate sc2 */
   sc2->n = -1;
   

   
   /* update times */
   if (sc2->startT < sc1->startT)
      sc1->startT = sc2->startT;
   if (sc2->endT > sc1->endT)
      sc1->endT = sc2->endT;
   

   sc2->prev->next = sc2->next;
   sc2->next->prev = sc2->prev;

   --cn->nClusters;
   /* #### delete sc2 */

   return 1;
}

/*GE
	UpdateClusterCand()
	- loop over list of clustering candidates 
	  and update all entries which were affected by 
	  last merger (sca/scb)
*/
ClusterCand *UpdateClusterCand(ConfNet *cn, ClusterCand *cl, 
				    SCluster *sca, SCluster *scb)
{
   ClusterCand *cc1, *cc2, *newcc=NULL;
   SCluster *sc1, *sc2;
   double dist;
   int count=0;
   int ncount=0;
   int dcount=0;

   for (cc1=cl; cc1; cc1=cc1->next) {
      /* sc1 and sc2 were cluster HEADS before combining sca and scb  */ 
      
      sc1=cc1->sc1;
      sc2=cc1->sc2;
      ++count;

      if ((sc1==sca) || (sc2==sca)
          ||(sc1==scb) || (sc2==scb)){
         /* entry was affected by last clustering */

         if (sc1 == scb)
            sc1 = cc1->sc1 = sca;
         if (sc2 == scb)
            sc2 = cc1->sc2 = sca;


         if ((sc1 == sc2)
             || getBV(sc1->n, sc2->predBV, cn->bvsize)
             || getBV(sc2->n, sc1->predBV, cn->bvsize)) {
            /* clusters are equal or in precedence => delete entry */
            cc1->l1 = cc1->l2=-1;
            cc1->sc1 = cc1->sc2 = NULL;
            ++dcount;
            /* #### delete entry! */
         }
         else {
            /* calc new score */
            dist = SimScore(cn, sc1, sc2);

            cc2 = (ClusterCand *) New (cn->heap, sizeof(ClusterCand));

            cc2->next = newcc;
            cc2->l1 = sc1->n;
            cc2->l2 = sc2->n;
            cc2->sc1 = sc1;
            cc2->sc2 = sc2;

            cc2->score = dist;
            newcc = cc2;
            
            cc1->l1 = cc1->l2 = -1;
            cc1->sc1 = cc1->sc2 = NULL;
            /* #### delete entry! */
            
            ++ncount;
         }
      }
   } /* for cc1 */
   
   
   if (trace&T_CN)
      printf("UCC:  %d entries and %d new entries   %d entries deleted\n",
             count, ncount, dcount);
   
   /* ## maybe delete deactivated entries here, instead of skipping 
      them in ProcessClusterCand? 
   */
   
   /* insert newcc entries into list */
   for(cc1 = newcc; cc1; cc1 = cc2) {
      cc2 = cc1->next;
      cc1->next = NULL;
      cl = InsertCC (cc1, cl);
   }

   count = 0;
   for (cc1 = cl; cc1; cc1 = cc2) {
      cc2 = cc1->next; 
      while (cc2 && (!cc2->sc1 || !cc2->sc2
                        || (cc2->sc1 == cc1->sc1 && cc2->sc2 == cc1->sc2))) {
         cc2 = cc2->next;
         ++count;
      }
      cc1->next = cc2;
   }
   if (trace&T_CN)
      printf ("purged %d entries\n", count);


   return cl;
}


/*
  ProcessClusterCand

*/
int ProcessClusterCand (ConfNet *cn, ClusterCand *ccList, int pass)
{ 
   ClusterCand *cc1, *cc2;
   SCluster *sc1, *sc2;
   int count;

   count=0;
   for (cc1 = ccList; cc1; cc1 = cc2) {
      if ((cc1->l1 != -1) && (cc1->l2 != -1)) {
        
         /* combine the two CLUSTERS corresponding to this entry */
         sc1 = cc1->sc1;
         sc2 = cc1->sc2;
         
         cc2 = cc1->next;
         
         if ((sc1 != sc2)
             && !(getBV(sc1->n, sc2->predBV, cn->bvsize)
                  || getBV(sc2->n, sc1->predBV, cn->bvsize))) {
#if 0
            printf("merge %d/%d %s (%.2f -- %.2f)",
                   clust1->l1, sc1->n, sc1->word->sym->name,
                   sc1->tstart, sc1->tend);
            printf(" %d/%d %s (%.2f -- %.2f)\n",
                   clust1->l2, sc2->n, sc2->word->sym->name,
                   sc2->tstart, sc2->tend);
#endif	   
            count += MergeSClusters (cn, sc1, sc2, pass==2 ? TRUE : FALSE);
            
            if (pass >= 2) /* #### maybe (inter>=1) ? */
               cc2 = UpdateClusterCand (cn, cc2, sc1, sc2);
         }
         else{
#if 0
            printf("skip %d/%d %s %.2f -- %.2f   j %d/%d %s %.2f -- %.2f\n", 
                   clust1->l1, sc1->n, sc1->word->sym->name,
                   sc1->word->start, sc1->word->end,
                   clust1->l2, sc2->n, sc2->word->sym->name,
                   sc2->word->start, sc2->word->end);
#endif
         }	   
      }
      else  /* deactivated entry */
         cc2 = cc1->next;
      
   }
   
   if (trace&T_CN)
      printf("ProcessClusterCand: combined %d links\n", count);
   
   return count;
}


void PruneConfNet (ConfNet *cn, LogDouble thresh)
{
   int pCount = 0;
   SCluster *sc;
   
   for (sc = cn->head->next; sc != cn->tail; sc = sc->next) {
      if (sc->arc->post < thresh) {
         ++pCount;
         sc->prev->next = sc->next;
         sc->next->prev = sc->prev;
      }
   }
   if (trace & T_CN)
      printf ("PruneConfNet: pruned %d entries\n", pCount);

   cn->nClusters -= pCount;
}


/* sc_cmp

     helper function for SortConfNet, called from qsort()
*/
static int bvSize = 0;      /* hack! */
static int sc_cmp(const void *v1,const void *v2)
{
   SCluster *sc1, *sc2;

   sc1 = *((SCluster **) v1);
   sc2 = *((SCluster **) v2);

   if (getBV (sc1->n, sc2->predBV, bvSize))
      return -1;
   else
      return 1;
}

/*
  SortConfNet

    sort the SCluster in ConfNet into precedence order.
*/
void SortConfNet (ConfNet *cn)
{ 
   int i;
   SCluster **scArray, *sc1;

   scArray = (SCluster **) New (cn->heap, cn->nClusters * sizeof (SCluster *));
   for (i = 0, sc1 = cn->head->next; sc1 != cn->tail; sc1 = sc1->next, ++i) {
      scArray[i] = sc1;
   }
   assert (i == cn->nClusters);

   if (cn->nClusters == 1)
      return;           /* no need to sort */

   bvSize = cn->bvsize;
   qsort (scArray, cn->nClusters, sizeof (SCluster *), sc_cmp);

   assert (cn->nClusters >= 1);

   cn->head->next = scArray[0];
   scArray[0]->prev = cn->head;
   scArray[0]->next = scArray[1];
   for (i = 1; i < cn->nClusters-1; ++i) {
      sc1 = scArray[i];
      sc1->prev = scArray[i-1];
      sc1->next = scArray[i+1];
   }

   i = cn->nClusters-1;
   cn->tail->prev = scArray[i];
   scArray[i]->prev = scArray[i-1];
   scArray[i]->next = cn->tail;
   
   Dispose (cn->heap, scArray);
   
#if 1           /* sanity check */
   /* check whether SClusters are in correct order */
   for (sc1 = cn->head->next; sc1 != cn->tail; sc1 = sc1->next) {
      if ((sc1->prev != cn->head) 
          && (!getBV(sc1->prev->n, sc1->predBV, cn->bvsize))) {
         printf("##### ERRROR sc %d does not precede sc %d\n",
                sc1->prev->n, sc1->n);
      }
   }
#endif
}


ConfNet *ClusterLat2ConfNet (MemHeap *heap, Lattice *lat)
{
   ConfNet *cn;
   ClusterCand *ccList;

   CalcPosteriors (lat);
      
   /* init clustering: combine arcs with same word&times into clusters */
   cn = InitConfNet (heap, lat);
   
   CalcPrecedence (cn, lat);
   
   PruneConfNet (cn, -10.0);

   /* pass 2 */
   if (trace&T_CN) {
      printf ("pass 2...");
      fflush (stdout);
   }
   ccList = FindClusterCand (cn, 2);
   ProcessClusterCand (cn, ccList, 2);
   
   /* PruneConfNet (cn, -5.0); */
   PruneConfNet (cn, confNetPrune);
   if (trace & T_CN)
      PrintConfNet (cn);

   /* pass 3 */
   if (trace&T_CN) {
      printf ("pass 3...");
      fflush (stdout);
   }
   ccList = FindClusterCand (cn, 3);
   ProcessClusterCand (cn, ccList, 3);


   /* pass 4 -- why do I need this? */
   ccList = FindClusterCand (cn, 4);
   if (ccList) {
      if (trace&T_CN) {
         printf ("pass 4...");
         fflush (stdout);
      }
      ProcessClusterCand (cn, ccList, 3);
   }   

   if (trace&T_CN) {
      printf ("done\n");
      fflush (stdout);
   }

   SortConfNet (cn);
   
   if (trace&T_CN)
      PrintConfNet (cn);

   return cn;
}


Transcription *TranscriptionFromConfNet (ConfNet *cn)
{
   Transcription *trans;
   LabList *lList;
   LLink lab;
   SCluster *sc;
   SCWord *scw, *bestSCW;
   LogDouble logsum, bestSCWlogpost;
   double post;
   int i;

   trans = CreateTranscription (&transHeap);
   lList = CreateLabelList (&transHeap, 0);

   sc = cn->head;
   for (i = 0; i < cn->nClusters; ++i) {
      sc = sc->next;

      bestSCW = NULL;
      bestSCWlogpost = LZERO;
      logsum = LZERO;

      /* find best word and calculate sum of posteriors */
      for (scw = sc->arc; scw; scw = scw->next) {
         logsum = LAdd (logsum, scw->post);
         if (scw->post > bestSCWlogpost) {
            bestSCWlogpost = scw->post;
            bestSCW = scw;
         }
      }

      post = L2F (bestSCWlogpost);
      /* word more likely than deleting this SC alltogether? */
      if (post > (1 - L2F(logsum))) {
         /* #### take outSym from first pron? */
         if (bestSCW->word->pron && bestSCW->word->pron->outSym) {

            lab = CreateLabel (&transHeap, 0);
            lab->labid = bestSCW->word->pron->outSym;
            lab->score = post;
            lab->start = sc->startT * 1.0e7;
            lab->end = sc->endT * 1.0e7;
            
            /* insert at end of label list */
            lab->pred = lList->tail->pred;
            lab->succ = lList->tail;
            lab->succ->pred = lab->pred->succ = lab;
         }
      }
   }
   assert (sc->next == cn->tail);

   /* fix times in lList */
   {
      HTime nextstart=0.0;
      for (lab = lList->head; lab; lab = lab->succ) {
         if (lab->labid) { /* ignore sentinels */
            lab->start = nextstart;
            
            if(lab->succ->labid)
               nextstart = lab->succ->start;
            else
               nextstart = lab->end;
            
            /* average current end and next start */
            lab->end = (lab->end + nextstart) / 2.0;
            if (lab->end < lab->start)
               lab->end = lab->start+1;
            
            nextstart = lab->end;
         }
      }
   }

   AddLabelList (lList, trans);
   return trans;
}

void WriteConfnet (ConfNet *cn, char *fn)
{
   FILE *SCF;
   Boolean isPipe;
   SCluster *sc1;
   SCWord *scw;
   LogDouble sum;
   int k;
   
   SCF=FOpen (fn, NetOFilter, &isPipe);
   if (!SCF)
      HError (4110, "cannot open scf file '%s'", fn);
   
   /* store  number of Sausages */
   OutputIntField('N', cn->nClusters, FALSE, "%d", SCF);
   fprintf(SCF, "\n");
   
   /* for historical reasons the SCFiles contain the SClusters
      in reverse order! */
   for (sc1 = cn->tail->prev; sc1 != cn->head; sc1 = sc1->prev) {
      
      /* store number of words in Sausage */
      sum=LZERO;
      for (k = 0, scw = sc1->arc; scw; scw = scw->next) {
         ++k;
         sum = LAdd (sum, scw->post);
      }
      if ((sum < 0.0) && (addNullWord))
         ++k;
      
      OutputIntField('k', k, FALSE, "%d", SCF);
      fprintf(SCF, "\n");
      
      for (scw = sc1->arc; scw; scw = scw->next) {
         if (fprintf(SCF, "W=%-19s ", 
                     ReWriteString (scw->word->wordName->name, 
                                    NULL, ESCAPE_CHAR)) < 0)
            HError (4114, "WriteConfnet: write failed on scf file");
         OutputFloatField('s', scw->startT, FALSE, "%-7.2f", SCF);
         OutputFloatField('e', scw->endT, FALSE, "%-7.2f", SCF);
         OutputFloatField('p', scw->post, FALSE, "%-9.5f", SCF);
         fprintf(SCF, "\n");
      }
      
      if ((sum < 0.0)  && (addNullWord)) {
         /* output !NULL link */
         if (fprintf(SCF, "W=%-19s ", 
                     ReWriteString(vocab.nullWord->wordName->name,
                                   NULL, ESCAPE_CHAR)) < 0)
            HError (4114, "WriteConfnet: write failed on scf file");
         OutputFloatField('s', sc1->arc->startT, FALSE, "%-7.2f", SCF);
         OutputFloatField('e', sc1->arc->endT, FALSE, "%-7.2f", SCF);
         OutputFloatField('p', LSub(0.0, sum), FALSE, "%-9.5f", SCF);
         fprintf(SCF, "\n");
      }
   }
   
   FClose (SCF, isPipe);
}

void ConfNetClusterFile (char *latfn_in, char *latfn_ou, char *labfn_ou)
{
   Lattice *lat;
   char lfn[MAXSTRLEN];
   FILE *lf;
   Boolean isPipe;
   ConfNet *cn;
   Transcription *trans;

   if (trace & T_MEM) {
      printf("Memory State before processing confnet\n");
      PrintAllHeapStats();
   }


   MakeFN(latfn_in, latInDir, latInExt, lfn);
  
   if ((lf = FOpen(lfn,NetFilter,&isPipe)) == NULL)
      HError(4110,"HLConf: Cannot open Lattice file %s", lfn);
  
   lat = ReadLattice (lf, &latHeap, &vocab, FALSE, FALSE);
   FClose(lf, isPipe);
   if (!lat)
      HError (4113, "HLConf: can't read lattice");

   /* cz277 - scale conf score */
   /*lat->lmscale = lmScale;
   lat->wdpenalty = wordPen;
   lat->acscale = acScale;
   lat->prscale = prScale;*/
   lat->lmscale = lmScale * latScoreScale;
   lat->wdpenalty = wordPen * latScoreScale;
   lat->acscale = acScale * latScoreScale;
   lat->prscale = prScale * latScoreScale;

   if (fixPronProb)
      FixPronProbs (lat, &vocab);

   if (clampACLike)
      ClampACLike (lat);

   LatCheck (lat);

   cn = ClusterLat2ConfNet (&cnHeap, lat);

   trans = TranscriptionFromConfNet (cn);

   /* save 1-best transcription */
   /* the following is from HVite/HDecode */
   if (trans) {
      char labOutfn[MAXSTRLEN];
      
      if (labOutForm != NULL)
         FormatTranscription (trans, 100, FALSE, FALSE, /* #### fix frameDur! */
                              strchr(labOutForm,'X')!=NULL,
                              strchr(labOutForm,'N')!=NULL,strchr(labOutForm,'S')!=NULL,
                              strchr(labOutForm,'C')!=NULL,strchr(labOutForm,'T')!=NULL,
                              strchr(labOutForm,'W')!=NULL,strchr(labOutForm,'M')!=NULL);
      
      MakeFN (labfn_ou, labOutDir, labOutExt, labOutfn);

      if (LSave (labOutfn, trans, ofmt) < SUCCESS)
         HError(4114, "ConfNetClusterFile: Cannot save file %s", labOutfn);
      if (trace & T_TRAN) {
         PrintTranscription (trans, "1-best hypothesis");
         printf ("\n");
      }
   }

   if (writeConfNet) {
      char scfFN[MAXSTRLEN];

      MakeFN (latfn_ou, labOutDir, "scf", scfFN);

      WriteConfnet (cn, scfFN);

   }

   if (trace & T_MEM) {
      printf("Memory State after processing confnet\n");
      PrintAllHeapStats();
   }
   
   ResetHeap (&latHeap);
   ResetHeap (&cnHeap);
   ResetHeap (&transHeap);
}


/* ------------------------- End of HLConf.c ------------------------- */

