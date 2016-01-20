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
/* ----------------------------------------------------------- */
/*           Copyright: Cambridge University                   */
/*                      Engineering Department                 */
/*            2000-2015 Cambridge, Cambridgeshire UK           */
/*                      http://www.eng.cam.ac.uk               */
/*                                                             */
/*   Use of this software is governed by a License Agreement   */
/*    ** See the file License for the Conditions of Use  **    */
/*    **     This banner notice must not be removed      **    */
/*                                                             */
/* ----------------------------------------------------------- */
/*         File: HDecode.c  HTK large vocabulary decoder       */
/* ----------------------------------------------------------- */

char *hdecode_version = "!HVER!HDecode:   3.5.0 [CUED 12/10/15]";
char *hdecode_sccs_id = "$Id: HDecode.c,v 1.2 2015/10/12 12:07:24 cz277 Exp $";

/* this is just the tool that handles command line arguments and
   stuff, all the real magic is in HLVNet and HLVRec */


#include "config.h"
#ifdef IMKL
#include "mkl.h"
#endif
#ifdef CUDA
#include "HCUDA.h"
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
#include "HUtil.h"
#include "HTrain.h"
#include "HAdapt.h"
#include "HNet.h"       /* for Lattice */
#include "HArc.h"
#include "HLat.h"       /* for Lattice */
#include "HFBLat.h"
#include "HNCache.h"

#include "lvconfig.h"

#include "HLVNet.h"
#include "HLVRec.h"
#include "HLVLM.h"

#include <time.h>

/* -------------------------- Trace Flags & Vars ------------------------ */

#define T_TOP 00001		/* Basic progress reporting */
#define T_OBS 00002		/* Print Observation */
#define T_ADP 00004		/* Adaptation */
#define T_MEM 00010		/* Memory usage, start and finish */

static int trace = 0;

/* -------------------------- Global Variables etc ---------------------- */


static char *langfn;		/* LM filename from commandline */
static char *dictfn;		/* dict filename from commandline */
static char *hmmListfn;		/* model list filename from commandline */
static char *hmmDir = NULL;     /* directory to look for HMM def files */
static char *hmmExt = NULL;     /* HMM def file extension */

static FileFormat ofmt = UNDEFF;	/* Label output file format */
static char *labDir = NULL;	/* output label file directory */
static char *labExt = "rec";	/* output label file extension */
static char *labForm = NULL;	/* output label format */

static Boolean latRescore = FALSE; /* read lattice for each utterance and rescore? */
static char *latInDir = NULL;   /* lattice input directory */
static char *latInExt = "lat";  /* latttice input extension */
static char *latFileMask = NULL; /* mask for reading lattice */
/* from mjfg, cz277 - 141022 */
static char *latOFileMask = NULL; /* mask for writing lattice */
static char *labOFileMask = NULL; /* mask for writing labels */

static Boolean latGen = FALSE;  /* output lattice? */
static char *latOutDir = NULL;  /* lattice output directory */
static char *latOutExt = "lat"; /* latttice output extension */
static char *latOutForm = NULL;  /* lattice output format */

static FileFormat dataForm = UNDEFF; /* data input file format */

static Vocab vocab;		/* wordlist or dictionary */
static HMMSet hset;		/* HMM set */
static FSLM *lm;                /* language model */
static LexNet *net;             /* Lexicon network of all required words/prons */

static char *startWord = "<s>"; /* word used at start of network */
static LabId startLab;          /*   corresponding LabId */
static char *endWord = "</s>";  /* word used at end of network */
static LabId endLab;            /*   corresponding LabId */

static char *spModel = "sp";    /* model used as word end Short Pause */
static LabId spLab;             /*   corresponding LabId */
static char *silModel = "sil";  /* model used as word end Silence */
static LabId silLab;            /*   corresponding LabId */

static Boolean silDict = FALSE; /* does dict contain -/sp/sil variants with probs */

static LogFloat insPen = 0.0;   /* word insertion penalty */

static float acScale = 1.0;     /* acoustic scaling factor */
static float pronScale = 1.0;   /* pronunciation scaling factor */
static float lmScale = 1.0;     /* LM scaling factor */

static int maxModel = 0;        /* max model pruning */
static LogFloat beamWidth = - LZERO;     /* pruning global beam width */
static LogFloat weBeamWidth = - LZERO;   /* pruning wordend beam width */
static LogFloat zsBeamWidth = - LZERO;   /* pruning z-s beam width */
static LogFloat relBeamWidth = - LZERO;  /* pruning relative beam width */
static LogFloat latPruneBeam = - LZERO;  /* lattice pruning beam width */
static LogFloat latPruneAPS = 0;;        /* lattice pruning arcs per sec limit */

static LogFloat fastlmlaBeam = - LZERO;  /* do fast LM la outside this beam */

static int nTok = 32;           /* number of different LMStates per HMM state */
static Boolean useHModel = FALSE; /* use standard HModel OutP functions */
static int outpBlocksize = 1;   /* number of frames for which outP is calculated in one go */
static Observation *obs;        /* array of Observations */

/* cz277 - ANN */
/*static int batchSamples;*/
static LabelInfo labelInfo;
static DataCache *cache[SMAX];

/* transforms/adaptatin */
/* information about transforms */
static XFInfo xfInfo;


/* info for comparing scores from alignment of 1-best with search */
static char *bestAlignMLF;      /* MLF with 1-best alignment */

/* -------------------------- Heaps ------------------------------------- */

static MemHeap modelHeap;
static MemHeap netHeap;
static MemHeap lmHeap;
static MemHeap inputBufHeap;
static MemHeap transHeap;
static MemHeap regHeap;
/* cz277 - ANN */
static MemHeap cacheHeap;

/* -------------------------- Prototypes -------------------------------- */
void SetConfParms (void);
void ReportUsage (void);
DecoderInst *Initialise (void);
void DoRecognition (DecoderInst *dec, char *fn);
Boolean UpdateSpkrModels (char *fn);

/* ---------------- Configuration Parameters ---------------------------- */

static ConfParam *cParm[MAXGLOBS];
static int nParm = 0;		/* total num params */


/* ---------------- Debug support  ------------------------------------- */

#if 0
FILE *debug_stdout = stdout;
FILE *debug_stderr = stderr;
#endif

/* ---------------- Process Command Line ------------------------- */

/* SetConfParms: set conf parms relevant to this tool */
void
SetConfParms (void)
{
   int i;
   double f;
   Boolean b;
   char buf[MAXSTRLEN];

   nParm = GetConfig ("HDECODE", TRUE, cParm, MAXGLOBS);
   if (nParm > 0) {
      if (GetConfInt (cParm, nParm, "TRACE", &i))
	 trace = i;
      if (GetConfStr (cParm, nParm, "STARTWORD", buf))
         startWord = CopyString (&gstack, buf);
      if (GetConfStr (cParm, nParm, "ENDWORD", buf))
         endWord = CopyString (&gstack, buf);
      if (GetConfFlt (cParm, nParm, "LATPRUNEBEAM", &f))
         latPruneBeam  = f;
      if (GetConfFlt (cParm, nParm, "FASTLMLABEAM", &f))
         fastlmlaBeam  = f;
      if (GetConfFlt (cParm, nParm, "LATPRUNEAPS", &f))
         latPruneAPS  = f;
      if (GetConfStr (cParm, nParm, "BESTALIGNMLF", buf))
         bestAlignMLF = CopyString (&gstack, buf);
      if (GetConfBool (cParm, nParm, "USEHMODEL",&b)) useHModel = b;
      if (GetConfStr(cParm,nParm,"LATFILEMASK",buf)) {
         latFileMask = CopyString(&gstack, buf);
      }
      /* from mjfg, cz277 - 141022 */
      if (GetConfStr(cParm,nParm,"LATOFILEMASK",buf)) {
         latOFileMask = CopyString(&gstack, buf);
      }
      if (GetConfStr(cParm,nParm,"LABOFILEMASK",buf)) {
         labOFileMask = CopyString(&gstack, buf);
      }
   }
}

void
ReportUsage (void)
{
   printf ("\nUSAGE: HDecode [options] VocabFile HMMList DataFiles...\n\n");
   printf (" Option                                   Default\n\n");
   printf (" -m      enable XForm and use inXForm        off\n");

   printf (" -d s    dir to find hmm definitions       current\n");
   printf (" -i s    Output transcriptions to MLF s      off\n");
   /*printf (" -k i    block size for outP calculation     1\n");*/
   printf (" -l s    dir to store label files	    current\n");
   printf (" -o s    output label formating NCSTWMX      none\n");
   printf (" -h s    speaker name pattern                none\n");
   printf (" -p f    word insertion penalty              0.0\n");
   printf (" -a f    acoustic scale factor               1.0\n");
   printf (" -r f    pronunciation scale factor          1.0\n");
   printf (" -s f    LM scale factor                     1.0\n");
   printf (" -t f    pruning beam width                  none\n");
   printf (" -u i    max model pruning                   0\n");
   printf (" -v f    wordend beam width                  0.0\n");
   printf (" -n i    number of tokens per state          32\n");
   printf (" -w s    use language model                  none\n");
   printf (" -x s    extension for hmm files             none\n");
   printf (" -y s    output label file extension         rec\n");
   printf (" -z s    generate lattices with extension s  off\n");
   printf (" -q s    output lattices format ABtvaldmnr  tvaldmr\n");
   printf (" -R s    best align MLF                      off\n");
   printf (" -X ext  set input lattice extension         lat\n");
   PrintStdOpts ("EJFHLSTP");
   printf ("\n\n");

   printf ("build-time options: ");
#ifdef MODALIGN
   printf ("MODALIGN ");
#endif   
#ifdef TSIDOPT
   printf ("TSIDOPT ");
#endif   
   printf ("\n  sizes: PronId=%lu  LMId=%lu \n", sizeof (PronId), sizeof (LMId));
}

int
main (int argc, char *argv[])
{
   char *s, *datafn;
   DecoderInst *dec;
   /* cz277 - ANN */
   int i;
   char fnbuf[MAXSTRLEN];

   if (InitShell (argc, argv, hdecode_version, hdecode_sccs_id) < SUCCESS)
      HError (3900, "HDecode: InitShell failed");

   InitMem ();
#ifdef CUDA
    InitCUDA();
#endif   
   InitMath ();
   InitSigP ();
   InitWave ();
   InitLabel ();
   InitAudio ();
   InitANNet();		/* cz277 - ANN */
   InitModel ();
   if (InitParm () < SUCCESS)
      HError (3900, "HDecode: InitParm failed");
   InitUtil ();
   InitDict ();
   InitLVNet ();
   InitNet();
   InitLVLM ();
   InitLVRec ();
   /* cz277 - xform */
   /*InitAdapt (&xfInfo);*/
   InitAdapt();
   InitXFInfo(&xfInfo);
   
   InitLat ();
   InitNCache();	/* cz277 - ANN */

   if (!InfoPrinted () && NumArgs () == 0)
      ReportUsage ();
   if (NumArgs () == 0)
      Exit (0);

   SetConfParms ();

   /* init model heap & set early to support loading MMFs */
   CreateHeap(&modelHeap, "Model heap",  MSTAK, 1, 0.0, 100000, 800000 );
   CreateHMMSet(&hset,&modelHeap,TRUE); 


   while (NextArg () == SWITCHARG) {
      s = GetSwtArg ();
      if (strlen (s) != 1)
	 HError (3919, "HDecode: Bad switch %s; must be single letter", s);
      switch (s[0]) {
      case 'd':
	 if (NextArg() != STRINGARG)
	    HError(3919,"HDecode: HMM definition directory expected");
	 hmmDir = GetStrArg(); 
	 break;
      case 'x':
	 if (NextArg() != STRINGARG)
	    HError(3919,"HDecode: HMM file extension expected");
	 hmmExt = GetStrArg(); 
	 break;
	 
      case 'i':
	 if (NextArg () != STRINGARG)
	    HError (3919, "HDecode: Output MLF file name expected");
	 if (SaveToMasterfile (GetStrArg ()) < SUCCESS)
	    HError (3914, "HDecode: Cannot write to MLF");
	 break;

      case 'P':
	 if (NextArg () != STRINGARG)
	    HError (3919, "HDecode: Target Label File format expected");
	 if ((ofmt = Str2Format (GetStrArg ())) == ALIEN)
	    HError (-3989, "HDecode: Warning ALIEN Label output file format set");
	 break;

      case 'l':
	 if (NextArg () != STRINGARG)
	    HError (3919, "HDecode: Label/Lattice output directory expected");
	 labDir = GetStrArg ();
         latOutDir = labDir;
	 break;
      case 'o':
	 if (NextArg () != STRINGARG)
	    HError (3919, "HDecode: Output label format expected");
	 labForm = GetStrArg ();
	 break;
      case 'y':
	 if (NextArg () != STRINGARG)
	    HError (3919, "HDecode: Output label file extension expected");
	 labExt = GetStrArg ();
	 break;

      case 'X':
	 if (NextArg () != STRINGARG)
	    HError (3919, "HDecode: Input Lattice file extension expected");
	 latInExt = GetStrArg ();
	 break;
      case 'L':
	 if (NextArg () != STRINGARG)
	    HError (3919, "HDecode: Input Lattice directory expected");
	 latInDir = GetStrArg ();
	 break;

      case 'q':
	 if (NextArg () != STRINGARG)
	    HError (3919, "HDecode: Output lattice format expected");
	 latOutForm = GetStrArg ();
	 break;
      case 'z':
         latGen = TRUE;
	 if (NextArg () != STRINGARG)
	    HError (3919, "HDecode: Output lattice file extension expected");
	 latOutExt = GetStrArg ();
	 break;

      case 'p':
	 if (NextArg () != FLOATARG)
	    HError (3919, "HDecode: word insertion penalty expected");
         insPen = GetFltArg ();
	 break;
      case 'a':
	 if (NextArg () != FLOATARG)
	    HError (3919, "HDecode: acoustic scale factor expected");
	 acScale = GetFltArg ();
	 break;
      case 'r':
	 if (NextArg () != FLOATARG)
	    HError (3919, "HDecode: pronunciation scale factor expected");
	  pronScale = GetFltArg ();
          silDict = TRUE;
	 break;
      case 's':
	 if (NextArg () != FLOATARG)
	    HError (3919, "HDecode: LM scale factor expected");
         lmScale= GetFltArg ();
	 break;


      case 'u':
	 if (NextArg () != INTARG)
	    HError (3919, "HDecode: max model pruning limit expected");
         maxModel = GetIntArg ();
	 break;

      case 't':
	 if (NextArg () != FLOATARG)
	    HError (3919, "HDecode: beam width expected");
	 beamWidth = GetFltArg ();
         if (latPruneBeam == -LZERO)
            latPruneBeam = beamWidth;
         relBeamWidth = beamWidth;
         if (NextArg () == FLOATARG)
            relBeamWidth = GetFltArg ();
	 break;

      case 'v':
	 if (NextArg () != FLOATARG)
	    HError (3919, "HDecode: wordend beam width expected");
         weBeamWidth = GetFltArg ();
         zsBeamWidth = weBeamWidth;
	 if (NextArg () == FLOATARG)
            zsBeamWidth = GetFltArg ();
         break;

      case 'w':
	 if (NextArg() != STRINGARG) {
            latRescore = TRUE;
         }
         else
            langfn = GetStrArg();
	 break;

      case 'n':
	 nTok = GetChkedInt (0, 1024, s);
	 break;
      case 'k':
         HError(-3919, "HDecode: -k option no long supported, ignored");
         break;
      /*case 'k':
	 outpBlocksize = GetChkedInt (0, MAXBLOCKOBS, s);
	 break;*/
      case 'H':
	 if (NextArg() != STRINGARG)
	    HError (3919,"HDecode: MMF File name expected");
	 AddMMF (&hset, GetStrArg()); 
	 break;
      case 'T':
	 trace = GetChkedInt (0, 1000, s);
	 break;

      case 'h':
         if (NextArg()!=STRINGARG)
	    HError (3919, "HDecode: Speaker name pattern expected");
         xfInfo.outSpkrPat = GetStrArg();
         if (NextArg()==STRINGARG) {
            xfInfo.inSpkrPat = GetStrArg();
            if (NextArg()==STRINGARG)
               xfInfo.paSpkrPat = GetStrArg(); 
         }
         if (NextArg() != SWITCHARG)
	    HError (3919, "HDecode: cannot have -h as the last option");
         break;
      case 'm':
	 xfInfo.useInXForm = TRUE;
         break;
      case 'E':
         if (NextArg()!=STRINGARG)
            HError(3919,"HDecode: parent transform directory expected");
	 xfInfo.usePaXForm = TRUE;
         xfInfo.paXFormDir = GetStrArg(); 
         if (NextArg()==STRINGARG)
            xfInfo.paXFormExt = GetStrArg(); 
	 if (NextArg() != SWITCHARG)
            HError(3919,"HDecode: cannot have -E as the last option");	  
         break;              
      case 'J':
         if (NextArg()!=STRINGARG)
            HError(3919,"HDecode: input transform directory expected");
         AddInXFormDir(&hset,GetStrArg());
         if (NextArg()==STRINGARG)
            xfInfo.inXFormExt = GetStrArg(); 
	 if (NextArg() != SWITCHARG)
            HError(3919,"HDecode: cannot have -J as the last option");	  
         break;              
      case 'K':
         HError(3919,"HDecode: transform estimation (-K option) not supported yet");	  
         if (NextArg()!=STRINGARG)
            HError(3919,"HDecode: output transform directory expected");
         xfInfo.outXFormDir = GetStrArg(); 
	 xfInfo.useOutXForm = TRUE;
         if (NextArg()==STRINGARG)
            xfInfo.outXFormExt = GetStrArg(); 
	 if (NextArg() != SWITCHARG)
            HError(3919,"HDecode: cannot have -K as the last option");	  
         break;              
      case 'N':
         HError (3919, "HDecode: old style fv transform not supported!");
	 break;
      case 'Q':
         HError (3919, "HDecode: old style mllr transform not supported!");
	 break;
      case 'R':
	 if (NextArg () != STRINGARG)
	    HError (3919, "HDecode: best align MLF name expected");
	 bestAlignMLF = GetStrArg ();
	 break;
      default:
	 HError (3919, "HDecode: Unknown switch %s", s);
      }
   }

   if (NextArg () != STRINGARG)
      HError (3919, "HDecode Vocab file name expected");
   dictfn = GetStrArg ();

   if (NextArg () != STRINGARG)
      HError (3919, "HDecode model list file name expected");
   hmmListfn = GetStrArg ();

   if (beamWidth > -LSMALL)
      HError (3919, "main beam is too wide!");

   if (xfInfo.useInXForm) {
      if (!useHModel) {
         HError (-3919, "HDecode: setting USEHMODEL to TRUE.");
         useHModel = TRUE;
      }
      /*if (outpBlocksize != 1) {
         HError (-4019, "HDecode: outP blocksize >1 not supported with new XForm code! setting to 1.");
         outpBlocksize = 1;
      }*/
   }   

#ifdef CUDA
    StartCUDA();
#endif
    /* cz277 - 151020 */
#ifdef MKL
    StartMKL();
#endif
   /* load models and initialise decoder */
   dec = Initialise ();
#ifdef CUDA
    ShowGPUMemUsage();
#endif

   /* load 1-best alignment */
   if (bestAlignMLF)
      LoadMasterFile (bestAlignMLF);

   /* cz277 - ANN */
   if (trace & T_TOP) {
      if (hset.annSet == NULL) {
         printf("Processing data directly from the input files");
      }
      else {
         printf("Proccesing data through the cache\n");
      }
   }

   while (NumArgs () > 0) {
      if (NextArg () != STRINGARG)
	 HError (3919, "HDecode: Data file name expected");
      datafn = GetStrArg ();
      /* cz277 - ANN */
      strcpy(fnbuf, datafn);
      if (trace & T_TOP) {
	 printf ("File: %s\n", datafn);
	 fflush (stdout);
      }
      DoRecognition (dec, fnbuf);
      /*DoRecognition (dec, datafn);*/
      /* perform recognition */
   }

   if (trace & T_MEM) {
      printf ("Memory State on Completion\n");
      PrintAllHeapStats ();
   }

   /* maybe output transforms for last speaker */
   UpdateSpkrStats(&hset,&xfInfo, NULL); 

   /* cz277 - ANN */
   /* remove the ANNSet matrices and vectors */
   if (hset.annSet != NULL) {
       FreeANNSet(&hset);
       for (i = 1; i <= hset.swidth[0]; ++i) {
           FreeCache(cache[i]);
       }
   }

#ifdef CUDA
    StopCUDA();
#endif

   Exit(0);             /* maybe print config and exit */
   return (0);
}

DecoderInst *Initialise (void)
{
   int i;
   DecoderInst *dec;
   Boolean eSep;
   Boolean modAlign;
   /* cz277 - ANN */
   FILE *script;
   int scriptcount;

   /* init Heaps */
   CreateHeap (&netHeap, "Net heap", MSTAK, 1, 0,100000, 800000);
   CreateHeap (&lmHeap, "LM heap", MSTAK, 1, 0,1000000, 10000000);
   CreateHeap(&transHeap,"Transcription heap",MSTAK,1,0,8000,80000);

   /* Read dictionary */
   if (trace & T_TOP) {
      printf ("Reading dictionary from %s\n", dictfn);
      fflush (stdout);
   }

   InitVocab (&vocab);
   if (ReadDict (dictfn, &vocab) < SUCCESS)
      HError (3913, "Initialise: ReadDict failed");

   /* Read accoustic models */
   if (trace & T_TOP) {
      printf ("Reading acoustic models...");
      fflush (stdout);
   }

   if (MakeHMMSet (&hset, hmmListfn) < SUCCESS) 
      HError (3900, "Initialise: MakeHMMSet failed");
   if (LoadHMMSet (&hset, hmmDir, hmmExt) < SUCCESS) 
      HError (3900, "Initialise: LoadHMMSet failed");
   CreateTmpNMat(hset.hmem);

   /* cz277 - ANN */
   if (hset.hsKind == HYBRIDHS || hset.feaMix[1] != NULL) {    /* for Tandem and Hybrid systems, use a different way of batch processing */
      outpBlocksize = 1;
   }
   
   /* convert to INVDIAGC */
   ConvDiagC (&hset, TRUE);
   ConvLogWt (&hset);
   
   if (trace&T_TOP) {
      printf("Read %d physical / %d logical HMMs\n",
	     hset.numPhyHMM, hset.numLogHMM);  
      /* cz277 - ANN */
      if (hset.annSet != NULL) {
         if (hset.hsKind == HYBRIDHS)
            printf("Hybrid ANN set: ");
         else
            printf("Tandem ANN set: ");
         ShowANNSet(&hset);
      }
      fflush (stdout);
   }

   SetupNMatRPLInfo(&hset);
   SetupNVecRPLInfo(&hset);

   /* process dictionary */
   startLab = GetLabId (startWord, FALSE);
   if (!startLab) 
      HError (3920, "HDecode: cannot find STARTWORD '%s'\n", startWord);
   endLab = GetLabId (endWord, FALSE);
   if (!endLab) 
      HError (3920, "HDecode: cannot find ENDWORD '%s'\n", endWord);

   spLab = GetLabId (spModel, FALSE);
   if (!spLab)
      HError (3920, "HDecode: cannot find label 'sp'");
   silLab = GetLabId (silModel, FALSE);
   if (!silLab)
      HError (3920, "HDecode: cannot find label 'sil'");

   if (silDict) {    /* dict contains -/sp/sil variants (with probs) */
      ConvertSilDict (&vocab, spLab, silLab, startLab, endLab);

      /* check for skip in sp model */
      { 
         LabId spLab;
         HLink spHMM;
         MLink spML;
         int N;

         spLab = GetLabId ("sp", FALSE);
         if (!spLab)
            HError (3921, "cannot find 'sp' model.");

         spML = FindMacroName (&hset, 'l', spLab);
         if (!spML)
            HError (3921, "cannot find model for sp");
         spHMM = spML->structure;
         N = spHMM->numStates;

         if (spHMM->transP[1][N] > LSMALL)
            HError (3922, "HDecode: using -/sp/sil dictionary but sp contains tee transition!");
      }
   }
   else {       /* lvx-style dict (no sp/sil at wordend */
      MarkAllProns (&vocab);
   }
   
   if (!latRescore) {

      if (!langfn)
         HError (3919, "HDecode: no LM or lattice specified");

      /* mark all words  for inclusion in Net */
      MarkAllWords (&vocab);

      /* create network */
      net = CreateLexNet (&netHeap, &vocab, &hset, startWord, endWord, silDict);
      
      /* Read language model */
      if (trace & T_TOP) {
         printf ("Reading language model from %s\n", langfn);
         fflush (stdout);
      }
      
      lm = CreateLM (&lmHeap, langfn, startWord, endWord, &vocab);
   }
   else {
      net = NULL;
      lm = NULL;
   }

   modAlign = FALSE;
   if (latOutForm) {
      if (strchr (latOutForm, 'd'))
         modAlign = TRUE;
      if (strchr (latOutForm, 'n'))
         HError (3901, "DoRecognition: likelihoods for model alignment not supported");
   }

   /* create Decoder instance */
   dec = CreateDecoderInst (&hset, lm, nTok, TRUE, useHModel, outpBlocksize,
                            bestAlignMLF ? TRUE : FALSE,
                            modAlign);

   /* create buffers for observations */
   SetStreamWidths (hset.pkind, hset.vecSize, hset.swidth, &eSep);

   obs = (Observation *) New (&gcheap, outpBlocksize * sizeof (Observation));	/* TODO: for Tandem system, might need an extra obs */
   for (i = 0; i < outpBlocksize; ++i) {
      obs[i] = MakeObservation (&gcheap, hset.swidth, hset.pkind, 
                                (hset.hsKind == DISCRETEHS), eSep);
   }

   CreateHeap (&inputBufHeap, "Input Buffer Heap", MSTAK, 1, 1.0, 80000, 800000);

   /* Initialise adaptation */

   /* sort out masks just in case using adaptation */
   if (xfInfo.inSpkrPat == NULL) xfInfo.inSpkrPat = xfInfo.outSpkrPat; 
   if (xfInfo.paSpkrPat == NULL) xfInfo.paSpkrPat = xfInfo.outSpkrPat; 

   if (xfInfo.useOutXForm) {
      CreateHeap(&regHeap,   "regClassStore",  MSTAK, 1, 0.5, 1000, 8000 );
      /* This initialises things - temporary hack - THINK!! */
      CreateAdaptXForm(&hset, "tmp");

      /* online adaptation not supported yet! */
   }

   /* cz277 - ANN */
   /* ANN and data cache related code */
   /* set label info */
   if (hset.annSet != NULL) {
      labelInfo.labelKind = LABLK;
      labelInfo.labFileMask = NULL;
      labelInfo.labDir = labDir;
      labelInfo.labExt = labExt;
      labelInfo.latFileMask = NULL;
      labelInfo.latMaskNum = NULL;
      labelInfo.numLatDir = NULL;
      labelInfo.nNumLats = 0;
      labelInfo.numLatSubDirPat = NULL;
      labelInfo.latMaskDen = NULL;
      labelInfo.denLatDir = NULL;
      labelInfo.nDenLats = 0;
      labelInfo.denLatSubDirPat = NULL;
      labelInfo.latExt = NULL;
      /* get script info */
      script = GetTrainScript(&scriptcount);
      /* initialise the cache heap */
      CreateHeap(&cacheHeap, "cache heap", CHEAP, 1, 0, 100000000, ULONG_MAX);
      /* initialise DataCache structure */
      for (i = 1; i <= hset.swidth[0]; ++i) {
         /*cache[i] = CreateCache(&cacheHeap, script, scriptcount, &hset, &obs[0], 1, -1, NONEVK, &xfInfo, NULL, TRUE);*/
         cache[i] = CreateCache(&cacheHeap, script, scriptcount, &hset, &obs[0], 1, GetDefaultNCacheSamples(), NONEVK, &xfInfo, NULL, TRUE);
         InitCache(cache[i]);
      }
   }

#ifdef LEGACY_CUHTK2_MLLR
   /* initialise adaptation */
   if (mllrTransDir) {
      CreateHeap(&regHeap,   "regClassStore",  MSTAK, 1, 0.5, 80000, 80000 );
      rt = (RegTransInfo *) New(&regHeap, sizeof(RegTransInfo));
      rt->nBlocks = 0;
      rt->classKind = DEF_REGCLASS;
      rt->adptSil = TRI_UNDEF;
      rt->nodeOccThresh = 0.0;

      /*# legacy CU-HTK adapt: create RegTree from INCORE/CLASS files
          and strore in ~r macro */
      LoadLegacyRegTree (&hset);
      
      InitialiseTransform(&hset, &regHeap, rt, FALSE);
   }
#endif

   return dec;
}


/**********  align best code  ****************************************/

/* linked list storing the info about the 1-best alignment read from BESTALIGNMLF 
   one bestInfo struct per model */
typedef struct _BestInfo BestInfo;
struct _BestInfo {
   int start;           /* frame numbers */
   int end;
   LexNode *ln;
   LLink ll;           /* get rid of this? currently start/end are redundant */
   BestInfo *next;
};


/* find the LN_MODEL lexnode following ln that has label lab
   step over LN_CON and LN_WORDEND nodes.
   return NULL if not found
*/
BestInfo *FindLexNetLab (MemHeap *heap, LexNode *ln, LLink ll, HTime frameDur)
{
   int i;
   LexNode *follLN;
   MLink m;
   BestInfo *info, *next;

   if (!ll->succ) {
      info = New (heap, sizeof (BestInfo));
      info->next = NULL;
      info->ll = NULL;
      info->ln = NULL;
      info->start = info->end = 0;
      return info;
   }
   
   for (i = 0; i < ln->nfoll; ++i) {
      follLN = ln->foll[i];
      if (follLN->type == LN_MODEL) {
         m = FindMacroStruct (&hset, 'h', follLN->data.hmm);
         if (m->id == ll->labid) {
            /*            printf ("found  %8.0f %8.0f %8s  %p\n", ll->start, ll->end, ll->labid->name, follLN); */
            next = FindLexNetLab (heap, follLN, ll->succ, frameDur);
            if (next) {
               info = New (heap, sizeof (BestInfo));
               info->next = next;
               info->start = ll->start / (frameDur*1.0e7);
               info->end = ll->end / (frameDur*1.0e7);
               info->ll = ll;
               info->ln = follLN;
               return info;
            }
            /*            printf ("damn got 0 back searching for %8s\n", ll->labid->name); */
         }
      }
      else {
         /*         printf ("searching for %8s recursing\n", ll->labid->name); */
         next = FindLexNetLab (heap, follLN, ll, frameDur);
         if (next) {
            info = New (heap, sizeof (BestInfo));
            info->next = next;
            info->start = info->end = ll->start / (frameDur*1.0e7);
            info->ll = ll;
            info->ln = follLN;
            return info;
         }
         /*         printf ("damn got 0 back from recursion\n"); */
      }
   }
   
   return NULL;
}

BestInfo *CreateBestInfo (MemHeap *heap, char *fn, HTime frameDur)
{
   char alignFN[MAXFNAMELEN];
   Transcription *bestTrans;
   LLink ll;
   LexNode *ln;
   MLink m;
   BestInfo *bestAlignInfo;
   LabId lnLabId;

   MakeFN (fn, "", "rec", alignFN);
   bestTrans = LOpen (&transHeap, alignFN, HTK);
      
   /* delete 'sp' or 'sil' before final 'sil' if it is there
      these are always inserted by HVite but not possible in HDecode's net structure*/
   if (bestTrans->head->tail->pred->pred->labid == spLab ||
       bestTrans->head->tail->pred->pred->labid == silLab) {
      LLink delLL;
      
      delLL = bestTrans->head->tail->pred->pred;
      /* add sp's frames (if any) to final sil */
      delLL->succ->start = delLL->pred->end;
      
      delLL->pred->succ = delLL->succ;
      delLL->succ->pred = delLL->pred;
   }
   
   ln = net->start;
   assert (ln->type == LN_MODEL);
   m = FindMacroStruct (&hset, 'h', ln->data.hmm);
   lnLabId = m->id;
   
   /* info for net start node */
   ll = bestTrans->head->head->succ;
#if 0
   printf ("%8.0f %8.0f %8s   ln %p %8s\n", ll->start, ll->end, ll->labid->name, 
           ln, lnLabId->name);
#endif
   if(ll->labid != lnLabId)
     HError(3901,"CreateBestInfo: labels differ");
   bestAlignInfo = New (&transHeap, sizeof (BestInfo));
   bestAlignInfo->start = ll->start / (frameDur*1.0e7);
   bestAlignInfo->end = ll->end / (frameDur*1.0e7);
   bestAlignInfo->ll = ll;
   bestAlignInfo->ln = ln;
   
   
   /* info for all the following nodes */
   bestAlignInfo->next = FindLexNetLab (&transHeap, ln, ll->succ, frameDur);
   
   {
      BestInfo *b;
      for (b = bestAlignInfo; b->next; b = b->next)
         printf ("%d %d %8s %p\n", b->start, b->end, b->ll->labid->name, b->ln);
   }

   return bestAlignInfo;
}

void PrintAlignBestInfo (DecoderInst *dec, BestInfo *b)
{
   LexNodeInst *inst;
   TokScore score;
   int l;
   LabId monoPhone;
   LogDouble phonePost;

   inst = b->ln->inst;
   score = inst ? inst->best : LZERO;

   if (b->ln->type == LN_MODEL) {
      monoPhone =(LabId) b->ln->data.hmm->hook;
      phonePost = dec->phonePost[(unsigned long int) monoPhone->aux];
   } else
      phonePost = 999.99;

   l = dec->nLayers-1;
   while (dec->net->layerStart[l] > b->ln) {
      --l;
      assert (l >= 0);
   }
   
   printf ("BESTALIGN frame %4d best %.3f alignbest %d -> %d ln %p layer %d score %.3f phonePost %.3f\n", 
           dec->frame, dec->bestScore, 
           b->start, b->end, b->ln, l, score, phonePost);
}

void AnalyseSearchSpace (DecoderInst *dec, BestInfo *bestInfo)
{
   BestInfo *b;
   LabId monoPhone;

   monoPhone =(LabId) dec->bestInst->node->data.hmm->hook;
   printf ("frame %4d best %.3f phonePost %.3f\n", dec->frame, 
           dec->bestScore,dec->phonePost[(unsigned long int)monoPhone->aux]);
 
   for (b = bestInfo; b; b = b->next) {
      if (b->start < dec->frame && b->end >= dec->frame) 
         break;
   }
   if (b) {
      PrintAlignBestInfo (dec, b);
      for (b = b->next; b && b->start == b->end && b->start == dec->frame; b = b->next) {
         PrintAlignBestInfo (dec, b);
      }
   }
   else {
      printf ("BESTALIGN ERROR\n");
   }
}

/*****************  main recognition function  ************************/

/* cz277 - ANN */
static void LoadCacheVec(DecoderInst *dec, int frameNum, HMMSet *hset) {
    int s, S, i, j, offset;
    NMatrix *srcMat;
    FELink feaElem;
    LELink layerElem;

    S = hset->swidth[0];
    for (i = 0; i < frameNum; ++i) {
        for (s = 1; s <= S; ++s) {
            offset = 1;
            if (dec->decodeKind == HYBRIDDK) {  /* hybrid models, cache the outputs */
                layerElem = hset->annSet->outLayers[s];
                srcMat = hset->annSet->llhMat[s];
                CopyNFloatSeg2FloatSeg(srcMat->matElems + i * layerElem->nodeNum, layerElem->nodeNum, dec->cacheVec[i][s] + offset);
            }
            else if (dec->decodeKind == TANDEMDK) {    /* tandem models, cache the features */
                for (j = 0; j < hset->feaMix[s]->elemNum; ++j) {
                    feaElem = hset->feaMix[s]->feaList[j];
                    srcMat = feaElem->feaMats[1];
                    CopyNFloatSeg2FloatSeg(srcMat->matElems + i * feaElem->srcDim + feaElem->dimOff, feaElem->extDim, dec->cacheVec[i][s] + offset); 
                    offset += feaElem->extDim;
                }
            }
            else {
                HError(3990, "LoadCacheVec: DataCache is only applicable for hybrid and tandem systems");
            }
        }
    }
}

void DoRecognition (DecoderInst *dec, char *fn)
{
    char buf1[MAXSTRLEN], buf2[MAXSTRLEN];
    ParmBuf parmBuf;
    BufferInfo pbInfo;
    int frameN, frameProc, i, bs, uttCnt;
    Transcription *trans;
    Lattice *lat;
    clock_t startClock, endClock;
    double cpuSec;
    Observation *obsBlock[MAXBLOCKOBS];
    BestInfo *bestAlignInfo = NULL;
    /* cz277 - ANN */
    int cUttLen, uttLen, nLoaded;
    LELink layerElem;
    /* cz277 - clock */
    clock_t fwdStClock, fwdClock = 0, decStClock, decClock = 0, loadStClock, loadClock = 0;
    double fwdSec = 0.0, decSec = 0.0, loadSec = 0.0;
    /* cz277 - xform */
    UttElem *uttElem;

    /* This handles the initial input transform, parent transform setting
       and output transform creation */
    { 
#if 0
         Boolean changed;

	 changed =
#endif
         UpdateSpkrStats(&hset, &xfInfo, fn);

#if 0   /* not neccessary if for USEHMODEL=T */
      if (changed)
         dec->si = ConvertHSet (&modelHeap, &hset, dec->useHModel);
#endif
    }

    startClock = clock();

    /* get transcrition of 1-best alignment */
    if (bestAlignMLF)
        bestAlignInfo = CreateBestInfo(&transHeap, fn, pbInfo.tgtSampRate / 1.0e7);
   
    parmBuf = OpenBuffer(&inputBufHeap, fn, 50, dataForm, TRI_UNDEF, TRI_UNDEF);
    if (!parmBuf)
        HError(3910, "HDecode: Opening input failed");

    GetBufferInfo(parmBuf, &pbInfo);
    if (pbInfo.tgtPK != hset.pkind)
        HError(3923, "HDecode: Incompatible parm kinds %s vs. %s", ParmKind2Str(pbInfo.tgtPK, buf1), ParmKind2Str(hset.pkind, buf2));
              
    if (latRescore) {
        /* read lattice and create LM */
        char latfn[MAXSTRLEN], buf3[MAXSTRLEN];
        FILE *latF;
        Boolean isPipe;
        Lattice *lat;

        /* clear out previous LexNet, Lattice and LM structures */
        ResetHeap(&lmHeap);
        ResetHeap(&netHeap);
      
        if (latFileMask != NULL) {  /* support for rescoring lattoce masks */
            if (!MaskMatch(latFileMask, buf3 , fn))
                HError(3919,"HDecode: mask %s has no match with segemnt %s", latFileMask, fn);
            MakeFN(buf3, latInDir, latInExt, latfn);
        } else {
            MakeFN(fn, latInDir, latInExt, latfn);
        }
      
        if (trace & T_TOP)
            printf("Loading Lattice from %s\n", latfn);
      
        {
            latF = FOpen(latfn, NetFilter, &isPipe);
            if (!latF)
                HError(3910, "DoRecognition: Cannot open lattice file %s\n", latfn);
            /* #### maybe separate lattice heap? */
            lat = ReadLattice(latF, &lmHeap, &vocab, FALSE, FALSE);
            FClose(latF, isPipe);
            if (!lat)
                HError(3913, "DoRecognition: Cannot read lattice file %s\n", latfn);
        }
      
        /* mark prons of all words in lattice */
        UnMarkAllWords(&vocab);
        MarkAllWordsfromLat(&vocab, lat, silDict);

        /* create network of all the words/prons marked (word->aux and pron->aux == 1) */
        if (trace & T_TOP)
            printf("Creating network\n");
        net = CreateLexNet(&netHeap, &vocab, &hset, startWord, endWord, silDict);

        /* create LM based on pronIds defined by CreateLexNet */
        if (trace & T_TOP)
            printf("Creating language model\n");
        lm = CreateLMfromLat(&lmHeap, latfn, lat, &vocab);
        dec->lm = lm;
    }

    if (weBeamWidth > beamWidth)
        weBeamWidth = beamWidth;
    if (zsBeamWidth > beamWidth)
        zsBeamWidth = beamWidth;

    InitDecoderInst(dec, net, pbInfo.tgtSampRate, beamWidth, relBeamWidth, weBeamWidth, zsBeamWidth, maxModel, insPen, acScale, pronScale, lmScale, fastlmlaBeam);

    net->vocabFN = dictfn;
    dec->utterFN = fn;

    frameN = frameProc = 0;
    /* cz277 - ANN */
    if (hset.annSet == NULL) { /* use conventional way */
        while (BufferStatus(parmBuf) != PB_CLEARED) {
            ReadAsBuffer(parmBuf, &obs[frameN % outpBlocksize]);
      
#ifdef LEGACY_CUHTK2_MLLR
            if (fvTransMat) {
                if (trace & T_OBS)
                    printf ("apply full variance transform\n");

                MultBlockMat_Vec(fvTransMat, obs[frameN % outpBlocksize].fv[1], obs[frameN % outpBlocksize].fv[1]);
            } 
#endif

            if (frameN + 1 >= outpBlocksize) {  
                if (trace & T_OBS)
                    PrintObservation(frameProc + 1, &obs[frameProc % outpBlocksize], 13);
                for (i = 0; i < outpBlocksize; ++i)
                    obsBlock[i] = &obs[(frameProc + i) % outpBlocksize];

#ifdef DEBUG_TRACE
                fprintf(stdout, "\nProcessing frame %d :\n", frameProc);
                fflush(stdout);
#endif

                ProcessFrame(dec, obsBlock, outpBlocksize, xfInfo.inXForm, -1); /* cz277 - ANN */

                if(bestAlignInfo)
                    AnalyseSearchSpace(dec, bestAlignInfo);
                ++frameProc;
            }
            ++frameN;
        }

        /* process remaining frames (no full blocks available anymore) */
        for (bs = outpBlocksize - 1; bs >= 1; --bs) {
            if (trace & T_OBS)
                PrintObservation(frameProc + 1, &obs[frameProc % outpBlocksize], 13);
            for (i = 0; i < bs; ++i)
                obsBlock[i] = &obs[(frameProc + i) % outpBlocksize];
      
            ProcessFrame(dec, obsBlock, bs, xfInfo.inXForm, -1);    /* cz277 - ANN */
            if (bestAlignInfo)
                AnalyseSearchSpace(dec, bestAlignInfo);
            ++frameProc;
        }
        assert(frameProc == frameN);
    }
    else {  /* if hset.feaMix is not empty */
        /* get utterance name in cache */
        if (strcmp(GetCurUttName(cache[1]), fn) != 0) 
            HError(3924, "Mismatched utterance in the cache and script file");
        uttElem = GetCurUttElem(cache[1]);	/* cz277 - xform */
        InstallOneUttNMatRPLs(uttElem);
        InstallOneUttNVecRPLs(uttElem);
        /* check the observation vector number */
        uttCnt = 1;
        uttLen = ObsInBuffer(parmBuf);
        cUttLen = GetCurUttLen(cache[1]);
        if (cUttLen != uttLen) 
            HError(3991, "Unequal utterance length in the cache and the original feature file");
        /* initialise the obsBlock to hack ProcessFrame */
        assert(outpBlocksize == 1);
        obsBlock[0] = &obs[0];
        /* process each block of frame */
        while (TRUE) {
            /* load a data batch */
            loadStClock = clock();  /* cz277 - clock */
            for (i = 1; i <= hset.swidth[0]; ++i) {

                FillAllInpBatch(cache[i], &nLoaded, &uttCnt);
                /* cz277 - mtload */
                /*UpdateCacheStatus(cache[i]);*/
                LoadCacheData(cache[i]);
            }
            loadClock += clock() - loadStClock;   /* cz277 - clock */
            /* forward these frames */
            fwdStClock = clock();   /* cz277 - clock */
            ForwardProp(hset.annSet, nLoaded, cache[1]->CMDVecPL);
            /* apply log transform */
            for (i = 1; i <= hset.swidth[0]; ++i) {
                layerElem = hset.annSet->outLayers[i];
                ApplyLogTrans(layerElem->yFeaMats[1], nLoaded, layerElem->nodeNum, hset.annSet->llhMat[i]);
                AddNVectorTargetPen(hset.annSet->llhMat[i], hset.annSet->penVec[i], nLoaded, hset.annSet->llhMat[i]);
#ifdef CUDA
                SyncNMatrixDev2Host(hset.annSet->llhMat[i]);
#endif
            }
            fwdClock += clock() - fwdStClock;   /* cz277 - clock */
            /* load the ANN outputs into dec->cacheVecs */
            decStClock = clock();   /* cz277 - clock */
            LoadCacheVec(dec, nLoaded, &hset);
            /* decode these frames */
            for (i = 0; i < nLoaded; ++i) 
                ProcessFrame(dec, obsBlock, outpBlocksize, xfInfo.inXForm, i);
            decClock += clock() - decStClock;   /* cz277 - clock */
            /* analyse the search space */
            if (bestAlignInfo) 
                AnalyseSearchSpace(dec, bestAlignInfo);
            /* update the frame count */
            frameN += nLoaded;
            frameProc += GetNBatchSamples();
            /* whether continue the loop or not */
            if (frameProc >= uttLen) 
                break;
        }
        /* check the frame number loaded */
        if (frameN != uttLen) 
            HError(3991, "The number of frame loaded does not match the utterance length");
        /* cz277 - mtload */
        for (i = 1; i <= hset.swidth[0]; ++i) 
            UnloadCacheData(cache[i]);
        /* cz277 - xform */
        ResetNMatRPL();
        ResetNVecRPL();
    }

    /* close the buffer */
    CloseBuffer(parmBuf);
   
    endClock = clock();
    cpuSec = (endClock - startClock) / (double) CLOCKS_PER_SEC;
    printf ("CPU time %f  utterance length %f  RT factor %f\n", cpuSec, frameN * dec->frameDur, cpuSec / (frameN * dec->frameDur));

    /* cz277 - clock */
    if (hset.annSet != NULL) {
        fwdSec = fwdClock / (double) CLOCKS_PER_SEC;
        decSec = decClock / (double) CLOCKS_PER_SEC;
        loadSec = loadClock / (double) CLOCKS_PER_SEC;
        printf("\tForwarding time is %f, which takes %4.2f%% of the time cost    RT factor %f\n", fwdSec, fwdSec / cpuSec * 100, fwdSec / (frameN * dec->frameDur));
        printf("\tDecoding time is %f, which takes %4.2f%% of the time cost    RT factor %f\n", decSec, decSec / cpuSec * 100, decSec / (frameN * dec->frameDur));
        printf("\tCache loading time is %f, which takes %4.2f%% of the time cost    RT factor %f\n", loadSec, loadSec / cpuSec * 100, loadSec / (frameN * dec->frameDur));
        fflush(stdout);
    }
    trans = TraceBack(&transHeap, dec);
    /* save 1-best transcription */
    /* the following is from HVite.c */
    if (trans) {
        char labfn[MAXSTRLEN], lfn[MAXSTRLEN];

        if (labForm != NULL)
            ReFormatTranscription(trans, pbInfo.tgtSampRate, FALSE, FALSE,
                                    strchr(labForm, 'X') != NULL,
                                    strchr(labForm, 'N') != NULL, strchr(labForm, 'S') != NULL,
                                    strchr(labForm, 'C') != NULL, strchr(labForm, 'T') != NULL,
                                    strchr(labForm, 'W') != NULL, strchr(labForm, 'M') != NULL);

        /* from mjfg, cz277 - 141022 */
        if (labOFileMask) {
            if (!MaskMatch (labOFileMask, lfn, fn))
                HError(3919,"DoRecognition: LABOFILEMASK %s has no match with segemnt %s", labOFileMask, fn);
        } else {     
            strcpy (lfn, fn);
        }
        MakeFN(lfn, labDir, labExt, labfn);
        /*MakeFN(fn, labDir, labExt, labfn);*/

        if (LSave(labfn, trans, ofmt) < SUCCESS)
            HError(3911, "DoRecognition: Cannot save file %s", labfn);
        if (trace & T_TOP)
            PrintTranscription(trans, "1-best hypothesis");

        Dispose(&transHeap, trans);
    }

    if (latGen) {
        lat = LatTraceBack(&transHeap, dec);
        /* prune lattice */
        if (lat && latPruneBeam < -LSMALL) {
            lat = LatPrune (&transHeap, lat, latPruneBeam, latPruneAPS);
        }

        /* the following is from HVite.c */
        if (lat) {
            char latfn[MAXSTRLEN], lfn[MAXSTRLEN];
            char *p;
            Boolean isPipe;
            FILE *file;
            LatFormat form;
         
            /* from mjfg, cz277 - 141022 */
            if (latOFileMask) {
                if (!MaskMatch (latOFileMask, lfn, fn))
                    HError(3919,"DoRecognition: LATOFILEMASK %s has no match with segemnt %s", latOFileMask, fn);
            } else {
                strcpy (lfn, fn);
            }
            MakeFN(lfn, latOutDir, latOutExt, latfn);          
            /*MakeFN(fn, latOutDir, latOutExt, latfn);*/
 
            file = FOpen(latfn, NetOFilter, &isPipe);
            if (!file) 
                HError (3913, "DoRecognition: Could not open file %s for lattice output", latfn);
            if (!latOutForm)
                form = (HLAT_DEFAULT & ~HLAT_ALLIKE) | HLAT_PRLIKE;
            else {
                for (p = latOutForm, form=0; *p != 0; p++) {
                    switch (*p) {
                        case 'A': form |= HLAT_ALABS;   break;
                        case 'B': form |= HLAT_LBIN;    break;
                        case 't': form |= HLAT_TIMES;   break;
                        case 'v': form |= HLAT_PRON;    break;
                        case 'a': form |= HLAT_ACLIKE;  break;
                        case 'l': form |= HLAT_LMLIKE;  break;
                        case 'd': form |= HLAT_ALIGN;   break;
                        case 'm': form |= HLAT_ALDUR;   break;
                        case 'n': form |= HLAT_ALLIKE; 
                            HError(3901, "DoRecognition: likelihoods for model alignment not supported");
                            break;
                        case 'r': form |= HLAT_PRLIKE;  break;
                    }
                }
            }
            if (WriteLattice(lat, file, form) < SUCCESS)
                HError(3913, "DoRecognition: WriteLattice failed");
         
            FClose(file, isPipe);
            Dispose(&transHeap, lat);
        }
    }


#ifdef COLLECT_STATS
    printf("Stats: nTokSet %lu\n", dec->stats.nTokSet);
    printf("Stats: TokPerSet %f\n", dec->stats.sumTokPerTS / (double) dec->stats.nTokSet);
    printf("Stats: activePerFrame %f\n", dec->stats.nActive / (double) dec->stats.nFrames);
    printf("Stats: activateNodePerFrame %f\n", dec->stats.nActivate / (double) dec->stats.nFrames);
    printf("Stats: deActivateNodePerFrame %f\n\n", dec->stats.nDeActivate / (double) dec->stats.nFrames);
#if 0
    printf ("Stats: LMlaCacheHits %ld\n", dec->stats.nLMlaCacheHit);
    printf ("Stats: LMlaCacheMiss %ld\n", dec->stats.nLMlaCacheMiss);
#endif
#ifdef COLLECT_STATS_ACTIVATION
    {
        int i;
        for (i = 0; i <= STATS_MAXT; ++i)
            printf("T %d Dead %lu Live %lu\n", i, dec->stats.lnDeadT[i], dec->stats.lnLiveT[i]);
    }
#endif
#endif

    if (trace & T_MEM) {
        printf("memory stats at end of recognition\n");
        PrintAllHeapStats();
    }

    ResetHeap(&inputBufHeap);
    ResetHeap(&transHeap);
    CleanDecoderInst(dec);

}

#ifdef LEGACY_CUHTK2_MLLR
void ResetFVTrans (HMMSet *hset, BlockMatrix transMat)
{
   HError (3901, "HDecode: switching speakers/transforms not supprted, yet");
}

void LoadFVTrans (char *fn, BlockMatrix *transMat)
{
   Source src;
   int blockSize, bs, i;
   short nblocks;
   char buf[MAXSTRLEN];
   Boolean binary = FALSE;
   
   InitSource (fn, &src, NoFilter);

   /* #### the file input should use HModel's/HAdapt's scanner  */
   ReadUntilLine (&src, "~k \"globalSemi\"");
   if (!ReadString (&src, buf) || strcmp (buf, "<SEMICOVAR>"))
      HError (3919, "LoadSemiTrans: expected <SEMICOVAR> tag in file '%s'");
   ReadShort (&src, &nblocks, 1, binary);

   ReadInt (&src, &blockSize, 1, binary);
   for(i = 2; i <= nblocks; i++) {
      ReadInt (&src, &bs, 1, binary);
      if (bs != blockSize)
         HError (3901, "LoadSemiTrans: BlockMats with different size blocks not supported");
   }
   if(!*transMat)
      *transMat = CreateBlockMat (&gcheap, nblocks * blockSize, nblocks);
   if (!ReadBlockMat (&src, *transMat, binary))
      HError (3913, "LoadSemiTrans: cannot read transform matrix");
      
   CloseSource(&src);
}

void FVTransModels (HMMSet *hset, BlockMatrix transMat)
{
   int i = 0;
   HMMScanState hss;

   NewHMMScan (hset, &hss);
   do{
      while(GoNextMix (&hss, FALSE)){
         MultBlockMat_Vec (transMat, hss.mp->mean, hss.mp->mean);
         ++i;
      }
   }while (GoNextHMM (&hss));
   EndHMMScan (&hss);

   if (trace & T_ADP)
      printf ("applied full-var transform to %d mixture means", i);
}


/* UpdateSpkrModels

     apply speaker specific transforms
*/
Boolean UpdateSpkrModels (char *fn)
{
   char spkrName[MAXSTRLEN] = "";
   char fvTransFN[MAXSTRLEN] = "";
   char mllrTransFN[MAXSTRLEN] = "";
   Boolean changed = FALSE;
   
   /* full-variance transform: apply to means & feature space */
   if (!MaskMatch (spkrPat, spkrName, fn))
      HError (3919, "UpdateSpkrModels: non-matching speaker mask '%s'", spkrPat);
   
   if (!curSpkrName || strcmp (spkrName, curSpkrName)) {
      if (trace & T_ADP)
         printf ("new speaker %s, adapting...\n", spkrName);

      /* MLLR transform */
      if (mllrTransDir) {
         if (curSpkrName) {
            /* apply back tranform */
            HError (3901, "UpdateSpkrModels: switching speakers not supported, yet!");
         }
         if (trace & T_ADP)
            printf (" applying MLLR transform");

         MakeFN (spkrName, mllrTransDir, NULL, mllrTransFN);
         LoadLegacyTransformSet (&hset, mllrTransFN, rt);

         ApplyTransforms (rt);
         changed = TRUE;
      }

      if (fvTransDir) {
         if (trace & T_ADP)
            printf (" applying full-var transform");

         /* full-variance transform */
         if (fvTransMat)
            ResetFVTrans (&hset, fvTransMat);
         MakeFN (spkrName, fvTransDir, NULL, fvTransFN);
         LoadFVTrans (fvTransFN, &fvTransMat);
         FVTransModels (&hset, fvTransMat);
         /* # store per frame offset log(BlkMatDet(transMat)) */

         changed = TRUE;
      }
      curSpkrName = spkrName;
   }

   return changed;
}
#else
Boolean UpdateSpkrModels (char *fn)
{
   HError (3901, "MLLR or FV transforms not supported");
   return FALSE;
}
#endif


/* ----------------------------------------------------------- */
/*                      END:  HDecode.c                        */
/* ----------------------------------------------------------- */

