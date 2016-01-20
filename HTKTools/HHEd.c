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
/*           Entropic Cambridge Research Laboratory            */
/*           (now part of Microsoft)                           */
/*                                                             */
/* ----------------------------------------------------------- */
/*           Copyright: Microsoft Corporation                  */
/*            1995-2000 Redmond, Washington USA                */
/*                      http://www.microsoft.com               */
/*                                                             */
/*           Copyright: Cambridge University                   */
/*                      Engineering Department                 */
/*            2001-2015 Cambridge, Cambridgeshire UK           */
/*                      http://www.eng.cam.ac.uk               */
/*                                                             */
/*   Use of this software is governed by a License Agreement   */
/*    ** See the file License for the Conditions of Use  **    */
/*    **     This banner notice must not be removed      **    */
/*                                                             */
/* ----------------------------------------------------------- */
/*           File: HHEd  HMM source definition editor          */
/* ----------------------------------------------------------- */

char *hhed_version = "!HVER!HHEd:   3.5.0 [CUED 12/10/15]";
char *hhed_vc_id = "$Id: HHEd.c,v 1.3 2015/10/12 12:07:24 cz277 Exp $";

/*
   This program is used to read in a set of HMM definitions
   and then edit them according to the contents of a 
   file of edit commands.  See the routine called Summary for
   a list of these or run HHEd with the command option
*/

#include "HShell.h"
#include "HMem.h"
#include "HMath.h"
#include "HSigP.h"
#include "HAudio.h"
#include "HWave.h"
#include "HVQ.h"
#include "HParm.h"
#include "HLabel.h"
#include "HANNet.h"
#include "HModel.h"
#include "HUtil.h"
#include "HTrain.h"
#include "HAdapt.h"
/* cz277 - aug */
/*#include "HNCache.h"*/

#define FLOAT_MAX 1E10      /* Limit for float arguments */
#define BIG_FLOAT 1E20      /* Limit for float arguments */
#define MAX_ITER  500        /* Maximum number of iterations */

                             /* Lowest level of tracing reset at end of each command block */
#define T_BAS 0x0001        /* Basic progess tracing */
#define T_INT 0x0002        /* Intermediate progress tracing */
#define T_DET 0x0004        /* Detailed progress tracing */
#define T_ITM 0x0008        /* Show item lists */
#define T_MEM 0x0010        /* Show state of memory (once only) */

/* General tracing flags from command line */
#define T_MAC 0x0040        /* Trace changes to macro definitions */
#define T_SIZ 0x0080        /* Trace changes to stream widths */

/* Detailed control over clustering output */
#define T_CLUSTERS   0x0100 /* Show contents of clusters */
#define T_QST        0x0200 /* Show items specified by questions */
#define T_TREE_ANS   0x0400 /* Trace best unseen triphone tree filtering */
#define T_TREE_BESTQ 0x0800 /* Trace best question for splitting each node */
#define T_TREE_BESTM 0x1000 /* Trace best merge of terminal nodes  */
#define T_TREE_OKQ   0x2000 /* Trace all questions exceeding threshold */
#define T_TREE_ALLQ  0x4000 /* Trace all questions */
#define T_TREE_ALLM  0x8000 /* Trace all possible merges */

/* Extra flags to allow easier multiple checks */
#define T_TREE       0xf400 /* Any specific tree tracing */
#define T_IND        0x0006 /* Intermediate or detailed tracing */
#define T_BID        0x0007 /* Basic, intermediate or detailed tracing */
#define T_MD         0x10000 /* Trace mix down detail merge */

/* cz277 - aug */
#define MAXAUGFEAS   5

static MemHeap questHeap;   /* Heap holds all questions */
static MemHeap hmmHeap;     /* Heap holds all hmm related info */
static MemHeap tmpHeap;     /* Temporary (duration of command or less) heap */

/* Global Settings */

static char * hmmDir = NULL;     /* directory to look for hmm def files */
static char * hmmExt = NULL;     /* hmm def file extension */
static char * newDir = NULL;     /* directory to store new hmm def files */
static char * newExt = NULL;     /* extension of new edited hmm files */
static Boolean noAlias = FALSE;  /* set to zap all aliases in hmmlist */
static Boolean inBinary = FALSE; /* set to save models in binary */
static char * mmfFn  = NULL;     /* output MMF file, if any */
static int  cmdTrace    = 0;     /* trace level from command line */
static int  trace    = 0;        /* current trace level */

static char *nmfFn = NULL;	 /* cz277 - ANN */

/* Global Data Structures */

static HMMSet hSet;        /* current HMM set */
static HMMSet *hset;       /* current HMM set */
static int fidx;           /* current macro file id */
static int maxStates;      /* max number of states in current HMM set */
static int maxMixes;       /* max number of mixes in current HMM set */
/* cz277 - ANN */
static HMMSet *auxSet = NULL;	/* the HMMSet used for auxiliary purpose */
static char *auxFn = NULL;	/* the file path of the auxiliry HMMSet */

static Source source;      /* the current input file */

static MixtureElem *joinSet;           /* current join Mix Set */
static int nJoins;                     /* current num mixs in joinSet */
static int joinSize=0;                 /* number of mixes in a joined pdf */
static float joinFloor;                /* join mix weight floor (* MINMIX) */
static Boolean badGC = FALSE;          /* set TRUE if gConst out of date */
static float meanGC,stdGC;             /* mean and stdev of GConst */
static Boolean occStatsLoaded = FALSE; /* set when RO/LS has loaded occ stats */
static float outlierThresh = -1.0;     /* outlier threshold set by RO cmd */

static int thisCommand;                /* index of current command */
static int lastCommand=0;              /* index of previous command */
static Boolean equivState = TRUE;      /* TRUE if states can be equivalent */
                                       /*  but not identical */
static Boolean useModelName = TRUE;    /* Use base-phone name as tree name */
static Boolean saveHMMSet   = TRUE;    /* Save the HMMSet */

/* ---------------- Configuration Parameters --------------------- */

static ConfParam *cParm[MAXGLOBS];
static int nParm = 0;            /* total num params */
static Boolean treeMerge = TRUE; /* After tree spltting merge leaves */
static char tiedMixName[MAXSTRLEN] = "TM"; /* Tied mixture base name */
static char mmfIdMask[MAXSTRLEN] = "*"; /* MMF Id Mask for baseclass */
static Boolean useLeafStats = TRUE; /* Use leaf stats to init macros */
static Boolean applyVFloor = TRUE; /* apply modfied varFloors to vars in model set */ 

/* ------------------ Process Command Line -------------------------- */

void SetConfParms(void)
{
   Boolean b;
   int i;

   nParm = GetConfig("HHED", TRUE, cParm, MAXGLOBS);
   if (nParm>0) {
      if (GetConfInt(cParm,nParm,"TRACE",&i)) trace = i;
      if (GetConfBool(cParm,nParm,"TREEMERGE",&b)) treeMerge = b;
      if (GetConfBool(cParm,nParm,"USELEAFSTATS",&b)) useLeafStats = b;
      if (GetConfBool(cParm,nParm,"APPLYVFLOOR",&b)) applyVFloor = b;
      if (GetConfBool(cParm,nParm,"USEMODELNAME",&b)) useModelName = b;
      GetConfStr(cParm,nParm,"TIEDMIXNAME",tiedMixName);
      GetConfStr(cParm,nParm,"MMFIDMASK",mmfIdMask);
   }
}

void Summary(void)
{
   printf("\nHHEd Command Summary\n\n");
   printf("AF macro feamixDef   - Add Feature mixture macro defined by feamixDef\n");
   printf("AM macro matrixDef   - Add Matrix macro defined by matrixDef\n");
   printf("AT i j prob itemlist - Add Transition from i to j in given mats\n");
   printf("AV macro vectorDef   - Add Vector macro defined by vectorDef\n");
   printf("AU hmmlist           - Add Unseen triphones in given hmmlist to\n");
   printf("                       currently loaded HMM list using previously\n");
   printf("                       built decision trees.\n");
   printf("CA macro actfunDef   - Change the Activation function of layer macro to actfunDef\n");
   printf("CD macro o i         - Change the output/input Dims of layer macro and its component to o/i");
   printf("CF lmacro fmacro ... - Change the Feature mixture of layer lmacro to fmacro\n");
   printf("CH mmf lst macro ... - Connect ANN model macro defined in mmf and lst to current HMM set");
   printf("CL hmmList           - CLone hmms to give new hmmList\n");
   printf("CO newHmmList        - COmpact identical HMM's by sharing same phys model\n");
   printf("CP mmf lst ...       - CoPy parameters of a macro in mmf to a macro in current model set\n");
   printf("DL macro             - DeLete an ANN layer macro from all ANN models\n");
   printf("DP s n id ...        - Duplicate the hmm set n times using id to differentiate\n");
   printf("                       the new hmms and macros.  Only macros the type of which\n");
   printf("                       appears in s will be duplicated, others will be shared.\n");
   printf("EF macro ...         - Extend the Feature mixture macro or of layer macro with a new element\n");
   printf("                       if macro is a layer, then its feature mixture will be cloned first.\n");
   printf("EL macro             - Erase (re-initialise) an ANN Layer\n");
   printf("                       different random value generation formulas used for different layers.\n");
   printf("FA f                 - Set variance floor to average within state variance * f\n");
   printf("FV vFloorfile        - Load variance floor from file\n");
   printf("FC                   - Convert diagonal variances to full covariances\n");
   printf("HK hsetkind          - change current set to hsetkind\n");
   printf("JO size floor        - set size and min mix weight for a JOin\n");
   printf("IL nmacro pos lmacro - Insert an ANN Layer lmacro at pos of ANN model nmacro\n");
   printf("LS statsfile         - load named statsfile\n");
   printf("LT filename          - Load Questions and Trees from filename\n");
   printf("MD n itemlist        - MixDown command, change mixtures in itemlist to n\n");
   printf("MM s itemlist        - make each item in list into a macro with usage==1\n");
   printf("MT triHmmList        - Make Triphones from loaded biphones\n");
   printf("MU n itemlist        - MixUp command, change mixtures in itemlist to n\n");
   printf("NC n macro itemlist  - N-Cluster specified components and tie\n");
   printf("QS name itemlist     - define a question as a list of model names\n");
   printf("RC n id [itemList]   - Build n regression classes (for adaptation purposes)\n");
   printf("                       this disables the storing of the models\n");  
   printf("PR                   - Convert model-set with PROJSIZE to compact form\n");
   printf("                       also supplying a regression tree identifier/label name\n");
   printf("                       Optional itemList to specify non-speech sounds\n");
   printf("RM hmmfile           - rem mean in state 2, mix 1 of hmmfile from all \n");
   printf("                       loaded models nb. whole mean is removed incl. dels\n");
   printf("RN hmmSetIdentifier  - Rename the hmm mmf with a new identifier name\n");
   printf("                       If omitted and MMF contains no identifier, then\n");
   printf("                       the MMF is given the identifier \"Standard\"\n");
   printf("RO f [statsfile]     - Remove outliers with counts < f as the\n");
   printf("                       final phase in the TC/NC commands.  If statsfile\n");
   printf("                       is omitted, it must be already loaded (see LS)\n");
   printf("RT i j itemlist      - Rem Transition from i to j in given mats\n");
   printf("SH                   - show the current HMM set (for debugging)\n");
   printf("SK sk                - Set sample kind of all models to sk\n");
   printf("SM macro <VALUES>... - Set each value of the ANN matrix macro to the input ones\n");
   printf("SS n                 - Split into n data Streams\n");
   printf("ST filename          - Save Questions and Trees to filename\n");
   printf("SU n w1 .. wn        - Split into user defined stream widths\n");
   printf("SV macro <VALUES>... - Set each value of the ANN vector macro to the input ones\n");
   printf("SW s n               - Set width of stream s to n\n");
   printf("TB f macro itemlist  - Tree build using QS questions and likelihood\n");
   printf("                       based clustering criterion.\n");
   printf("TC f macro itemlist  - Thresh Cluster specified comps to thresh f and tie\n");
   printf("TI macro itemlist    - TIe the specified components\n");
   printf("TR n                 - set trace level to n (overrides -T option)\n");
   printf("UT itemlist          - UnTie the specified components\n");
   printf("XF filename          - Set the Input Xform to filename\n");
   Exit(0);
}

void ReportUsage(void)
{
   printf("\nUSAGE: HHEd [options] editF hmmList\n\n");
   printf(" Option                                       Default\n\n");
   printf(" -d s    dir to find hmm definitions          current\n");
   printf(" -n mmf  Save all ANNs macro file mmf s       as source\n");
   printf(" -o s    extension for new hmm files          as source\n");
   printf(" -w mmf  Save all HMMs to macro file mmf s    as source\n");
   printf(" -x s    extension for hmm files              none\n");
   printf(" -z      zap aliases in hmmList\n");
   PrintStdOpts("BHMQ");
}

int main(int argc, char *argv[])
{
   char *s, *editFn;
   void DoEdit(char * editFn);
   void ZapAliases(void);
   void Initialise(char *hmmListFn);
   
   if(InitShell(argc,argv,hhed_version,hhed_vc_id)<SUCCESS)
      HError(2600,"HHEd: InitShell failed");
   InitMem();   InitLabel();
   InitMath();  InitSigP();
   InitWave();  InitAudio();
   InitVQ();    InitModel();
   if(InitParm()<SUCCESS)  
      HError(2600,"HHEd: InitParm failed");
   InitUtil();

   if (!InfoPrinted() && NumArgs() == 0)
      ReportUsage();
   if (NumArgs() == 0) Exit(0);
   SetConfParms();
 
   CreateHeap(&hmmHeap,"Model Heap",MSTAK,1,1.0,40000,400000);
   CreateHMMSet(&hSet,&hmmHeap,TRUE);hset=&hSet;fidx=0;

   while (NextArg() == SWITCHARG) {
      s = GetSwtArg();
      if (strlen(s)!=1) 
         HError(2619,"HHEd: Bad switch %s; must be single letter",s);
      switch(s[0]) {
      case 'd':
         if (NextArg()!=STRINGARG)
            HError(2619,"HHEd: Input HMM definition directory expected");
         hmmDir = GetStrArg(); break;  
      case 'n':
         if (NextArg() != STRINGARG)
            HError(2619, "HHEd: Output ANN def file name expected");
         nmfFn = GetStrArg();
         break;
      case 'o':
         if (NextArg()!=STRINGARG)
            HError(2619,"HHEd: Output HMM file extension expected");
         newExt = GetStrArg(); break;
      case 'w':
         if (NextArg()!=STRINGARG)
            HError(2619,"HHEd: Output MMF file name expected");
         mmfFn = GetStrArg();
         break;
      case 'x':
         if (NextArg()!=STRINGARG)
            HError(2619,"HHEd: Input HMM file extension expected");
         hmmExt = GetStrArg(); break;
      case 'z':
         noAlias = TRUE; break;
      case 'B':
         inBinary=TRUE; break;
      case 'J':
         if (NextArg()!=STRINGARG)
            HError(2319,"HERest: input transform directory expected");
         AddInXFormDir(hset,GetStrArg());
         break;              
      case 'H':
         if (NextArg()!=STRINGARG)
            HError(2619,"HHEd: Input MMF file name expected");
         AddMMF(hset,GetStrArg());
         break;
      case 'M':
         if (NextArg()!=STRINGARG)
            HError(2619,"HHEd: Output HMM definition directory expected");
         newDir = GetStrArg(); break;  
      case 'Q':
         Summary(); break;
      case 'T':
         trace = cmdTrace = GetChkedInt(0,0x3FFFF,s); break;
      default:
         HError(2619,"HHEd: Unknown switch %s",s);
      }
   } 
   if (NextArg()!=STRINGARG)
      HError(2619,"HHEd: Edit script file name expected");
   editFn = GetStrArg();
   if (NextArg() != STRINGARG)
      HError(2619,"HHEd: HMM list file name expected");
   if (NumArgs()>1)
      HError(2619,"HHEd: Unexpected extra args on command line");

   Initialise(GetStrArg());

   if (hset->logWt == TRUE) HError(2690,"HHEd requires linear weights");
   DoEdit(editFn);
   Exit(0);
   return (0);          /* never reached -- make compiler happy */
}

/* ----------------------- Lexical Routines --------------------- */

int ChkedInt(char *what,int min,int max)
{
   int ans;

   if (!ReadInt(&source,&ans,1,FALSE))
      HError(2650,"ChkedInt: Integer read error - %s",what);
   if (ans<min || ans>max)
      HError(2651,"ChkedInt: Integer out of range - %s",what);
   return(ans);
}

float ChkedFloat(char *what,float min,float max)
{
   float ans;

   if (!ReadFloat(&source,&ans,1,FALSE))
      HError(2650,"ChkedFloat: Float read error - %s",what);
   if (ans<min || ans>max)
      HError(2651,"ChkedFloat: Float out of range - %s",what);
   return(ans);
}

char *ChkedAlpha(char *what,char *buf)
{
   if (!ReadString(&source,buf))
      HError(2650,"ChkedAlpha: String read error - %s",what);
   return(buf);
}

/* ------------- Question Handling for Tree Building ----------- */

typedef struct _IPat{
   char *pat;
   struct _IPat *next;
}IPat;

typedef struct _QEnt *QLink;   /* Linked list of Questions */
typedef struct _QEnt{           /* each question stored as both pattern and  */
   LabId qName;                 /* an expanded list of model names */
   IPat *patList;               
   ILink ilist;
   QLink next;
}QEnt;

static QLink qHead = NULL;      /* Head of question list */
static QLink qTail = NULL;      /* Tail of question list */

/* TraceQuestion: output given questions */
void TraceQuestion(char *cmd, QLink q)
{
   IPat *ip;
   
   printf("   %s %s: defines %d % d models\n       ",
          cmd,q->qName->name,NumItems(q->ilist),hset->numLogHMM);
   for (ip=q->patList; ip!=NULL; ip=ip->next)
      printf("%s ",ip->pat); 
   printf("\n");
   fflush(stdout);
}

/* ParseAlpha: get next string from src and store it in s */
/* This is a copy of ParseString with extra terminators */
char *ParseAlpha(char *src, char *s)
{
   static char term[]=".,)";      /* Special string terminators */
   Boolean wasQuoted;
   int c,q=0;

   wasQuoted=FALSE;
   while (isspace((int) *src)) src++;
   if (*src == DBL_QUOTE || *src == SING_QUOTE){
      wasQuoted = TRUE; q = *src;
      src++;
   }
   while(*src) {
      if (wasQuoted) {
         if (*src == q) return (src+1);
      } else {
         if (isspace((int) *src) || strchr(term,*src)) return (src);
      }
      if (*src==ESCAPE_CHAR) {
         src++;
         if (src[0]>='0' && src[1]>='0' && src[2]>='0' &&
             src[0]<='7' && src[1]<='7' && src[2]<='7') {
            c = 64*(src[0] - '0') + 8*(src[1] - '0') + (src[2] - '0');
            src+=2;
         } else
            c = *src;
      }
      else c = *src;
      *s++ = c;
      *s=0;
      src++;
   }
   return NULL;
}

/* LoadQuestion: store given question in question list */
void LoadQuestion(char *qName, ILink ilist, char *pattern)
{
   QLink q,c;
   LabId labid;
   IPat *ip;
   char *p,*r,buf[MAXSTRLEN];
   
   q=(QLink) New(&questHeap,sizeof(QEnt));
   q->ilist=ilist;
   labid=GetLabId(qName,TRUE);
   for (c=qHead;c!=NULL;c=c->next) if (c->qName==labid) break;
   if (c!=NULL)
      HError(2661,"LoadQuestion: Question name %s invalid",qName);
   q->qName=labid; labid->aux=q;
   q->next = NULL; q->patList = NULL;
   if (qHead==NULL) {
      qHead = q; qTail = q;
   } else {
      qTail->next = q; qTail = q;
   }
   for (p=pattern;*p && isspace((int) *p);p++);
   if (*p!='{')
      if (p==NULL) HError(2660,"LoadQuestion: no { in itemlist");
   ++p;
   for (r=pattern+strlen(pattern)-1;r>=pattern && isspace((int) *r);r--);
   if (*r!='}') HError(2660,"LoadQuestion: no } in itemlist"); 
   *r = ',';
   do {                         /* pick up model patterns from item list */
      p=ParseAlpha(p,buf);
      while(isspace((int) *p)) p++;
      if (*p!=',')
         HError(2660,"LoadQuestion: missing , in itemlist"); 
      p++;
      ip=(IPat*) New(&questHeap,sizeof(IPat));
      ip->pat = NewString(&questHeap,strlen(buf));
      strcpy(ip->pat,buf);
      ip->next = q->patList; q->patList = ip;
   } while (p<r);
}

/* QMatch: return true if given name matches question */
Boolean QMatch(char *name, QLink q)
{
   IPat *ip;
   
   for (ip=q->patList;ip!=NULL;ip=ip->next)
      if (DoMatch(name,ip->pat)) return TRUE;
   return FALSE;
}

/* ----------------------- HMM Management ---------------------- */

typedef enum { baseNorm=0, baseLeft, baseRight, baseMono } baseType;

/* FindBaseModel: return the idx in hList of base model corresponding 
   to id, type determines type of base model wanted and is 
   baseMono (monophone); baseRight (right biphone); baseLeft (left biphone) */
HLink FindBaseModel(HMMSet *hset,LabId id,baseType type)
{
   char baseName[255],buf[255],*p,*p2;
   LabId baseId;
   MLink ml;
   
   strcpy(baseName,id->name);
   if (type==baseMono || type==baseRight) { /* strip Left context */
      strcpy(buf,baseName);
      if ((p = strchr(buf,'-')) != NULL)
         strcpy(baseName,p+1);
   }
   if (type==baseMono || type==baseLeft) { /* strip Right context */
      strcpy(buf,baseName);
      if ((p = strrchr(buf,'+')) != NULL) {
         *p = '\0';
         strcpy(baseName,buf);
      }
   }
   if ((baseId = GetLabId(baseName,FALSE))==NULL) {
      /* problem could be due to mappings ... try no context/no tone */
      strcpy(buf,baseName);
      if ((p = strrchr(buf,'^')) != NULL) {
         *p = '\0';
         baseId = GetLabId(buf,FALSE);
         if (baseId == NULL) {
            strcpy(buf,baseName);            
            if ((p2 = strrchr(buf,';')) != NULL) {
               /* either tone by itself or position by itself */
               /* try position first */
               *p2 = '\0';
               baseId = GetLabId(buf,FALSE);
               if (baseId== NULL) {
                  /* then try tone */
                  strcpy(buf,baseName);            
                  while (*p2 != '\0') {
                     *p = *p2;
                     p++; p2++;
                  }
                  *p = '\0';
                  baseId = GetLabId(buf,FALSE);
               }
            }
         }
      } else if ((p = strrchr(buf,';')) != NULL) {
         *p = '\0';
         baseId = GetLabId(buf,FALSE);
      }
      if (baseId == NULL)
         HError(2635,"FindBaseModel: No Base Model %s for %s",baseName,id->name);
      strcpy(baseName,buf);
   }
   ml = FindMacroName(hset,'l',baseId);
   if (ml==NULL)
      HError(2635,"FindBaseModel: Cannot Find HMM %s in Current List",baseName);
   return ((HLink) ml->structure);
}

/* OutMacro: output the name of macro associated with structure */
void OutMacro(char type,Ptr structure)
{
   MLink ml;
   
   if ((ml = FindMacroStruct(hset,type,structure))==NULL)
      HError(2635,"OutMacro: Cannot find macro to print");
   printf(" ~%c:%s",type,ml->id->name);
}

/* ShowWhere: print state, stream, mix if necessary */
void ShowWhere(int state, int stream, int mix)
{
   static int j = -1;
   static int s = -1;
   static int m = -1;
   
   if (state != j) {
      printf("\n  State %d: ",state);
      s = -1; m = -1; j = state;
   }
   if (stream != s && stream != -1) {
      printf("\n    Stream %d: ",stream);
      m = -1; s = stream;
   }
   if (mix != m && mix != -1) {
      printf("\n      Mix %d: ",mix);
      m = mix;
   }
}


/* ShowMacros: list macros used by given HMM */
void ShowMacros(HMMDef *hmm)
{
   StateElem *se;
   StateInfo *si;
   MixtureElem *me;
   StreamElem *ste;
   MixPDF *mp;
   Ptr strct;
   int i,j,s;
   char type;
   
   if (GetUse(hmm->transP) > 0) {
      printf("\n   TransP: ");
      OutMacro('t',hmm->transP);
   }
   se = hmm->svec+2;
   for (i=2;i<hmm->numStates;i++,se++) {
      si = se->info;
      if (si->nUse > 0) {
         ShowWhere(i,-1,-1);
         OutMacro('s',si);
      }     
      if (si->dur != NULL && GetUse(si->dur) > 0) {
         ShowWhere(i,-1,-1);
         OutMacro('d',si->dur);
      }
      if (si->weights != NULL && GetUse(si->weights) > 0) {
         ShowWhere(i,-1,-1);
         OutMacro('w',si->weights);
      }
      ste = si->pdf +1;

      if (hset->hsKind==PLAINHS || hset->hsKind==SHAREDHS)
         for (s=1; s<=hset->swidth[0]; s++,ste++) {
            me = ste->spdf.cpdf + 1;
            for (j=1; j<=ste->nMix;j++,me++) {
               if (me->weight > MINMIX) {
                  mp = me->mpdf;
                  if (mp->nUse>0) {
                     ShowWhere(i,s,j); OutMacro('m',mp);
                  }
                  if (GetUse(mp->mean) > 0) {
                     ShowWhere(i,s,j); OutMacro('u',mp->mean);
                  }
                  switch(mp->ckind) {
                  case FULLC:
                     strct=mp->cov.inv; type='i'; break;
                  case LLTC:
                     strct=mp->cov.inv; type='c'; break;
                  case XFORMC:
                     strct=mp->cov.xform; type='x'; break;
                  case DIAGC:
                  case INVDIAGC:
                     strct=mp->cov.var; type='v'; break;
                  default:
                     strct=NULL; type=0; break;
                  }
                  if (type && GetUse(strct) > 0) {
                     ShowWhere(i,s,j);
                     OutMacro(type,strct);
                  }
               }
            }
         }
   }
}

/* ShowHMMSet: print a summary of currently loaded HMM Set */
void ShowHMMSet(void)
{
   HMMScanState hss;
   LabId id;
   int h=0;
   
   printf("Summary of Current HMM Set\n");
   NewHMMScan(hset,&hss);
   do {
      id = hss.mac->id;
      h++;
      printf("\n%d.Name: %s",h,id->name);
      ShowMacros(hss.hmm);
   }
   while(GoNextHMM(&hss));
   printf("\n------------\n"); fflush(stdout);
}

/* ZapAliases: reduce hList to just distinct physical HMM's */
void ZapAliases(void)
{
   int h;
   MLink q;
   HLink hmm;
   
   for (h=0; h<MACHASHSIZE; h++)
      for (q=hset->mtab[h]; q!=NULL; q=q->next)
         if (q->type=='l')
            DeleteMacro(hset,q);

   for (h=0; h<MACHASHSIZE; h++)
      for (q=hset->mtab[h]; q!=NULL; q=q->next)
         if (q->type=='h') {
            hmm=(HLink) q->structure;
            NewMacro(hset,fidx,'l',q->id,hmm);
            hmm->nUse=1;
         }
}

/* EquivMix: return TRUE if both states are identical */
Boolean EquivMix(MixPDF *a, MixPDF *b)
{
   if (a->mean != b->mean ||
       a->cov.var != b->cov.var ) return FALSE;
   return TRUE;
}

/* EquivStream: return TRUE if both streams are identical */
Boolean EquivStream(StreamElem *a, StreamElem *b)
{
   int m,M;
   MixtureElem *mea,*meb;
   MixPDF *ma,*mb;
   
   mea = a->spdf.cpdf+1; meb = b->spdf.cpdf+1;
   M = a->nMix;
   for (m=1; m<=M; m++,mea++,meb++) {   
      /* This really should search b for matching mixture */
      if (M>1 && mea->weight != meb->weight)
         return FALSE;
      ma = mea->mpdf; mb = meb->mpdf;
      if (ma != mb && !EquivMix(ma,mb))
         return FALSE;
   }
   return TRUE;
}

/* EquivState: return TRUE if both states are identical (only CONT) */
Boolean EquivState(StateInfo *a, StateInfo *b, int S)
{
   int s;
   StreamElem *stea,*steb;
   
   
   stea = a->pdf+1; steb = b->pdf+1;
   for (s=1; s<=S; s++,stea++,steb++) {
      if ((stea->nMix!=steb->nMix) || !EquivStream(stea,steb))
         return FALSE;
   }
   return TRUE;
}

/* EquivHMM: return TRUE if both given hmms are identical */
Boolean EquivHMM(HMMDef *a, HMMDef *b)
{
   int i;
   StateInfo *ai,*bi;
   
   if (a->numStates!=b->numStates ||
       a->transP!=b->transP) return FALSE;
   for (i=2;i<a->numStates;i++) {
      ai=a->svec[i].info; bi=b->svec[i].info;
      if (ai!=bi && (!equivState ||
                     hset->hsKind==TIEDHS || hset->hsKind==DISCRETEHS ||
                     !EquivState(ai, bi, hset->swidth[0])))
         return FALSE;
   }
   return TRUE;
}

/* PurgeMacros: purge all unused macros */
void PurgeMacros(HMMSet *hset)
{
   int h,n;
   MLink q;
   
   /* First mark all macros with usage > 0 */
   ResetHooks(hset,NULL);
   SetVFloor(hset,NULL,0.0);
   for (h=0; h<MACHASHSIZE; h++)
      for (q=hset->mtab[h]; q!=NULL; q=q->next) {
         if ((q->type=='*') || (q->type=='a')|| (q->type=='a'))  continue;
         n=GetMacroUse(q);
         if (n>0 && !IsSeen(n)) {
            Touch(&n);
            SetMacroUse(q,n);
         }
      }

   /* Delete unused macros in next pass */
   for (h=0; h<MACHASHSIZE; h++)
      for (q=hset->mtab[h]; q!=NULL; q=q->next) {
         if ((q->type=='*') || (q->type=='a')|| (q->type=='a')) continue;
         n=GetMacroUse(q);
         if (!IsSeen(n)) {
            SetMacroUse(q,0);
            DeleteMacro(hset,q);
         }
      }

   /* Finally unmark all macros */
   for (h=0; h<MACHASHSIZE; h++)
      for (q=hset->mtab[h]; q!=NULL; q=q->next) {
         if ((q->type=='*') || (q->type=='a')|| (q->type=='a')) continue;
         n=GetMacroUse(q);
         if (IsSeen(n)) {
            Untouch(&n);
            SetMacroUse(q,n);
         }
      }

}

/* -------------------- Vector/Matrix Resizing Operations -------------------- */

/* SwapMacro: swap macro from old to new, and delete old */
void SwapMacro(HMMSet *hset,char type,Ptr o, Ptr n)
{
   MLink ml;
   
   ml = FindMacroStruct(hset,type,o);
   if (trace & (T_MAC | T_DET)) {
      printf("   Swapping Macro %s to resized vector",ml->id->name);
      fflush(stdout);
   }
   DeleteMacro(hset,ml);
   NewMacro(hset, fidx, type, ml->id, n);
}

void SetVSize(SVector v,int n)
{
   *((int*) v)=n;
}

void SetMSize(STriMat m,int n)
{
   *((int*) m)=n;
}

/* ResizeSVector: resize the given vector to n components, if
   extension is needed, new elements have pad value */
SVector ResizeSVector(HMMSet *hset,SVector v, int n, char type, float pad)
{
   int i,u,z;
   SVector w;

   u = GetUse(v);
   if (u==0 || !IsSeen(u)) {
      if (u!=0)
         TouchV(v);
      z = VectorSize(v);
      if (z >= n) {
         w = v;
         SetVSize(v,n);
      }
      else {
         w = CreateSVector(&hmmHeap,n);
         for (i=1;i<=z;i++) w[i] = v[i];
         for (i=z+1;i<=n;i++) w[i] = pad;
         if (u!=0)
            SwapMacro(hset,type,v,w);
      }
   }
   else {
      w=(SVector) GetHook(v);
      IncUse(w);
   }
   return(w);
}

/* ResizeSTriMat: resize the given matrix to be n x n, if
   extension is needed, new diag elements have pad value
   and the rest are zero */
STriMat ResizeSTriMat(HMMSet *hset,STriMat m, int n, char type, float pad)
{
   int i,j,u,z;
   STriMat w;

   u = GetUse(m);
   if (u==0 || !IsSeen(u)) {
      if (u!=0)
         TouchV(m);
      z = NumRows(m);
      if (z >= n) {
         w = m;
         SetMSize(m,n);
      }
      else {
         w = CreateSTriMat(&hmmHeap,n);
         CovInvert(m,m);
         ZeroTriMat(w);
         for (i=1; i<=z; i++)
            for (j=1;j<=i; j++)
               w[i][j] = m[i][j];
         for (i=z+1; i<=n; i++) 
            w[i][i] = pad;
         CovInvert(w,w);
         if (u!=0)
            SwapMacro(hset,type,m,w);
      }
   }
   else {
      w=(STriMat) GetHook(m);
      IncUse(w);
   }
   return(w);
}

/* ----------------- Stream Splitting Operations ------------------- */

/* SliceVector: return vec[i..j] as a vector */
Vector SliceVector(Vector vec, int i, int j)
{
   Vector w;
   int k,l;
   
   w = CreateSVector(&hmmHeap,j-i+1);
   for (l=1,k=i; k<=j; k++,l++)
      w[l] = vec[k];
   return w;
}

/* SliceTriMat: return submatrix mat[i..j,i..j] */
Matrix SliceTriMat(Matrix mat, int i, int j)
{
   TriMat w;
   int k,l,m,n;
   
   w = CreateSTriMat(&hmmHeap,j-i+1);
   for (l=1,k=i; k<=j; k++,l++)
      for (n=1,m=i; m<=k; m++,n++)
         w[l][n] = mat[k][m];
   return w;
}

/* ChopVector: return i,j,k elements of vec as a vector,
   element k is optional */
Vector ChopVector(Vector vec, int i, int j, int k)
{
   Vector w;
   int size;
   
   size =  (k>0) ? 3 : 2;
   w = CreateSVector(&hmmHeap,size);
   w[1] = vec[i]; w[2] = vec[j];
   if (k>0) w[3] = vec[k];
   return w;
}

/* ChopTriMat: return submatrix formed by extracting the intersections
   of i, j, and k from trimat mat, k'th is optional */
TriMat ChopTriMat(TriMat mat, int i, int j, int k)
{
   TriMat w;
   int size;
   
   size =  (k>0) ? 3 : 2;
   w = CreateSTriMat(&hmmHeap,size);
   w[1][1] = mat[i][i];
   w[2][1] = mat[j][i]; w[2][2] = mat[j][j];
   if (k>0) {
      w[1][3] = mat[i][k];
      w[2][3] = mat[j][k];
      w[3][3] = mat[k][k];
   }
   return w;
}

/* SplitStreams: split streams of given HMM as per swidth info */
void SplitStreams(HMMSet *hset,StateInfo *si,Boolean simple,Boolean first)
{
   int j,s,S,m,M,width,next;
   StreamElem *ste,*oldste;
   MixtureElem *me,*oldme;
   MixPDF *mp, *oldmp;
   Boolean hasN;
   int epos[4];
   int eposIdx;
   
   S = hset->swidth[0];
   hasN = HasNulle(hset->pkind);
   oldste = si->pdf+1;
   ste = (StreamElem *)New(hset->hmem,S*sizeof(StreamElem));
   si->pdf = ste - 1;
   next = 1;      /* start of next slice to take from src vector */
   eposIdx = 0;
   for (j=0; j<3; j++) epos[j] = 0;
   for (s=1;s<=S;s++,ste++) {
      width = hset->swidth[s];
      M = ste->nMix = oldste->nMix;
      ste->hook = NULL;
      me = (MixtureElem *) New(hset->hmem,M*sizeof(MixtureElem));
      ste->spdf.cpdf = me-1;
      oldme=oldste->spdf.cpdf+1;
      for (m=1; m<=M; m++, me++,oldme++) {
         oldmp = oldme->mpdf;
         if (oldmp->nUse>0)
            HError(2640,"SplitStreams: MixPDF is shared");
         if (GetUse(oldmp->mean) > 0)
            HError(2640,"SplitStreams: Mean is shared");
         switch(oldmp->ckind) {
         case DIAGC:
            if (GetUse(oldmp->cov.var) > 0)
               HError(2640,"SplitStreams: variance is shared");
            break;
         case LLTC:
         case FULLC:
            if (GetUse(oldmp->cov.inv) > 0)
               HError(2640,"SplitStreams: covariance is shared");
            break;
         default:
            HError(2640,"SplitStreams: not implemented for CovKind==%d",
                   oldmp->ckind);
            break;
         }
         me->weight = oldme->weight;
         mp = (MixPDF *) New(hset->hmem,sizeof(MixPDF));
         mp->nUse = 0; mp->hook = NULL; mp->gConst = LZERO;
         mp->ckind = oldmp->ckind;
         me->mpdf = mp;
         if (simple || s<S) {    
            /* straight segmentation of original stream */
            if ((trace & (T_SIZ | T_IND)) && first)
               printf("  Slicing  at stream %d, [%d->%d]\n",
                      s,next,next+width-1);
            mp->mean = SliceVector(oldmp->mean,next,next+width-1);
            if (mp->ckind == DIAGC)
               mp->cov.var = SliceVector(oldmp->cov.var,next,next+width-1);
            else
               mp->cov.inv = SliceTriMat(oldmp->cov.inv,next,next+width-1);
         } else {
            if ((trace & (T_SIZ | T_DET)) && first)
               printf("  Chopping at stream %d, [%d,%d,%d]\n",
                      s,epos[0],epos[1],epos[2]);
            mp->mean = ChopVector(oldmp->mean,epos[0],epos[1],epos[2]);
            if (oldmp->ckind == DIAGC)
               mp->cov.var = ChopVector(oldmp->cov.var,
                                        epos[0],epos[1],epos[2]);
            else
               mp->cov.inv = ChopTriMat(oldmp->cov.inv,
                                        epos[0],epos[1],epos[2]);
         }  
      }
      next += width;
      if (!simple && !(hasN && s==1)) {   /* remember positions of E coefs */
         epos[eposIdx++] = next; ++next;
      }
   }
   si->weights = CreateSVector(hset->hmem,S);
   for (s=1; s<=S; s++)
      si->weights[s] = 1.0;
}


/* -------------------- Tying Operations --------------------- */

/*
  TypicalState: extract a typical state from given list ie the
  state with largest total gConst in stream 1 (indicating broad 
  variances) and as few as possible zero mixture weights
*/
ILink TypicalState(ILink ilist, LabId macId)
{
   ILink i,imax;
   LogFloat gsum,gmax;
   int m,M;
   StateInfo *si;
   StreamElem *ste;
   
   gmax = LZERO; imax = NULL;
   for (i=ilist; i!=NULL; i=i->next) {
      si = ((StateElem *)(i->item))->info;
      ste = si->pdf+1;
      M = ste->nMix;
      gsum = 0;
      for (m=1; m<=M; m++)
         if (ste->spdf.cpdf[m].weight>MINMIX)
            gsum += ste->spdf.cpdf[m].mpdf->gConst;
         else
            gsum -= 200.0;      /* arbitrary penaltly */
      if (gsum>gmax) {
         gmax = gsum; imax = i;
      }
   }
   if (imax==NULL) {
      HError(-2638,"TypicalState: No typical state for %s",macId->name);
      imax=ilist;
   }
   return imax;
}

/* TieState: tie all state info fields in ilist to typical state */
void TieState(ILink ilist, LabId macId)
{
   ILink i,ti;
   StateInfo *si,*tsi;
   StateElem *se;
   
   if (badGC) {
      FixAllGConsts(hset);         /* in case any bad gConsts around */
      badGC=FALSE;
   }
   if (hset->hsKind==TIEDHS || hset->hsKind==DISCRETEHS)
      ti=ilist;
   else
      ti = TypicalState(ilist,macId);
   se = (StateElem *) ti->item;
   tsi = se->info;
   tsi->nUse = 1;
   NewMacro(hset,fidx,'s',macId,tsi);
   for (i=ilist; i!=NULL; i=i->next) {
      se = (StateElem *)i->item; si = se->info;
      if (si != tsi) {
         se->info = tsi;
         ++tsi->nUse;
      }
   }
}

/* TieTrans: tie all transition matrices in ilist to 1st */
void TieTrans(ILink ilist, LabId macId)
{
   ILink i;
   SMatrix tran;
   HMMDef *hmm;
   
   hmm = ilist->owner;
   tran = hmm->transP;
   SetUse(tran,1); 
   NewMacro(hset,fidx,'t',macId,tran);
   for (i=ilist; i!=NULL; i=i->next) {
      hmm = i->owner;
      if (hmm->transP != tran) {
         hmm->transP = tran;
         IncUse(tran);
      }
   }
}

/* TieDur: tie all dur vectors in ilist to 1st */
void TieDur(ILink ilist, LabId macId)
{
   ILink i;
   StateInfo *si;
   SVector v;
   
   si = (StateInfo *)ilist->item;
   if ((v=si->dur)==NULL)
      HError(2630,"TieDur: Attempt to tie null duration as macro %s",
             macId->name);
   SetUse(v,1);
   NewMacro(hset,fidx,'d',macId,v);
   for (i=ilist->next; i!=NULL; i=i->next) {
      si = (StateInfo *)i->item;
      if (si->dur==NULL)
         HError(-2630,"TieDur: Attempt to tie null duration to macro %s",
                macId->name);
      if (si->dur != v) {
         si->dur = v;
         IncUse(v);
      }
   }
}

/* TieWeights: tie all stream weight vectors in ilist to 1st */
void TieWeights(ILink ilist, LabId macId)
{
   ILink i;
   StateInfo *si;
   SVector v;
   
   si = (StateInfo *)ilist->item;
   v = si->weights;
   if (v==NULL)
      HError(2630,"TieWeights: Attempt to tie null stream weights as macro %s",
             macId->name);
   SetUse(v,1);
   NewMacro(hset,fidx,'w',macId,v);
   for (i=ilist->next; i!=NULL; i=i->next) {
      si = (StateInfo *)i->item;
      if (si->weights==NULL)
         HError(-2630,"TieWeights: Attempt to tie null stream weights to macro %s",macId->name);
      if (si->weights != v) {
         si->weights = v;
         IncUse(v);
      }
   }
}

/* VAdd: add vector b to vector a */
void VAdd(Vector a, Vector b)
{
   int k,V;
   
   V = VectorSize(a);
   for (k=1; k<=V; k++) a[k] += b[k];
}

/* VMax: max vector b to vector a */
void VMax(Vector a, Vector b)
{
   int k,V;
   
   V = VectorSize(a);
   for (k=1; k<=V; k++) 
      if (b[k]>a[k]) a[k] = b[k];
}

/* VNorm: divide vector a by n */
void VNorm(Vector a, int n)
{
   int k,V;
   
   V = VectorSize(a);
   for (k=1; k<=V; k++) a[k] /= n;
}

/* TieMean: tie all mean vectors to average */
void TieMean(ILink ilist, LabId macId)
{
   ILink i;
   MixPDF *mp;
   Vector tmean;
   int tSize,vSize;
   
   mp = (MixPDF *)ilist->item; tmean = mp->mean;
   tSize = VectorSize(tmean);
   SetUse(tmean,1);
   NewMacro(hset,fidx,'u',macId,tmean);
   for (i=ilist->next; i!=NULL; i=i->next) {
      mp = (MixPDF *)i->item;
      if (mp->mean != tmean) {
         vSize = VectorSize(mp->mean);
         if (tSize != vSize)
            HError(2630,"TieMean: Vector size mismatch %d vs %d",
                   tSize, vSize);
         VAdd(tmean,mp->mean);
         mp->mean = tmean;
         IncUse(tmean);
      }
   }
   VNorm(tmean,GetUse(tmean));
}

/* TieVar: tie all variance vectors in ilist to max */
void TieVar(ILink ilist, LabId macId)
{
   ILink i;
   MixPDF *mp;
   Vector tvar;
   int tSize,vSize;
   
   mp = (MixPDF *)ilist->item; tvar = mp->cov.var;
   tSize = VectorSize(tvar);
   SetUse(tvar,1);
   NewMacro(hset,fidx,'v',macId,tvar);
   for (i=ilist->next; i!=NULL; i=i->next) {
      mp = (MixPDF *)i->item;
      if (mp->cov.var!=tvar) {
         vSize = VectorSize(mp->cov.var);
         if (tSize != vSize)
            HError(2630,"TieVar: Vector size mismatch %d vs %d",
                   tSize, vSize);
         VMax(tvar,mp->cov.var);
         mp->cov.var = tvar; IncUse(tvar);
      }
   }
   badGC=TRUE;                  /* gConsts no longer valid */
}

/* TieInv: tie all invcov mats in ilist to first */
void TieInv(ILink ilist, LabId macId)
{
   ILink i;
   MixPDF *mp;
   STriMat tinv;
   int tSize,mSize;
   
   mp = (MixPDF *)ilist->item; tinv = mp->cov.inv;
   tSize = NumRows(mp->cov.inv);
   SetUse(tinv,1);
   NewMacro(hset,fidx,'i',macId,tinv);
   for (i=ilist->next; i!=NULL; i=i->next) {
      mp = (MixPDF *)i->item;
      if (mp->cov.inv!=tinv) {
         mSize = NumRows(mp->cov.inv);
         if (tSize != mSize)
            HError(2630,"TieInv: Matrix size mismatch %d vs %d",
                   tSize, mSize);
         mp->cov.inv = tinv; IncUse(tinv);
      }
   }
   badGC=TRUE;                  /* gConsts no longer valid */
}

/* TieXform: tie all xform mats in ilist to first */
void TieXform(ILink ilist, LabId macId)
{
   ILink i;
   MixPDF *mp;
   SMatrix txform;
   int tr,tc,mr,mc;
   
   mp = (MixPDF *)ilist->item; txform = mp->cov.xform;
   tr = NumRows(mp->cov.xform); tc = NumCols(mp->cov.xform);
   SetUse(txform,1);
   NewMacro(hset,fidx,'x',macId,txform);
   for (i=ilist->next; i!=NULL; i=i->next) {
      mp = (MixPDF *)i->item;
      if (mp->cov.xform!=txform) {
         mr = NumRows(mp->cov.xform); mc = NumCols(mp->cov.xform);
         if (tc != mc || tr != mr)
            HError(2630,"TieXform: Matrix size mismatch");
         mp->cov.xform = txform; IncUse(txform);
      }
   }
   badGC=TRUE;                  /* gConsts no longer valid */
}

/* TieMix: tie all mix pdf's in ilist to 1st */
void TieMix(ILink ilist, LabId macId)
{
   ILink i;
   MixtureElem *me;
   MixPDF *tmpdf;
   
   me = (MixtureElem *)ilist->item; /* should choose max(Sigma) for */
   tmpdf = me ->mpdf;           /* consistency with tie states */
   tmpdf->nUse = 1;
   NewMacro(hset,fidx,'m',macId,tmpdf);
   for (i=ilist->next; i!=NULL; i=i->next) {
      me = (MixtureElem *)i->item;
      if (me->mpdf!=tmpdf) {
         me->mpdf = tmpdf;
         ++tmpdf->nUse;
      }
   }
}

/* CreateJMixSet: create an array of 'joinSize' mixture elems */
MixtureElem * CreateJMixSet(void)
{
   MixtureElem *me;
   int m;
   
   me = (MixtureElem*) New(&hmmHeap,sizeof(MixtureElem)*joinSize);
   --me;
   for (m=1;m<=joinSize;m++) {
      me[m].weight = 0.0;
      me[m].mpdf = NULL;
   }
   return me;
}

/* AddJMix: add given mixture elem to joinSet inserting it in
   order of mixture weight */
void AddJMix(MixtureElem *me)
{
   int i,j;
   
   if (nJoins<joinSize || me->weight > joinSet[nJoins].weight) {
      if (nJoins == joinSize) /* if full discard last mix */
         --nJoins;
      i = 1;                    /* find insertion point */
      while (me->weight < joinSet[i].weight) i++;
      for (j=nJoins; j>=i; j--) /* make space */
         joinSet[j+1] = joinSet[j];
      joinSet[i] = *me;         /* insert new mix */
      ++nJoins;
   }
}

/* RemTop: remove and return top mix in join set */
MixtureElem RemTop(void)
{
   int j;
   MixtureElem me;
   
   if (nJoins==0)
      HError(2691,"RemTop: Attempt to rem top of empty join set");
   me = joinSet[1];
   for (j=1;j<nJoins;j++)
      joinSet[j] = joinSet[j+1];
   joinSet[nJoins].weight = 0.0;
   --nJoins;
   return me;
}

/* SplitMix: split given mix mi into two */
void SplitMix(MixtureElem *mi,MixtureElem *m01,MixtureElem *m02,int vSize)
{
   const float pertDepth = 0.2;
   int k,splitcount;
   float x;
   TriMat mat=NULL;
   
   splitcount = (int) (long int)mi->mpdf->hook + 1;
   m01->mpdf = CloneMixPDF(hset,mi->mpdf,FALSE); 
   m02->mpdf = CloneMixPDF(hset,mi->mpdf,FALSE);
   m01->weight = m02->weight = mi->weight/2.0;
   m01->mpdf->hook = m02->mpdf->hook = (void *)(long int)splitcount;
   if (mi->mpdf->ckind==FULLC || mi->mpdf->ckind==LLTC) {
      mat = CreateTriMat(&tmpHeap,vSize);
      CovInvert(mi->mpdf->cov.inv,mat);
   }
   for (k=1;k<=vSize;k++) {     /* perturb them */
      if (mi->mpdf->ckind == FULLC || mi->mpdf->ckind==LLTC)
         x = sqrt(mat[k][k])*pertDepth;
      else
         x = sqrt(mi->mpdf->cov.var[k])*pertDepth;
      m01->mpdf->mean[k] += x;
      m02->mpdf->mean[k] -= x;
   }
   if (mat!=NULL) FreeTriMat(&tmpHeap,mat);
}

/* FixWeights: Fix the weights of me using the old stream info */
void FixWeights(MixtureElem *me, HMMDef *owner, StreamElem *ste)
{
   MixtureElem *sme;
   double w,p,wSum,fSum,maxW,floor,lFloor;
   int i=0,m,sm;
   
   if (trace & T_DET)
      printf("   For  %-12s  ",HMMPhysName(hset,owner));
   maxW = LZERO;
   for (m=1;m<=joinSize;m++) {  /* set weights to SOutP(owner) */
      w=LZERO;
      for (sm=1,sme=ste->spdf.cpdf+1;sm<=ste->nMix;sm++,sme++) {
         Untouch(&me[m].mpdf->nUse);
         p = MOutP(me[m].mpdf->mean,sme->mpdf);
         w = LAdd(w,log(sme->weight)+p);
      }
      me[m].weight = w;
      if (w > maxW) maxW = w; 
   }
   wSum = 0.0; fSum = 0.0; 
   floor = joinFloor*MINMIX; lFloor = log(floor*joinSize);
   for (m=1;m<=joinSize;m++) {  /* make all weights > floor */
      w = me[m].weight-maxW;
      if (w < lFloor) {
         me[m].weight = floor; fSum += floor;
      } else {
         w = exp(w); me[m].weight = w;
         wSum += w;
      }
   }
   if (fSum>=0.9)
      HError(2634,"FixWeights: Join Floor too high");
   if (wSum==0.0)
      HError(2691,"FixWeights: No positive weights");
   wSum /= (1-fSum);            /* finally normalise */
   for (m=1;m<=joinSize;m++) {   
      w = me[m].weight;
      if (w>floor) {
         w /= wSum; me[m].weight = w;
         if (trace & T_DET) {
            if ((++i%4)==0) printf("\n    ");
            printf(" %3d == %7.4e",m,w);
         }
      }
   }
   if (trace & T_DET) {
      printf("\n");
      fflush(stdout);
   }
}

/* Create macros for each mix in joinSet & reset nUse */
void CreateJMacros(LabId rootMacId)
{
   int m;
   char buf[255];
   MixPDF *mp;
   
   for (m=1;m<=joinSize;m++) {
      sprintf(buf,"%s%d",rootMacId->name,m);
      mp = joinSet[m].mpdf;
      mp->nUse = 0;
      NewMacro(hset,fidx,'m',GetLabId(buf,TRUE),mp);
      if (trace & (T_DET | T_MAC)) {
         printf("   Join Macro: ~m:%s created\n",buf);
         fflush(stdout);
      }
   }
}

/* TiePDF: join mixtures for set of 'joinSize' tied pdfs in ilist
   This is standard 'tied mixture' case. Min mix weight
   is joinFloor * MINMIX   */
void TiePDF(ILink ilist, LabId macId)
{
   ILink i;
   int m,nSplit,vs,vSize=0;
   StreamElem *ste;
   MixtureElem *me, mix,mix1,mix2;
   
   if (badGC) {
      FixAllGConsts(hset);         /* in case any bad gConsts around */
      badGC=FALSE;
   }
   if (joinSize==0)
      HError(2634,"TiePDF: Join size and floor not set - use JO command");
   nJoins = 0; joinSet = CreateJMixSet();
   for (i=ilist; i!=NULL; i=i->next) {
      ste = (StreamElem *) i->item;
      vs = VectorSize(ste->spdf.cpdf[1].mpdf->mean);
      if (vSize==0)
         vSize = vs;
      else if (vs != vSize)
         HError(2630,"TiePDF: incompatible vector sizes %d vs %d",vs,vSize);
      for (m=1;m<=ste->nMix;m++) {
         me = ste->spdf.cpdf+m;
         if (IsSeen(me->mpdf->nUse)) continue;
         Touch(&me->mpdf->nUse); /* Make sure we don't add twice */
         AddJMix(me);
      }
   }
   if (trace & T_BID) {
      printf("  Joined %d mixtures.",nJoins);
      if (nJoins==joinSize) 
         printf("  [Full set]\n");
      else
         printf("  [%d more needed]\n",joinSize-nJoins);
      fflush(stdout);
   }
   nSplit=0;
   while (nJoins<joinSize) {    /* split best mixes if needed */
      mix = RemTop();
      SplitMix(&mix,&mix1,&mix2,vSize);
      AddJMix(&mix1); AddJMix(&mix2);
      ++nSplit;
   }

   CreateJMacros(macId);
   for (i=ilist; i!=NULL; i=i->next) { /* replace pdfs */
      ste = (StreamElem *) i->item;
      if (IsSeen(ste->nMix)) continue;
      me = CreateJMixSet();
      for (m=1;m<=joinSize;m++) {
         me[m].mpdf = joinSet[m].mpdf;
         ++(joinSet[m].mpdf->nUse);
      }
      FixWeights(me,i->owner,ste);
      ste->spdf.cpdf = me; ste->nMix = joinSize;
      Touch(&ste->nMix);
   }
   for (i=ilist; i!=NULL; i=i->next) { /* reset nMix flags */
      ste = (StreamElem *) i->item;
      Untouch(&ste->nMix);
   }
   if (joinSize>maxMixes) maxMixes=joinSize;
}

/* TieHMMs: tie all hmms in ilist to 1st */
void TieHMMs(ILink ilist,LabId macId)
{
   ILink i;
   HLink hmm,cur;
   MLink q;
   int h,c,j;
   char type;

   hmm=ilist->owner;    /* get first in list */
   q=FindMacroStruct(hset,'h',hmm);
   DeleteMacro(hset,q);
   NewMacro(hset,fidx,'h',macId,hmm);
   for (i=ilist->next,c=1; i!=NULL; i=i->next,c++)
      i->owner->owner=NULL;
   hmm->owner=hset;    /* In case of duplicates in ilist */
   for (h=0,j=0; h<MACHASHSIZE; h++)
      for (q=hset->mtab[h]; q!=NULL; q=q->next) {
         if (q->type!='h' && q->type!='l') continue;
         cur = (HLink) q->structure; 
         if (cur->owner==NULL) {
            type = q->type;
            DeleteMacro(hset,q);
            if (type=='l')
               NewMacro(hset,fidx,'l',q->id,hmm);
            hmm->nUse++;
            j++;
         }
      }
}

/* ApplyTie: tie all structures in list of give type */
void ApplyTie(ILink ilist, char *macName, char type)
{
   LabId macId;
   int use;
   
   if (ilist == NULL) {
      HError(-2631,"ApplyTie: Macro %s has nothing to tie of type %c",
             macName,type);
      return;
   }
   macId = GetLabId(macName,TRUE);
   switch (type) {
   case 's':   TieState(ilist,macId); break;
   case 't':   TieTrans(ilist,macId); break;
   case 'm':   TieMix(ilist,macId); break;
   case 'u':   TieMean(ilist,macId); break;
   case 'v':   TieVar(ilist,macId); break;
   case 'i':   TieInv(ilist,macId); break;
   case 'x':   TieXform(ilist,macId); break;
   case 'p':   TiePDF(ilist,macId); break;
   case 'w':   TieWeights(ilist,macId); break;
   case 'd':   TieDur(ilist,macId); break;
   case 'h':   TieHMMs(ilist,macId); break;
   }
   if ((trace & (T_DET | T_MAC)) && type!='p') {
      use=GetMacroUse(FindMacroName(hset,type,macId));
      printf("   Macro: ~%c:%s created for %d/%d items\n",
             type,macName,use,NumItems(ilist));
      fflush(stdout);
   }
}

/* ------------------ Untying Operations --------------------- */

/* UntieState: untie (ie clone) all items in ilist */
void UntieState(ILink ilist)
{
   ILink i;
   StateElem *se;
   StateInfo *si;
   int nu;
   
   for (i=ilist; i!=NULL; i=i->next) {
      se = (StateElem *)i->item;
      si = se->info;
      nu = si->nUse; 
      si->nUse = 0;
      if (nu==1)
         DeleteMacroStruct(hset,'s',si);
      else if (nu>1) {
         se->info = CloneState(hset,si,FALSE);
         si->nUse = nu-1;
      }
   }
}

/* UntieMix: untie (ie clone) all items in ilist */
void UntieMix(ILink ilist)
{
   ILink i;
   MixtureElem *me;
   MixPDF *mp;
   int nu;
   
   for (i=ilist; i!=NULL; i=i->next) {
      me = (MixtureElem *)i->item;
      if (me->weight > MINMIX) {
         mp = me->mpdf;
         nu = mp->nUse; 
         mp->nUse = 0;
         if (nu==1)
            DeleteMacroStruct(hset,'m',mp);
         else if (nu>1) {
            me->mpdf = CloneMixPDF(hset,mp,FALSE);
            mp->nUse = nu-1;
         }
      }
   }
}

/* UntieMean: untie (ie clone) all items in ilist */
void UntieMean(ILink ilist)
{
   ILink i;
   MixPDF *mp;
   Vector v;
   int nu;
   
   for (i=ilist; i!=NULL; i=i->next) {
      mp = (MixPDF *)i->item;
      v = mp->mean;
      nu = GetUse(v); 
      SetUse(v,0);
      if (nu==1)
         DeleteMacroStruct(hset,'u',v);
      else if (nu>1) {
         mp->mean = CloneSVector(hset->hmem,v,FALSE);
         SetUse(v,nu-1);
      }
   }
}

/* UntieVar: untie (ie clone) all items in ilist */
void UntieVar(ILink ilist)
{
   ILink i;
   MixPDF *mp;
   Vector v;
   int nu;
   
   for (i=ilist; i!=NULL; i=i->next) {
      mp = (MixPDF *)i->item;
      v = mp->cov.var;
      nu = GetUse(v); 
      SetUse(v,0);
      if (nu==1)
         DeleteMacroStruct(hset,'v',v);
      else if (nu>1) {
         mp->cov.var = CloneSVector(hset->hmem,v,FALSE);
         SetUse(v,nu-1);
      }
   }
}

/* UntieInv: untie (ie clone) all items in ilist */
void UntieInv(ILink ilist)
{
   ILink i;
   MixPDF *mp;
   SMatrix m;
   int nu;
   
   for (i=ilist; i!=NULL; i=i->next) {
      mp = (MixPDF *)i->item;
      m = mp->cov.inv;
      nu = GetUse(m);
      SetUse(m,0);
      if (nu==1)
         DeleteMacroStruct(hset,'i',m);
      else if (nu>1) {
         mp->cov.inv = CloneSTriMat(hset->hmem,m,FALSE);
         SetUse(m,nu-1);
      }
   }
}

/* UntieXform: untie (ie clone) all items in ilist */
void UntieXform(ILink ilist)
{
   ILink i;
   MixPDF *mp;
   SMatrix m;
   int nu;
   
   for (i=ilist; i!=NULL; i=i->next) {
      mp = (MixPDF *)i->item;
      m = mp->cov.xform;
      nu = GetUse(m);
      SetUse(m,0);
      if (nu==1)
         DeleteMacroStruct(hset,'x',m);
      else if (nu>0) {
         mp->cov.xform = CloneSMatrix(hset->hmem,m,FALSE);
         SetUse(m,nu-1);
      }
   }
}

/* UntieTrans: untie (ie clone) all items in ilist */
void UntieTrans(ILink ilist)
{
   ILink i;
   HMMDef *hmm;
   SMatrix m;
   int nu;
   
   for (i=ilist; i!=NULL; i=i->next) {
      hmm = (HMMDef *)i->item;
      m = hmm->transP;
      nu = GetUse(m); 
      SetUse(m,0);
      if (nu==1)
         DeleteMacroStruct(hset,'t',m);
      else if (nu>0) {
         hmm->transP = CloneSMatrix(hset->hmem,m,FALSE);
         SetUse(m,nu-1);
      }
   }
}

/* -------------------- Cluster Operations --------------------- */

typedef struct _CRec *CLink;
typedef struct _CRec{
   ILink item;                  /* a single item in this cluster (group) */
   int idx;                     /* index of this item */
   Boolean ans;                 /* answer to current question */
   short state;                 /* state of clink */
   CLink next;                  /* next item in group */
}CRec;

/* TDistance: return mean sq diff between Tied Mix weights */
float TDistance(StreamElem *s1, StreamElem *s2)
{
   int m,M;
   float *mw1,*mw2;
   float x,sum=0.0;
   
   mw1 = s1->spdf.tpdf+1; mw2 = s2->spdf.tpdf+1; 
   M = s1->nMix;
   for (m=1; m<=M; m++, mw1++, mw2++) {
      x = (*mw1 - *mw2);
      sum += x*x;
   }
   return sqrt(sum/M);
}

/* DDistance: return mean sq log diff between Discrete Probabilities */
float DDistance(StreamElem *s1, StreamElem *s2)
{
   int m,M;
   short *mw1,*mw2;
   float x,sum=0.0;
   
   mw1 = s1->spdf.dpdf+1; mw2 = s2->spdf.dpdf+1; 
   M = s1->nMix;
   for (m=1; m<=M; m++, mw1++, mw2++) {
      x = (*mw1 - *mw2);
      sum += x*x;
   }
   return sqrt(sum/M);
}

/* Divergence: return divergence between two Gaussians */
float Divergence(StreamElem *s1, StreamElem *s2)
{
   int k,V;
   MixPDF *m1,*m2;
   float x,v1,v2,sum=0.0;
   
   m1 = (s1->spdf.cpdf+1)->mpdf; m2 = (s2->spdf.cpdf+1)->mpdf;
   V = VectorSize(m1->mean);
   for (k=1; k<=V; k++) {
      x = m1->mean[k] - m2->mean[k];
      v1 = m1->cov.var[k]; v2 = m2->cov.var[k];
      sum += x*x / sqrt(v1*v2);
      /* The following is closer to the true divergence but the */
      /* increased sensitivity to undertrained variances is undesirable */
      /*    sum += v1/v2 + v2/v1 - 2.0 + (1.0/v1 + 1.0/v2)*x*x; */
   }
   return sqrt(sum/V);
}

/* GDistance: return general distance between two arbitrary pdfs
   by summing the log probabilities of each mixture mean with
   respect to the other pdf  */
float GDistance(int s, StreamElem *s1, StreamElem *s2)
{
   int m,M;
   MixtureElem *me;
   Observation dummy;
   float sum=0.0;
   
   M = s1->nMix;
   for (m=1,me = s1->spdf.cpdf+1; m<=M; m++, me++) {
      dummy.fv[s]=me->mpdf->mean;
      sum += SOutP(hset,s,&dummy,s2);
   }
   M = s2->nMix;
   for (m=1,me = s2->spdf.cpdf+1; m<=M; m++, me++) {
      dummy.fv[s]=me->mpdf->mean;
      sum += SOutP(hset,s,&dummy,s1);
   }
   return -(sum/M);
}

/* StateDistance: return distance between given states */
float StateDistance(ILink i1, ILink i2)
{
   StateElem *se1, *se2;
   StateInfo *si1, *si2;
   StreamElem *ste1,*ste2;
   float x = 0.0;
   int s,S;
   
   se1 = (StateElem *)i1->item; si1 = se1->info;
   se2 = (StateElem *)i2->item; si2 = se2->info;
   S = hset->swidth[0];
   ste1 = si1->pdf+1; ste2 = si2->pdf+1;
   for (s=1;s<=S;s++,ste1++,ste2++)
      if (hset->hsKind == TIEDHS)
         x += TDistance(ste1,ste2);
      else if (hset->hsKind==DISCRETEHS)
         x += DDistance(ste1,ste2);
      else if (maxMixes == 1 && ste1->spdf.cpdf[1].mpdf->ckind==DIAGC && 
               ste2->spdf.cpdf[1].mpdf->ckind==DIAGC) 
         x += Divergence(ste1,ste2);
      else
         x += GDistance(s,ste1,ste2);
   return x/S;
}

/* SetGDist: compute inter group distances */
void SetGDist(CLink *cvec, Matrix id, Matrix gd, int N)
{
   int i,j;
   CLink p,q,pp;
   float maxd;
   
   for (i=1; i<=N; i++) gd[i][i]=0.0;
   for (i=1; i<N; i++) {
      p = cvec[i];
      for (j=i+1; j<=N; j++) {
         q = cvec[j];
         maxd = 0.0;
         while (q != NULL) {
            pp = p;
            while (pp != NULL) {
               if (id[pp->idx][q->idx] > maxd)
                  maxd = id[pp->idx][q->idx];
               pp = pp->next;
            }
            q = q->next;
         }
         gd[i][j] = gd[j][i] = maxd;
      }
   }
}

/* SetIDist: compute inter item distances */
void SetIDist(CLink *cvec, Matrix id, int N, char type)
{
   ILink ii,jj;
   int i,j;
   float dist=0.0;
   
   for (i=1; i<=N; i++) id[i][i]=0.0;
   for (i=1; i<N; i++) {
      ii=cvec[i]->item;
      for (j=i+1; j<=N; j++) {
         jj = cvec[j]->item;
         switch(type) {
         case 's': 
            dist = StateDistance(ii,jj);
            break;
         default:
            HError(2640,"SetIDist: Cant compute distances for %c types",type);
         }
         id[i][j] = id[j][i] = dist;
      }
   }
}

/* MinGDist: find min inter group distance */
float MinGDist(Matrix g, int *ix, int *jx, int N)
{
   int mini,minj;
   float min;
   int i,j;
   
   min = g[1][2]; mini=1; minj=2;
   for (i=1; i<N; i++)
      for (j=i+1; j<=N; j++)
         if (g[i][j]<min) {
            min = g[i][j]; mini = i; minj = j;
         }
   *ix = mini; *jx = minj; 
   return min;
}


/* MergeGroups:  merge the two specified groups */
void MergeGroups(int i, int j, CLink *cvec, int N)
{
   int k;
   CLink p;
   
   p = cvec[i];
   while(p->next != NULL) 
      p = p->next;
   p->next = cvec[j];
   for (k=j; k<N; k++) 
      cvec[k] = cvec[k+1];
}

/* BuildCVec: allocate space for cvec and create item sized groups */
CLink *BuildCVec(int numClust, ILink ilist)
{
   CLink *cvec,p;
   int i;
   
   /* Allocate extra space to allow for call to Dispose(&tmpHeap,cvec) */
   cvec=(CLink*) New(&tmpHeap,(numClust+1)*sizeof(CLink));
   for (i=1;i<=numClust; i++) {
      p=(CLink) New(&tmpHeap,sizeof(CRec));
      if ((p->item = ilist) == NULL)
         HError(2690,"BuildCVec: numClust<NumItems(ilist) %d",i);
      ilist = ilist->next;
      p->idx = i; p->next = NULL;
      cvec[i] = p;
   }
   if (ilist!=NULL)
      HError(2690,"BuildCVec: numClust>NumItems(ilist)");
   return cvec;
}

/* SetOccSums: return a vector of cluster occupation sums.  The occ count
   for each state was stored in the hook of the StateInfo rec by
   the RO command */
Vector SetOccSums(CLink *cvec, int N)
{
   int i;
   float sum,x;
   CLink p;
   Vector v;
   StateElem *se;
   StateInfo *si;
   ILink ip;
   
   v = CreateVector(&tmpHeap,N);
   for (i=1; i<=N; i++) {
      sum = 0.0;
      for (p=cvec[i]; p != NULL; p = p->next) {
         ip = p->item;
         se = (StateElem *)ip->item; 
         si = se->info;
         memcpy(&x,&(si->hook),sizeof(float));
         sum += x;
      }
      v[i] = sum;
   }
   return v;
}

/* UpdateOccSums: update the occSum array following MergeGroups */
void UpdateOccSums(int i, int j, Vector occSum, int N)
{
   int k;
   
   occSum[i] += occSum[j];
   for (k=j; k<N; k++)
      occSum[k] = occSum[k+1];
}

/* MinOccSum: return index of min occ sum */
int MinOccSum(Vector occSum, int N)
{
   float min;
   int mini,i;
   
   mini = 1; min = occSum[mini];
   for (i=2; i<=N; i++)
      if (occSum[i]<min) {
         mini = i; min = occSum[mini];
      }
   return mini;
}

/* RemOutliers: remove any cluster for which the total state occupation
   count is below the 'outlierThresh' set by the RO command */
void RemOutliers(CLink *cvec, Matrix idist, Matrix gdist, int *numClust, 
                 Vector occSum)
{
   int N;                       /* current num clusters */
   int sparsest,i,mini;
   Vector gd;
   float min;
   
   N = *numClust;
   sparsest = MinOccSum(occSum,N);
   while (N>1 && occSum[sparsest] < outlierThresh) {
      gd = gdist[sparsest];     /* find best merge */
      mini = (sparsest==1)?2:1;
      min = gd[mini];
      for (i=2; i<=N; i++)
         if (i != sparsest && gd[i]<min) {
            mini = i; min = gd[mini];
         }
      MergeGroups(sparsest,mini,cvec,N);
      UpdateOccSums(sparsest,mini,occSum,N);
      --N;
      SetGDist(cvec,idist,gdist,N);
      sparsest = MinOccSum(occSum,N);
   }
   *numClust = N;
}

/* Clustering: split ilist into sublists where each sublist contains one
   cluster of items. Return list of sublists in cList.  It uses a 
   simple 'Furthest neighbour hierarchical cluster algorithm' */
void Clustering(ILink ilist, int *numReq, float threshold,
                char type, char *macName)
{
   Vector occSum=NULL;          /* array[1..N] of cluster occupation sum */
   int numClust;                /* current num clusters */
   CLink *cvec;                 /* array[1..numClust] of ->CRec */
   Matrix idist;                /* item distance matrix */
   Matrix gdist;                /* group cluster matrix */
   CLink p;
   ILink l;
   float ming,min;
   int i,j,k,numItems;
   char buf[40];

   if (badGC) {
      FixAllGConsts(hset);         /* in case any bad gConsts around */
      badGC=FALSE;
   }
   numItems = NumItems(ilist);
   numClust = numItems;         /* each item is separate cluster initially */
   if (trace & T_IND) {
      printf(" Start %d items\n",numClust);
      fflush(stdout);
   }
   
   cvec = BuildCVec(numClust,ilist);
   idist = CreateMatrix(&tmpHeap,numClust,numClust);
   gdist = CreateMatrix(&tmpHeap,numClust,numClust);
   SetIDist(cvec,idist,numClust,type); /* compute inter-item distances */
   CopyMatrix(idist,gdist);     /* 1 item per group so dmats same */
   ming = MinGDist(gdist,&i,&j,numClust);
   while (numClust>*numReq && ming<threshold) { /* merge closest two groups */
      MergeGroups(i,j,cvec,numClust);
      --numClust;
      SetGDist(cvec,idist,gdist,numClust); /* recompute gdist */
      ming = MinGDist(gdist,&i,&j,numClust);
   }
   if (occStatsLoaded) {
      if (trace & T_IND) {
         printf(" Via %d items before removing outliers\n",numClust);
         fflush(stdout);
      }
      occSum = SetOccSums(cvec,numClust);
      RemOutliers(cvec,idist,gdist,&numClust,occSum);
   }
   *numReq = numClust;          /* in case this is thresh limited case */
   if (trace & T_IND) {
      printf(" End %d items\n",numClust);
      fflush(stdout);
   }
   for (i=1; i<=numClust; i++) {
      if (trace & T_CLUSTERS) {
         for (j=1,min=99.999,k=0;j<=numClust;j++)
            if (i!=j && gdist[i][j]<min) min=gdist[i][j],k=j;
         printf("  C.%-2d MinG %6.3f[%d]",i,min,k); 
         if (occSum != NULL) printf(" (%.1f) ==",occSum[i]);
         for (p=cvec[i]; p!=NULL; p=p->next)
            printf(" %s",HMMPhysName(hset,p->item->owner));
         printf("\n");
         fflush(stdout);
      }
      sprintf(buf,"%s%d",macName,i);    /* construct macro name */
      for (p=cvec[i],l=NULL;p!=NULL;p=p->next)
         p->item->next=l,l=p->item;     /* and item list */
      ApplyTie(l,buf,type);             /* and tie it */
      FreeItems(&l);                    /* free items in sub list */
   }

   Dispose(&tmpHeap,cvec);
}

/* ---------- Up Num Mixtures Operations --------------------- */

/* SetGCStats: scan loaded models and compute average gConst */
void SetGCStats(void)
{
   HMMScanState hss;
   LogDouble sum = 0.0, sumsq = 0.0;
   float x;
   int count = 0;
   
   FixAllGConsts(hset);         /* in case any bad gConsts around */
   badGC=FALSE;
   
   NewHMMScan(hset,&hss);
   while(GoNextMix(&hss,FALSE)) {
      x = hss.me->mpdf->gConst;
      sum += x; sumsq += x*x;
      count++;
   }
   EndHMMScan(&hss);
   meanGC = sum / count;
   stdGC = sqrt(sumsq / count - meanGC*meanGC);
   if (trace & T_IND) {
      printf("  Mean GC = %f, Std Dev GC = %f\n",meanGC,stdGC);
      fflush(stdout);
   }
}

/* HeaviestMix: find the heaviest mixture in me, the hook field of each mix
   holds a count of the number of times that the mixture has been
   split.  Each mixture is then scored as:            
   mixture weight - num splits
   but if the mix gConst is less than 4 x stddev below average
   then the score is reduced by 5000 making it very unlikely to 
   be selected */
int HeaviestMix(char *hname, MixtureElem *me, int M)
{
   float max,w,gThresh;
   int m,maxm;
   MixPDF *mp;
   
   gThresh = meanGC - 4.0*stdGC;
   maxm = 1;  mp = me[1].mpdf;
   max = me[1].weight - (long int)mp->hook;
   if ((long int)mp->hook < 5000 && mp->gConst < gThresh) {
      max -= 5000.0; mp->hook = (void *)5000;
      HError(-2637,"HeaviestMix: mix 1 in %s has v.small gConst [%f]",
             hname,mp->gConst);
   }
   for (m=2; m<=M; m++) {   
      mp = me[m].mpdf;
      w = me[m].weight - (long int)mp->hook;
      if ((long int)mp->hook < 5000 && mp->gConst < gThresh) {
         w -= 5000.0; mp->hook = (void *)5000;
         HError(-2637,"HeaviestMix: mix %d in %s has v.small gConst [%f]",
                m,hname,mp->gConst);
      }
      if (w>max) {
         max = w; maxm = m;
      }
   }
   if (me[maxm].weight<=MINMIX)
      HError(2697,"HeaviestMix:  heaviest mix is defunct!");
   if (trace & T_DET) {
      printf("               : Split %d (weight=%.3f, count=%ld, score=%.3f)\n",
             maxm, me[maxm].weight, (long int)me[maxm].mpdf->hook, max);
      fflush(stdout);
   }
   return maxm;
}

/* UpMix: increase number of mixes in stream from oldM to newM */
void UpMix(char *hname, StreamElem *ste, int oldM, int newM)
{
   MixtureElem *me,m1,m2;
   int m,count,vSize;
   
   me = (MixtureElem*) New(&hmmHeap,sizeof(MixtureElem)*newM);
   --me;
   vSize = VectorSize(ste->spdf.cpdf[1].mpdf->mean);
   for (m=1;m<=oldM;m++)
      me[m] = ste->spdf.cpdf[m];
   count=oldM;
   while (count<newM) {
      m = HeaviestMix(hname,me,count);
      ++count;
      SplitMix(me+m,&m1,&m2,vSize);
      me[m] = m1; me[count] = m2;
   }
   ste->spdf.cpdf = me; ste->nMix = newM;
}

/* CountDefunctMix: return number of defunct mixtures in given stream */
int CountDefunctMix(StreamElem *ste)
{
   int m,defunct;
   
   for (m=1,defunct=0; m<=ste->nMix; m++)
      if (ste->spdf.cpdf[m].weight <= MINMIX)
         ++defunct;
   return defunct;
}

/* FixDefunctMix: restore n defunct mixtures by successive mixture splitting */
void FixDefunctMix(char *hname,StreamElem *ste, int n)
{
   MixtureElem *me,m1,m2;
   int m,M,l,count,vSize;

   me = ste->spdf.cpdf; M = ste->nMix;
   vSize = VectorSize(me[1].mpdf->mean);
   for (count = 0; count<n; ++count) {
      for (l=1; l<=M; l++)
         if (me[l].weight <= MINMIX)
            break;
      m = HeaviestMix(hname,me,M);
      SplitMix(me+m,&m1,&m2,vSize);
      me[m] = m1; me[l] = m2;
   }
}

/* ------------------- MixDown Operations --------------------- */

/* MixMergeCost: compute cost of merging me1 and me2 */
float MixMergeCost(MixtureElem *me1,MixtureElem *me2)
{
   float w1,w2,v,d=0.0,v1k,v2k;
   int vs,k;
   Vector m1,m2,v1,v2;
   
   /* normalise weight */
   w1=me1->weight/(me1->weight+me2->weight);
   w2=me2->weight/(me1->weight+me2->weight);
   m1=me1->mpdf->mean;
   m2=me2->mpdf->mean;
   v1=me1->mpdf->cov.var;
   v2=me2->mpdf->cov.var;
   vs=VectorSize(m1);

   for (k=1; k<=vs; k++) {
      v1k = v1[k]; v2k = v2[k];
      v = w1*v1k+w2*v2k+m1[k]*m1[k]*(w1-w1*w1)+m2[k]*m2[k]*(w2-w2*w2)-
         2*m1[k]*m2[k]*w1*w2;
      d += -0.5*(me1->weight*(log(TPI*v)-log(TPI*v1k))+
                 me2->weight*(log(TPI*v)-log(TPI*v2k)));
   }
   return(d);
}
   
/* MergeMix: merge components p and q from given stream elem.  If inPlace
   overwrite existing components, otherwise create new mix component */
void MergeMix(StreamElem *ste,int p,int q, Boolean inPlace)
{
   float w1,w2,m,v,p0,p1,p2,v1k,v2k;
   int vs,k;
   Vector m1,m2,v1,v2,mt,vt;
   MixtureElem me, *meq, *mep;

   if (q!=ste->nMix) {
      if (p==ste->nMix) {
         p=q; q=ste->nMix;
      }
      else {
         me=ste->spdf.cpdf[q];
         ste->spdf.cpdf[q]=ste->spdf.cpdf[ste->nMix];
         ste->spdf.cpdf[ste->nMix]=me;
         q=ste->nMix;
      }
   }
   /* the mixuture components */ 
   meq = ste->spdf.cpdf + q;                   mep = ste->spdf.cpdf + p;
   w1=mep->weight/(mep->weight+meq->weight);   w2=meq->weight/(mep->weight+meq->weight);
   m1=mep->mpdf->mean;                         m2=meq->mpdf->mean;
   v1=mep->mpdf->cov.var;                      v2=meq->mpdf->cov.var;
   vs=VectorSize(m1);

   if (inPlace) {
      mt = m1; vt = v1;
   } else {
      mep->mpdf->mean = mt = CreateSVector(&hmmHeap,vs);
      vt = CreateSVector(&hmmHeap,vs);
      mep->mpdf->cov.var = vt;
   }

   p0=p1=p2=0.0;
   for (k=1;k<=vs;k++) {
      v1k = v1[k]; v2k = v2[k];
      m=w1*m1[k]+w2*m2[k];
      v=w1*v1k+w2*v2k+m1[k]*m1[k]*(w1-w1*w1)+m2[k]*m2[k]*(w2-w2*w2)-
         2*m1[k]*m2[k]*w1*w2;
      if (trace & T_DET) {
         p0+=-0.5*(1.0+log(TPI*v));
         p1+=-0.5*(1.0+log(TPI*v1k));
         p2+=-0.5*(1.0+log(TPI*v2k));
      }
      mt[k]=m;
      vt[k]=v;
   }
   
   mep->weight += meq->weight;
   if (trace & T_DET)
      printf("     Likelihood from %.2f,%.2f to %.2f\n",p1,p2,p0);
}

/* DownMixSingle
   replace the mixture with a single Gaussian
 */
void DownMixSingle(StreamElem *ste,Boolean inPlace)
{
    int k,i,vs;
    float w;
    SVector mt,vt;
    MixPDF *m;
    MixtureElem *me;
    
    me = ste->spdf.cpdf; m=(me+1)->mpdf;
    vs=VectorSize(m->mean);
    mt=CreateSVector(&hmmHeap,vs);
    vt=CreateSVector(&hmmHeap,vs);
    for (k=1;k<=vs;k++) {
       /* mean */
       mt[k] = 0.0; 
       for (i=1; i<=ste->nMix; i++) {
          m=(me+i)->mpdf;w=(me+i)->weight;
          if ( m->ckind != DIAGC ) /* only support DIAGC */
             HError(2640,"DownMix: covariance type not supported");
          mt[k] += w*m->mean[k]; 
       }
       /* variance */
       vt[k] = 0.0;
       for (i=1; i<=ste->nMix; i++) { 
          m=(me+i)->mpdf; w=(me+i)->weight;
          vt[k] +=w*m->cov.var[k]-w*m->mean[k]*(mt[k]-m->mean[k]);
       }
    }
    /* decrement use of mean and var */
    for(i=2;i<=ste->nMix;i++) {
       m=(me+i)->mpdf;
       DecUse(m->mean); DecUse(m->cov.var);
    }
    m=(me+1)->mpdf;
    if (inPlace) {
       /* copy vectors */
       for (k=1;k<=vs;k++) {
          m->mean[k] = mt[k]; m->cov.var[k] = vt[k];
       }
       FreeSVector(&hmmHeap,vt);
       FreeSVector(&hmmHeap,mt);
    }
    else {
       DecUse(m->mean); DecUse(m->cov.var);
       m->mean = mt; m->cov.var = vt; 
    }
    (me+1)->weight=1.0;
    ste->nMix=1;
}

/* DownMix: merge mixtures in stream to maxMix. If inPlace then existing
   mean and variance vectors will be overwritten.  Otherwise, new vectors
   are created for changed components */
void DownMix(char *hname, StreamElem *ste, int maxMix, Boolean inPlace)
{
   int i,j,m,p,q,oldM,k,V;
   float bc,mc;
   MixPDF *m1,*m2;
   float x,v1,v2,sum=0.0;
    
   oldM=m=ste->nMix;
   if (trace & T_IND)
      printf("   Mix Down for %s\n",hname);

   /* special case mixdown to single Gaussian */
   if (maxMix==1)
      DownMixSingle(ste,inPlace);

   while(ste->nMix>maxMix) { /* merge pairs until maxMix */
      p=q=0;
      bc = LZERO;
      /* find paur with lowest cost */ 
      for (i=1; i<=ste->nMix; i++) {
         /* only support for CovKind  DIAGC */
         if ( (MixtureElem*)(ste->spdf.cpdf+i)->mpdf->ckind != DIAGC )
            HError(2640,"DownMix: covariance type not supported");
         for (j=i+1; j<=ste->nMix; j++) {
            mc=MixMergeCost(ste->spdf.cpdf+i,ste->spdf.cpdf+j);
            if (trace & T_MD) {
               printf("Could merge %d (%.2f) with %d (%.2f) for %f\n",
                      i,ste->spdf.cpdf[i].weight,j,ste->spdf.cpdf[j].weight,mc);
               m1 = (ste->spdf.cpdf+i)->mpdf; m2 = (ste->spdf.cpdf+j)->mpdf;
               V = VectorSize(m1->mean);
               for (k=1; k<=V; k++){
                  x = m1->mean[k] - m2->mean[k];
                  v1 = m1->cov.var[k]; v2 = m2->cov.var[k];
                  sum += x*x / sqrt(v1*v2);
               }
               printf("Divergence %.2f\n",sqrt(sum/V));
            }
            if (mc>bc) {
               p=i; q=j; bc=mc;
            }
         }
      }
      if (trace & T_DET)
         printf("    Merging %d with %d for %f\n",p,q,bc);
      MergeMix(ste,p,q,inPlace);
      ste->nMix--;
   }
   if ((trace & T_DET) && (ste->nMix!=oldM))
      printf("    %s: mixdown %d -> %d\n",hname,oldM,ste->nMix);
}

/* ------------------- Tree Building Routines ------------------- */

typedef struct _AccSum {        /* Accumulator Record for storing */
   Vector sum;                  /* the sum, sqr and occupation  */
   Vector sqr;                  /* count statistics             */
   float  occ;
} AccSum;

typedef struct _Node {          /* Tree Node */
   CLink clist;                 /* list of cluster items */
   float occ;                   /* total occupation count */
   float tProb;                 /* likelihood of total cluster */
   float sProb;                 /* likelihood of split cluster */
   QLink quest;                 /* question used to do split */
   Boolean ans;                 /* TRUE = yes, FALSE = no */
   short snum;
   MLink macro;                 /* macro used for tie */
   struct _Node *parent;        /* parent of this node */
   struct _Node *yes;           /* yes subtree */
   struct _Node *no;            /* no subtree */
   struct _Node *next;          /* doubly linked chain of */
   struct _Node *prev;          /* leaf nodes */
}Node;

typedef struct _Tree{           /* A tree */
   LabId baseId;                /* base phone name */
   int state;                   /* state (or 0 if hmm tree) */
   int size;                    /* number of non terminal nodes */
   Node *root;                  /* root of tree */
   Node *leaf;                  /* chain of leaf nodes */
   struct _Tree *next;          /* next tree in list */
}Tree;

static Tree *treeList = NULL;   /* list of trees */
static AccSum yes,no;           /* global accs for yes - no branches */
static float occs[2];           /* array[Boolean]of occupation counts */
static float  cprob;            /* complete likelihood at current node */
static int numTreeClust;        /* number of clusters in tree */

/* -------------------- Tree Name Mapping routines ----------------------- */

static char treeName[256] = "";
static void SetTreeName(char *name) {
   strcpy(treeName,name);
   strcat(treeName,"tree");   
}

/* MapTreeName: If USEMODELNAME=FALSE, the macRoot prefix passed into
   the TB command will be used instead.   
*/
void MapTreeName(char *buf) {
   if (!useModelName) {
      strcpy(buf,treeName);
   }
}

/* CreateTree: create a new tree */
Tree *CreateTree(ILink ilist,LabId baseId,int state)
{
   Tree *tree,*t;
   char buf[256];
   LabId id;
   HMMDef *hmm;
   ILink i;
   int n;
   
   tree = (Tree*) New(&questHeap,sizeof(Tree));
   tree->root = tree->leaf = NULL; tree->next = NULL; tree->size=0;
   if (treeList) { 
      for (t=treeList;t->next;t=t->next);
      t->next=tree;
   }
   else treeList=tree;
   tree->baseId = baseId;
   tree->state = state;
   if (ilist!=NULL) {
      hmm = ilist->owner;
      n=hmm->numStates;
      for (i=ilist; i!=NULL; i=i->next) {
         hmm = i->owner; strcpy(buf,HMMPhysName(hset,hmm)); FTriStrip(buf);
         MapTreeName(buf);
         id = GetLabId(buf,FALSE);
         if (tree->baseId != id)
            HError(-2663,"CreateTree: different base phone %s in item list for %s",
                   buf,tree->baseId->name);
         if (state>0) {
            if (hmm->svec+state != (StateElem *)i->item)
               HError(2663,"CreateTree: attempt to cluster different states");
         }
         else if (hmm->numStates != n)
            HError(2663,"CreateTree: attempt to cluster different size hmms");
      }
   }
   return tree;
}

/* CreateTreeNode: create a new tree node */
Node *CreateTreeNode(CLink clist, Node *parent)
{
   Node *n;

   n=(Node*) New(&questHeap,sizeof(Node));
   n->clist = clist;
   n->ans = FALSE;
   n->yes = n->no = NULL;
   n->prev = n->next = NULL;
   n->parent = parent;
   n->macro = NULL;
   n->quest = NULL;
   n->snum = -1;
   return n;
}

/* ChkTreeObject: check that item is a valid state or hmm, 
   and make sure that MixPDF hooks are NULL */
void ChkTreeObject(ILink obj)
{
   HMMDef *hmm;
   StateElem *se;
   int j;

   hmm = obj->owner;
   if (hmm==NULL)
      HError(2690,"ChkTreeObject: null hmm owner");
   if (obj->item == obj->owner) {
      for (j=2;j<hmm->numStates;j++) {
         se = hmm->svec+j;
         if (se->info->pdf[1].nMix!=1 || 
             se->info->pdf[1].spdf.cpdf[1].mpdf->ckind!=DIAGC)
            HError(2663,"ChkTreeObject: TB only valid for 1 mix diagonal covar models");
         se->info->pdf[1].spdf.cpdf[1].mpdf->hook = NULL;
      }
   }
   else {
      se = (StateElem *)obj->item;
      if (se == NULL)
         HError(2690,"ChkTreeObject: null state elem ptr");
      if (se->info->pdf[1].nMix!=1 || 
          se->info->pdf[1].spdf.cpdf[1].mpdf->ckind!=DIAGC)
         HError(2663,"ChkTreeObject: TB only valid for 1 mix diagonal covar models");
      se->info->pdf[1].spdf.cpdf[1].mpdf->hook = NULL;
   }
}

/* InitTreeAccs:  attach and AccSum record to the MixPDF hook and
   initialise with sum and sqr calculated from the 
   occupation counts deposited in the StateInfo hook */
void InitTreeAccs(StateElem *se, int l)
{
   StateInfo *si;
   MixPDF *mp;
   int j, k;
   short nMix;
   float x, w;
   AccSum *acc;

   si=se->info ;
   nMix = si->pdf[1].nMix ;
   for (j = 1; j <= nMix; j++) {
      mp=si->pdf[1].spdf.cpdf[j].mpdf;
      w = si->pdf[1].spdf.cpdf[j].weight;
      memcpy(&x,&(si->hook),sizeof(float));
      if (mp->hook==NULL && x>0.0) {
         acc=(AccSum*) New(&tmpHeap,sizeof(AccSum));
         mp->hook=acc;
         acc->sum=CreateVector(&tmpHeap,l);
         acc->sqr=CreateVector(&tmpHeap,l);
         x *= w;
         acc->occ=x;
         for (k=1;k<=l;k++) {
            acc->sum[k]=mp->mean[k]*x;
            acc->sqr[k]=(mp->cov.var[k]+mp->mean[k]*mp->mean[k])*x;
         }
      }
   }
}

/* ZeroAccSum: zero given accumulator */
void ZeroAccSum(AccSum *acc)
{
   ZeroVector(acc->sum);
   ZeroVector(acc->sqr);
   acc->occ=0.0;
}

/* AccSumProb: return log likelihood for given statistics */
float AccSumProb(AccSum *acc)
{
   double variance,prob;
   int k,l;

   prob=0.0;
   if (acc->occ > 0.0) {
      l=VectorSize(acc->sum);
      for (k=1;k<=l;k++) {
         variance=(acc->sqr[k]-(acc->sum[k]*acc->sum[k]/acc->occ))/acc->occ;
         if (variance<=MINLARG) return(LZERO);
         prob+=-0.5*acc->occ*(1.0+log(TPI*variance));
      }
   }
   return(prob);
}

/* IncSumSqr: add sums and sqrs from given state into yes or no depending
   on value of ans */
void IncSumSqr(StateInfo *si, Boolean ans, AccSum *no, AccSum *yes, int l)
{
   AccSum *acc,*tacc;
   int k;
   
   acc = (AccSum *) si->pdf[1].spdf.cpdf[1].mpdf->hook;
   if (!acc) return;
   tacc =  (ans && yes != NULL) ? yes : no;
   tacc->occ+=acc->occ;
   for (k=1;k<=l;k++) {
      tacc->sum[k] += acc->sum[k];
      tacc->sqr[k] += acc->sqr[k];
   }   
}

/* ClusterLogL: return log likelihood of given cluster, this is used for both
   the complete and split clusters.  In the former case, the yes accumulator
   will be NULL and only the no accumulator is used */
float ClusterLogL(CLink clist,AccSum *no,AccSum *yes,float *occs)
{
   CLink p;
   float prob;
   StateElem *se;
   int l,i;

   l=VectorSize(no->sum);
   if (clist->item->item == clist->item->owner) {
      prob=0.0;
      occs[FALSE]=0.0;occs[TRUE]=0.0;
      for (i=2;i<clist->item->owner->numStates;i++) {
         ZeroAccSum(no);
         if (yes != NULL) ZeroAccSum(yes);
         for(p=clist;p!=NULL;p=p->next) {
            IncSumSqr(p->item->owner->svec[i].info,p->ans,no,yes,l);
         }
         prob += AccSumProb(no);
         if (yes != NULL) prob += AccSumProb(yes);
         if (occs != NULL) {
            occs[FALSE] += no->occ;
            if (yes != NULL)
               occs[TRUE] += yes->occ;
         }
      }
   }
   else {
      ZeroAccSum(no);
      if (yes != NULL) ZeroAccSum(yes);
      for(p=clist;p!=NULL;p=p->next) {
         se=(StateElem*)p->item->item;
         IncSumSqr(se->info,p->ans,no,yes,l);
      }
      prob = AccSumProb(no);
      if (yes != NULL) prob += AccSumProb(yes);
      if (occs != NULL) {
         occs[FALSE] = no->occ;
         occs[TRUE] = (yes != NULL) ? yes->occ : 0.0;
      }
   }
   return(prob);
}

/* AnswerQuestion: set ans field in each cluster item in preparation for
   a possible split */
void AnswerQuestion(CLink clist,QLink q)
{
   CLink p;
   ILink i;
  
   for (p=clist;p!=NULL;p=p->next) 
      p->ans = FALSE;
   if (q!=NULL)
      for (i=q->ilist;i!=NULL;i=i->next)
         if ((p=(CLink) i->item)!=NULL)
            p->ans=TRUE;
}

/* ValidProbNode: set tProb and sProb of given node according to best
   possible question which is stored in quest field.  */
void ValidProbNode(Node *node,float thresh)
{
   QLink q,qbest;
   float best,sProb;
   
   node->tProb = ClusterLogL(node->clist,&no,NULL,occs);
   node->occ = occs[FALSE];
   if (trace & T_TREE_BESTQ) {
      char buf[20];
      if (node->parent==NULL)
         sprintf(buf," ROOT ");
      else
         sprintf(buf,"%3d[%c]",node->parent->snum,
                 (node->parent->yes==node?'Y':'N'));
      
      printf("    Node %s:                  LogL=%5.3f (%.1f)\n",
             buf,node->tProb/node->occ,node->occ);
      fflush(stdout);
   }
   qbest = NULL;
   best = node->tProb;
   for (q=qHead;q!=NULL;q=q->next) {
      AnswerQuestion(node->clist,q);
      sProb = ClusterLogL(node->clist,&no,&yes,occs);
      if (node->occ<=0.0 || (outlierThresh >= 0.0 &&  
                             (occs[FALSE]<outlierThresh || occs[TRUE]<outlierThresh)))
         sProb=node->tProb;

      if (trace & T_TREE_ALLQ || 
          ((trace & T_TREE_OKQ) && (sProb-node->tProb)>thresh)) {
         printf("       Q %20s    LogL=%-7.3f  Imp = %8.2f (%.1f,%.1f)\n",
                q->qName->name,sProb/node->occ,sProb-node->tProb,
                occs[0],occs[1]);
         fflush(stdout);
      }
      if (sProb>best) {
         best=sProb;
         qbest=q;
      }
   }
   if (trace & T_TREE_BESTQ) {
      if (qbest != NULL)
         printf("    BestQ %20s    q LogL %6.3f Imp %8.2f\n",
                qbest->qName->name,best/node->occ,best-node->tProb);
      else
         printf("    BestQ [ NULL ]\n");
      fflush(stdout);
   }
   node->sProb=best; node->quest=qbest;
}

/* SplitTreeNode: split the given node */
void SplitTreeNode(Tree *tree, Node *node)
{
   CLink cl,nextcl;

   if (node->quest == NULL) return;
   cprob += node->sProb - node->tProb;

   AnswerQuestion(node->clist,node->quest);
   node->yes = CreateTreeNode(NULL,node);
   node->yes->ans= TRUE;
   node->no = CreateTreeNode(NULL,node);
   node->no->ans=TRUE;

   for(cl=node->clist;cl!=NULL;cl=nextcl) {
      nextcl=cl->next;
      switch(cl->ans) {
      case FALSE: cl->next=node->no->clist;
         node->no->clist=cl;
         break;
      case TRUE:  cl->next=node->yes->clist;
         node->yes->clist=cl;
         break;
      default:
         HError(2694,"SplitTreeNode: Unspecified question result");
      }
   }
   node->clist = NULL;

   /* unlink node from leaf chain and link in yes and no */
   node->yes->next = node->no;
   node->no->prev = node->yes;

   node->no->next = node->next;
   node->yes->prev = node->prev;

   if (node->next!=NULL) node->next->prev=node->no;
   if (node->prev==NULL) 
      tree->leaf = node->yes;
   else
      node->prev->next=node->yes;

   numTreeClust++;
}

/* FindBestSplit: find best node to split */
Node *FindBestSplit(Node *first, float threshold)
{
   Node *best,*node;
   float sProb,imp;

   best=NULL;imp=0.0;
   for (node=first;node!=NULL;node=node->next) {
      sProb = node->sProb - node->tProb;
      if (sProb>imp && sProb>threshold) { 
         best=node;imp=sProb;
      }
   }
   return(best);
}

/* MergeCost: return logL reduction if node b is merged with node a.
   On entry, atail points to last cluster item in node a clist. */
float MergeCost(Node *a, Node *b, CLink atail)
{
   float combProb;              /* combined logL */
   
   atail->next = b->clist;
   combProb = ClusterLogL(a->clist,&no,NULL,occs);
   atail->next = NULL;
   return a->tProb + b->tProb - combProb;
}

/* MergeNode: find the as yet unseen node which when merged with
   given node gives smallest reduction in LogL.  If this reduction
   is less than threshold, merge the nodes and return TRUE */
Boolean MergeNode(Node *node, float threshold)
{
   Node *p, *min;
   CLink ctail;
   float minCost,cost;

   ctail = node->clist;         /* set ctail to last cluster item */
   if (ctail == NULL)
      HError(2690,"MergeNode: empty cluster list");
   while (ctail->next != NULL) ctail = ctail->next;

   /* find bestnode to merge */
   minCost = threshold;min=NULL;
   for (p=node->next;p!=NULL;p=p->next) {
      cost = MergeCost(node,p,ctail);
      if (trace & T_TREE_ALLM) {
         char buf1[20],buf2[20];
         if (node->parent==NULL) sprintf(buf1," ROOT ");
         else sprintf(buf1,"%3d[%c]",node->parent->snum,
                      (node->parent->yes==node?'Y':'N'));
         if (min->parent==NULL) sprintf(buf2," ROOT ");
         else sprintf(buf2,"%3d[%c]",min->parent->snum,
                      (min->parent->yes==min?'Y':'N'));
         printf("       M  %s (%.3f,%.1f) + %s (%.3f,%.1f)   Imp %.1f\n",
                buf1,node->tProb/node->occ,node->occ,
                buf2,min->tProb/min->occ,min->occ,-minCost);
      }
      if (cost < minCost) {
         minCost = cost; min = p;
      }
   }
   if (minCost < threshold) {   /* Merge nodes */
      if (trace & T_DET) {
         char buf1[20],buf2[20];
         if (node->parent==NULL) sprintf(buf1," ROOT ");
         else sprintf(buf1,"%3d[%c]",node->parent->snum,
                      (node->parent->yes==node?'Y':'N'));
         if (min->parent==NULL) sprintf(buf2," ROOT ");
         else sprintf(buf2,"%3d[%c]",min->parent->snum,
                      (min->parent->yes==min?'Y':'N'));
         printf("    BestM  %s (%.3f,%.1f) + %s (%.3f,%.1f)   Imp -%4.1f\n",
                buf1,node->tProb/node->occ,node->occ,
                buf2,min->tProb/min->occ,min->occ,minCost);
         fflush(stdout);
      }
      ctail->next = min->clist;
      node->tProb += min->tProb - minCost;
      node->occ += min->occ;
      if (min->parent->yes == min)     min->parent->yes = node;
      else if (min->parent->no == min) min->parent->no = node;
      else
         HError(2695,"MergeNode: cannot find min parent link");
      if (min->prev != NULL) min->prev->next = min->next;
      if (min->next != NULL) min->next->prev = min->prev;
      --numTreeClust;
      cprob-=minCost;
      return TRUE;
   } else
      return FALSE;
}

/* MergeLeaves: merge any pair of leaf nodes in given tree for which 
   reduction in combined LogL is less than threshold.  Note that 
   after this call, the parent links are no longer valid. */
void MergeLeaves(Tree *tree, float threshold)
{
   Node *node;
   
   for (node=tree->leaf;node!=NULL;node=node->next)
      MergeNode(node,threshold);
}

/* TieLeafNodes: tie all items in the leaf nodes of given tree */
void TieLeafNodes(Tree *tree, char *macRoot)
{
   float occ;
   ILink ilist,i;
   CLink cl;
   char clnum[20];              /* cluster number */
   int clidx;                   /* cluster index */
   char buf[255];               /* construct mac name in this */
   Node *node;
   int numItems = 0;
   LabId id;
   SVector vf[SMAX];
   SVector mean, var;
   int l=0;
   Boolean vfSet=FALSE; 

   if (trace & T_CLUSTERS) printf("\n Nodes for %s\n",macRoot);
   if (useLeafStats) { 
      int k;
      /* get vFloor vectors and flag if present in hset */
      SetVFloor(hset,vf,-1.0);
      vfSet=TRUE;
      l=VectorSize(vf[1]);
      for (k=1;k<=l;k++)
         if (vf[1][k]<0.0)
            vfSet = FALSE;
   }
   cprob=0.0; occ=0.0;
   clidx = numTreeClust;
   for (node=tree->leaf;node!=NULL;node=node->next) {
      cprob += ClusterLogL(node->clist,&no,NULL,occs); /*==node->tProb;*/
      occ += occs[FALSE];
      sprintf(clnum,"%d",clidx--); /* construct macro name */
      strcpy(buf,macRoot);
      strcat(buf,clnum);
      if (trace & T_CLUSTERS) {
         char buf[20];
         if (node->parent==NULL) sprintf(buf,"ROOT");
         else sprintf(buf,"%d[%c]",node->parent->snum,
                      (node->parent->yes==node?'Y':'N'));
         printf("  %-6s== %-3d LogL %6.3f (%.1f) ==",buf,clidx,
                node->tProb/node->occ,node->occ);
      }
      ilist = NULL;
      for (cl=node->clist;cl!=NULL;cl=cl->next) {
         i = cl->item;
         if (trace & T_CLUSTERS)
            printf(" %s",HMMPhysName(hset,i->owner));

         i->next = ilist;
         ilist = i;
      }
      numItems += NumItems(ilist);
      if (trace & T_CLUSTERS)
          printf("\n");
      if ((ilist->item != ilist->owner) && useLeafStats) {
         /* get cluster stats */
         l = VectorSize(no.sum);
         ZeroAccSum(&no);
         for(cl=node->clist;cl!=NULL;cl=cl->next) 
            IncSumSqr(((StateElem*)cl->item->item)->info,FALSE,&no,NULL,l);
      }
      ApplyTie(ilist,buf,(ilist->item==ilist->owner?'h':'s'));
      id=GetLabId(buf,FALSE);
      node->macro = FindMacroName(hset,((ilist->item==ilist->owner)?'h':'s'),id);
      if ((ilist->item != ilist->owner) && useLeafStats) {
         int k;
         StreamElem *ste;
         MixPDF *mp;
         ste = (((StateElem *)ilist->item)->info->pdf)+1;
         mp = ste->spdf.cpdf[1].mpdf;
         mean = mp->mean;
         var  = mp->cov.var;
         for (k=1;k<=l;k++) {
            mean[k] = no.sum[k]/no.occ;
            var[k] = no.sqr[k]/no.occ - mean[k]*mean[k];
            /* floor variances */
            if (vfSet && (var[k] < vf[1][k]) )
               var[k] = vf[1][k];
         }
      }
      FreeItems(&ilist);        /* free items at this node */
      node->clist=NULL;
   }
}

/* BuildTree: build a tree for objects stored in ilist and return
   the actual clusters in rlist, the number of clusters
   in numCl.  The value of threshold determines when to stop.
   macRoot is the root prefix of the name to use in the tie  */
void BuildTree(ILink ilist,float threshold, char *macRoot)
{
   int i,j,l,snum,state,numItems;
   char buf[MAXSTRLEN];
   HMMDef *hmm;
   CLink clHead,cl;
   ILink p;
   QLink q;
   LabId labid;
   Node *node;
   Tree *tree;
   static int totalItems = 0;   /* for overall statistics */
   static int totalClust = 0;
   
   SetTreeName(macRoot);
   l = hset->swidth[1];
   /* Initialise Global AccSums for yes - no branches */
   yes.sum=CreateVector(&tmpHeap,l);  yes.sqr=CreateVector(&tmpHeap,l);
   no.sum=CreateVector(&tmpHeap,l);   no.sqr=CreateVector(&tmpHeap,l);

   /* Check object consistency */
   hmm = ilist->owner;          /* any HMM will do */
   for(p=ilist;p!=NULL;p=p->next) 
      ChkTreeObject(p);

   /* Create a new tree in tree list */
   /* Find base name and state if any */
   hmm = ilist->owner;
   strcpy(buf,HMMPhysName(hset,hmm)); FTriStrip(buf);
   MapTreeName(buf);
   labid = GetLabId(buf,TRUE);

   if (ilist->owner==ilist->item)
      state=-1;
   else
      for (j=2,state=0; j<hmm->numStates;j++)
         if (hmm->svec+j == (StateElem *)ilist->item) {
            state = j; break;
         }
   if (state==0)
      HError(2663,"BuildTree: cannot find state index");
   tree = CreateTree(ilist,labid,state);

   /* Create a single cluster, owner->hook links to cluster record */
   i = 1; clHead = NULL;
   for (p=ilist;p!=NULL;p=p->next,i++) {
      if (p->owner == p->item)
         for (j=2;j<p->owner->numStates;j++)
            InitTreeAccs(p->owner->svec+j, l);
      else
         InitTreeAccs((StateElem*)p->item, l);
      cl=(CLink) New(&tmpHeap,sizeof(CRec));
      p->owner->hook = cl; /* HMM hook links to CRec */
      cl->item = p;  cl->ans = FALSE;
      cl->idx = i; cl->next = clHead;
      clHead = cl;
   }
   
   /* For each question make each hmm (item) point to its */
   /*  corresponding cluster member (CREC) via hmm->hook */
   for (q=qHead; q!=NULL; q=q->next)
      for (p=q->ilist; p!=NULL; p=p->next)
         p->item = p->owner->hook;
   for (p=ilist;p!=NULL;p=p->next,i++)
      p->owner->hook=NULL;

   /* Create the root of the tree */
   node = tree->leaf = tree->root = CreateTreeNode(clHead,NULL);
   cprob = node->tProb = ClusterLogL(node->clist,&no,NULL,occs);
   node->occ = occs[FALSE];
   numTreeClust=1; numItems=NumItems(ilist);
   if (trace & T_IND) {
      if (state<0) 
         printf(" Start  %4s : %d  have LogL=%6.3f occ=%4.1f\n",
                tree->baseId->name,
                numItems,cprob/node->occ,node->occ);
      else
         printf(" Start %3s[%d] : %d  have LogL=%6.3f occ=%4.1f\n",
                tree->baseId->name,state,numItems,cprob/node->occ,node->occ);
      fflush(stdout);
   }
   ValidProbNode(node,threshold);
   node->prev = NULL;

   /* Now build the actual tree by repeated node splitting */
   snum=0;
   node=FindBestSplit(tree->leaf,threshold);
   while(node != NULL) {
      node->snum=snum++;
      if (trace & T_DET) {
         char buf[20];
         if (node->parent==NULL) sprintf(buf," ROOT ");
         else sprintf(buf,"%3d[%c]",node->parent->snum,
                      (node->parent->yes==node?'Y':'N'));
         printf("\n  Split %s == %-3d (%.1f)\n",
                buf,node->snum,node->occ);
         printf("          %20s   LogL %7.3f -> %7.3f  Imp %.2f\n",
                node->quest->qName->name,node->tProb/node->occ,
                node->sProb/node->occ,node->sProb-node->tProb);
         fflush(stdout);
      }
      SplitTreeNode(tree,node);
      ValidProbNode(node->yes,threshold);
      ValidProbNode(node->no,threshold);
      if (trace & T_TREE_BESTQ)
         printf("   Node %-3d  [Y] LogL %.3f (%.1f)    [N] LogL %.3f (%.1f)\n",
                node->snum,node->yes->tProb/node->yes->occ,node->yes->occ,
                node->no->tProb/node->no->occ,node->no->occ);
      node=FindBestSplit(tree->leaf,threshold);
   }
   tree->size=snum;
   
   /* Merge any similar leaf nodes */
   if (treeMerge) {
      if (trace & T_IND) {
         if (state<0) 
            printf("\n Via    %4s : %d gives LogL=%6.3f occ=%4.1f\n",
                   tree->baseId->name,numTreeClust,
                   cprob/tree->root->occ,tree->root->occ);
         else
            printf("\n Via   %3s[%d] : %d gives LogL=%6.3f occ=%4.1f\n",
                   tree->baseId->name,state,numTreeClust,cprob/tree->root->occ,
                   tree->root->occ);
         fflush(stdout);
      }
      MergeLeaves(tree,threshold);
   }
   
   /* Finally assign a macro to each leaf node and do tie */
   totalItems += numItems; totalClust += numTreeClust;
   if (trace & T_IND) {
      if (state<0) 
         printf("\n End    %4s : %d gives LogL=%6.3f occ=%4.1f\n",
                tree->baseId->name,numTreeClust,
                cprob/tree->root->occ,tree->root->occ);
      else
         printf("\n End   %3s[%d] : %d gives LogL=%6.3f occ=%4.1f\n",
                tree->baseId->name,state,numTreeClust,cprob/tree->root->occ,
                tree->root->occ);
      fflush(stdout);
   }
   TieLeafNodes(tree,macRoot);
   Dispose(&tmpHeap,yes.sum);
   if (trace & T_BID) {
      printf("\n TB: Stats %d->%d [%.1f%%]  { %d->%d [%.1f%%] total }\n",
             numItems,numTreeClust,(float)numTreeClust*100.0/(float)numItems,
             totalItems,totalClust,(float)totalClust*100.0/(float)totalItems);
      fflush(stdout);
   }
   SetTreeName("");
}

/* AssignState: find shared state info for given id and state idx */
Ptr AssignStructure(LabId id, int state)
{
   int len=0;
   char buf[256];
   LabId tid;
   Tree *tree;
   Node *node;
   Boolean isYes;
   MLink m;
   
   /* First find Tree to use */
   strcpy(buf,id->name); FTriStrip(buf);
   MapTreeName(buf); 
   tid = GetLabId(buf,FALSE);
   for (tree=treeList;tree!=NULL;tree=tree->next) {
      if (tree->baseId == tid && tree->state == state) break;
   }
   if (tree==NULL && state<0) return(NULL);
   if (tree==NULL) {
      if (useModelName) {
         HError(2662,"AssignStructure: cannot find tree for %s state %d",
                id->name,state);
      } else {
         /* not using model name - must be a global tree - match state */
         for (tree=treeList;tree!=NULL;tree=tree->next) {
            if (tree->state == state) break;
         }
      }
   }      
   /* Then move down tree until state is found */
   node = tree->root;
   if (trace & T_DET) {
      if (trace & T_TREE_ANS) printf("\n     ");
      if (state>0)
         printf(" [%d] ",state);
      fflush(stdout);
   }
   while (node->yes != NULL) {
      isYes = QMatch(id->name,node->quest);
      if (trace & T_TREE_ANS) printf("%s ",isYes?"yes":" no"),len+=4;
      node = (isYes)?node->yes:node->no;
   }
   m = (MLink)node->macro;
   if (trace & T_DET) {
      if (trace & T_TREE_ANS) printf("%*s",48-len," ");
      printf("=~%c %-10s ",(state>0?'s':'h'),node->macro->id->name);
      fflush(stdout);
   }
   return(m->structure);
}

/* FindProtoModel: find an allophone of given model in hList */
HLink FindProtoModel(LabId model)
{
   char phone[256], buf[256];
   int h;
   MLink q;

   strcpy(phone,model->name);
   FTriStrip(phone);
   for (h=0; h<MACHASHSIZE; h++)
      for (q=hset->mtab[h]; q!=NULL; q=q->next) 
         if (q->type=='h') {
            strcpy(buf,q->id->name);
            FTriStrip(buf);
            if (strcmp(phone,buf)==0) return ((HLink) q->structure);
         }
   if (useModelName) {
      HError(2662,"FindProtoModel: no proto for %s in hSet",model->name);
   } else {
      for (h=0; h<MACHASHSIZE; h++)
         for (q=hset->mtab[h]; q!=NULL; q=q->next) 
            if (q->type=='h') {
               printf("%s\n",q->id->name);
               fflush(stdout);
               return ((HLink) q->structure);      
            }
   }
   return NULL;
}

/* SynthStates: synthesise a HMM from state based trees */
HMMDef *SynthModel(LabId id)
{
   HMMDef *hmm,*proto;
   StateElem *se;
   int i,N;
   
   if (trace & T_DET) {
      printf("   Synth %-11s",id->name);
      fflush(stdout);
   }
   /* First find Tree to use */
   if ((hmm=(HMMDef *) AssignStructure(id,-1)) == NULL) {
      proto=FindProtoModel(id);
      hmm=(HLink) New(&hmmHeap,sizeof(HMMDef));
      N=hmm->numStates=proto->numStates;
      se=(StateElem*) New(&hmmHeap,sizeof(StateElem)*N);
      hmm->svec=se-2;hmm->owner=hset;
      hmm->dur=proto->dur;
      hmm->nUse=0; hmm->hook=NULL;
      hmm->transP=CloneSMatrix(hset->hmem,proto->transP,TRUE);
      hmm->nUse = 1; hmm->hook = NULL; N = hmm->numStates;
      hmm->svec = se - 2;
      for (i=2; i<N; i++,se++) {
         se->info = (StateInfo *) AssignStructure(id,i);
         if (se->info->nUse==0)
            HError(2695,"SynthModel: untied state found for state %d",i);
         se->info->nUse++;
      }
      NewMacro(hset,fidx,'h',id,hmm);
   }
   if (trace & T_DET) {
      printf("\n");
      fflush(stdout);
   }
   return hmm;
}

/* TreeFilter: for each new model in newList, pass it thru trees and 
   add it to the currently loaded set. */
void TreeFilter(HMMSet *newSet)
{
   MLink p,q;
   HLink hmm;
   int seen,unseen,h;

   seen=unseen=0;
   for (h=0; h<MACHASHSIZE; h++)
      for (q=newSet->mtab[h]; q!=NULL; q=q->next) 
         if (q->type=='l') {
            p=FindMacroName(hset,'l',q->id);
            if (p==NULL) {
               hmm=SynthModel(q->id);
               unseen++;
               NewMacro(hset,fidx,'l',q->id,hmm);
               q->structure=hmm;
               /* Can do this as never use FindMacroStruct with newSet */
            }
            else {
               seen++;
               hmm=(HLink) p->structure;
               hmm->nUse++;
               q->structure=hmm;
            }
         }
   for (h=0; h<MACHASHSIZE; h++)
      for (q=newSet->mtab[h]; q!=NULL; q=q->next) 
         if (q->type=='h') {
            p=FindMacroName(newSet,'l',q->id);
            q->structure=p->structure;
         }
}

void DownTree(Node *node,Node **array) 
{
   if (node->yes) {
      if (node->snum>=0)
         array[node->snum]=node;
      DownTree(node->yes,array);
      DownTree(node->no,array);
   }
}

/* ShowTrees: print a summary of currently constructed trees */
void ShowTreesCommand(void)
{
   Tree *tree;
   Node *node,**array;
   QLink q;
   IPat *p;
   int i;
   FILE *file;
   char fn[256];
  
   ChkedAlpha("ST trees file name",fn);         /* get name of trees file */
   if (trace & T_BID) {
      printf("\nST %s\n Writing current questions/trees to file\n",fn);
      fflush(stdout);
   }
   if ((file=fopen(fn,"w"))==NULL)
      HError(2611,"ShowTreesCommand: Cannot open trees file %s",fn);
   for (q=qHead;q;q=q->next) {
      fprintf(file,"QS %s ",ReWriteString(q->qName->name,NULL,SING_QUOTE));
      fprintf(file,"{ %s",ReWriteString(q->patList->pat,NULL,DBL_QUOTE));
      for (p=q->patList->next;p;p=p->next)
         fprintf(file,",%s",ReWriteString(p->pat,NULL,DBL_QUOTE));
      fprintf(file," }\n");
   }
   fprintf(file,"\n");
   for (tree=treeList;tree!=NULL;tree=tree->next) {
      if (tree->state>0)
         fprintf(file,"%s[%d]\n",
                 ReWriteString(tree->baseId->name,NULL,ESCAPE_CHAR),
                 tree->state);
      else
         fprintf(file,"%s\n",tree->baseId->name);
      if (tree->size==0) {
         fprintf(file,"   %s\n\n",
                 ReWriteString(tree->root->macro->id->name,NULL,DBL_QUOTE));
      }
      else {
         array=(Node**) New(&tmpHeap,sizeof(Node*)*tree->size);
         for (i=0;i<tree->size;i++) array[i]=NULL;
         DownTree(tree->root,array);
         fprintf(file,"{\n");
         for (i=0;i<tree->size;i++) {
            node=array[i];
            if (node==NULL) 
               HError(2695,"ShowTreesCommand: Cannot find node %d",i);
            if (node->yes==NULL || node->no==NULL) 
               HError(2695,"ShowTreesCommand: Node %d has no children",i);
            fprintf(file," %3d %24s ",-i,
                    ReWriteString(node->quest->qName->name,NULL,SING_QUOTE));
            if (node->no->yes) fprintf(file,"  %5d    ",-node->no->snum);
            else fprintf(file," %9s ",
                         ReWriteString(node->no->macro->id->name,NULL,DBL_QUOTE));
            if (node->yes->yes) fprintf(file,"  %5d    ",-node->yes->snum);
            else fprintf(file," %9s ",
                         ReWriteString(node->yes->macro->id->name,NULL,DBL_QUOTE));
            fprintf(file,"\n");
         }
         fprintf(file,"}\n\n");
         Dispose(&tmpHeap,array);
      }
   }
   fclose(file);
}

static Node *GetNode(Node *node,int n)
{
   Node *ret;
  
   ret=NULL;
   if (node->snum==n) return(node);
   else {
      if (node->yes) {
         ret=GetNode(node->yes,n);
         if (ret) return(ret);
      }
      if (node->no) {
         ret=GetNode(node->no,n);
         if (ret) return(ret);
      }
   }
   return(NULL);
}    

Tree *LoadTree(char *name,Source *src)
{
   int n,nt,num_no,num_yes;
   char type;
   LabId lab_no,lab_yes,qname;
   Tree *tree;
   Node *node;
   char buf[256],bname[64],*p;
  
   strcpy(bname,name);
   if (name[strlen(bname)-1]==']' && strrchr(bname,'[')!=NULL) {
      p=strchr(bname,'[');
      *p++=0;
      if (sscanf(p,"%d",&n)!=1)
         HError(2660,"LoadTree: Cannot parse tree state %s %s",name,p);
      type='s';
   }
   else
      n=-1,type='h';
   if (GetLabId(bname,TRUE)==NULL)
      HError(2660,"LoadTree: Cannot parse tree name %s (base-phone)",name);

   tree=CreateTree(NULL,GetLabId(bname,FALSE),n);
   nt=0;tree->size=0;
   tree->root=CreateTreeNode(NULL,NULL);
   tree->root->snum=0;
   ReadString(src,buf);
   if (strcmp(buf,"{")) {
      if ((tree->root->macro=FindMacroName(hset,type,GetLabId(buf,FALSE)))==NULL)
         HError(2661,"LoadTree: Macro %s not recognised",buf);
   }
   else {
      while (ReadString(src,buf),strcmp("}",buf)!=0) {
         tree->size++;
         sscanf(buf,"%d",&n);

         ReadString(src,buf);
         qname=GetLabId(buf,FALSE);
         if (qname==NULL?TRUE:qname->aux==0)
            HError(2661,"LoadTree: Question %s not recognised",buf);

         if ((node=GetNode(tree->root,-n))==NULL)
            HError(2661,"LoadTree: Node %d not in tree",n);

         node->quest=(QLink) qname->aux;
         node->yes=CreateTreeNode(NULL,node);
         node->yes->ans=TRUE;
         node->no=CreateTreeNode(NULL,node);
         node->no->ans=FALSE;

         ReadString(src,buf);
         if (sscanf(buf,"%d",&num_no)==1)
            lab_no=NULL;
         else
            if ((lab_no=GetLabId(buf,FALSE))==NULL)
               HError(2661,"LoadTree: Macro %s not recognised",buf);
         ReadString(src,buf);
         if (sscanf(buf,"%d",&num_yes)==1)
            lab_yes=NULL;
         else
            if ((lab_yes=GetLabId(buf,FALSE))==NULL)
               HError(2661,"LoadTree: Macro %s not recognised",buf);
         if (lab_no) {
            node->no->macro=FindMacroName(hset,type,lab_no);
            node->no->snum=--nt;
            node->no->next=tree->leaf;
            if (tree->leaf) tree->leaf->prev=node;
            node->no->prev=NULL;
            tree->leaf=node;
         }
         else 
            node->no->snum=-num_no,node->no->macro=NULL;
         if (lab_yes) {
            node->yes->macro=FindMacroName(hset,type,lab_yes);
            node->yes->snum=--nt;
            node->yes->next=tree->leaf;
            if (tree->leaf) tree->leaf->prev=node;
            node->yes->prev=NULL;
            tree->leaf=node;
         }
         else
            node->yes->snum=-num_yes,node->yes->macro=NULL;
      }
   }
   return(tree);
}

void LoadTreesCommand(void)
{
   Source src;
   char qname[256],buf[4096],info[256];
   char fn[256];
  
   ChkedAlpha("LT trees files name",fn);        /* get name of trees file */
   if (trace & T_BID) {
      printf("\nLT %s Loading questions/trees from file\n",fn);
      fflush(stdout);
   }
   if(InitSource(fn,&src,NoFilter)<SUCCESS)
      HError(2610,"LoadTreesCommand: Can't open file %s", fn);

   while(ReadString(&src,info)) {
      if (strcmp(info,"QS")!=0)
         break;
      ReadString(&src,qname);
      ReadLine(&src,buf);
    
      LoadQuestion(qname,NULL,buf);
   }
   if (qHead==NULL)
      HError(2660,"LoadTreesCommand: Questions Expected");

   do {
      LoadTree(info,&src);
   }
   while (ReadString(&src,info));
   CloseSource(&src);
}

/* ----------------- TR - Set Trace Level Command --------------- */

/* SetTraceCommand: set the trace level */
void SetTraceCommand(void)
{
   int m;

   m = ChkedInt("Trace level",0,31);
   if (trace & T_BID) {
      printf("\nTR %d\n Adjusting trace level\n",m);
      fflush(stdout);
   }
   if (m & T_MEM)
      PrintAllHeapStats();
   trace = ((cmdTrace&0xfff0) | (m&0xf));
}

/* ----------------- RN - Renaming HMMSet Command --------------- */

/* RenameHMMSetIdCommand: Change the HMM Set identifier */
void RenameHMMSetIdCommand(void)
{
   char buf[MAXSTRLEN];

   ChkedAlpha("RN Rename the HMM Set identifier",buf);
   hset->hmmSetId=CopyString(hset->hmem,buf);

}

/* -------------------- CL - Clone Command ---------------------- */


void SwapLists(HMMSet *set,HMMSet *list)
{
   MLink q;
   int h;

   /* First delete old HMM names */
   for (h=0; h<MACHASHSIZE; h++)
      for (q=set->mtab[h]; q!=NULL; q=q->next)
         if (q->type=='l' || q->type=='h')
            DeleteMacro(set,q);
   /* Then add new ones */
   for (h=0; h<MACHASHSIZE; h++)
      for (q=list->mtab[h]; q!=NULL; q=q->next) 
         if (q->type=='l' || q->type=='h')
            NewMacro(set,fidx,q->type,q->id,q->structure);
}

/* CloneCommand: clone the current HMM set to make the target
   set.  The 'basename' of every target hmm must be in the
   current HMMlist. */
void CloneCommand(void)
{
   char buf[255];
   HMMSet *tmpSet;
   HLink oldHMM,newHMM;
   MLink p,q;
   int h;
   
   ChkedAlpha("CL hmmlist file name",buf);
   if (trace & T_BID) {
      printf("\nCL %s\n Cloning current hmms to produce new set\n",buf);
      fflush(stdout);
   }
   tmpSet=(HMMSet*) New(&hmmHeap,sizeof(HMMSet));
   CreateHMMSet(tmpSet,&hmmHeap,TRUE);
   if(MakeHMMSet(tmpSet,buf)<SUCCESS)
      HError(2628,"CloneCommand: MakeHMMSet failed");
   
   for (h=0; h<MACHASHSIZE; h++)
      for (q=tmpSet->mtab[h]; q!=NULL; q=q->next) 
         if (q->type=='l') {
            if ((p=FindMacroName(hset,'l',q->id))==NULL)
               oldHMM=FindBaseModel(hset,q->id,baseMono);
            else
               oldHMM=(HLink) p->structure;
            newHMM=(HLink) q->structure;
            CloneHMM(oldHMM,newHMM,TRUE);
            newHMM->nUse=1;
         }
   SwapLists(hset,tmpSet);
}

/* -------------------- CM - Concatenate Command ---------------------- */

/* ConcatenateCommand: Concatenate hmms to synthesize a new hmm */
void ConcatenateCommand(void)
{
   int i = 0, j = 0, k = 0, n = 0, ns = 0, d = 0;
   char buf[255], baset[255];
   char **buf2 = NULL;
   LabId Id = NULL, tId = NULL;
   StateElem *t = NULL;
   MLink macroName = NULL;
   HLink *src = NULL, tgt = NULL;

   ChkedAlpha("target HMM name", buf);
   n = ChkedInt("number of HMMs to concantenate", 1, 16);
   buf2 = (char **)New(&gstack, (n+1) * sizeof(char *));
   /* list of source HMMs */
   src = (HLink *)New(&gstack, (n+1) * sizeof(HLink));
   for (i=1; i<=n; i++) {
      buf2[i] = (char *)New(&gstack, 256 * sizeof(char));
      ChkedAlpha("source HMM name", buf2[i]);
      Id = GetLabId(buf2[i], FALSE);
      macroName = FindMacroName(hset, 'l', Id);
      if(macroName == NULL) {
         HError(2674,"ConcatenateCommand: Unknown label %s", Id->name);
      }
      src[i] = (HLink) macroName->structure;
      ns += src[i]->numStates;
   }
   /* total number states of new HMM */
   ns = ns - n*2 + 2;

   if (trace & T_BID) {
      fprintf(stdout, "\nCM synthesizing %d state HMM %s using %d HMMs: ", ns, buf, n);
      for (i=1; i<=n; i++) {
         fprintf(stdout, "%s ", buf2[i]);
      }
      fprintf(stdout, "\n");
      fflush(stdout);
   }

   Id = GetLabId(buf, TRUE);
   strcpy(baset, Id->name);
   TriStrip(baset);
   strcat(baset, "_t");
   tgt = (HLink) New(hset->hmem, sizeof(HMMDef));
   tgt->owner = hset;

   tgt->numStates = ns;
   tgt->dur = src[1]->dur;
   t = (StateElem *)New(hset->hmem, (tgt->numStates - 2) * sizeof(StateElem));
   tgt->svec = t-2;
   /* state vector of new HMM */
   for (i=1; i<=n; i++) {
      for (j=2; j<src[i]->numStates; j++) {
         t->info =  src[i]->svec[j].info;
         t++;
      }
   }

   /* block diagnoanal structure transition matrix of new HMM */
   tId = GetLabId(baset, FALSE);
   if (tId == NULL) {
      tId = GetLabId(baset, TRUE);
   }
   macroName = FindMacroName(hset, 't', tId);
   /* create new transition matrix if necessary */
   if (macroName == NULL) {
      tgt->transP = CreateSMatrix(hset->hmem, ns, ns);
      for (i=1; i<=ns; i++) {
         for (j=1; j<=ns; j++) {
            tgt->transP[i][j] = LZERO;
         }
      }
      d = 0;
      for (i=1; i<=n; i++) {
         if (i == 1) {
            for (j=1; j<=src[i]->numStates; j++) {
               for (k=1; k<=src[i]->numStates; k++) {
                  tgt->transP[j][k] = src[i]->transP[j][k];
               }
            }
            d += (src[i]->numStates-1);
         }
         else {
            for (j=2; j<=src[i]->numStates; j++) {
               for (k=2; k<=src[i]->numStates; k++) {
                  tgt->transP[j-1+d][k-1+d] = src[i]->transP[j][k];
               }
            }
            d += (src[i]->numStates-2);
         }
      }
      NewMacro(hset, fidx, 't', tId, tgt->transP);
   }
   /* use found ones */
   else {
      tgt->transP = (SMatrix) macroName->structure;
   }
   IncUse(tgt->transP);

   tgt->hook = NULL; tgt->nUse = 1;
   NewMacro(hset, fidx, 'h', Id, tgt);

   if (trace & T_BID) {
      ShowMacros(tgt);
   }

   Dispose(&gstack, buf2);
}


/* -------------------- DP - Duplicate Command ---------------------- */

SVector DupSVector(SVector v)
{
   SVector w;

   if (v==NULL) return(NULL);
   if (GetUse(v)>0) {   /* vector is shared */
      if ((w=(SVector) GetHook(v))==NULL)
         HError(2692,"DupSVector: could not find replacement vector macro");
      IncUse(w);
   } else {                     /* vector not shared so Dup it */
      w = CloneSVector(&hmmHeap,v,FALSE);
   }
   return(w);
}

SMatrix DupSMatrix(SMatrix m)
{
   SMatrix w;

   if (m==NULL) return(NULL);
   if (GetUse(m)>0) {   /* matrix is shared */
      if ((w=(SMatrix) GetHook(m))==NULL)
         HError(2692,"DupSMatrix: could not find replacement matrix macro");
      IncUse(w);
   } else {                     /* matrix not shared so Dup it */
      w = CloneSMatrix(&hmmHeap,m,FALSE);
   }
   return(w);
}

STriMat DupSTriMat(STriMat m)
{
   STriMat w;

   if (m==NULL) return(NULL);
   if (GetUse(m)>0) {   /* TriMat is shared */
      if ((w=(STriMat) GetHook(m))==NULL)
         HError(2692,"DupSTriMat: could not find replacement TriMat macro");
      IncUse(w);
   } else {                     /* TriMat not shared so Dup it */
      w = CloneSTriMat(&hmmHeap,m,FALSE);
   }
   return(w);
}

/* DupMixPDF: return a Dup of given MixPDF */
MixPDF *DupMixPDF(MixPDF *s, Boolean frc)
{
   MixPDF *t;                   /* the target */
   
   if (s->nUse>0 && !frc) {     /* shared struct so just return ptr to it */
      if ((t=(MixPDF *) s->hook)==NULL)
         HError(2692,"DupMixPDF: could not find dup mixpdf macro");
      ++t->nUse;
      return t;
   }
   t = (MixPDF*) New(&hmmHeap,sizeof(MixPDF));
   t->nUse = 0; t->hook = NULL;
   t->ckind=s->ckind; t->gConst = s->gConst;
   t->mean=DupSVector(s->mean);
   switch(s->ckind) {
   case DIAGC:
   case INVDIAGC:
      t->cov.var = DupSVector(s->cov.var); break;
   case LLTC:
   case FULLC:
      t->cov.inv = DupSTriMat(s->cov.inv); break;
   case XFORMC:
      t->cov.xform = DupSMatrix(s->cov.xform); break;
   }
   return t;
}

/* DupStream: return a Dup of given stream */
MixtureElem *DupStream(StreamElem *ste)
{
   int m,M;
   MixtureElem *sme,*tme,*t;

   M = ste->nMix;
   tme = (MixtureElem*) New(&hmmHeap,sizeof(MixtureElem)*M);
   t = tme-1;
   sme = ste->spdf.cpdf + 1;
   for (m=1; m<=M; m++,sme++,tme++) {
      tme->weight = sme->weight;
      if (tme->weight > MINMIX)
         tme->mpdf = DupMixPDF(sme->mpdf,FALSE);
      else
         tme->mpdf = NULL;
   }
   return t;
}

/* DupState: return a Dup of given State */
StateInfo *DupState(StateInfo *si, Boolean frc)
{
   StateInfo *t;                /* the target */
   StreamElem *tste,*sste;
   int s;
   
   if (si->nUse>0 && !frc) {    /* shared struct so just return ptr to it */
      if ((t=(StateInfo *) si->hook)==NULL)
         HError(2692,"DupState: could not find dup stateInfo macro");
      ++t->nUse;
      return t;
   }
   t = (StateInfo*) New(&hmmHeap,sizeof(StateInfo));
   t->nUse = 0; t->hook = NULL;
   tste = (StreamElem*) New(&hmmHeap,sizeof(StreamElem)*hset->swidth[0]);
   t->pdf = tste-1; sste = si->pdf + 1;
   for (s=1; s<=hset->swidth[0]; s++,tste++,sste++) {
      tste->nMix = sste->nMix; 
      tste->hook = NULL;
      tste->spdf.cpdf = DupStream(sste);
   }
   t->dur = DupSVector(si->dur);
   t->weights = DupSVector(si->weights);
   return t;
}

/* DupHMM: if src hasnt been Dupd before then just return
   a pointer to it, otherwise copy it.  The hook field of
   the HMM def is used to flag that a first copy has already 
   been taken */
HMMDef *DupHMM(HMMDef *src)
{
   HMMDef *tgt;
   StateElem *s,*t;
   int i;
  
   tgt = (HLink) New(&hmmHeap,sizeof(HMMDef));
   t = (StateElem*) New(&hmmHeap,sizeof(StateElem)*(src->numStates-2));
   tgt->owner = src->owner;
   tgt->numStates = src->numStates;
   tgt->dur = DupSVector(src->dur);
   tgt->transP = DupSMatrix(src->transP);
   tgt->svec = t-2; s = src->svec+2;
   for (i=2; i<tgt->numStates; i++,s++,t++) {
      if (hset->hsKind==TIEDHS || hset->hsKind==DISCRETEHS)
         t->info = CloneState(hset, s->info, FALSE);
      else
         t->info = DupState(s->info,FALSE);
   }
   tgt->hook = NULL; tgt->nUse = 1;
   src->hook = tgt;
   return tgt;
}

void DupMacro(MLink ml,LabId labid)
{
   Vector iv,ov;
   Matrix im,om;
   MixPDF *imp,*omp;
   StateInfo *isi,*osi;
   Ptr ostct=NULL;
    
   switch(ml->type) {
   case 's':
      isi=(StateInfo *) ml->structure;
      osi=DupState(isi,TRUE);
      ostct=osi;
      isi->hook=osi;
      break;
   case 'm':
      imp=(MixPDF *) ml->structure;
      omp=DupMixPDF(imp,TRUE);
      ostct=omp;
      imp->hook=omp;
      break;
   case 'c':
   case 'i':
      im=(Matrix) ml->structure;
      om=DupSTriMat(im);
      ostct=om;
      SetHook(ml->structure,ostct);
      break;
   case 't':
   case 'x':
      im=(Matrix) ml->structure;
      om=DupSMatrix(im);
      ostct=om;
      SetHook(ml->structure,ostct);
      break;
   case 'v':
   case 'u':
   case 'w':
   case 'd':
      iv=(Vector) ml->structure;
      ov=DupSVector(iv);
      ostct=ov;
      SetHook(ml->structure,ostct);
      break;
   }
   if (trace & (T_DET | T_MAC)) {
      printf("   Dup Macro: ~%c:%-20s (from %s)\n",
             ml->type,labid->name,ml->id->name);
      fflush(stdout);
   }
   NewMacro(hset,fidx,toupper(ml->type),labid,ostct);
}

/* DuplicateCommand: clone the current HMM set to make the target
   set.  The 'basename' of every target hmm must be in the
   current HMMlist. */
void DuplicateCommand(void)
{
   char buf[255],name[255],*p,id[255],what[255];
   MLink ml;
   HLink hmm,newHMM;
   Ptr str;
   LabId labid;
   int i,N,h,nphmm=0,nlhmm=0;
   
   ChkedAlpha("DP macro types",what);
   N = ChkedInt("No. Duplicates",1,INT_MAX);
   if (trace & (T_BID | T_MAC)) {
      printf("\nDP %s %d 'id1' ..\n Duplicating current HMMSet\n",what,N);
      fflush(stdout);
   }

   ResetHooks(hset,NULL);
   occStatsLoaded = TRUE;  /* We just over-wrote them */
   
   for (i=1;i<=N;i++) {
      ChkedAlpha("DP macro name extension",id);
      if (trace & T_IND) {
         printf("  Duplicating for id == '%s'\n",id);
         fflush(stdout);
      }
    
      for (h=0; h<MACHASHSIZE; h++)
         for (ml=hset->mtab[h]; ml!=NULL; ml=ml->next) {
            if (ml->type=='*' || isupper((int) ml->type)) continue;
            if (ml->type=='l' || ml->type=='h') str=NULL;
            else if (strchr(what,ml->type)) str=NULL;
            else str=ml->structure;
            SetMacroHook(ml,str);
            if (ml->type=='m' || ml->type=='s' || !strchr(what,ml->type))
               continue;
            strcpy(buf,ml->id->name);
            strcat(buf,id);
            DupMacro(ml,GetLabId(buf,TRUE));
         }
      if (strchr(what,'m'))
         for (h=0; h<MACHASHSIZE; h++)
            for (ml=hset->mtab[h]; ml!=NULL; ml=ml->next) {
               if (ml->type!='m') continue;
               strcpy(buf,ml->id->name);
               strcat(buf,id);
               DupMacro(ml,GetLabId(buf,TRUE));
            }
      if (strchr(what,'s'))
         for (h=0; h<MACHASHSIZE; h++)
            for (ml=hset->mtab[h]; ml!=NULL; ml=ml->next) {
               if (ml->type!='s') continue;
               strcpy(buf,ml->id->name);
               strcat(buf,id);
               DupMacro(ml,GetLabId(buf,TRUE));
            }
      
      for (h=0; h<MACHASHSIZE; h++)
         for (ml=hset->mtab[h]; ml!=NULL; ml=ml->next)
            if (ml->type=='l' || ml->type=='h') {
               strcpy(buf,ml->id->name);
               if ((p=strchr(buf,'+'))!=NULL) {
                  *p++=0;
                  sprintf(name,"%s%s+%s",buf,id,p);
               }
               else
                  sprintf(name,"%s%s",ml->id->name,id);
               labid = GetLabId(name,TRUE);
               hmm = (HLink) ml->structure;

               if (hmm->hook==NULL) {
                  newHMM = DupHMM(hmm);
                  hmm->hook = newHMM;
               }
               else newHMM = (HLink) hmm->hook;
               
               NewMacro(hset,fidx,toupper(ml->type),labid,newHMM);
               if (trace & T_DET) {
                  printf("   Dup Model: %s\n",labid->name);
                  fflush(stdout);
               }
               if (ml->type=='h') nphmm++;
               else nlhmm++;
            }
   }
   for (h=0; h<MACHASHSIZE; h++)
      for (ml=hset->mtab[h]; ml!=NULL; ml=ml->next) {
         if (ml->type=='*') continue;
         ml->type=tolower(ml->type);
         SetMacroHook(ml,NULL);
      }
   if (trace & T_BID) {
      printf(" DP: HMMSet Duplicated for %d ids for %d / %d new models\n",
             N,nlhmm,nphmm);
      fflush(stdout);
   }
}

/* -------------------- MT - Make Triphone Command --------------- */

/* -------------------- Triphone Recording ------------------ */

typedef struct {
   HLink left,right;            /* physical names of constituent biphones */
   MLink ml;
}TriRec;

static int triSize;        /* num items in triTab */
static TriRec *triTab;     /* array[0..triSize-1] of TriRec; */

/* SameTriphone: return the existing HMMEntry for left+right, if any */
MLink SameTriphone(HLink left, HLink right)
{
   int i;
   TriRec *tr;
   
   for (i=0,tr=triTab;i<triSize;i++,tr++)
      if (tr->left == left && tr->right==right)
         return tr->ml;
   return NULL;
}

/* RecordTriphone: in next free slot of triTab */
void RecordTriphone(HLink left, HLink right, MLink ml)
{
   triTab[triSize].left = left;
   triTab[triSize].right = right;
   triTab[triSize].ml = ml;
   ++triSize;
}

/* MakeTriCommand: make each phone in given list from currently
   loaded models.  If phone is not a triphone then it must 
   be already loaded in which case it is just cloned.
   Otherwise, the triphone is synthesised from left and right
   biphones which must already be loaded.  The synthesis for
   A-B+C is to clone B+C then replace state 2 by that of
   A-B (this is intended for 3 state models).
   */
void MakeTriCommand(void)
{
   char fn[255], *s;
   HMMSet *tmpSet;
   HLink hmm,left,right;
   MLink q,match;
   int h,mcount=0,tcount=0;
   
   ChkedAlpha("MT hmmlist file name",fn);
   if (trace & T_BID) {
      printf("\nMT %s\n Converting current biphones to triphones\n",fn);
      fflush(stdout);
   }
   tmpSet=(HMMSet*) New(&hmmHeap,sizeof(HMMSet));
   CreateHMMSet(tmpSet,&hmmHeap,TRUE);
   if(MakeHMMSet(tmpSet,fn)<SUCCESS)
      HError(2628,"MakeTriCommand: MakeHMMSet failed");
   
   triTab=(TriRec*) New(&tmpHeap,tmpSet->numLogHMM*sizeof(TriRec));
   triSize = 0;
   for (h=0; h<MACHASHSIZE; h++)
      for (q=tmpSet->mtab[h]; q!=NULL; q=q->next) {
         if (q->type != 'l') continue;
         s = q->id->name;
         if (strchr(s,'-') != NULL && strrchr(s,'+') != NULL ) { /*triphone*/
            right = FindBaseModel(hset,q->id,baseRight);
            left = FindBaseModel(hset,q->id,baseLeft);
            hmm=(HLink) q->structure;
            if ((match = SameTriphone(left,right)) == NULL) {
               RecordTriphone(left,right,q);
               CloneHMM(right,hmm,TRUE);
               tcount++;
               hmm->svec[2].info=CloneState(hset,left->svec[2].info,TRUE);
            } else {            /* this triphone is already made */
               hmm = (HLink) (q->structure = match->structure);
               hmm->nUse++;
            }
         } 
         else {                 /* not a triphone so just clone it */
            hmm = FindBaseModel(hset,q->id,baseNorm);
            CloneHMM(hmm, (HLink) q->structure, TRUE);
            mcount++;
         }
      }
   SwapLists(hset,tmpSet);
   if (trace & T_BID) {
      printf(" MT: %d distinct triphones, %d non-triphones\n",
             tcount,mcount);
      fflush(stdout);
   }
}

/* ----------------- AT/RT - Add/Remove Transition Command ---------------- */

/* EditTransMat: add/rem a transition in list of transP */
void EditTransMat(Boolean adding)
{
   ILink t,ilist = NULL;        /* list of items to tie */
   char type = 't';             /* type of items must be t */
   int i,j,k,use;
   float prob=0.0,sum,x;
   SMatrix tran;
   int N,nedit=0;
   Vector row;
   HMMDef *hmm;
   
   i = ChkedInt("From state",1,maxStates-1);
   j = ChkedInt("To state",2,maxStates);
   if (adding) prob = ChkedFloat("AT transition prob",0.0,0.999999);
   else prob = 0.0;
   if (trace & T_BID) {
      if (adding)
         printf("\nAT %d %d %.3f {}\n Adding transitions to transP\n",
                i,j,prob);
      else
         printf("\nRT %d %d {}\n Removing transitions from transP\n",i,j);
      fflush(stdout);
   }
   PItemList(&ilist,&type,hset,&source,trace&T_ITM);
   if (ilist == NULL) {
      HError(-2631,"EditTransMat: No trans mats to edit!");
      return;
   }
   for (t=ilist; t!=NULL; t=t->next) {
      hmm = t->owner;
      tran = hmm->transP;
      use = GetUse(tran);
      if (IsSeen(use)) continue;
      Touch(&use);SetUse(tran,use);
      N = hmm->numStates;
      if (i<1 || i>N-1 || j<2 || j>N)
         HError(2632,"EditTransMat: States (%d->%d) out of range for %s",
                i,j,HMMPhysName(hset,hmm));
      row = tran[i];
      row[j] = prob;
      sum = 0.0;
      for (k=1;k<=N;k++)
         if (k!=j) {
            x = row[k];
            row[k]=(x<LSMALL)?0.0:exp(x);
            sum += row[k];
         }
      sum /= 1.0-row[j];
      for (k=1;k<=N;k++) {
         x = row[k];
         if (k!=j) x /= sum;
         row[k] = (x<MINLARG)?LZERO:log(x);
      }
      nedit++;
   }
   ClearSeenFlags(hset,CLR_HMMS);
   FreeItems(&ilist);
   if (trace & T_BID) {
      printf(" %cT: %d transP matrices adjusted\n",(adding?'A':'R'),nedit);
      fflush(stdout);
   }
}

/* -------------------- MU - Mix Up Command ---------------------- */

/* MixUpCommand: increase num mixtures in itemlist, note
   that this also restores defunct mixtures.  When a mix is
   split, 1 is added to the hook of the mpdf struct and this 
   is used to weight against repeated splitting of the same
   mixture */
void MixUpCommand(void)
{
   ILink i,ilist = NULL;        /* list of items to mixup */
   char type = 'p';             /* type of items must be p */
   int ch,j,m,trg,M,mDefunct,n2fix,totm=0,totM=0;
   StreamElem *ste;
   HMMDef *hmm;
   char *hname;
   
   SkipWhiteSpace(&source);
   ch = GetCh(&source);
   if (ch=='+') {
      trg = -ChkedInt("Mix increment",1,INT_MAX);
   }
   else {
      UnGetCh(ch,&source);
      trg = ChkedInt("Mix target",1,INT_MAX);
   }
   if (trace & T_BID) {
      if (trg>0)
         printf("\nMU %d {}\n Mixup to %d components per stream\n",trg,trg);
      else
         printf("\nMU +%d {}\n Mixup by %d components per stream\n",-trg,-trg);
      fflush(stdout);
   }
   if (hset->hsKind==TIEDHS || hset->hsKind==DISCRETEHS)
      HError(2640,"MixUpCommand: MixUp only possible for continuous models");
   PItemList(&ilist,&type,hset,&source,trace&T_ITM);
   if (ilist == NULL) {
      HError(-2631,"MixUpCommand: No mixtures to increase!");
      return;
   }
   SetGCStats();
   for (i=ilist; i!=NULL; i=i->next) {
      ste = (StreamElem *)i->item;
      M = ste->nMix;
      if (M>0) ste->nMix=-M;
   }
   for (i=ilist; i!=NULL; i=i->next) {
      hmm = i->owner; 
      ste = (StreamElem *)i->item; M = ste->nMix;
      if (M<0) {
         M = ste->nMix = -M;
         hname = HMMPhysName(hset,hmm);
         for (j=1; j<=M; j++)   /* reset the split counters */
            ste->spdf.cpdf[j].mpdf->hook = (void *) 0;
         mDefunct = CountDefunctMix(ste);
         if (trg<0) m=M-trg;
         else m=trg;
         if (m > M-mDefunct) {
            n2fix = m-M+mDefunct;
            if (n2fix>mDefunct) n2fix = mDefunct;
            if (n2fix>0) {
               FixDefunctMix(hname,ste,n2fix);
            }
         }
         else n2fix=0;
         if (trace & T_IND) {
            if (n2fix>0)
               printf("  %12s : mixup %d -> %d (%d defunct restored)\n",
                      hname,M,m,n2fix);
            else
               printf("  %12s : mixup %d -> %d\n",hname,M,m);
         }
         if (m>maxMixes) maxMixes = m;
         if (m>M) UpMix(hname,ste,M,m);
         totm+=m;totM+=M;
         for (j=1; j<=ste->nMix; j++) /* restore the hooks */
            ste->spdf.cpdf[j].mpdf->hook = NULL;
      }
   }
   FreeItems(&ilist);
   if (trace & T_BID) {
      printf(" MU: Number of mixes increased from %d to %d\n",
             totM,totm);
      fflush(stdout);
   }
}

/* -------------------- TI - Tie Command ---------------------- */

/* TieCommand: tie all components in following itemlist */
void TieCommand(void)
{
   ILink ilist = NULL;          /* list of items to tie */
   char type = ' ';             /* type of items to tie */
   char macName[255];           /* name of macro to use */
   
   ChkedAlpha("TI macro name",macName);
   if (strlen(macName) > 20 )
      HError(-2639,"TieCommand: %s is rather long for a macro name!",macName);
   PItemList(&ilist,&type,hset,&source,trace&T_ITM);
   if (trace & (T_BID | T_MAC)) {
      printf("\nTI %s {}\n Tie items\n",macName);
      fflush(stdout);
   }
   ApplyTie(ilist,macName,type);
   FreeItems(&ilist);
}

/* -------------------- TI - Tie Command ---------------------- */

/* MakeIntoMacrosCommand: tie all components in following itemlist */
void MakeIntoMacrosCommand(void)
{
   ILink ilist = NULL;          /* list of items to tie */
   char type = ' ';             /* type of items to tie */
   char macName[255],buf[255];  /* name of macro to use */
   ItemRec itemp,*i;
   MixPDF *mp;
   StateInfo *si;
   StateElem *se;
   HMMDef *hmm;
   int use,n=0;
   
   ChkedAlpha("MM macro name",macName);
   if (trace & (T_BID | T_MAC)) {
      printf("\nMM %s {}\n Make each item into macro\n",macName);
      fflush(stdout);
   }
   if (strlen(macName) > 20 )
      HError(-2639,"MakeIntoMacrosCommand: %s is rather long for a macro name!",macName);
   PItemList(&ilist,&type,hset,&source,trace&T_ITM);
   if (type=='p' || type=='h')
      HError(2640,"MakeIntoMacros: Cannot convert PDFs or HMMs into macro");
   for (i=ilist;i;i=i->next) {
      se=(StateElem*)i->item; si=(StateInfo *)i->item; 
      mp=(MixPDF *)i->item; hmm=(HMMDef*)i->owner;
      switch(type) {
      case 's': use=se->info->nUse; break;
      case 'x': use=GetUse(mp->cov.xform); break;
      case 'v': use=GetUse(mp->cov.var); break;
      case 'i': use=GetUse(mp->cov.inv); break;
      case 'c': use=GetUse(mp->cov.inv); break;
      case 'u': use=GetUse(mp->mean); break;
      case 't': use=GetUse(hmm->transP); break;
      case 'm': use=mp->nUse; break;
      case 'w': use=GetUse(si->weights); break;
      case 'd': use=GetUse(si->dur); break;
      default : use=-1; break;
      }
      if (use==0) {
         itemp.owner=i->owner;
         itemp.item=i->item;
         itemp.next=NULL;
         sprintf(buf,"%s%d",macName,++n);
         ApplyTie(&itemp,buf,type);
      }
   }
   if (trace & T_BID) {
      printf(" MM: Made %d ~%c macros for %d items\n",
             n,type,NumItems(ilist));
      fflush(stdout);
   }
   FreeItems(&ilist);
}

/* -------------------- UT - Untie Command ---------------------- */

/* UntieCommand: untie all components in following itemlist */
void UntieCommand(void)
{
   ILink ilist = NULL;          /* list of items to untie */
   char type = ' ';             /* type of items to untie */
   
   if (trace & (T_BID | T_MAC)) {
      printf("\nUT {}\n Untie previously tied structures\n");
      fflush(stdout);
   }
   PItemList(&ilist,&type,hset,&source,trace&T_ITM);
   if (ilist==NULL) {
      HError(-2631,"UntieCommand: No items to untie");
      return;
   }
   switch (type) {
   case 's': UntieState(ilist); break;
   case 'm': UntieMix(ilist);   break;
   case 't': UntieTrans(ilist); break;
   case 'u': UntieMean(ilist);  break;
   case 'v': UntieVar(ilist);   break;
   case 'i': UntieInv(ilist);   break;
   case 'x': UntieXform(ilist); break;
   case 'h':
      HError(2640,"UntieCommand: Joined HMMs cannot be untied");
   case 'p':
      HError(2640,"UntieCommand: Joined PDF's cannot be untied");
   default:
      HError(2640,"UntieCommand: Cannot untie %c's",type);
   }
   if (trace & (T_BID | T_MAC)) {
      printf(" UT: Untied %d ~%c structures\n",NumItems(ilist),type);
      fflush(stdout);
   }
   FreeItems(&ilist);
}

/* -------------------- JO - Set Join Size Command ---------------- */

/* JoinSizeCommand: set number of tied mixtures */
void JoinSizeCommand(void)
{
   joinSize = ChkedInt("JoinSize",1,INT_MAX);
   joinFloor = ChkedFloat("JoinFloor",0.0,FLOAT_MAX);
   if (trace & T_BID) {
      printf("\nJO %d %.2f\n Set size and floor for pdf tying\n",
             joinSize,joinFloor);
      fflush(stdout);
   }
}

/* ----------- TC/NC - Threshold/Number Cluster Command ---------------------- */

/* ClusterCommand: cluster all components in following itemlist 
   and then tie each cluster.  The i'th cluster
   is called macroi.  If macro = 'check' then
   just the cluster info is output and no tying
   is done.  If nCluster is set then clustering
   converges when the required number is reached
   otherwise a threshold for ming is used */
void ClusterCommand(Boolean nCluster)
{
   ILink ilist = NULL;          /* list of items to tie */
   char type = ' ';             /* type of items to tie */
   char macName[255];           /* name of macro to use */
   int numItems,numClust=1;     /* num clusters required */
   float thresh = 1.0E15;
   static int totalItems = 0;   /* for overall statistics */
   static int totalClust = 0;
   
   if (nCluster)
      numClust = ChkedInt("No. clusters",1,INT_MAX);
   else
      thresh = ChkedFloat("Cluster threshold",0.0,FLOAT_MAX);
   ChkedAlpha((char *) (nCluster ? "NC macro name" : "TC macro name"), macName);
   if (trace & (T_BID | T_CLUSTERS)) {
      if (nCluster) 
         printf("\nNC %d %s {}\n Cluster items into N groups\n",
                numClust,macName);
      else
         printf("\nTC %.2f %s {}\n Cluster items until threshold exceeded\n",
                thresh,macName);
      fflush(stdout);
   }
   if (strlen(macName) > 20 )
      HError(-2639,"ClusterCommand: %s is rather long for a macro name",
             macName);
   PItemList(&ilist,&type,hset,&source,trace&T_ITM);
   if (ilist==NULL) {
      HError(-2631,"ClusterCommand: No items to cluster for %s\n",macName);
      return;
   }
   numItems = NumItems(ilist);
   Clustering(ilist, &numClust, thresh, type, macName);
   totalItems += numItems; totalClust += numClust;
   if (trace & T_BID) {
      printf("\n %cC: Stats %d->%d [%.1f%%] { %d->%d [%.1f%%] total }\n",
             (nCluster?'N':'T'),numItems,numClust,numClust*100.0/numItems,
             totalItems,totalClust,totalClust*100.0/totalItems);
      fflush(stdout);
   }
}

/* ------------- LS/RO - Load Stats/Remove Outliers Set-Up Command ----------- */

/* LoadStatsCommand: loads occupation counts for hmms */
void LoadStatsCommand(void)
{
   char statfile[256];
   
   ChkedAlpha("LS stats file name",statfile);
   if (trace & T_BID) {
      printf("\nLS %s\n",statfile);
      printf(" Loading state occupation stats\n");
      fflush(stdout);
   }
   LoadStatsFile(statfile, hset, trace & T_BID);
   occStatsLoaded = TRUE;
}


/* RemOutliersCommand: used in conjunction with NC/TC.  After clustering
   any group with a total occupation count less than the specified 
   threshold is merged with the closest group
   */
void RemOutliersCommand(void)
{
   char statfile[256];
   int ch;
   
   outlierThresh = ChkedFloat("Outlier Occupancy Threshold",0.0,FLOAT_MAX);
   if (trace & T_BID) {
      printf("RO %.2f ''\n",outlierThresh);
      printf(" Setting outlier threshold for clustering\n");
      fflush(stdout);
   }
   do {
      ch = GetCh(&source);
      if (ch=='\n') break;
   }
   while(ch!=EOF && isspace(ch));
   UnGetCh(ch,&source);
   if (ch!='\n') { /* Load Stats File */
      ChkedAlpha("RO stats file name",statfile);
      if (trace & T_BID) {
         printf(" RO->LS %s\n",statfile);
         printf("  and loading state occupation stats\n");
         fflush(stdout);
      }
      if (occStatsLoaded)
         HError(-2655,"RemOutliersCommand: Stats already loaded");
      else
         LoadStatsFile(statfile,hset,trace&T_BID);
      occStatsLoaded = TRUE;
   }
}

/* ---------------- CO - Compact HMM Set Command ---------------- */

/* CompactCommand: Find all sets of identical HMM's in current
   hmm set and merge each into a single physical hmm */
void CompactCommand(void)
{
   ILink seen=NULL,i;
   MLink q;
   HLink hmm;
   int h;
   Boolean seenMean,seenVar;
   char fn[255];
   
   ChkedAlpha("CO hmmlist file name",fn);  /* get name of new hmm list */
   if (trace & T_BID) {
      printf("\nCO %s\n Create compact HMMList\n",fn);
      fflush(stdout);
   }
   
   ResetHooks(hset,"h");
   seenMean=seenVar=FALSE;
   for (h=0; h<MACHASHSIZE; h++)
      for (q=hset->mtab[h]; q!=NULL; q=q->next)  {
         if (q->type=='m') seenMean=seenVar=TRUE;
         if (q->type=='u') seenMean=TRUE;
         if (q->type=='v' || q->type=='i' || q->type=='c' || q->type=='x')
            seenVar=TRUE;
         if (seenMean && seenVar) break;
      }
   if (seenMean && seenVar) equivState=TRUE;
   else equivState=FALSE;

   for (h=0; h<MACHASHSIZE; h++)
      for (q=hset->mtab[h]; q!=NULL; q=q->next) 
         if (q->type=='h') {
            hmm=(HLink) q->structure;
            for (i=seen;i!=NULL;i=i->next)
               if (EquivHMM(hmm,i->owner)) break;
            if (i!=NULL) {
               hmm->hook=i->owner;
               i->owner->nUse++;
               DeleteMacro(hset,q);
               if (trace & T_DET)
                  printf("  %12s == %s\n",q->id->name,
                         ((MLink)i->item)->id->name);
            }
            else {
               if (trace & T_DET)
                  printf("  %12s ++\n",q->id->name);
               AddItem(hmm,q,&seen);
            }
         }

   for (h=0; h<MACHASHSIZE; h++)
      for (q=hset->mtab[h]; q!=NULL; q=q->next) 
         if (q->type=='l') {
            hmm=(HLink) q->structure;
            if (hmm->hook!=NULL && hmm->hook!=hmm) {
               NewMacro(hset,fidx,'1',q->id,hmm->hook);	/* cz277 - ANN */
               DeleteMacro(hset,q);
            }
         }
   
   for (h=0; h<MACHASHSIZE; h++)
      for (q=hset->mtab[h]; q!=NULL; q=q->next) 
         if (q->type=='1') {	/* cz277 - ANN */
            hset->numLogHMM++;hset->numMacros--;
            q->type='l';	/* cz277 - ANN */
         }
   ResetHooks(hset,"h");
   
   if(SaveHMMList(hset,fn)<SUCCESS)
      HError(2611,"CompactCommand: SaveHMMList failed");
   if (trace & T_BID) {
      printf(" CO: HMMs %d Logical %d Physical\n",
             hset->numLogHMM,hset->numPhyHMM);
      fflush(stdout);
   }
}


/* ------------------ SS - SplitStream Command ------------------ */

/* SplitStreamCommand: split input vectors into n independent data streams */
void SplitStreamCommand(Boolean userWidths)
{
   int s,next,S,V,ewidth,vchk,i,nedit=0;
   int epos[4];
   char buf[20],c=' ';
   HLink hmm=NULL;
   HMMScanState hss;
   short swidth[SMAX];
   ParmKind pk;
   Vector vf[SMAX],v;
   Boolean simple=TRUE,vfSet,hasE,hasN,hasD,hasA,has0;

   S = ChkedInt("No. streams",2,SMAX-1);
   V = hset->vecSize;  pk = hset->pkind;
   swidth[0] = S;
   if (userWidths)
      for (s=1;s<=S;s++)
         swidth[s] = ChkedInt("Stream width",1,V);
   if (trace & (T_BID | T_MAC | T_SIZ)) {
      if (userWidths) {
         printf("\nSU %d",S);
         for (s=1;s<=S;s++) printf(" %d",swidth[s]);
         printf("\n splitting into user defined stream widths\n");
      }
      else
         printf("SS %d\n splitting into standard stream widths\n",S);
      fflush(stdout);
   }
   if (hset->hsKind==TIEDHS || hset->hsKind==DISCRETEHS)
      HError(2640,"SplitStreamsCommand: Only implemented for continuous");
   if (hset->swidth[0] != 1)
      HError(2640,"SplitStreamCommand: Cannot split multiple streams");
   
   /* Calculate Stream widths */
   hasN=HasNulle(pk);
   if (!userWidths) {
      hasD=HasDelta(pk);hasA=HasAccs(pk);
      hasE= HasEnergy(pk); has0 = HasZeroc(pk);
      if (hasE && has0)
         HError(2641,"SplitStreamCommand: Source data cant have _0 and _E");
      hasE |= has0;     /* dont care which it is any more */
      if (!hasD && !hasE)
         HError(2641,"SplitStreamCommand: Source data must have _D or _E");
      if (S==2) {
         if (hasA)
            HError(2641,"SplitStreamCommand: need 3 streams min when _A");  
         if (hasD) {
            swidth[1] = V / 2;
            swidth[2] = V - swidth[1];
         } else {
            swidth[1] = V -1; swidth[2] = 1;
         }
      }
      else if (S==3) {
         if (!hasD)
            HError(2641,"SplitStreamCommand: Source data must have _D");
         if (!hasE && !hasA)
            HError(2641,"SplitStreamCommand: Source data must have _E or _A");
         if (hasA) {
            swidth[1] = V / 3;
            swidth[2] = swidth[3] = (V - swidth[1]) / 2;
         } else {
            if (hasN) ewidth=1; else ewidth=2;
            swidth[1] = swidth[2] = (V-ewidth) / 2;
            swidth[3] = ewidth;
         }
      }
      else if (S==4) {
         if (!(hasE && hasA))
            HError(2641,"SplitStreamCommand: Source data must have _E and _A");
         if (hasN) ewidth=2; else ewidth=3;
         swidth[1] = swidth[2] = swidth[3] = (V-ewidth) / 3;
         swidth[4] = ewidth;
      }
      else
         HError(2641,"SplitStreamCommand: cant manage %d streams",S);
      if (trace & (T_BID | T_SIZ)) {
         printf(" Calculated %d stream widths: ",S);
         for (s=1;s<=S;s++) printf(" %d",swidth[s]);
         printf("\n");
         fflush(stdout);
      }
      simple = !hasE || (swidth[S]<2) || (S==2) || (hasA && S==3) ;
   }
   
   /* Get varFloor */
   SetVFloor(hset,vf,-1.0);
   for (i=1,vfSet=TRUE;i<=V;i++)
      if (vf[1][i]<0.0) {
         vfSet=FALSE;
         break;
      }

   /* Check Widths consistent with VecSize */
   vchk = 0;
   hset->swidth[0]=swidth[0];
   for (i=1; i<=S; i++) {
      vchk += swidth[i];
      hset->swidth[i]=swidth[i];
   }
   if (vchk != V) {
      if (userWidths)
         HError(2630,"SplitStreamCommand: Sum of stream widths = %d, vecSize=%d",vchk,V);     
      else
         HError(2696,"SplitStreamCommand: Bug in stream width allocation");
   
   }
   /* Do Splitting */
   NewHMMScan(hset,&hss);
   while(GoNextState(&hss,FALSE)) {
      SplitStreams(hset,hss.si,simple,nedit==0);
      if (trace & (T_SIZ | T_DET)) {
         if (nedit==0) printf(" For model.state["),hmm=NULL;
         if (hmm!=hss.hmm)
            printf("]\n  %12s.state",hss.mac->id->name),c='[',hmm=hss.hmm;
         printf("%c%d",c,hss.i);c=',';
         fflush(stdout);
      }
      nedit++;
   }
   if (trace & (T_SIZ | T_DET))
      printf("]\n"),fflush(stdout);
   EndHMMScan(&hss);
   /* Now do varFloor */
   if (vfSet) {
      if (trace & (T_SIZ | T_DET | T_MAC))
         printf(" For varFloor\n"),fflush(stdout);
      for (i=0; i<3; i++) epos[i] = 0;
      v=vf[1];
      DeleteMacroStruct(hset,'v',v);
      for (s=1,next=1;s<=S;s++) {
         sprintf(buf,"varFloor%d",s);
         if (simple || s<S) 
            vf[s] = SliceVector(v,next,next+swidth[s]-1);
         else
            vf[s] = ChopVector(v,epos[0],epos[1],epos[2]);
         NewMacro(hset,fidx,'v',GetLabId(buf,TRUE),vf[s]);
         if (trace & (T_DET | T_MAC)) {
            printf("   varFloor Macro: ~m:%s created [%d]\n",buf,swidth[s]);
         }
         next += swidth[s];
         if (!simple && !(hasN && s==1))
            epos[s-(hasN?2:1)] = next++;
      }
      SetVFloor(hset,vf,1.0);
   }
         
      
   badGC = TRUE;
   if (trace & (T_BID | T_SIZ)) {
      printf(" S%c: %d state's stream widths adjusted\n",
             (userWidths?'U':'S'),nedit);
      fflush(stdout);
   }
}

/* ---------------- SW - Set Stream Width Command ---------------- */

/* SetStreamWidthCommand: change width of stream s to n */
void SetStreamWidthCommand(void)
{
   int i, size, s, n, nedit=0;
   char c=' ';
   HMMScanState hss;
   HLink hmm=NULL;
   MixPDF *mp;
   Vector vf[SMAX];

   s = ChkedInt("Stream",1,SMAX-1);
   n = ChkedInt("Stream width",1,INT_MAX);
   if (trace & (T_BID | T_SIZ | T_MAC)) {
      printf("\nSW %d %d\n Changing stream width\n",s,n);
      fflush(stdout);
   }
   if (hset->swidth[s]==n) return;

   /* cz277 - ANN */
   if (hset->hsKind != HYBRIDHS) {
       NewHMMScan(hset,&hss);
       if (!(hss.isCont || (hss.hset->hsKind == TIEDHS))) {	
           HError(2640,"SetStreamWidthCommand: Can only resize continuous and tied systems");      
       }
       while(GoNextMix(&hss,FALSE)) {
           if (hss.s!=s) continue;
           mp = hss.me->mpdf;
           mp->mean=ResizeSVector(hset,mp->mean,n,'u',0.0);
           switch(mp->ckind) {
           case DIAGC:
               mp->cov.var=ResizeSVector(hset,mp->cov.var,n,'v',1.0);
               break;
           case FULLC:
               mp->cov.inv=ResizeSTriMat(hset,mp->cov.inv,n,'i',1.0);
               break;
           default: 
               HError(2640,"SetStreamWidthCommand: Can only resize DIAGC or FULLC");
           }
           if (trace & (T_SIZ | T_DET)) {
               if (nedit==0) printf(" For model.state["),hmm=NULL;
               if (hmm!=hss.hmm)
                   printf("]\n  %12s.state",hss.mac->id->name),c='[',hmm=hss.hmm;
               printf("%c%d",c,hss.i);c=',';
               fflush(stdout);
           }
           nedit++;
       }
       if (trace & (T_SIZ | T_DET))
           printf("]\n"),fflush(stdout);
       EndHMMScan(&hss);
       /* Now varFloor */
       SetVFloor(hset,vf,0.0);
       if (FindMacroStruct(hset,'v',vf[s])!=NULL) {
           if (trace & T_BID)
               printf(" Resizing varFloor\n");
           ResizeSVector(hset,vf[s],n,'v',0.0);
       }
       badGC = TRUE;
   }
   hset->swidth[s]=n;

   size=0;
   for (i = 1; i <= hset->swidth[0]; i++)
      size += hset->swidth[i];
   hset->vecSize = size;
   
   if (trace & (T_BID | T_SIZ)) {
      printf(" SW: Stream width changed for %d mixes\n",nedit);
      fflush(stdout);
   }
}

/* ---------------- SK - Set Sample Kind Command ---------------- */

/* SetSampKindCommand: change skind of all loaded models */
void SetSampKindCommand(void)
{
   char s[256];
   ParmKind pk;
   
   ChkedAlpha("SK sample kind",s);
   if (trace & T_BID) {
      printf("\nSK %s\n Set parameter kind of currently loaded HMM set\n",s);
      fflush(stdout);
   }
   pk = Str2ParmKind(s);
   hset->pkind=pk;
}

/* ---------------- HK - Set HMMSet Kind Command ---------------- */

static int nconv;

static int cmpMix(const void *v1,const void *v2)
{
   MixPDF *m1,*m2;
   MLink ml1,ml2;
   int i1,i2;
   
   m1=*(MixPDF**)v1;m2=*(MixPDF**)v2;
   ml1=FindMacroStruct(hset,'m',m1);
   ml2=FindMacroStruct(hset,'m',m2);
   if (ml1==NULL || ml2==NULL) return(m1-m2);
   i1=strlen(ml1->id->name);
   i2=strlen(ml2->id->name);
   if (i1==i2)
      return(strcmp(ml1->id->name,ml2->id->name));
   else
      return(i1-i2);
}
   
void CreateTMRecs(void)
{
   HMMScanState hss;
   TMixRec *p;
   MixPDF **mpp;
   TMProb *tm;
   LabId labid;
   char buf[80];
   int s,M;
   
   for (s=1;s<=hset->swidth[0];s++)
      hset->tmRecs[s].nMix=0;
   NewHMMScan(hset,&hss);
   while(GoNextMix(&hss,FALSE))
      hset->tmRecs[hss.s].nMix++;
   EndHMMScan(&hss);
   
   for (s=1;s<=hset->swidth[0];s++) {
      M = hset->tmRecs[s].nMix;
      p = hset->tmRecs+s;
      sprintf(buf,"%s_%d_",tiedMixName,s);
      labid=GetLabId(buf,TRUE);
      p->mixId = labid;
      mpp = (MixPDF **)New(hset->hmem,sizeof(MixPDF *) * M);
      p->mixes = mpp-1;
      tm = (TMProb *)New(hset->hmem,sizeof(TMProb)*M);
      p->probs = tm-1;
      p->topM = 1;
   }
   NewHMMScan(hset,&hss);
   while(GoNextMix(&hss,FALSE)) {
      p = hset->tmRecs+hss.s;
      p->mixes[p->topM]=hss.me->mpdf;
      p->topM++;
   }
   for (s=1;s<=hset->swidth[0];s++) {
      p = hset->tmRecs+s;
      qsort(p->mixes+1,p->topM-1,sizeof(MixPDF*),cmpMix);
   }
   EndHMMScan(&hss);
}

/* CalcTMWeights: Fix the weights of me using the old stream info */
Vector CalcTMWeights(int s, StreamElem *ste, double tFloor)
{
   DVector v;
   Vector tpdf;
   MixtureElem *sme;
   double w,p,wSum,fSum,maxW,floor,lFloor;
   int m,sm,M;
   
   M = hset->tmRecs[s].nMix;

   v = CreateDVector(&tmpHeap,M);
   tpdf = CreateVector(hset->hmem,M);

   maxW = LZERO;
   if (M==ste->nMix) {
      for (m=1;m<=M;m++) {  /* copy weights */
         for (sm=1,sme=ste->spdf.cpdf+1;sm<=ste->nMix;sm++,sme++)
            if (sme->mpdf==hset->tmRecs[s].mixes[m]) break;
         if (sm>ste->nMix)
            HError(2693,"CalcTMWeights: Cannot find matching mixture");
         if (sme->weight>MINMIX)
            w = v[m] = log(sme->weight);
         else
            w = v[m] = LZERO;
         if (w > maxW) maxW = w; 
      }
   }
   else {
      for (m=1;m<=M;m++) {  /* set weights to SOutP(owner) */
         w=LZERO;
         for (sm=1,sme=ste->spdf.cpdf+1;sm<=ste->nMix;sm++,sme++) {
            if (sme->weight>MINMIX) {
               p = MOutP(hset->tmRecs[s].mixes[m]->mean,sme->mpdf);
               w = LAdd(w,log(sme->weight)+p);
            }
         }
         if (w > maxW) maxW = w; 
         v[m] = w;
      }
   }
   wSum = 0.0; fSum = 0.0; 
   floor = tFloor*MINMIX; 
   if (floor>0.0)
      lFloor = log(floor*M);
   else
      lFloor = LZERO;
   for (m=1,sm=0;m<=M;m++) {  /* make all weights > floor */
      w = v[m]-maxW;
      if (w < lFloor) {
         v[m] = floor; fSum += floor;sm++;
      } else {
         w = exp(w); v[m] = w;
         wSum += w;
      }
   }
   if (fSum>=0.9)
      HError(2634,"CalcTMWeights: Join Floor too high");
   if (wSum==0.0)
      HError(2691,"CalcTMWeights: No positive weights");
   wSum /= (1-fSum);            /* finally normalise */
   fSum=0.0;
   for (m=1;m<=M;m++) {
      w = v[m];
      if (w>floor) w /= wSum;
      tpdf[m]=w;
      fSum+=w;
   }
   if (trace & T_DET) {
      printf(" Sum == %.3f [%d floored]\n",fSum,sm);
      fflush(stdout);
   }
   nconv++;
   FreeDVector(&tmpHeap,v);
   return(tpdf);
}


void ConvertCont2Tied(void)
{
   Vector tpdf;
   MLink q;
   StreamElem *ste;
   HLink hmm;
   LabId labid;
   char buf[80];
   int h,i,j,s,M;

   /* Create the TMRecs */
   CreateTMRecs();

   /* Rebuild every stream element */
   /* Can't use Scan because will break */
   for (h=0; h<MACHASHSIZE; h++)
      for (q=hset->mtab[h]; q!=NULL; q=q->next) 
         if (q->type=='h') {
            hmm=(HLink) q->structure;
            for (j=2;j<hmm->numStates;j++) {
               for (s=1;s<=hset->swidth[0];s++) {
                  ste=hmm->svec[j].info->pdf+s;
                  if (IsSeen(ste->nMix)) continue;
                  if (trace & T_DET)
                     printf("   For  %-12s[%d].stream[%d]:  ",
                            HMMPhysName(hset,hmm),j,s);
                  M = hset->tmRecs[s].nMix;
                  tpdf=CalcTMWeights(s,ste,joinFloor);
                  ste->spdf.tpdf=tpdf;
                  ste->nMix=M;
                  Touch(&ste->nMix);
               }
            }
         }
   for (h=0; h<MACHASHSIZE; h++)
      for (q=hset->mtab[h]; q!=NULL; q=q->next) 
         if (q->type=='h') {
            hmm=(HLink) q->structure;
            for (j=2;j<hmm->numStates;j++)
               for (s=1;s<=hset->swidth[0];s++) {
                  ste=hmm->svec[j].info->pdf+s;
                  Untouch(&ste->nMix);
               }
         }
                                  
   /* Purge any mix macros */
   for (h=0; h<MACHASHSIZE; h++)
      for (q=hset->mtab[h]; q!=NULL; q=q->next) 
         if (q->type=='m')
            DeleteMacro(hset,q);

   /* Give the tied mixture records names */
   for (s=1;s<=hset->swidth[0];s++) {
      M = hset->tmRecs[s].nMix;
      for (i=1;i<=M;i++) {
         sprintf(buf,"%s%d",hset->tmRecs[s].mixId->name,i);
         labid=GetLabId(buf,TRUE);
         q=NewMacro(hset,fidx,'m',labid,hset->tmRecs[s].mixes[i]);
         SetMacroUse(q,1);
      }
   }
}

/* SetHSetKindCommand: change the hset kind of the current HMM set  */
void SetHSetKindCommand(void)
{
   char s[256];
   HSetKind hk;
   
   ChkedAlpha("HK HMMSet kind",s);
   if (trace & T_BID) {
      printf("\nHK %s\n Set HMMSet kind of currently loaded HMM set\n",s);
      fflush(stdout);
   }
   if (strcmp(s,"PLAINHS")==0) hk=PLAINHS;
   else if (strcmp(s,"SHAREDHS")==0) hk=SHAREDHS;
   else if (strcmp(s,"TIEDHS")==0) hk=TIEDHS;
   else if (strcmp(s,"DISCRETEHS")==0) hk=DISCRETEHS;
   else {
      hk=PLAINHS;
      HError(2632,"SetHSetKindCommand: Unknown HMMSet kind %s",s);
   }

   if (hset->hsKind==hk)
      return;
   nconv=0;
   switch(hset->hsKind) {
   case PLAINHS:
      switch(hk) {
      case SHAREDHS:
         break;
      case TIEDHS:
         ConvertCont2Tied();
         break;
      case DISCRETEHS:
         HError(2640,"SetHSetKindCommand: conversion to %s not implemented",s);
         break;
      }
      break;
   case SHAREDHS:
      switch(hk) {
      case PLAINHS:
         /* Should untie all tied parameters */
         HError(2640,"SetHSetKindCommand: conversion to %s not implemented",s);
         break;
      case TIEDHS:
         ConvertCont2Tied();
         break;
      case DISCRETEHS:
         HError(2640,"SetHSetKindCommand: conversion to %s not implemented",s);
         break;
      }
      break;
   case TIEDHS:
      switch(hk) {
      case PLAINHS:
         /* Should copy each tm to each slot */
      case SHAREDHS:
         /* Should make macros from tm and copy to slots */
      case DISCRETEHS:
         /* Should make code-book and discretise system */
         HError(2640,"SetHSetKindCommand: conversion to %s not implemented",s);
      }
      break;
   case DISCRETEHS:
      switch(hk) {
      case PLAINHS:
         /* Should make macros from code book */
         /* Should copy each tm to each slot */
      case SHAREDHS:
         /* Should make macros from code book */
         /* Should make macros from tm and copy to slots */
      case TIEDHS:
         /* Should make tmrecs from code book */
         HError(2640,"SetHSetKindCommand: conversion to %s not implemented",s);
      }
      break;
   }

   hset->hsKind=hk;
   if (trace & (T_BID | T_SIZ)) {
      printf(" HK: Rebuilt %d streams\n",nconv);
      fflush(stdout);
   }
}

/* ------------------ RM - Remove Mean Command ------------------ */

/* RemMean: subtract src from tgt */
void RemMean(Vector src, Vector tgt)
{
   int i,n;

   n = VectorSize(src);
   i = VectorSize(tgt);
   if (n != i)
      HError(2630,"RemMean: vector sizes incompatible %d vs %d",n,i);
   for (i=1; i<=n; i++)
      tgt[i] -= src[i];
}

/* RemMeansCommand: remove mean given by state 1, mix 1 of given
   hmm def, if all then remove means from energy too */
void RemMeansCommand(void)
{
   HMMScanState hss;
   HMMSet tmpSet;
   HLink hmm;
   MLink ml;
   Vector sv[SMAX];
   int s,nedit=0;
   char buf[256];

   ChkedAlpha("RM hmm file name",buf);
   if (trace & T_BID) {
      printf("\nRM %s\n Subracting HMM Mean from HMMSet means\n",buf);
      fflush(stdout);
   }

   CreateHMMSet(&tmpSet,&hmmHeap,TRUE);
   if(MakeOneHMM(&tmpSet,buf)<SUCCESS)
      HError(2628,"RemMeansCommand: MakeOneHMM failed");
   if(LoadHMMSet(&tmpSet,NULL,NULL)<SUCCESS)
      HError(2628,"RemMeansCommand: LoadHMMSet failed");

   if (hset->hsKind==TIEDHS || hset->hsKind==DISCRETEHS)
      HError(2640,"RemMeansCommand: HMM is not continuous");
   if (hset->swidth[0] != tmpSet.swidth[0])
      HError(2630,"RemMeansCommand: num streams incompatible");
   if (hset->pkind != tmpSet.pkind)
      HError(2630,"RemMeansCommand: different parameter kinds");
   ml = FindMacroName(&tmpSet,'h',GetLabId(buf,FALSE));
   hmm = (HLink) ml->structure;
   for (s=1; s<=hset->swidth[0]; s++) {
      if (hset->swidth[s] != tmpSet.swidth[s])
         HError(2630,"RemMeansCommand: stream %d different size",s);
      sv[s] = hmm->svec[2].info->pdf[s].spdf.cpdf[1].mpdf->mean;
   }
   NewHMMScan(hset,&hss);
   while(GoNextMix(&hss,FALSE)) {
      RemMean(sv[hss.s],hss.me->mpdf->mean);
      nedit++;
   }
   EndHMMScan(&hss);
   if (trace & (T_BID | T_SIZ)) {
      printf(" RM: Mean subtracted from %d mixes\n",nedit);
      fflush(stdout);
   }
}

/* --------------- QS - Load Question Set Commands ------------ */

/* LoadQuestionCommand: question is an item list of model names */
void QuestionCommand(void)
{
   ILink ilist=NULL;
   char type='h';
   char qName[255];
   char *pattern;

   ChkedAlpha("QS question name",qName);
   /* get copy of original item list whilst parsing it */
   pattern=PItemList(&ilist,&type,hset,&source,trace&T_ITM);
   if (trace & T_QST) {
      printf("\nQS %s %s Define question\n",qName,pattern);
      fflush(stdout);
   }
   if (ilist==NULL)
      HError(-2631,"QuestionCommand: No items for question %s\n",qName);
   else {
      LoadQuestion(qName,ilist,pattern);
      if (trace & T_QST) {
         printf(" QS: Refers to %d of %d models\n",
                NumItems(ilist),hset->numLogHMM);
         fflush(stdout);
      }
   }
   
}

/* --------------- TB - Tree Based Clustering Command ------------ */

void TreeBuildCommand(void)
{
   ILink ilist = NULL;          /* list of items to tie */
   char type = ' ';             /* type of items to tie */
   char macName[255];           /* name of macro to use */
   float thresh = 0.0;
   
   thresh = ChkedFloat("Tree build threshold",0.0,FLOAT_MAX);
   ChkedAlpha("TB macro name",macName);
   if (trace & (T_BID | T_MAC | T_CLUSTERS | T_TREE)) {
      printf("\nTB %.2f %s {}\n Tree based clustering\n",thresh,macName);
      fflush(stdout);
   }
   
   if (treeList!=NULL && thisCommand!=lastCommand)
      HError(2640,"TreeBuildCommand: TB commands must be in sequence");
   if (!occStatsLoaded)
      HError(2655,"TreeBuildCommand: No stats loaded - use LS command");
   if (strlen(macName) > 20 )
      HError(-2639,"TreeBuildCommand: %s is rather long for a macro name",
             macName);
   PItemList(&ilist,&type,hset,&source,trace&T_ITM);
   if (ilist==NULL) {
      HError(-2631,"TreeBuildCommand: No items to cluster for %s\n",macName);
      return;
   }
   if (type!='h' && type!='s')
      HError(2640,"TreeBuildCommand: Type %c not implemented",type);

   /* Do Tree based clustering */
   BuildTree(ilist,thresh,macName);
}

/* ------------- AU - Add Unseen Triphones Command --------- */

void AddUnseenCommand(void)
{
   char newListFn[255];         /* name of hmmList containing unseen models */
   HMMSet newSet;
   int oldL = hset->numLogHMM;
   int oldP = hset->numPhyHMM;

   ChkedAlpha("AU hmmlist file name",newListFn);
   if (trace & (T_BID | T_TREE_ANS)) {
      printf("\nAU %s\n Creating HMMset using trees to add unseen triphones\n",
             newListFn);
      fflush(stdout);
   }
   if (treeList == NULL)
      HError(2662,"AddUnseenCommand: there are no existing trees");
   CreateHMMSet(&newSet,&hmmHeap,TRUE);
   if(MakeHMMSet(&newSet,newListFn)<SUCCESS)
      HError(2628,"AddUnseenCommand: MakeHMMSet failed");

   TreeFilter(&newSet);
   SwapLists(hset,&newSet);
   if (trace & T_BID) {
      printf(" AU: %d Log/%d Phys created from %d Log/%d Phys\n",
             hset->numLogHMM,hset->numPhyHMM,oldL,oldP);
      fflush(stdout);
   }
}

/* ------------- UF - Use File Command --------- */

void UseCommand(void)
{
   MILink mil;
   char newMMF[256];   /* Name of MMF file to store macros in */

   ChkedAlpha("UF file name",newMMF);
   if (trace & T_BID) {
      printf("\nUF %s\n Using MMF file to store new macros",newMMF);
      fflush(stdout);
   }
   mil=AddMMF(hset,newMMF);
   mil->isLoaded=TRUE; /* Just to make sure we actually save to file */
   fidx=mil->fidx;
}

/* -------------------------- FA Command ------------------------- */
void FloorAverageCommand(void)
{
   HMMScanState hss;
   StateInfo *si   ;
   StreamElem *ste;
   SVector var , mean;
   DVector *varAcc;
   double occAcc;
   float weight;
   float occ; 
   int l=0,k,i,s,S;
   float varScale;

   /*  need stats for operation */
   if (!occStatsLoaded)
      HError(1,"FloorAverageCommand: stats must be loaded before calling FA");
   varScale = ChkedFloat("Variance Scale Value",0.0,FLOAT_MAX);
   if (hset->hsKind==TIEDHS || hset->hsKind==DISCRETEHS)
      HError(2640,"FloorAverageCommand: Only possible for continuous models");
   if (hset->ckind != DIAGC)
      HError(2640,"FloorAverageCommand: Only implemented for DIAGC models");
   S= hset->swidth[0];

   /* allocate accumulators */
   varAcc = (DVector*)New(&gstack,(S+1)*sizeof(DVector));
   for (s=1;s<=S;s++) {
      varAcc[s] = CreateDVector(&gstack,hset->swidth[s]);
      ZeroDVector(varAcc[s]);
   }
   NewHMMScan(hset,&hss);
   occAcc = 0.0;
   while(GoNextState(&hss,FALSE)) {
      si = hss.si;
      memcpy(&occ,&(si->hook),sizeof(float));
      occAcc += occ;
      s=0;
      while (GoNextStream(&hss,TRUE)) {
         l=hset->swidth[hss.s];
         ste = hss.ste;
         s++;
         if ( hss.M > 1 ) { 
            for (k=1;k<=l;k++) { /* loop over k-th dimension */
               double    rvar  = 0.0;
               double    rmean = 0.0;
               for (i=1; i<= hss.M; i++) {
                  weight = ste->spdf.cpdf[i].weight;
                  var  = ste->spdf.cpdf[i].mpdf->cov.var;
                  mean = ste->spdf.cpdf[i].mpdf->mean;
                  
                  rvar  += weight * ( var[k] + mean[k] * mean[k] );
                  rmean += weight * mean[k];
               }
               rvar -= rmean * rmean;
               varAcc[s][k] += rvar * occ ;
            }
         }
         else { /* single mix */
            var  = ste->spdf.cpdf[1].mpdf->cov.var;
            for (k=1;k<=l;k++) 
               varAcc[s][k] += var[k]*occ;
         }
      }
   }
   EndHMMScan(&hss);
   /* normalisation */
   for (s=1;s<=S;s++)
      for (k=1;k<=hset->swidth[s];k++)
         varAcc[s][k] /= occAcc;
   /* set the varFloorN macros */
   for (s=1; s<=S; s++){
      int size;
      LabId id;
      SVector v;
      char mac[MAXSTRLEN];
      MLink m;
      sprintf(mac,"varFloor%d",s);
      id = GetLabId(mac,FALSE);
      if (id != NULL  && (m=FindMacroName(hset,'v',id)) != NULL){
         v = (SVector)m->structure;
         SetUse(v,1);
         /* check vector sizes */
         size = VectorSize(v);
         if (size != hset->swidth[s])
            HError(7023,"FloorAverageCommand: Macro %s has vector size %d, should be %d",
                   mac,size,hset->swidth[s]);
      }
      else { /* create new macro */
         id = GetLabId(mac,TRUE);
         v = CreateSVector(hset->hmem,hset->swidth[s]);
         NewMacro(hset,hset->numFiles,'v',id,v);
      }
      /* scale and store */
      for (k=1;k<=l;k++)
         v[k] = varAcc[s][k]*varScale;
   }
   /* and apply floors */
   if (applyVFloor)
       ApplyVFloor(hset);
   else
       HError(-7023,"FloorAverageCommand: variance floors have not been applied to mixes");
   Dispose(&gstack,varAcc);
}

/* -------------------------- FV Command ------------------------- */
void FloorVectorCommand(void)
{
   /* read macro file as generated by HCompV 
       and copy floor vectors to model set */
   char fn[MAXFNAMELEN],mac[MAXSTRLEN],buf[MAXSTRLEN];
   Source src;
   int s,S,n,i;
   char c,h;
   LabId id;
   SVector v;
   MLink m;

   ChkedAlpha("FV file containing variance floor value(s)",fn);
   if (trace & T_BID) {
      printf("\nFV: loading variance floor(s) from file %s\n",fn);
      fflush(stdout);
   }
   /* check model set */
   if (hset->hsKind==TIEDHS || hset->hsKind==DISCRETEHS)
      HError(2640,"FloorVectorCommand: Only possible for continuous models");
   S=hset->swidth[0];

   /* load variance floor macro */
   if(InitSource(fn, &src, NoFilter)<SUCCESS)
      HError(2610,"FloorVectorCommand: Can't open file");
   do {
      while ( (c=GetCh(&src)) != '~' && c!= EOF);
      c = tolower(GetCh(&src));
      if (c == 'v' ) { 
         /* found variance vector */
         if (!ReadString(&src,buf))
            HError(2613,"FloorVectorCommand: cannot parse file %s",fn);
         strcpy(mac,"varFloor");
         h=buf[strlen(mac)];
         buf[strlen(mac)]='\0';
         if (strcmp(mac,buf)!=0) /* not a varFloor */
            continue;
         /*assert(SMAX<10);*/ s=h-'0';
         if(s<1||s>S)
            HError(2613,"FloorVectorCommand: undefined stream %d in HMM set",s);
         mac[strlen(buf)]=h;
         mac[strlen(buf)+1]='\0';
         if (trace & T_BID) {
            printf("loading vector %s\n",mac);
         }
          
         /* obtain vector */
         id = GetLabId(mac,FALSE);
         if (id != NULL  && (m=FindMacroName(hset,'v',id)) != NULL){
            v = (SVector)m->structure;
            SetUse(v,1);
         }
         else { /* create new macro */
            id = GetLabId(mac,TRUE);
            v = CreateSVector(hset->hmem,hset->swidth[s]);
            NewMacro(hset,hset->numFiles,'v',id,v);
         }
         /* load vector */
         if (!ReadString(&src,buf))
            HError(2613,"FloorVectorCommand: cannot parse file %s",fn);
         for (n=strlen(buf),i=0;i<n;i++)
            buf[i]=tolower(buf[i]);
         if (strcmp("<variance>",buf)!=0)
            HError(2613,"FloorVectorCommand: variance vector expected",fn);
         if (!ReadInt(&src,&n,1,FALSE))
            HError(2613,"FloorVectorCommand: cannot parse file %s",fn);
         if (n!=hset->swidth[s])
            HError(2613,"FloorVectorCommand: stream width mismatch (stream %d)",s);
         if (!ReadFloat(&src,v+1,n,FALSE))
            HError(2613,"FloorVectorCommand: cannot load variance vector");
      }
   } while(c!=EOF);
   CloseSource(&src);

   /* and apply floors */
   if (applyVFloor)
       ApplyVFloor(hset);
   else
       HError(-7023,"FloorVectorCommand: variance floors have not been applied to mixes");
}

#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))
#define ABS(a) ((a)>0?(a):-(a))

/* Sets size of state... */
int SetSize(char *hname, StreamElem *ste /*nMix must be +ve*/, int tgt){ /*returns nDefunct*/
   int nDefunct=0,sign = (ste->nMix > 0 ? 1 : -1); 
   int m, M;
   ste->nMix *= sign; M=ste->nMix;

   /* (1) remove defunct mixture compoonents */
   for (m=1; m<=M; m++)
      if (ste->spdf.cpdf[m].weight <= MINMIX) {
         MixPDF *mp=ste->spdf.cpdf[m].mpdf;
         if (mp->nUse) mp->nUse--; /* decrement MixPDF use */
         ste->spdf.cpdf[m] = ste->spdf.cpdf[M]; M--; 
         nDefunct++;
      }
   /* (2) get to desired size.. */
   ste->nMix = M;
   if(tgt < M) DownMix(hname,ste,tgt,TRUE);
   else if(tgt > M) UpMix(hname, ste, M, tgt);
   ste->nMix *= sign;
   return nDefunct;
}


/* ------------- PS Command --------------- */

void PowerSizeCommand(void)
{
   int ch;
   int nDefunct=0,m=0,M=0,maxM=0,newM=0,newMIdeal=0,nTooBig=0;  int S=0;
   float minweight = 1;
   float sum=0; float factor;

   HMMScanState hss;
   float AvgNumMix = ChkedFloat("AvgNumMix",1.1, 1000);
   float Power = ChkedFloat("Power",0.1, 10);
   float NumItsRemaining = 1.0;   /* A float but will normally be an int.
                                     Set this to e.g. 4,3,2,1 in succession
                                     for 4 its of up-mixing. */
   
   do { 
      ch = GetCh(&source);
      if (ch=='\n') break;
   } while(ch!=EOF && isspace(ch));
   UnGetCh(ch,&source);
   if (ch!='\n') { /* Get nIts remaining... */
      NumItsRemaining = ChkedFloat("NumItsRemaining",1, 100);
   }


   if(!occStatsLoaded)
      HError(2672, "ControlSizeCommand: Use LoadStats (LS <whatever-dir/stats>) before doing this.");

   printf("Running PowerSize command, tgt #mix/state = %f, power of #frames=%f, NumItsRemaining=%f\n",
          AvgNumMix,Power,NumItsRemaining); fflush(stdout);

   NewHMMScan(hset,&hss);
   while(GoNextStream(&hss,FALSE)){
      float stateocc;
      memcpy(&stateocc,&(hss.si->hook),sizeof(float));
      sum += exp(log(MAX(stateocc,MINLARG)) * Power);
      S++; M+=hss.M;  maxM=MAX(maxM,hss.M); 
      for (m=1;m<=hss.M;m++){ minweight = MIN(hss.ste->spdf.cpdf[m].weight, minweight); }
   }  /*count states. */
   EndHMMScan(&hss);

   if(minweight > 0){
      printf("Min weight in this HMM set is %f, perhaps because weights were floored\n"
             "If so, use HERest with -w 0 or no -w option to get unfloored weights.\n", minweight);
   }
   factor = AvgNumMix / (sum/S);  

   NewHMMScan(hset,&hss);
   while(GoNextStream(&hss,FALSE)){
      float stateocc; float tgt; int itgt;
      memcpy(&stateocc,&(hss.si->hook),sizeof(float));
      tgt = factor * exp(log(MAX(stateocc,MINLARG)) * Power);
      itgt = (int)(tgt+0.5); 
      if(itgt<1) itgt=1;

      newMIdeal += itgt;
      if(itgt > ABS(hss.M) && NumItsRemaining!=1.0){
         /* If NumItsRemaining>0, then don't mix up all at once. */
         int nummix = ABS(hss.M);
         int newitgt = (int) (0.5 +  nummix * exp( 1.0/NumItsRemaining * log(itgt/(float)nummix)));
         if(newitgt <= nummix)   newitgt = nummix+1;
         itgt = newitgt;
      }
      newM += itgt;
      if(itgt > hss.M*2) nTooBig++;
      nDefunct += SetSize(HMMPhysName(hset,hss.hmm), hss.ste, itgt);
   }
   EndHMMScan(&hss);
   
   if(nTooBig>0) HError(-2631/*?*/, "%d(/%d) mixtures are more than doubled in size; you may want to mix up \n"
                        "in multiple iterations as: PS #mix power n, where n = nIts remaining (e.g. 3,2,1)", nTooBig, S);
   printf("Initial %d mixes, final %d mixes, avg %f/state, nDefunct=%d\n",M,newM,newM/(float)S,
          nDefunct);
   if(NumItsRemaining!=1.0)
      printf("Aiming for final nMixes = %d, nIts remaining = %f\n",newMIdeal, (NumItsRemaining-1.0));  
}


/* -------------------------- MD Command ------------------------- */

void MixDownCommand(void)
{
   ILink i,ilist = NULL;		/* list of items to mixdown */
   char type = 'p';		        /* type of items must be p */
   int trg;
   int m,M;
   int totm=0,totM=0;
   StreamElem *ste;
   HMMDef *hmm;
   char *hname;
   int nDefunct,nDefunctMix;
    
   trg = ChkedInt("MD: Mix Target",1,INT_MAX);
   if (trace & T_BID){
      printf("Executing Mix Down Command to %d\n",trg);
      fflush(stdout);
   }
   /* check compatibility with model set */
   if (hset->hsKind==TIEDHS || hset->hsKind==DISCRETEHS)
      HError(2640,"MixDownCommand: MixDown only possible for continuous models");

   /* get itemlist */
   PItemList(&ilist,&type,hset,&source,trace&T_ITM);
   if (ilist == NULL) {
      HError(-2631,"MixDownCommand: No mixtures to decrease!");
      return;
   }
   /* touch all mixture models to avoid duplicates */
   nDefunct = nDefunctMix= 0; 
   for (i=ilist; i!=NULL; i=i->next) {
      ste = (StreamElem *)i->item;
      M = ste->nMix;
      if (M>0) ste->nMix=-M;
   }

   /* mix down all pdf's */
   nDefunct = nDefunctMix= 0; 
   for (i=ilist; i!=NULL; i=i->next){
      hmm = i->owner;
      ste = (StreamElem *)i->item; 
      M = ste->nMix;
      if (M<0) { 
         /* mix not yet processed */
         M = ste->nMix = -M;
         hname = HMMPhysName(hset,hmm);
         /* remove defunct mixture compoonents */
         for (m=1; m<=M; m++)
            if (ste->spdf.cpdf[m].weight <= MINMIX) {
               MixPDF *mp=ste->spdf.cpdf[m].mpdf;
               if (mp->nUse) mp->nUse--; /* decrement MixPDF use */
               ste->spdf.cpdf[m] = ste->spdf.cpdf[M]; M--; 
               nDefunct++;
            }
         if(ste->nMix>M) {
            if ( trace & T_DET) 
               printf("MD %12s : %d out of %d mixes defunct\n",hname,ste->nMix-M,ste->nMix);
            nDefunctMix++;
            ste->nMix = M;
         }
         /* downmix if necessary */
         if (trg<M) {
            DownMix(hname,ste,trg,TRUE);
            totm+=trg;
         }
         else 
            totm+=M;
         totM+=M;
      }
   }
   FreeItems(&ilist);
   if (trace & T_BID) {
      if (nDefunct>0)
         printf("MD: deleted %d defunct mix comps in %d pdfs \n",nDefunct,nDefunctMix);
      else
         printf("MD: no defunct mixcomps found \n");
      printf("MD: Number of mixes decreased from %d to %d\n",totM,totm);
      fflush(stdout);
   }
}

/* ------------- FC - FullCovar Command ----------------- */

void FullCovarCommand(void)
{
   HMMScanState hss;
   int i,u,z;
   SVector var;
   STriMat invcov;
   MLink ml;

   if (hset->hsKind==TIEDHS || hset->hsKind==DISCRETEHS)
      HError(2640,"FullCovarCommand: Only possible for continuous models");
   if (hset->ckind != DIAGC)
      HError(2640,"FullCovarCommand: Only implemented for DIAGC models");

   ConvDiagC(hset,TRUE);

   NewHMMScan(hset,&hss);
   while(GoNextMix(&hss,FALSE)) {
      var = hss.me->mpdf->cov.var;
      hss.me->mpdf->ckind = FULLC;
      u = GetUse(var);
      if (u==0 || !IsSeen(u)) {
         z = VectorSize(var);
         invcov = CreateSTriMat(&hmmHeap,z);
         ZeroTriMat(invcov);
         for (i=1; i<=z; i++) 
            invcov[i][i] = var[i];
         if (u!=0) {
            ml = FindMacroStruct(hset,'v',var);
            DeleteMacro(hset,ml);                  /* this needs to delete */
            NewMacro(hset,fidx,'i',ml->id,invcov); /* the macro name also */
            TouchV(var);
            SetUse(invcov,u);
            SetHook(var,invcov);
         }
      }
      else  /* a macro that we have already seen */
         invcov = GetHook(var);
      hss.me->mpdf->cov.inv = invcov;
   }
   EndHMMScan(&hss);
   hset->ckind = FULLC;
}

/* ------------- PR - ProjectCommand   ----------------- */

void ProjectCommand(void)
{
   AdaptXForm *xform;

   xform = hset->semiTied;
   if ((xform == NULL) || (hset->projSize == 0))
      HError(2691,"Can not only project with semitied XForm and PROJSIZE>0");

   /* check that this is a reasonable command to run */
   if (xform->bclass->numClasses != 1)
      HError(2691,"Can only store as input XForm with global transform");
   if (hset->swidth[0] != 1) 
      HError(2691,"Can only store as input XForm with single stream");
   if (hset->xf != NULL)
      HError(2691,"Can not store as input XForm if HMMSet already has an input XForm");

   UpdateProjectModels(hset,newDir);
}

/* ----------------------- IL/PL/EL - Add/Pop/Erase an ANN Layer --------------------- */


static NVecBundle *AddOneNVecBundle(char *invoker, char *kwd, HMMSet *curset, Boolean existmacro) {
    LabId id;
    MLink m;
    char buf[MAXSTRLEN];
    int intVal;
    NVecBundle *bundle;

    if (kwd == NULL)
        ChkedAlpha("Next keyword", buf);  
    else
        strcpy(buf, kwd);
    if (strcmp(buf, "~V") == 0) {
        ChkedAlpha("NVecBundle macro name expected", buf);
        id = GetLabId(buf, FALSE);
        if (id != NULL && (m = FindMacroName(curset, 'V', id)) != NULL) {
            if (existmacro == TRUE)
                return (NVecBundle *) m->structure;
            else
                HError(2676, "%s: Macro named ~V \"%s\" already exists", invoker, id->name);
        }
        if (id == NULL)
            id = GetLabId(buf, TRUE);
        ChkedAlpha("<VECTOR> key word", buf);
    }
    else {	/* anonymous macro */
        id = GetNextANNMacroName(invoker, curset, 'V');
        if (trace & T_BAS)
            HError(-2675, "%s: Anonymous NVecBundle macro created, auto-name %s assigned", invoker, id->name);
    }
    /* read the rest things as usual */
    if (strcmp("<VECTOR>", buf) != 0) 
        HError(2650, "%s: <VECTOR> keyword expected, but %s was read in", invoker, buf);
    intVal = ChkedInt("Vector size expected", 1, INT_MAX);
    bundle = FetchNVecBundle(curset, id->name);
    bundle->kind = SIBK;
    bundle->variables = CreateNVector(curset->hmem, intVal);
    ClearNVector(bundle->variables);
    /* register the new macro */
    NewMacro(curset, fidx, 'V', id, bundle);
    /* saves the model */
    saveHMMSet = TRUE;

    return bundle;
}   

void SetOneNVecBundle(HMMSet *hset) {
   NVecBundle *bundle;
   int intVal, i, j;
   float floatVal=FLOAT_MAX;
   char buf[MAXSTRLEN];

   bundle = AddOneNVecBundle("SetOneNVecBundle", NULL, hset, TRUE);
   ChkedAlpha("Next keyword", buf);
   if (strcmp("<VALUES>", buf) != 0) {
       intVal = ChkedInt("Number of values", 1, bundle->variables->vecLen);
       for (i = 0, j = 0; i < bundle->variables->vecLen; ++i) {
           if (j < intVal)
               floatVal = ChkedFloat("initial value", -FLOAT_MAX, FLOAT_MAX);
           bundle->variables->vecElems[i] = floatVal;
#ifdef CUDA
           SyncNVectorHost2Dev(bundle->variables);
#endif
       }
   }
   /* saves the model */
   saveHMMSet = TRUE;
}

static void ChangeDimOneNVecBundle(char *invoker, NVecBundle *bundle, int nlen) {
    NVector *vector;

    if (nlen != bundle->variables->vecLen) {
        if (bundle->nUse > 1)
            HError(2640, "%s: Changing dimension of shared NVecBundle is not allowed by v3.5");
        vector = bundle->variables;
        bundle->variables = CreateNVector(hset->hmem, nlen);
        ClearNVector(bundle->variables);
        CopyPartialNVector2NVector(vector, bundle->variables);
        /*if (trace & T_INT)
            InformByNVecBundleTrace(invoker, hset, bundle);*/
        /* saves the model */
        saveHMMSet = TRUE;
    }
}

/*void ChangeDimOneMacroNVecBundle(HMMSet *hset) {
    MLink m;
    NVecBundle *bundle;
    int nlen;

    nlen = ChkedInt("New vector length", 1, INT_MAX);
    bundle = AddOneNVecBundle("ChangeDimOneNVecBundle", NULL, hset, TRUE);
    ChangeDimOneNVecBundle("ChangeDimOneMacroNVecBundle", bundle, nlen);
}*/

/*static void InformByNMatBundleTrace(char *invoker, HMMSet *hset, NMatBundle *bundle) {
    MLink m;
    BTLink trace;
    LELink layerElem;

    trace = (BTLink) bundle->hook;
    while (trace != NULL) {
        m = FindMacroStruct(hset, 'L', trace->layerElem);
        if (m == NULL)
            HError(9999, "%s: Fail to find the macro associated linked to the bundle");
        layerElem = (LELink) layerElem->structure;
        if (layerElem->nodeNum != bundle->variables->rowNum || layerElem->inputDim != bundle->variables->colNum)
            printf("%s: ~L \"%s\" probably need to be modified as well\n", m->id->name);
        trace = trace->nextTrace;
    }
}*/

static NMatBundle *AddOneNMatBundle(char *invoker, char *kwd, HMMSet *curset, Boolean existmacro) {
    LabId id;
    MLink m = NULL;
    char buf[MAXSTRLEN];
    int intVal[2];
    NMatBundle *bundle;

    if (kwd == NULL)
        ChkedAlpha("Next keyword", buf);
    else
        strcpy(buf, kwd);
    if (strcmp(buf, "~M") == 0) {
        ChkedAlpha("NMatBundle macro name expected", buf);
        id = GetLabId(buf, FALSE);
        if (id != NULL && (m = FindMacroName(curset, 'M', id)) != NULL) {
            if (existmacro == TRUE)
                return (NMatBundle *) m->structure;
            else
                HError(2676, "%s: Macro named ~M \"%s\" already exists", invoker, id->name);
        }
        if (id == NULL)
            id = GetLabId(buf, TRUE);
        ChkedAlpha("<MATRIX> key word", buf);
    }
    else {      /* anonymous macro */
        id = GetNextANNMacroName(invoker, curset, 'M');
        if (trace & T_BAS)
            HError(-2675, "%s: Anonymous NMatBundle macro created, auto-name %s assigned", invoker, id->name);
    }
    /* read the rest things as usual */
    if (strcmp("<MATRIX>", buf) != 0)
        HError(2650, "%s: <MATRIX> keyword expected, but %s was read in", invoker, buf);
    intVal[0] = ChkedInt("Matrix row number expected", 1, INT_MAX);
    intVal[1] = ChkedInt("Matrix column number expected", 1, INT_MAX);
    bundle = FetchNMatBundle(curset, id->name);
    bundle->kind = SIBK;
    bundle->variables = CreateNMatrix(curset->hmem, intVal[0], intVal[1]);
    ClearNMatrix(bundle->variables, intVal[0]);
    /* register the new macro */
    NewMacro(curset, fidx, 'M', id, bundle);
    /* saves the model */
    saveHMMSet = TRUE;

    return bundle;
}

void SetOneNMatBundle(HMMSet *hset) {
   NMatBundle *bundle;
   int intVal, i, j, n;
   float floatVal=FLOAT_MAX;
   char buf[MAXSTRLEN];

   bundle = AddOneNMatBundle("SetOneNMatBundle", NULL, hset, TRUE);
   ChkedAlpha("Next keyword", buf);
   if (strcmp("<VALUES>", buf) != 0) {
       n = bundle->variables->rowNum * bundle->variables->colNum;
       intVal = ChkedInt("Number of values", 1, n);
       for (i = 0, j = 0; i < n; ++i) {
           if (j < intVal)
               floatVal = ChkedFloat("initial value", -FLOAT_MAX, FLOAT_MAX);
           bundle->variables->matElems[i] = floatVal;
#ifdef CUDA
           SyncNMatrixHost2Dev(bundle->variables);
#endif
       }
   }
   /* saves the model */
   saveHMMSet = TRUE;
}

static void ChangeDimOneNMatBundle(char *invoker, NMatBundle *bundle, int nrows, int ncols) {
    NMatrix *matrix;

    if (nrows != bundle->variables->rowNum || ncols != bundle->variables->colNum) {
        if (bundle->nUse > 1)
            HError(2640, "%s: Changing dimensions of shared NMatBundle is not allowed by v3.5", invoker);
        matrix = bundle->variables;
        bundle->variables = CreateNMatrix(hset->hmem, nrows, ncols);
        ClearNMatrix(bundle->variables, nrows);
        CopyPartialNMatrix2NMatrix(matrix, bundle->variables);
        /*if (trace & T_INT)
            InformByNMatBundleTrace(invoker, hset, bundle);*/
        /* saves the model */
        saveHMMSet = TRUE;
    }
}

/*void ChangeDimOneNMatBundle(HMMSet *hset) {
    MLink m;
    NMatBundle *bundle;
    int nrows, ncols;
    NMatrix matrix;

    nrows = ChkedInt("New matrix row number", 0, INT_MAX);
    ncols = ChkedInt("New matrix column number", 0, INT_MAX);
    bundle = AddOneNMatBundle("ChangeDimOneNMatBundle", NULL, hset, TRUE);
    if (nrows == 0)
        nrows = bundle->variables->rowNum;
    if (ncols == 0)
        ncols = bundle->variables->colNum;
    ChangeDimOneNMatBundle("ChangeDimOneNMatBundle", bundle, nrows, ncols);
}*/


/* cz277 - 1007 */
static ADLink GetANNDefByMacName(char *invoker, char *kwd, HMMSet *curset) {
    LabId id;
    MLink m;
    ADLink annDef=NULL;
    char buf[MAXSTRLEN];

    if (kwd == NULL) 
        ChkedAlpha("Next keyword", buf);
    else 
        strcpy(buf, kwd);
    if (strcmp(buf, "~N") == 0) 
        ChkedAlpha("Target ANN macro name", buf);
    id = GetLabId(buf, FALSE);
    if (id != NULL && (m = FindMacroName(curset, 'N', id)) != NULL) 
        annDef = (ADLink) m->structure;
    else 
        HError(2635, "%s: Cannot find target ANN model", invoker);

    return annDef;
}

/* owner and pos could be NULL */
static LELink GetLayerElemByMacName(char *invoker, char *kwd, HMMSet *curset, ADLink *owner, int *pos) {
    LELink layerElem=NULL;
    int i=-1;
    ADLink annDef=NULL;
    LabId id;
    MLink m;
    char buf[MAXSTRLEN];

    if (kwd == NULL) 
        ChkedAlpha("Target layer macro type", buf);
    else 
        strcpy(buf, kwd);
    if (strcmp(buf, "~L") == 0) {
        ChkedAlpha("ANN layer macro name", buf);
        id = GetLabId(buf, FALSE);
        if (id != NULL && (m = FindMacroName(curset, 'L', id)) != NULL) 
            layerElem = (LELink) m->structure;
        else 
            HError(2635, "%s: Cannot find the target ANN layer", invoker);
        annDef = layerElem->ownerHead->annDef;	/* at the moment, not layer tying  */
        for (i = 0; i < annDef->layerNum; ++i) 
            if (annDef->layerList[i] == layerElem)
                break;
    }
    else 
        HError(2632, "%s: Unsupported target ANN layer macro type", invoker);
    if (owner != NULL){
        if(annDef==NULL) HError(2632,"%s: ANN definition not set",invoker);
        *owner = annDef;
    }
    if (pos != NULL){
        if(i<0) HError(2632,"%s: position not set",invoker);
        *pos = i;
    }
    if(!layerElem) HError(2632,"%s: layer element not set",invoker);

    return layerElem;
}

/* cz277 - 1007 */
static FELink ParseNextFeaElem(char *invoker, char *kwd, int *feaIdx, int *feaDim) {
    char buf[MAXSTRLEN];
    FELink feaElem;
    int i, streamIdx=-1, intVal;

    /* <FEATURE> */
    if (kwd == NULL) 
        ChkedAlpha("Next keyword", buf);
    else 
        strcpy(buf, kwd);
    if (strcmp(buf, "<FEATURE>") != 0) 
        HError(2650, "%s: <FEATURE> keyword expected", invoker);
    feaElem = (FELink) New(hset->hmem, sizeof(FeaElem)); 
    feaElem->nUse = 0;
    feaElem->dimOff = 0;
    intVal = ChkedInt("Feature element index", -1, INT_MAX);
    if (feaIdx != NULL)
        *feaIdx = intVal;
    feaElem->feaDim = ChkedInt("Feature element dimension", 1, INT_MAX);
    if (feaDim != NULL)
        *feaDim = feaElem->feaDim; 
    /* <SOURCE> */
    ChkedAlpha("Next keyword", buf);
    if (strcmp(buf, "<SOURCE>") == 0) {
        ChkedAlpha("Feature element source type", buf);
        if (strcmp(buf, "<AUGFEATURE>") == 0) {
            feaElem->inputKind = AUGFEAIK;
            feaElem->feaSrc = NULL;
            feaElem->augFeaIdx = ChkedInt("Augmented feature source index", 1, MAXAUGFEAS);
            streamIdx = 1;
        }
        else if (strcmp(buf, "~L") == 0) {
            feaElem->inputKind = ANNFEAIK;
            feaElem->feaSrc = GetLayerElemByMacName(invoker, buf, hset, NULL, NULL);
            ++feaElem->feaSrc->nDrv;
        }
        else if (strcmp(buf, "<STREAM>") == 0) {
            feaElem->inputKind = INPFEAIK;
            feaElem->feaSrc = NULL;
            streamIdx = ChkedInt("Feature stream index", 1, hset->swidth[0]);
        }
        else {  /* PARMKIND */
            if (hset->pkind != Str2ParmKind(buf)) 
                HError(2632, "%s: Input parmkind different from the input parmKind to the model", invoker);
            feaElem->inputKind = INPFEAIK;
            feaElem->feaSrc = NULL;
            streamIdx = 1;
        }
        ChkedAlpha("Next keyword", buf);
    }
    else 
        HError(2650, "%s: <SOURCE> keyword expected", invoker);
    if (strcmp(buf, "<DIMRANGE>") == 0) { 
        feaElem->dimOff = ChkedInt("Start dimension", 1, INT_MAX) - 1;
        intVal = ChkedInt("End dimension", 2, INT_MAX);
        if (feaElem->feaDim != intVal - feaElem->dimOff) 
            HError(2677, "%s: Inconsistent feature element dimension range", invoker);
        ChkedAlpha("Next keyword", buf);
    }
    if (strcmp(buf, "<CONTEXTSHIFT>") == 0) {
        intVal = ChkedInt("Context feature expansion vector size", 1, INT_MAX);
        feaElem->ctxMap = CreateIntVec(hset->hmem, intVal);
        for (i = 1; i <= intVal; ++i) 
           feaElem->ctxMap[i] = ChkedInt("A context shift", -INT_MAX, INT_MAX);
    }
    else 
        HError(2650, "%s: <CONTEXTSHIFT> keyword expected", invoker);
    /* set extDim and srcDim fields*/
    feaElem->extDim = feaElem->feaDim * IntVecSize(feaElem->ctxMap);
    if (feaElem->inputKind == ANNFEAIK) 
        feaElem->srcDim = feaElem->feaSrc->nodeNum;
    else 
        feaElem->srcDim = feaElem->extDim;
    /* add this feature element to hset->nInp, if needed */
    if (feaElem->inputKind == INPFEAIK || feaElem->inputKind == AUGFEAIK) {
        if(streamIdx<0) HError(2678,"%s: stream index not set",invoker);
        for (i = 0; i < hset->nInp[streamIdx]; ++i) 
            if (CmpFeaElem(hset->inpElem[streamIdx][i], feaElem))
                break;
        if (i == hset->nInp[streamIdx]) {
            if (i >= MAXINPUSE) 
                HError(2678, "%s: Maximum number of input feature usage reached, increase MAXINPUSE", invoker);
            hset->inpElem[streamIdx][hset->nInp[streamIdx]++] = feaElem;
        }
    }

    return feaElem;
}

/* cz277 - 1007 */
FeaMix *AddOneFeaMix(char *invoker, char *kwd, Boolean existmacro) {
    char buf[MAXSTRLEN];
    LabId id;
    FeaMix *feaMix;
    int i, feaIdx, feaDim;
    MLink m = NULL;

    if (kwd == NULL) 
        ChkedAlpha("Next keyword", buf);
    else 
        strcpy(buf, kwd);
    if (strcmp(buf, "~F") == 0) {
        ChkedAlpha("Macro name", buf);
        id = GetLabId(buf, FALSE);
        if (id != NULL && (m = FindMacroName(hset, 'F', id)) != NULL) {
            if (existmacro == TRUE) 
                return (FeaMix *) m->structure;
            else
                HError(2676, "%s: Feature mixture macro name %s occupied", invoker, id->name);
        }
        if (id == NULL)
            id = GetLabId(buf, TRUE);
        ChkedAlpha("Next keyword", buf);
    }
    else {
        id = GetNextANNMacroName(invoker, hset, 'F');
        if (trace & T_BAS)
            HError(-2675, "%s: Anonymous feature mixture macro created, auto-name %s assigned\n", invoker, id->name);
    }
    feaMix = (FeaMix *) New(hset->hmem, sizeof(FeaMix));
    if (strcmp(buf, "<NUMFEATURES>") == 0) {
        feaMix->elemNum = ChkedInt("The elements in this feature mixture", 1, INT_MAX);
        feaMix->mixDim = ChkedInt("The dimension of this feature mixture", 1, INT_MAX);
        feaMix->feaList = (FELink *) New(hset->hmem, sizeof(FELink) * feaMix->elemNum);
        for (i = 0; i < feaMix->elemNum; ++i) 
            feaMix->feaList[i] = ParseNextFeaElem(invoker, NULL, &feaIdx, &feaDim);
    }
    else if (strcmp(buf, "<FEATURE>") == 0) {
        feaMix->elemNum = 1;
        feaMix->feaList = (FELink *) New(hset->hmem, sizeof(FELink) * 1);
        feaMix->feaList[0] = ParseNextFeaElem(invoker, buf, &feaIdx, &feaDim);
        feaMix->mixDim = feaMix->feaList[0]->extDim;
    }
    else 
        HError(2650, "%s: Incorrect keyword for input feature", invoker);
    /* reset owner num */
    feaMix->ownerNum = 0;
    /* add a new macro */
    if (m == NULL) 
        NewMacro(hset, fidx, 'F', id, feaMix);
    else 
        m->structure = (Ptr) feaMix;
    /* saves the model */
    saveHMMSet = TRUE;

    return feaMix;
}

/* cz277 - pact */
static void InitActParmVecs(HMMSet *hset, LELink layerElem) {
    int i, intVal;
    NVecBundle *hyper = NULL;
    char buf[MAXSTRLEN];

    switch (layerElem->actfunKind) {
    case AFFINEAF:       
    case HERMITEAF:
    case LHUCRELUAF:     
    case PRELUAF:        
    case PARMRELUAF:     
    case LHUCSIGMOIDAF: 
    case PSIGMOIDAF:   
    case PARMSIGMOIDAF:
    case LHUCSOFTRELUAF: 
    case PSOFTRELUAF:   
    case PARMSOFTRELUAF: 
        ChkedAlpha("Next keyword", buf);
        break;
    default:
        return;
    }
    if (strcmp(buf, "<HYPERPARAMETER>") == 0) 
        hyper = AddOneNVecBundle("InitActParmVecs", NULL, hset, TRUE);
    layerElem->actfunParmNum = 0;
    if (strcmp(buf, "<NUMPARAMETERS>") == 0) {
        layerElem->actfunParmNum = ChkedInt("Number of parameter vectors", 1, INT_MAX);
        ChkedAlpha("Next keyword", buf);
    } 
    else if (strcmp(buf, "<PARAMETER>") == 0) 
        layerElem->actfunParmNum = 1;
    else
        HError(2650, "InitActParmVecs: Unexpected keyword %s", buf);
    layerElem->actfunVecs = (NVecBundle **) New(hset->hmem, (layerElem->actfunParmNum + 1) * sizeof(NVecBundle *)); 
    memset(layerElem->actfunVecs, 0, (layerElem->actfunParmNum + 1) * sizeof(NVecBundle *));
    layerElem->actfunVecs[0] = hyper;
    for (i = 1; i <= layerElem->actfunParmNum; ++i) { 
        if (i != 1) {
            ChkedAlpha("Next keyword", buf);
            if (strcmp("<PARAMETER>", buf) != 0)
                HError(2650, "<PARAMETER> expected");
        }
        intVal = ChkedInt("Parameter vector index", 1, layerElem->actfunParmNum + 1);
        layerElem->actfunVecs[intVal] = AddOneNVecBundle("InitActParmVecs", NULL, hset, TRUE);
        ++layerElem->actfunVecs[intVal]->nUse;
    }
    CheckActFunParameters(layerElem);
    /* setup the tracing list */
    if (layerElem->actfunVecs != NULL) {
        for (i = 0; i<= layerElem->actfunParmNum; ++i) {
            layerElem->actfunVecs[i]->kind = SIBK;
            CreateBundleTrace(hset->hmem, layerElem, (BTLink *) &layerElem->actfunVecs[i]->hook);
        }
    }
}

/* IL */
void InsertOneANNLayer(void) {
    int i, j, layerPos, layerNum;
    ADLink annDef;
    LELink layerElem;
    LELink *layerList;
    char buf[MAXSTRLEN];
    LabId id;
    
    /* get the target ANN model */
    ChkedAlpha("Target ANN macro type", buf);
    if (strcmp(buf, "~N") != 0) 
        HError(2650, "InsertOneANNLayer: ~N macro type expected");
    ChkedAlpha("Target ANN macro name", buf);
    annDef = GetANNDefByMacName("InsertOneANNLayer", buf, hset);
    /* get target layer position */
    /* -1, add on the top */
    layerPos = ChkedInt("New layer insert position", -1 * annDef->layerNum, annDef->layerNum + 2);
    if (layerPos >= 0 && layerPos < 2)
        HError(2677, "InsertOneANNLayer: Illegal insert layer position");
    else if (layerPos >= 2)
        layerPos -= 2;
    else
        layerPos = annDef->layerNum + 1 + layerPos;
    /* get the layer element macro name */
    ChkedAlpha("Next keyword", buf); 
    if (strcmp(buf, "~L") == 0) {
        ChkedAlpha("~L macro name", buf); 
        id = GetLabId(buf, FALSE);
        /*if (id != NULL || (m = FindMacroName(hset, 'L', id)) != NULL)*/
        if (id != NULL)
            HError(2676, "InsertOneANNLayer: Layer element macro name %s occupied", buf);
        id = GetLabId(buf, TRUE);
        /* get the layer kind */
        ChkedAlpha("next keyword", buf);
    } 
    else {
        id = GetNextANNMacroName("InsertOneANNLayer", hset, 'L');
        if (trace & T_BAS)
            HError(-2675, "InsertOneANNLayer: Anonymous layer element macro created, auto-name %s assigned\n", id->name);
    }
    if (strcmp(buf, "<BEGINLAYER>") != 0)
        HError(2650, "InsertOneANNLayer: <BEGINLAYER> expected");
    ChkedAlpha("next keyword", buf);
    if (strcmp(buf, "<LAYERKIND>") != 0)
        HError(2650, "InsertOneANNLayer: <LAYERKIND> expected");
    ChkedAlpha("layer kind string", buf);
    layerElem = GenBlankLayer(hset->hmem);
    layerElem->layerKind = Str2LayerKind(buf);
    ChkedAlpha("Next key word", buf);
    if (strcmp("<INPUTFEATURE>", buf) != 0)
        HError(2650, "<INPUTFEATURE> expected");
    layerElem->feaMix = AddOneFeaMix("InsertOneANNLayer", NULL, TRUE); 
    layerElem->feaMix->ownerList[layerElem->feaMix->ownerNum++] = layerElem;
    ++layerElem->feaMix->nUse;
    for (i = 0; i < layerElem->feaMix->elemNum; ++i) {
        ++layerElem->feaMix->feaList[i]->nUse;
       if (layerElem->feaMix->feaList[i]->inputKind == ANNFEAIK)
            ++layerElem->feaMix->feaList[i]->feaSrc->nDrv;
    }
    switch (layerElem->layerKind) {
    case ACTIVATIONONLYLAK: HError(2640, "InsertOneANNLayer: Not implemented yet!"); break;
    case CONVOLUTIONLAK: HError(2640, "InsertOneANNLayer: Not implemented yet!"); break;
    case PERCEPTRONLAK: 
        ChkedAlpha("Next keyword", buf);
        if (strcmp("<WEIGHT>", buf) != 0)
            HError(2650, "InsertOneANNLayer: <WEIGHT> expected");
        layerElem->wghtMat = AddOneNMatBundle("InsertOneANNLayer", NULL, hset, TRUE);
        ++layerElem->wghtMat->nUse; 
        layerElem->nodeNum = layerElem->wghtMat->variables->rowNum;
        layerElem->inputDim = layerElem->wghtMat->variables->colNum;
        if (layerElem->inputDim != layerElem->feaMix->mixDim)
            HError(2677, "InsertOneANNLayer: Feature mixture and weight matrix dimension inconsistent");
        CreateBundleTrace(hset->hmem, layerElem, (BTLink *) &layerElem->wghtMat->hook);
        ChkedAlpha("Next keyword", buf);
        if (strcmp("<BIAS>", buf) != 0)
            HError(2650, "InsertOneANNLayer: <BIAS> expected");
        layerElem->biasVec = AddOneNVecBundle("InsertOneANNLayer", NULL, hset, TRUE);
        ++layerElem->biasVec->nUse;
        if (layerElem->nodeNum != layerElem->biasVec->variables->vecLen)
            HError(2677, "InsertOneANNLayer: Weight matrix and bias vector dimension inconsistent");
        CreateBundleTrace(hset->hmem, layerElem, (BTLink *) &layerElem->biasVec->hook);
        ChkedAlpha("Next keyword", buf);
        if (strcmp("<ACTIVATION>", buf) != 0)
            HError(2650, "InsertOneANNLayer: <ACTIVATION> expected");
        ChkedAlpha("Activation kind string", buf);
        layerElem->actfunKind = Str2ActFunKind(buf);
        InitActParmVecs(hset, layerElem);
        break;
    case SUBSAMPLINGLAK: HError(2640, "InsertOneANNLayer: Not implemented yet!"); break;
    default:
        HError(2691, "InsertOneANNLayer: Unknown layer kind");
    } 
    ++layerElem->nUse;
    NewMacro(hset, fidx, 'L', id, layerElem);
    layerElem->ownerHead = hset->annSet->defsHead;
    layerElem->ownerTail = hset->annSet->defsHead;
    ++layerElem->ownerCnt;
    /* insert the new layer */
    layerNum = annDef->layerNum + 1;
    layerList = (LELink *) New(hset->hmem, layerNum * sizeof(LELink));
    for (i = 0, j = 0; i < layerNum; ++i) {
        if (i == layerPos) 
            layerList[i] = layerElem;
        else 
            layerList[i] = annDef->layerList[j++];
    }
    annDef->layerList = layerList;
    annDef->layerNum = layerNum;
    /* <ENDLAYER> */
    ChkedAlpha("Next keyword", buf);
    if (strcmp("<ENDLAYER>", buf) != 0)
        HError(2650, "InsertOneANNLayer: <ENDLAYER> expected"); 
    /* saves the model */
    saveHMMSet = TRUE;
}

/* cz277 - 1007 */
/* seed uses to be the position of layerElem */
static void ChangeOneLayerDims(char *invoker, LELink layerElem, int seed, int nodeNum, int inputDim) {
    int i;
    MLink m;
   
    if (inputDim != layerElem->feaMix->mixDim) {
        m = FindMacroStruct(hset, 'F', layerElem->feaMix);
        if (m == NULL)
            HError(2675, "%s: Anonymous feature mixture not allowed", invoker);
        if (trace & T_BAS)
            printf("%s: ~F \"%s\" dimension for weights may need to change\n", invoker, m->id->name);
    }
    switch (layerElem->layerKind) {
    case ACTIVATIONONLYLAK: HError(2640, "%s: Not implemented yet!", invoker); break;
    case CONVOLUTIONLAK: HError(2640, "%s: Not implemented yet!", invoker); break;
    case PERCEPTRONLAK:
        ChangeDimOneNMatBundle(invoker, layerElem->wghtMat, nodeNum, inputDim);
        ChangeDimOneNVecBundle(invoker, layerElem->biasVec, nodeNum);
        switch (layerElem->actfunKind) {
        case AFFINEAF:
        case HERMITEAF:
        case LHUCRELUAF:
        case PRELUAF:
        case PARMRELUAF:
        case LHUCSIGMOIDAF:
        case PSIGMOIDAF:
        case PARMSIGMOIDAF:
        case LHUCSOFTRELUAF:
        case PSOFTRELUAF:
        case PARMSOFTRELUAF:
            for (i = 1; i <= layerElem->actfunParmNum; ++i) 
                ChangeDimOneNVecBundle(invoker, layerElem->actfunVecs[i], nodeNum);
        default:
            break;
        }
        break;
    case SUBSAMPLINGLAK: HError(2640, "%s: Not implemented yet!", invoker); break;
    default:
        HError(2691, "%s: Unknown layer kind", invoker); break;
    }
    layerElem->nodeNum = nodeNum;
    layerElem->inputDim = inputDim;
}

/* cz277 - 1007 */
/* CF */
void ChangeOneANNLayerInput(void) {
    char buf[MAXSTRLEN];
    FeaMix *feaMix;
    LELink layerElem=NULL;
    int i, j;

    /* original feature mixture */
    ChkedAlpha("Target layer macro type", buf);
    if (strcmp(buf, "~L") == 0) 
        layerElem = GetLayerElemByMacName("ChangeOneANNLayerInput", buf, hset, NULL, NULL);
    else 
        HError(2650, "ChangeOneANNLayerInput: Unsupported target ANN layer macro");
    /* deuse the old feature mixture */
    for (i = 0, j = 0; j < layerElem->feaMix->ownerNum; ++i, ++j) {
        if (layerElem->feaMix->ownerList[i] == layerElem)
            ++j;
        if (i != j)
            layerElem->feaMix->ownerList[i] = layerElem->feaMix->ownerList[j];
    }
    --layerElem->feaMix->ownerNum;
    for (i = 0; i < layerElem->feaMix->elemNum; ++i) {
        --layerElem->feaMix->feaList[i]->nUse;
        if (layerElem->feaMix->feaList[i]->inputKind == ANNFEAIK)
            --layerElem->feaMix->feaList[i]->feaSrc->nDrv;
    }
    --layerElem->feaMix->nUse;
    /* new feature mixture */
    feaMix = AddOneFeaMix("ChangeOneANNLayerInput", NULL, TRUE);
    for (i = 0; i < feaMix->elemNum; ++i) 
        if (feaMix->feaList[i]->inputKind == ANNFEAIK)
            ++feaMix->feaList[i]->feaSrc->nDrv;
    /*ChkedAlpha("Next keyword", buf);
    if (strcmp(buf, "~F") == 0) {
        feaMix = AddOneFeaMix("ChangeOneANNLayerInput", buf, TRUE);
        ChkedAlpha("Feature mixture macro name", buf);
        id = GetLabId(buf, FALSE);
        if (id != NULL && (m = FindMacroName(hset, 'F', id)) != NULL) 
            feaMix = (FeaMix *) m->structure;
        else 
            HError(9999, "ChangeOneANNLayerInput: Unexisted feature mixture macro");
        for (i = 0; i < feaMix->elemNum; ++i) 
            if (feaMix->feaList[i]->inputKind == ANNFEAIK)
                ++feaMix->feaList[i]->feaSrc->nDrv;;
    }
    else 
        feaMix = AddOneFeaMix("ChangeOneANNLayerInput", buf, TRUE);*/
    /* use the new feature mixture */
    feaMix->ownerList[layerElem->feaMix->ownerNum++] = layerElem;
    ++feaMix->nUse;
    for (i = 0; i < feaMix->elemNum; ++i) 
        ++feaMix->feaList[i]->nUse;
        /*if (feaMix->feaList[i]->inputKind == ANNFEAIK)
            --feaMix->feaList[i]->feaSrc->nDrv;*/
    layerElem->feaMix = feaMix;
}

/* CD */
void ChangeOneANNLayerDims(void) {
    int nodeNum, inputDim, layerPos;
    LELink layerElem;

    layerElem = GetLayerElemByMacName("ChangeOneANNLayerDims", NULL, hset, NULL, &layerPos);
    /* get the new nodeNum */
    nodeNum = ChkedInt("Node number of the new layer", 0, INT_MAX);
    if (nodeNum == 0)
        nodeNum = layerElem->nodeNum;
    /* get the new inputDim */
    inputDim = ChkedInt("Input dimension of the new layer", 0, INT_MAX);
    if (inputDim == 0)
        inputDim = layerElem->inputDim;
    /* change the dimensions */
    ChangeOneLayerDims("ChangeOneANNLayerDims", layerElem, layerPos, nodeNum, inputDim);
}

/* EL */
void EraseOneANNLayer(void) {
    int layerPos;
    LELink layerElem=NULL;
    float randScale = 1.0;
    char buf[MAXSTRLEN];

    /* original feature mixture */
    ChkedAlpha("Target layer macro type", buf);
    if (strcmp(buf, "~L") == 0) 
        layerElem = GetLayerElemByMacName("EraseOneANNLayer", buf, hset, NULL, &layerPos);
    else 
        HError(2650, "EraseOneANNLayer: Unsupported target ANN layer macro");
    /* take action */
    /*actfunKind = layerElem->actfunKind;*/
    RandANNLayer(layerElem, layerPos, randScale);
    /*layerElem->actfunKind = actfunKind;*/
    /* saves the model */
    saveHMMSet = TRUE;
}

/* DL */
/* The other feature mixtures referencing this layer are not removed */
void DeleteOneANNLayer(void) {
    LELink layerElem;
    FeaMix *feaMix;
    ADLink annDef;
    int i, layerPos;

    layerElem = GetLayerElemByMacName("DeleteOneANNLayer", NULL, hset, &annDef, &layerPos);
    /* remove feature mixture */
    if (layerElem->feaMix != NULL) {
        feaMix = layerElem->feaMix;
        /* decrease nUse */
        for (i = 0; i < feaMix->elemNum; ++i)
            --feaMix->feaList[i]->nUse;
        --feaMix->nUse;
        /* decrease nDrv */
        for (i = 0; i < feaMix->elemNum; ++i) 
            if (feaMix->feaList[i]->inputKind == ANNFEAIK)
                --feaMix->feaList[i]->feaSrc->nDrv;
        /* remove this owner */
        --feaMix->ownerNum;
        for (i = layerPos; i < feaMix->ownerNum; ++i) 
            feaMix->ownerList[i] = feaMix->ownerList[i + 1];
    }
    /* remove the rest */
    switch (layerElem->layerKind) {
    case ACTIVATIONONLYLAK: HError(2640, "DeleteOneANNLayer: Not implemented yet"); break;
    case CONVOLUTIONLAK: HError(2640, "DeleteOneANNLayer: Not implemented yet"); break;
    case PERCEPTRONLAK:
        CancelBundleTrace(hset->hmem, layerElem, (BTLink *) &layerElem->wghtMat->hook);
        --layerElem->wghtMat->nUse;
        CancelBundleTrace(hset->hmem, layerElem, (BTLink *) &layerElem->biasVec->hook);
        --layerElem->biasVec->nUse; 
        if (layerElem->actfunVecs != NULL) {
            for (i = 0; i <= layerElem->actfunParmNum; ++i) { 
                if (layerElem->actfunVecs[i] != NULL) {
                    CancelBundleTrace(hset->hmem, layerElem, (BTLink *) &layerElem->actfunVecs[i]->hook);
                    --layerElem->actfunVecs[i]->nUse;
                }
            }
        }
        break;
    case SUBSAMPLINGLAK: HError(2640, "DeleteOneANNLayer: Not implemented yet"); break;
    default:
        break;
    }
    /* remove the layer itself */
    layerElem->nUse = 0;
    /* remove this layer */
    --annDef->layerNum;
    for (i = layerPos; i < annDef->layerNum; ++i) 
        annDef->layerList[i] = annDef->layerList[i + 1];
    /* saves the model */
    saveHMMSet = TRUE;
}

/* CA */
void ChangeOneANNLayerActFun(void) {
    LELink layerElem;
    char buf[MAXSTRLEN];

    layerElem = GetLayerElemByMacName("ChangeANNOneLayerActFun", NULL, hset, NULL, NULL);
    ChkedAlpha("Next keyword", buf);
    if (strcmp(buf, "<ACTIVATION>") == 0) {
        ChkedAlpha("New hidden activation function", buf);
        layerElem->actfunKind = Str2ActFunKind(buf);
        /* cz277 - pact */
        InitActParmVecs(hset, layerElem);
    }
    else 
        HError(2650, "ChangeOneANNLayerActFun: <ACTIVATION> keyword expected");
   
    /* saves the model */
    saveHMMSet = TRUE;
}

static FeaMix *CloneOneFeaMix(FeaMix *srcMix) {
    FeaMix *dstMix;
    FELink srcElem, dstElem;
    int i, j, n;

    dstMix = (FeaMix *) New(hset->hmem, sizeof(FeaMix));
    memcpy(dstMix, srcMix, sizeof(FeaMix));
    dstMix->feaList = (FELink *) New(hset->hmem, sizeof(FELink) * srcMix->elemNum);
    for (i = 0; i < srcMix->elemNum; ++i) {
        dstMix->feaList[i] = (FELink) New(hset->hmem, sizeof(FeaElem));
        srcElem = srcMix->feaList[i];
        dstElem = dstMix->feaList[i];
        memcpy(dstElem, srcElem, sizeof(FeaElem));
        /* cz277 - many */
        n = IntVecSize(dstElem->ctxPool);
        for (j = 1; j <= n; ++j) 
            dstElem->feaMats[j] = CreateNMatrix(hset->hmem, srcElem->feaMats[j]->rowNum, srcElem->feaMats[j]->colNum);
        if (dstElem->hisMat != NULL) 
            dstElem->hisMat = CreateNMatrix(hset->hmem, GetNBatchSamples(), dstElem->hisLen * dstElem->feaDim);
    }
    /* cz277 - many */
    /*dstMix->mixMat = CreateNMatrix(hset->hmem, GetNBatchSamples(), dstMix->mixDim);*/
    n = IntVecSize(dstMix->ctxPool);
    for (i = 1; i <= n; ++i) 
        dstMix->mixMats[i] = CreateNMatrix(hset->hmem, GetNBatchSamples(), dstMix->mixDim);

    return dstMix;
}

/* EF */
void ExtendOneFeaMix(void) {
    FeaMix *feaMix=NULL;
    FELink incElem, *feaList;
    LELink layerElem;
    LabId id;
    int i, j, incIdx, incDim, elemNum, layerPos = -1, seed;
    char buf[MAXSTRLEN]; 
    MLink m;

    ChkedAlpha("Target macro type", buf);
    if (strcmp(buf, "~L") == 0) {
        layerElem = GetLayerElemByMacName("ExtendOneFeaMix", buf, hset, NULL, &layerPos);
        /* if needed, clone layerElem->feaMix and update its owners and nUse */
        if (layerElem->feaMix->nUse > 1) {	/* if shared */
            feaMix = CloneOneFeaMix(layerElem->feaMix); 
            --layerElem->feaMix->ownerNum;
            for (i = 0, j = 0; i < layerElem->feaMix->ownerNum; ++i, ++j) {
                if (layerElem->feaMix->ownerList[i] == layerElem)
                    ++j;
                if (i != j)
                    layerElem->feaMix->ownerList[i] = layerElem->feaMix->ownerList[j];
            }
            --layerElem->feaMix->nUse;
            for (i = 0; i < layerElem->feaMix->elemNum; ++i) 
                --layerElem->feaMix->feaList[i]->nUse;
            /* update feaMix's owners and nUse */
            feaMix->ownerNum = 1;
            feaMix->ownerList[0] = layerElem;
            feaMix->nUse = 1;
            for (i = 0; i < feaMix->elemNum; ++i) 
                feaMix->feaList[i]->nUse = 1;
            /* add it as a new macro */
            id = GetNextANNMacroName("ExtendOneFeaMix", hset, 'F');
            NewMacro(hset, fidx, 'F', id, feaMix);
            layerElem->feaMix = feaMix;
        }
        else 
            feaMix = layerElem->feaMix;
    }
    else if (strcmp(buf, "~F") == 0) {
        ChkedAlpha("Feature mixture macro name", buf);
        id = GetLabId(buf, FALSE);
        if (id != NULL && (m = FindMacroName(hset, 'F', id)) != NULL) 
            feaMix = (FeaMix *) m->structure;
        else 
            HError(2679, "ExtendOneFeaMix: Unexisted feature mixture macro");
    }
    else 
        HError(2650, "ExtendOneFeaMix: Unsupported target ANN layer macro");
    /* from now on, operate the feaMix pointer */
    incElem = ParseNextFeaElem("ExtendOneFeaMix", NULL, &incIdx, &incDim);
    if (incIdx > 0 && incIdx <= feaMix->elemNum) 
        HError(2677, "ExtendOneFeaMix: Unacceptable new feature element index, please use -1 instead");
    /* append the new feature element */
    elemNum = feaMix->elemNum;
    ++feaMix->elemNum;
    feaList = feaMix->feaList;
    feaMix->feaList = (FELink *) New(hset->hmem, sizeof(FELink) * feaMix->elemNum);
    for (i = 0; i < elemNum; ++i) 
        feaMix->feaList[i] = feaList[i];
    feaMix->feaList[i] = incElem;
    feaMix->mixDim += incElem->extDim;
    /* modify the relevant layers */
    for (i = 0; i < feaMix->ownerNum; ++i) {
        layerElem = feaMix->ownerList[i];
        /* the random seed (layerPos) is inconsistent, but this should not matter */
        if (layerPos < 0)
            seed = i;
        else
            seed = layerPos;
        ChangeOneLayerDims("ChangeOneANNLayerDims", layerElem, seed, layerElem->nodeNum, feaMix->mixDim);
    }
    /* saves the model */
    saveHMMSet = TRUE;
}



/* CP */
void CopyANNParameters(void) {
    char mmfPath[MAXSTRLEN], lstPath[MAXSTRLEN], buf[MAXSTRLEN], xformPath[MAXSTRLEN], macroname[MAXSTRLEN];
    char srctype, dsttype;
    int i, actfunParmNum;
    InputXForm *auxinp = NULL;
    MLink srcmacro, dstmacro;
    LabId srclabid, dstlabid;
    /*ANNUpdtKind updtFlag = WEIGHTUK | BIASUK | ACTFUNUK;*/
    int ACTFUNUK = 0x4, BIASUK = 0x2, WEIGHTUK = 0x1;
    int updtFlag = ACTFUNUK | BIASUK | WEIGHTUK;

    ChkedAlpha("Next keyword", buf);
    /* load the auxiliary model set */
    if (strcmp(buf, "<HMMSET>") == 0) {
        ChkedAlpha("Auxiliary MMF file path", mmfPath);
        ChkedAlpha("Auxiliary HMM list file path", lstPath);
        if (auxFn == NULL || strcmp(auxFn, mmfPath) != 0) {
            if (auxFn != NULL)
                ResetHMMSet(auxSet);
            auxFn = CopyString(&gcheap, mmfPath); 
            auxSet = (HMMSet *) New(&hmmHeap, sizeof(HMMSet));
            CreateHMMSet(auxSet, &hmmHeap, TRUE);
            AddMMF(auxSet, mmfPath);
            if (MakeHMMSet(auxSet, lstPath) < SUCCESS)
                HError(2600, "CopyANNParameters: MakeHMMSet failed");
            if (LoadHMMSet(auxSet, NULL, NULL) < SUCCESS)
                HError(2600, "CopyANNParameters: LoadHMMSet failed");
        }
    }
    else if (strcmp(buf, "<INPUTXFORM>") == 0) {
        ChkedAlpha("Auxiliary transform file path", xformPath);
        auxinp = LoadInputXForm(NULL, NameOf(xformPath, macroname), xformPath);
        if (auxinp == NULL)
            HError(2600, "CopyANNParameters: Load auxiliary input xform failed");
    }
    else
        HError(2650, "CopyANNParameters: Unsupported keyword");
    /* get the update flag */
    ChkedAlpha("Next keyword", buf);
    if (strcmp(buf, "<UPDATEFLAG>") == 0) {
        ChkedAlpha("Update flag", buf);
        updtFlag = 0;
        for (i = 0; i < strlen(buf); ++i) {
            switch (buf[i]) {
            case 'a': updtFlag |= ACTFUNUK; break;
            case 'b': updtFlag |= BIASUK;   break;
            case 'w': updtFlag |= WEIGHTUK; break;
            default:
                HError(2650, "CopyANNParameters: Unsupported flag %c for copying", buf[i]);
            }
        }
        ChkedAlpha("Next keyword", buf);
    }
    /* get the source macro entity */
    /*ChkedAlpha("Next keyword", buf);*/
    if (strcmp(buf, "<SOURCEMACRO>") != 0)
        HError(2650, "CopyANNParameters: <SOURCEMACRO> expected");
    ChkedAlpha("Macro type", buf);
    if (buf[0] != '~' || strlen(buf) != 2)
        HError(2650, "CopyANNParameters: ~? macro type expected");
    srctype = buf[1];
    ChkedAlpha("Macro name", buf);
    srclabid = GetLabId(buf, FALSE);
    if (srclabid == NULL || (srcmacro = FindMacroName(auxSet, srctype, srclabid)) == NULL)
        HError(2679, "CopyANNParameters: Cannot find the source macro ~%c \"%s\"", srctype, srclabid->name);
    /* get the target macro entity */
    ChkedAlpha("Next keyword", buf);
    if (strcmp(buf, "<TARGETMACRO>") != 0)
        HError(2650, "CopyANNParameters: <TARGETMACRO> expected");
    ChkedAlpha("Macro type", buf);
    if (buf[0] != '~' || strlen(buf) != 2)
        HError(2650, "CopyANNParameters: ~? macro type expected");
    dsttype = buf[1];
    ChkedAlpha("Macro name", buf);
    dstlabid = GetLabId(buf, FALSE);
    if (dstlabid == NULL || (dstmacro = FindMacroName(hset, dsttype, dstlabid)) == NULL)
        HError(2679, "CopyANNParameters: Cannot find the source macro ~%c \"%s\"", dsttype, dstlabid->name);
    /* safety check */
    switch (srctype) {
    case 'V':
    case 'M':
    case 'L':
        if (auxinp != NULL || auxSet == NULL)
            HError(2615, "CopyANNParameters: Source macro ~%c needs source HMMSet", srctype);
        break;
    case 'j':
        if (auxinp == NULL)
            HError(2615, "CopyANNParameters: Source macro ~%c needs input xform", srctype);
        break;
    default:
        HError(2691, "CopyANNParameters: Unsupported source macro type");
    }  
    /* do the copying */
    switch (dsttype) {
    case 'V':
        switch (srctype) {
        case 'V':
            if ((updtFlag & ACTFUNUK) || (updtFlag & BIASUK))
                CopyPartialNVector2NVector(((NVecBundle *) srcmacro->structure)->variables, ((NVecBundle *) dstmacro->structure)->variables);
            break;
        case 'j':
            if (updtFlag & BIASUK) {
                if (auxinp->xform->bias == NULL)
                    HError(2615, "CopyANNParameters: source ~j bias expected to copy to ~V"); 
                CopyVector2NVector(auxinp->xform->bias, ((NVecBundle *) dstmacro->structure)->variables);
            }
            break;
        default:
            HError(2680, "CopyANNParameters: Cannot copy ~%c macro to ~V", srctype);
        }
        break;
    case 'M':
        switch (srctype) {
        case 'M':
            if (updtFlag & WEIGHTUK)
                CopyPartialNMatrix2NMatrix(((NMatBundle *) srcmacro->structure)->variables, ((NMatBundle *) dstmacro->structure)->variables);
            break;
        case 'j':
            if (updtFlag & WEIGHTUK)
                CopyMatrix2NMatrix(auxinp->xform->xform[1], ((NMatBundle *) dstmacro->structure)->variables);
            break;
        default:
            HError(2680, "CopyANNParameters: Cannot copy ~%c macro to ~V", srctype);
        } 
        break;
    case 'L':
        switch (srctype) {
        case 'V':
            if (updtFlag & BIASUK)
                CopyPartialNVector2NVector(((NVecBundle *) srcmacro->structure)->variables, ((LELink) dstmacro->structure)->biasVec->variables);
            break;
        case 'M':
            if (updtFlag & WEIGHTUK)
                CopyPartialNMatrix2NMatrix(((NMatBundle *) srcmacro->structure)->variables, ((LELink) dstmacro->structure)->wghtMat->variables);
            break;
        case 'L':
            if ((updtFlag & BIASUK) != 0 && ((LELink) srcmacro->structure)->biasVec != NULL && ((LELink) dstmacro->structure)->biasVec != NULL)
                CopyPartialNVector2NVector(((LELink) srcmacro->structure)->biasVec->variables, ((LELink) dstmacro->structure)->biasVec->variables);
            if ((updtFlag & WEIGHTUK) != 0 && ((LELink) srcmacro->structure)->wghtMat != NULL && ((LELink) dstmacro->structure)->wghtMat != NULL)
                CopyPartialNMatrix2NMatrix(((LELink) srcmacro->structure)->wghtMat->variables, ((LELink) dstmacro->structure)->wghtMat->variables);
            if ((updtFlag & ACTFUNUK) != 0 && ((LELink) srcmacro->structure)->actfunVecs != NULL && ((LELink) dstmacro->structure)->actfunVecs != NULL) {
                actfunParmNum = ((LELink) dstmacro->structure)->actfunParmNum;
                if (actfunParmNum < ((LELink) srcmacro->structure)->actfunParmNum)
                    actfunParmNum = ((LELink) srcmacro->structure)->actfunParmNum;
                for (i = 0 ; i <= actfunParmNum; ++i)
                    if (((LELink) srcmacro->structure)->actfunVecs[i] != NULL && ((LELink) dstmacro->structure)->actfunVecs[i] != NULL)
                        CopyPartialNVector2NVector(((LELink) srcmacro->structure)->actfunVecs[i]->variables, ((LELink) dstmacro->structure)->actfunVecs[i]->variables);  
            }
            break;
        case 'j':
            if (updtFlag & BIASUK)
                CopyVector2NVector(auxinp->xform->bias, ((LELink) dstmacro->structure)->biasVec->variables);
            if (updtFlag & WEIGHTUK)
                CopyMatrix2NMatrix(auxinp->xform->xform[1], ((LELink) dstmacro->structure)->wghtMat->variables);
            break;
        default:
            HError(2680, "CopyANNParameters: Cannot copy ~%c macro to ~V", srctype);
        } 
        break;
    default:
        HError(2640, "CopyANNParameters: Unsupported target ANN parameters for copying"); 
    }

    /* saves the model */
    saveHMMSet = TRUE;
}

/*void CopyANNLayerParms(void) {
    char mmfPath[MAXSTRLEN], lstPath[MAXSTRLEN], buf[MAXSTRLEN], xformPath[MAXSTRLEN], macroname[MAXSTRLEN];
    LELink srcLayer, dstLayer;
    int i, j, k, opCnt;
    HMMSet *auxset = NULL; 
    InputXForm *auxinp = NULL;

    opCnt = ChkedInt("Operation count", 1, INT_MAX);
    ChkedAlpha("Next keyword", buf);
    if (strcmp(buf, "<HMMSET>") == 0) {
        ChkedAlpha("Auxiliary MMF file path", mmfPath);
        ChkedAlpha("Auxiliary HMM list file path", lstPath);
        auxset = (HMMSet *) New(&hmmHeap, sizeof(HMMSet));
        CreateHMMSet(auxset, &hmmHeap, TRUE);
        AddMMF(auxset, mmfPath);
        if (MakeHMMSet(auxset, lstPath) < SUCCESS)
            HError(9999, "CopyANNLayerParms: MakeHMMSet failed");
        if (LoadHMMSet(auxset, NULL, NULL) < SUCCESS)
            HError(9999, "CopyANNLayerParms: LoadHMMSet failed");
        ChkedAlpha("Next keyword", buf);
    }
    else if (strcmp(buf, "<INPUTXFORM>") == 0) {
        ChkedAlpha("Auxiliary transform file path", xformPath);
        auxinp = LoadInputXForm(NULL, NameOf(xformPath, macroname), xformPath);
        if (auxinp == NULL)
            HError(9999, "CopyANNLayerParms: Load auxiliary input xform failed");
        ChkedAlpha("Next keyword", buf);
    }
    else 
        auxset = hset;
    for (i = 0; i < opCnt; ++i) {
        if (i != 0)
            ChkedAlpha("Next keyword", buf);
        if (strcmp(buf, "<COPY>") == 0 && auxset != NULL) {
            srcLayer = GetLayerElemByMacName("CopyANNLayerParms", NULL, auxset, NULL, NULL);
            dstLayer = GetLayerElemByMacName("CopyANNLayerParms", NULL, hset, NULL, NULL);
            ChkedAlpha("Next keyword", buf);
            if (srcLayer != dstLayer) {
                for (j = 0; j < strlen(buf); ++j) {
                    switch (buf[j]) {
                    case 'a':
                        if (srcLayer->actfunVecs != NULL && dstLayer->actfunVecs != NULL) 
                            if (srcLayer->actfunKind == dstLayer->actfunKind) 
                                for (k = 0; k <= srcLayer->actfunParmNum; ++k)
                                    CopyPartialNVector2NVector(srcLayer->actfunVecs[k]->variables, dstLayer->actfunVecs[k]->variables);
                        break;
                    case 'b':
                        if (srcLayer->inputDim != dstLayer->inputDim) 
                            HError(-1, "CopyANNLayerParms: Unequal bias vector lenghts");
                        CopyPartialNVector2NVector(srcLayer->biasVec->variables, dstLayer->biasVec->variables);
                        break;
                    case 'w':
                        if (srcLayer->nodeNum != dstLayer->nodeNum || srcLayer->inputDim != dstLayer->inputDim)
                            HError(-1, "CopyANNLayerParms: Unequal weight matrix dimensions");
                        CopyPartialNMatrix2NMatrix(srcLayer->wghtMat->variables, dstLayer->wghtMat->variables);
                        break;
                    default:
                        HError(9999, "CopyANNLayerParms: Unknown update flag %c", buf[j]); 
                    }
                }
            }
        }
        else if (strcmp(buf, "<COPY>") == 0 && auxinp != NULL) {
            ChkedAlpha("Input transform macro type", buf);
            if (strcmp(buf, "~j") != 0)
                HError(9999, "CopyANNLayerParms: The input transform macro ~j expected");
            ChkedAlpha("Input transform macro name", buf);
            if (strcmp(buf, macroname) != 0)
                HError(9999, "CopyANNLayerParms: The input transform macro name unmatched");
            dstLayer = GetLayerElemByMacName("CopyANNLayerParms", NULL, hset, NULL, NULL);
            ChkedAlpha("Next keyword", buf);
            for (j = 0; j < strlen(buf); ++j) {
                switch (buf[j]) {
                case 'b':
                    if (auxinp->xform->bias != NULL) {
                        if (VectorSize(auxinp->xform->bias) != dstLayer->inputDim)
                            HError(-1, "CopyANNLayerParms: Unequal bias vector lenghts");
                        CopyVector2NVector(auxinp->xform->bias, dstLayer->biasVec->variables);
                    }
                    break;
                case 'w':
                    if (NumRows(auxinp->xform->xform[1]) != dstLayer->nodeNum || NumCols(auxinp->xform->xform[1]) != dstLayer->inputDim)
                        HError(-1, "CopyANNLayerParms: Unequal weight matrix dimensions");
                    CopyMatrix2NMatrix(auxinp->xform->xform[1], dstLayer->wghtMat->variables);
                    break;
                default:
                    HError(9999, "CopyANNLayerParms: Unsupported update flag %c for input transform", buf[j]);
                }
            }
        }
        else 
            HError(9999, "CopyANNLayerParms: <COPY> keyword expected");
    }    
    if (auxset != NULL && hset != auxset) 
        ResetHMMSet(auxset);

    saveHMMSet = TRUE;
}*/

/* cz277 - ANN */
/* SL */
void ShowANNLayerStats(void) {
    int i, len, layerPos;
    LELink layerElem=NULL;
    char buf[MAXSTRLEN];
    double mean, stddev, absmax, l1norm, l2norm, absval;

    /* original feature mixture */
    ChkedAlpha("Target layer macro type", buf);
    if (strcmp(buf, "~N") == 0 || strcmp(buf, "~L") == 0) {
        layerElem = GetLayerElemByMacName("ShowANNLayerStats", buf, hset, NULL, &layerPos);
    }
    else {
        HError(2650, "ShowANNLayerStats: Unsupported target ANN layer macro");
    }
    /* compute for weights */
    len = layerElem->nodeNum * layerElem->inputDim;
    absmax = 0.0;
    mean = 0.0;
    stddev = 0.0;
    l1norm = 0.0;
    l2norm = 0.0;
    for (i = 0; i < len; ++i) {
        absval = fabs(layerElem->wghtMat->variables->matElems[i]);
        if (absval > absmax) 
            absmax = absval;
        mean += layerElem->wghtMat->variables->matElems[i];
        l1norm += absval;
        l2norm += pow(layerElem->wghtMat->variables->matElems[i], 2.0);
    }
    mean /= len;
    l1norm /= layerElem->nodeNum;
    l2norm /= layerElem->nodeNum;
    for (i = 0; i < len; ++i) {
        stddev += pow(layerElem->wghtMat->variables->matElems[i] - mean, 2.0);
    }
    stddev = sqrt(stddev / len);
    /* output */
    printf("\tWeights:\n");
    printf("\t\tinput dim = %d, output dim = %d, param num = %d\n", layerElem->inputDim, layerElem->nodeNum, len);
    printf("\t\tmean = %e, std dev = %e, max abs = %e\n", mean, stddev, absmax);
    printf("\t\tl1 norm / output node = %e, l2 norm / output node = %e\n", l1norm, l2norm);
    /*printf("\n");*/
    /* compute for biases */
    len = layerElem->nodeNum;
    absmax = 0.0;
    mean = 0.0;
    stddev = 0.0;
    l1norm = 0.0;
    l2norm = 0.0;
    for (i = 0; i < len; ++i) {
        absval = fabs(layerElem->biasVec->variables->vecElems[i]);
        if (absval > absmax)
            absmax = absval;
        mean += layerElem->biasVec->variables->vecElems[i];
        l1norm += absval;
        l2norm += pow(layerElem->biasVec->variables->vecElems[i], 2.0);
    }
    mean /= len;
    l1norm /= layerElem->nodeNum;
    l2norm /= layerElem->nodeNum;
    for (i = 0; i < len; ++i) {
        stddev += pow(layerElem->biasVec->variables->vecElems[i] - mean, 2.0);
    }
    stddev = sqrt(stddev / len);
    /* output */
    printf("\tbiases:\n");
    printf("\t\toutput dim = %d, param num = %d\n", layerElem->nodeNum, len);
    printf("\t\tmean = %e, std dev = %e, max abs = %e\n", mean, stddev, absmax);
    printf("\t\tl1 norm / output node = %e, l2 norm / output node = %e\n", l1norm, l2norm);
    printf("\n");

    /* saves the model */
    saveHMMSet = FALSE;
}


/* ----------------------- RS - Replace an HMM state  --------------------- */

/*void ReplaceHMMState(void) {
    char buf[MAXSTRLEN], hbuf[MAXSTRLEN];
    int tIdx, sIdx = -1, i, j;
    LabId hmmId, stateId;
    HMMScanState hss;
    MLink macDef;
    HLink hlink;
    StateInfo *si;
    StreamElem se;
    LELink layerElem;

    if (hset->hsKind == ANNHS) 
        HError(2615, "ReplaceHMMState: No HMMs in the model file");
    // get target HMM structure 
    ChkedAlpha("Target HMM name", buf);
    if (strcmp(buf, "~h") == 0)
        ChkedAlpha("Target HMM name", buf);
    hmmId = GetLabId(buf, FALSE); 
    if (hmmId == NULL) 
        HError(2635, "ReplaceHMMState: Failed to find model for label \"%s\"", buf);
    if ((macDef = FindMacroName(hset, 'l', hmmId)) == NULL) 
        HError(2635, "ReplaceHMMState: Unknown HMM name %s", buf);
    hlink = (HLink) macDef->structure;
    tIdx = ChkedInt("Target state index", 2, hlink->numStates - 1);
    // get source state structure 
    ChkedAlpha("Source state name", buf);
    if (strcmp(buf, "~s") == 0)
        ChkedAlpha("Source state name", buf);
    stateId = GetLabId(buf, FALSE);
    if (stateId == NULL) {
        ExtractState(buf, hbuf, &sIdx);
        stateId = GetLabId(hbuf, FALSE);
        if ((macDef = FindMacroName(hset, 'l', stateId)) == NULL) 
            HError(2635, "ReplaceHMMState: Unknown HMM name %s", hbuf);
        si = ((HLink) macDef->structure)->svec[sIdx].info;
    }
    else {
        if ((macDef = FindMacroName(hset, 's', stateId)) == NULL) 
            HError(2635, "ReplaceHMMState: Unknown state name %s", buf);
        si = (StateInfo *) macDef->structure;
    }
    // change the use count
    --hlink->svec[tIdx].info->nUse;
    if (hlink->svec[tIdx].info->nUse == 0) 
        DeleteMacroStruct(hset, 's', hlink->svec[tIdx].info);
    ++si->nUse;
    // replace target state
    hlink->svec[tIdx].info = si;
    // remove the redundant targets and reset the target indexes 
    if (hset->hsKind == HYBRIDHS) {
        for (i = 1; i <= hset->swidth[0]; ++i) {
            se = si->pdf[i];
            tIdx = se.targetIdx - 1;
            layerElem = se.targetSrc;
            if (tIdx < layerElem->nodeNum - 1) {
                for (j = tIdx + 1; j < layerElem->nodeNum; ++j) {
                    CopyNSegment(layerElem->wghtMat->variables, j * layerElem->inputDim, layerElem->inputDim, layerElem->wghtMat->variables, (j - 1) * layerElem->inputDim);
                    CopyNVectorSegment(layerElem->biasVec->variables, j, 1, layerElem->biasVec->variables, j - 1);
                }
                --layerElem->nodeNum;
            }
            NewHMMScan(hset, &hss);
            while(GoNextState(&hss, FALSE)) {
                if (hss.si->pdf[i].targetIdx > tIdx + 1) {
                    --hss.si->pdf[i].targetIdx;
                    hss.si->pdf[i].targetIdx *= -1;
                }
            }
            EndHMMScan(&hss);
            NewHMMScan(hset, &hss);
            while(GoNextState(&hss, FALSE)) 
                if (hss.si->pdf[i].targetIdx < 0) 
                    hss.si->pdf[i].targetIdx *= -1;
        }
    }

    // saves the model 
    saveHMMSet = TRUE;
}*/

/* ----------------------- CH - Convert GMM-HMMs to an initial ANN-HMMs --------------------- */

static void BorrowOneANNMacro(char *invoker, char type, Ptr structure, HMMSet *srcset, HMMSet *dstset) {
    MLink m;
    LabId id;

    m = FindMacroStruct(dstset, type, structure);
    if (m == NULL) {
        m = FindMacroStruct(srcset, type, structure);
        if (m != NULL) {
            id = m->id;
            if (FindMacroName(dstset, type, id) != NULL)
                m = NULL;
        }
        if (m == NULL)
            id = GetNextANNMacroName(invoker, dstset, type);
        NewMacro(dstset, fidx, type, id, structure);
    }
}

void ConnectANNtoHMMs(void) {
    char buf[MAXSTRLEN], mmfPath[MAXSTRLEN], lstPath[MAXSTRLEN];
    HMMScanState hss;
    HMMSet *auxset;
    LELink layerElem;
    MLink m;
    AILink annInfo, curInfo;
    ADLink annDef;
    int i, j, h, targetCnt, streamIdx = 1;
    LabId id;

    if (hset->hsKind == ANNHS) 
        HError(2615, "ConnectANNtoHMMs: No HMMs found in the model set");

    /* load the ANN model */
    ChkedAlpha("Auxiliary MMF file path", mmfPath);
    ChkedAlpha("Auxiliary HMM list file path", lstPath);
    auxset = (HMMSet *) New(&hmmHeap, sizeof(HMMSet));
    CreateHMMSet(auxset, &hmmHeap, TRUE);
    AddMMF(auxset, mmfPath);
    if (MakeHMMSet(auxset, lstPath) < SUCCESS)
        HError(2600, "ConnectANNtoHMMs: MakeHMMSet failed");
    if (LoadHMMSet(auxset, NULL, NULL) < SUCCESS)
        HError(2600, "ConnectANNtoHMMs: LoadHMMSet failed");
    /* fetch the ANN model */
    ChkedAlpha("ANN macro type", buf);
    if (strcmp(buf, "~N") == 0)
        ChkedAlpha("ANN macro name", buf);
    annDef = GetANNDefByMacName("ConnectANNtoHMMs", buf, auxset);
    /* import each of its component to hset */
    m = FindMacroStruct(hset, 'N', annDef);
    if (m == NULL) {
        m = FindMacroStruct(auxset, 'N', annDef);
        if (m != NULL) {
            id = m->id;
            if (FindMacroName(hset, 'N', id) != NULL)
                m = NULL;
        }
        if (m == NULL) 
            id = GetNextANNMacroName("ConnectANNtoHMMs", hset, 'N');
        NewMacro(hset, fidx, 'N', id, annDef);
    }
    /*annDef->nUse |= 1;*/
    if (hset->annSet == NULL) 
        InitANNSet(hset);
    annInfo = (AILink) New(hset->hmem, sizeof(ANNInfo));
    annInfo->annDef = annDef;
    annInfo->index = hset->annSet->annNum++;
    annInfo->fidx = fidx;
    curInfo = hset->annSet->defsTail;
    if (curInfo == NULL) 
        hset->annSet->defsHead = annInfo;
    else {
        curInfo->next = annInfo;
        annInfo->prev = curInfo;
    }
    annInfo->prev = curInfo;
    annInfo->next = NULL;
    hset->annSet->defsTail = annInfo;
    for (i = 0; i < annDef->layerNum; ++i) {
        layerElem = annDef->layerList[i];
        BorrowOneANNMacro("ConnectANNtoHMMs", 'L', layerElem, auxset, hset);
        if (layerElem->feaMix != NULL)
            BorrowOneANNMacro("ConnectANNtoHMMs", 'F', layerElem->feaMix, auxset, hset);
        if (layerElem->wghtMat != NULL)
            BorrowOneANNMacro("ConnectANNtoHMMs", 'M', layerElem->wghtMat, auxset, hset);
        if (layerElem->biasVec != NULL)
            BorrowOneANNMacro("ConnectANNtoHMMs", 'V', layerElem->biasVec, auxset, hset);
        if (layerElem->actfunVecs != NULL)
            for (j = 0; j <= layerElem->actfunParmNum; ++i) 
                if (layerElem->actfunVecs[j] != NULL)
                    BorrowOneANNMacro("ConnectANNtoHMMs", 'V', layerElem->actfunVecs[j], auxset, hset);
    } 
    /* to do Tandem or Hybrid */
    ChkedAlpha("Next keyword", buf);
    if (strcmp(buf, "<STREAM>") == 0) {
        streamIdx = ChkedInt("Target stream index", 1, hset->swidth[0]);
        ChkedAlpha("Next keyword", buf);
    }
    if (strcmp(buf, "<HYBRID>") == 0) {
        hset->hsKind = HYBRIDHS;
        NewHMMScan(hset, &hss);
        while(GoNextState(&hss, FALSE)) 
            if (hss.si->pdf[streamIdx].targetIdx >= 0) 
                hss.si->pdf[streamIdx].targetIdx = -1;
        EndHMMScan(&hss);
        targetCnt = 0;
        layerElem = annDef->layerList[annDef->layerNum - 1];
        NewHMMScan(hset, &hss);
        while(GoNextState(&hss, FALSE)) {
            if (hss.si->pdf[streamIdx].targetIdx < 0) {
                hss.si->pdf[streamIdx].densKind = ANNDK;
                hss.si->pdf[streamIdx].targetSrc = layerElem;
                hss.si->pdf[streamIdx].targetIdx = ++targetCnt;
                hss.si->pdf[streamIdx].targetPen = 0.0;
                ++hss.si->pdf[streamIdx].targetSrc->nUse;
                hss.si->pdf[streamIdx].nMix = 1;
            }
        }
        EndHMMScan(&hss);
        if (targetCnt != layerElem->nodeNum) {
            if (trace & T_BAS)
                HError(-2681, "ConnectANNtoHMMs: Change target number to %d to match HMM state number", targetCnt);
            ChangeOneLayerDims("ChangeOneANNLayerDims", layerElem, annDef->layerNum - 1, targetCnt, layerElem->inputDim);
        }
        /* remove redundant macros */
        for (h = 0; h < MACHASHSIZE; ++h)
            for (m = hset->mtab[h]; m != NULL; m = m->next)
                if (m->type == 'v')
                    DeleteMacro(hset, m);
    }
    else if (strcmp(buf, "<TANDEM>") == 0) 
        HError(2640, "ConnectANNtoHMMs: Tandem system connection function not implemented yet");
    else 
        HError(2650, "ConnectANNtoHMMs: Unknown connection method, <HYBRID> or <TANDEM> expected");
    
    /* note, should not reset the auxset here*/
    /* saves the model */
    saveHMMSet = TRUE;

}

/* -----Regression Class Clustering Tree Building Routines -------------- */

/* linked list of components that are grouped into a cluster */
typedef struct _CoList {

   struct _CoList *next;     /* next component in the linked list */
   char   *hmmName;          /* Physical hmm name for this component */
   int    state;             /* state containing this component */
   int    stream;            /* stream for this component */
   int    mix;               /* mixture for this component */
   MixPDF *mp;               /* actual component */

} CoList;
  
/* node information in the regression class tree */
typedef struct {

   Vector aveMean;           /* node cluster mean */
   Vector aveCovar;          /* node cluster variance */
   float  clusterScore;      /* node cluster score */
   float  clustAcc;          /* accumulates in this cluster */
   int  nComponents;         /* number of components in this cluster */
   short  nodeIndex;         /* node index number */
   CoList *list;             /* linked list of the mixture components */

} RNode;

/* return the node of a tree */
static RNode *GetRNode(RegNode *n) 
{
   return ((RNode *) n->info);
}

/* Create a link for a cluster linked list */
CoList *CreateClusterLink(char *s, int state, int stream, int mix, 
                          MixPDF *mp) 
{
   CoList *c;

   c = (CoList *) New(&tmpHeap, sizeof(CoList));
   c->next = NULL;
   if (s == NULL)
      HError(2670, "CreateClusterLink: Physical name is NULL!?!");
   c->hmmName = s;
   c->state = state;
   c->stream = stream;
   c->mix = mix;
   c->mp = mp;

   return c;

}

/* Create a tree node for use in the regression tree */
RNode *CreateRegTreeNode(CoList *list, int vSize) 
{
   RNode *n;

   n = (RNode *) New(&tmpHeap, sizeof(RNode));
   n->aveMean  = CreateVector(&tmpHeap, vSize);
   n->aveCovar = CreateVector(&tmpHeap, vSize);
   ZeroVector(n->aveMean);
   ZeroVector(n->aveCovar);
   n->list = list;
   n->nComponents = 0;
   n->clusterScore = 0.0;
   n->nodeIndex = -1;

   return n;

}

void PrintNodeInfo(RNode *n, int vSize) 
{
   int i;

   printf("For cluster %d there are:\n", n->nodeIndex);
   printf("%d components with %f occupation count;\n  Cluster score of %e\n",
          n->nComponents, n->clustAcc, n->clusterScore);
   printf("MEAN:-\n");
   for (i = 1; i <= vSize; i++)
      printf("%e ", n->aveMean[i]);
   printf("\nCOVARIANCE:-\n");
   for (i = 1; i <= vSize; i++)
      printf("%e ", n->aveCovar[i]);
   printf("\n");
   fflush(stdout);
}


float Distance(Vector v1, Vector v2) 
{  
   int k, vSize;
   float x, dist = 0.0;

   vSize = VectorSize(v1);
   for (k = 1; k <= vSize; k++) {
      x = v1[k] - v2[k];
      x *= x;
      dist += x;
   }

   return ((float) sqrt((double) dist));

}

float CalcNodeScore(RNode *n, int vSize) 
{
   float score = 0.0;
   Vector mean;
   CoList *c;
   AccSum *acc;

   for(c = n->list; c != NULL; c = c->next) {
      mean = c->mp->mean;
      acc  = (AccSum *) c->mp->hook;
      if (acc != NULL)
         score += Distance(mean, n->aveMean) * acc->occ ;
   }

   return score;

}

/* calculation of the cluster distribution -> aveMean aveCovar */
void CalcClusterDistribution(RNode *n, int vSize) 
{
   CoList *c;
   AccSum *acc;
   int k;
   float occ;
   Vector sum, sqr;
  
   sum = CreateVector(&gstack, vSize);
   sqr = CreateVector(&gstack, vSize);
   occ = 0.0;
   ZeroVector(sum);
   ZeroVector(sqr);

   for (c = n->list; c != NULL; c = c->next) {
      /* scaled(mean) and scaled(mean*mean + covar) already stored in mp->hook */
      acc = (AccSum *) c->mp->hook;
      if (acc != NULL) {
         for (k = 1; k <= vSize; k++) {
            sum[k] += acc->sum[k];
            sqr[k] += acc->sqr[k];
         }
         occ += acc->occ;
      }
      else
         HError(-2671, "CalcClusterDistribution: Accumulates are non existant for hmm %s", 
                c->hmmName);
   }

   /* now calculate mean and covariance for this cluster */
   for (k = 1; k <= vSize; k++) {
      n->aveMean[k]  = sum[k] / occ;
      n->aveCovar[k] = (sqr[k] - sum[k]*sum[k]/occ) / occ;
   }
   n->clustAcc = occ;

   FreeVector(&gstack, sum);

}
    
/* Initialise a regression tree root node -- will also separate out
   speech and non-speech sounds */
RegTree *InitRegTree(HMMSet *hset, int *vSize, ILink ilist) 
{  
  HMMScanState hss;
  CoList *listSp=NULL,*listNonSp=NULL,*cs, *cn;
  RNode *r1=NULL, *r2=NULL, *rNode=NULL;
  RegNode *root;
  RegTree *rTree;
  int nComponentsSp = 0, nComponentsNonSp = 0;
  StreamElem *ste;
  ILink p=NULL;

  if (!occStatsLoaded)
    HError(2672, "InitRegTree: Building regression classes must load the stats file!\n");
  
  cs = cn = NULL;
  *vSize = hset->swidth[1];
  if (*vSize <= 0)
    HError(2673, "InitRegTree: Problem with vector Size = %d", vSize);

  NewHMMScan(hset,&hss);

  if (!(hss.isCont || (hss.hset->hsKind == TIEDHS))) {
     HError(2673,"InitRegTree: Can only resize continuous featured systems");
  }

  do {
    while (GoNextState(&hss,TRUE)) {
      InitTreeAccs(hss.se, *vSize);
      while (GoNextStream(&hss,TRUE)) {
	/* go through item list to see if non-speech sound */
	if (ilist != NULL)
	  for (p=ilist; p != NULL; p=p->next) {
	    ste = (StreamElem *) p->item;
	    if (ste == hss.ste)
	      break;
	  }
	while (GoNextMix(&hss,TRUE)) {
	  if (p == NULL) {
	    if (cs == NULL) {
	      cs = CreateClusterLink(HMMPhysName(hset, hss.hmm), 
				     hss.i, hss.s, hss.m, hss.mp);
	      listSp = cs;
	    }
	    else {
	      cs->next = CreateClusterLink(HMMPhysName(hset, hss.hmm), 
					   hss.i, hss.s, hss.m, hss.mp);
	      cs = cs->next;
	    }
	    nComponentsSp += 1;
	  }
	  else {
	    if (cn == NULL) {
	      cn = CreateClusterLink(HMMPhysName(hset, hss.hmm), 
				     hss.i, hss.s, hss.m, hss.mp);
	      listNonSp = cn;
	    }
	    else {
	      cn->next = CreateClusterLink(HMMPhysName(hset, hss.hmm), 
					   hss.i, hss.s, hss.m, hss.mp);
	      cn = cn->next;
	    }
	    nComponentsNonSp += 1;
	  }  
	}
      }
    }
  } while (GoNextHMM(&hss));
  EndHMMScan(&hss);

  cs = NULL; cn = NULL;
  r1 = CreateRegTreeNode(listSp, *vSize);  
  r1->nodeIndex = 1;
  r1->nComponents = nComponentsSp;
  CalcClusterDistribution(r1, *vSize);
  r1->clusterScore = CalcNodeScore(r1, *vSize);
  /* create an instance of the tree */
  rTree = (RegTree *) New(&tmpHeap, sizeof(RegTree));
  rTree->bclass = NULL;
  rTree->numNodes = 0;
  rTree->numTNodes = 0;
  root = rTree->root = (RegNode *) New(&tmpHeap, sizeof(RegNode));
  /* is there a speech/sil split at the root */
  if (nComponentsNonSp > 0) {
    /* reset the root node */
    root->nodeIndex = 1;
    rNode = CreateRegTreeNode(NULL, *vSize);
    rNode->nodeIndex = 1;
    root->info = rNode;
    root->numChild = 2;
    root->child = (RegNode **)New(&tmpHeap,3*sizeof(RegNode *));
    root->child[1] = (RegNode *) New(&tmpHeap, sizeof(RegNode));
    root->child[2] = (RegNode *) New(&tmpHeap, sizeof(RegNode));
    /* set-up information for first child */
    r2 = CreateRegTreeNode(listNonSp, *vSize);
    r2->nodeIndex = 2;
    r2->nComponents = nComponentsNonSp;
    CalcClusterDistribution(r2, *vSize);
    r2->clusterScore = CalcNodeScore(r2, *vSize);
    root->child[1]->info = r2;
    root->child[1]->numChild = 0;
    root->child[1]->child = NULL;
    /* set-up information for second child */
    r1->nodeIndex = 3;
    root->child[2]->info = r1;
    root->child[2]->numChild = 0;
    root->child[2]->child = NULL;
  }
  else {
    root->info = r1;
    root->numChild = 0;
    root->child = NULL;
  }
  return(rTree);
}


void PerturbMean(Vector mean, Vector covar, float pertDepth) 
{  
   float x;
   int k, vSize;

   vSize = VectorSize(mean);
   for(k = 1; k <= vSize; k++) {
      x = pertDepth * covar[k];
      mean[k] += x;
   }

}

void CalcDistance(CoList *list, RNode *ch1, RNode *ch2, int vSize)
{
   int k;
   CoList *c;
   AccSum *acc;
   float score1, score2;
   Vector mean, sum1, sum2;

   sum1 = CreateVector(&gstack, vSize);
   ZeroVector(sum1);
   sum2 = CreateVector(&gstack, vSize);
   ZeroVector(sum2);


   ch1->clusterScore = ch2->clusterScore = 0.0;
   ch1->clustAcc = ch2->clustAcc = 0.0;
  
   for(c = list; c != NULL; c = c->next) {
      mean = c->mp->mean;
      acc  = (AccSum *) c->mp->hook;
      score1 = Distance(mean, ch1->aveMean);
      score2 = Distance(mean, ch2->aveMean);
      if (score1 < score2) {
         if (acc != NULL) {
            ch1->clustAcc += acc->occ;
            ch1->clusterScore += (acc->occ * score1);
            for (k = 1; k <= vSize; k++)
               sum1[k] += acc->sum[k];
         }
      }
      else {
         if (acc != NULL) {
            ch2->clustAcc += acc->occ;
            ch2->clusterScore += (acc->occ * score2);
            for (k = 1; k <= vSize; k++)
               sum2[k] += acc->sum[k];
         }
      }
   }

   for (k = 1; k <= vSize; k++) {
      ch1->aveMean[k] = sum1[k] / ch1->clustAcc;
      ch2->aveMean[k] = sum2[k] / ch2->clustAcc;
   }

   FreeVector(&gstack, sum1);

}

void CreateChildNodes(CoList *list, RNode *ch1, RNode *ch2, int vSize) 
{
   CoList *c, *c1, *c2;
   float score1, score2;
   int numLeft, numRight;
   Vector mean;

   c1 = c2 = NULL;
   numLeft = numRight = 0;

   for(c = list; c != NULL; c = c->next) {
      mean = c->mp->mean;
      score1 = Distance(mean, ch1->aveMean);
      score2 = Distance(mean, ch2->aveMean);
      if (score1 < score2) {
         if (c1 == NULL)
            c1 = ch1->list = c;
         else {
            c1->next = c;
            c1 = c1->next;
         }
         numLeft += 1;
      }
      else {
         if (c2 == NULL)
            c2 = ch2->list = c;
         else {
            c2->next = c;
            c2 = c2->next;
         }
         numRight += 1;
      }
   }

   ch1->nComponents = numLeft;
   ch2->nComponents = numRight;
   if (c1 != NULL)
      c1->next = NULL;
   if (c2 != NULL)
      c2->next = NULL;

   /* calculate the average mean and variance of the clusters */
   CalcClusterDistribution(ch1, vSize);
   CalcClusterDistribution(ch2, vSize);
   ch1->clusterScore = CalcNodeScore(ch1, vSize);
   ch2->clusterScore = CalcNodeScore(ch2, vSize);

}

void ClusterChildren(RNode *parent, RNode *ch1, RNode *ch2, int vSize) 
{ 
   const float thresh = 1.0e-12;
   int iter=0;
   float oldDistance, newDistance=0.0 ;

   do {
      iter+=1;
      if (iter >= MAX_ITER)
         break;
      CalcDistance(parent->list, ch1, ch2, vSize);
      if (iter == 1)
         oldDistance = ((ch1->clusterScore + ch2->clusterScore) / 
            (ch1->clustAcc + ch2->clustAcc)) + thresh + 1;
      else
         oldDistance = newDistance ;
      newDistance = (ch1->clusterScore + ch2->clusterScore) / 
         (ch1->clustAcc + ch2->clustAcc) ;
      if (trace & T_CLUSTERS) {
         if (iter == 1)
            printf("Iteration %d: Distance = %e\n", iter, newDistance);
         else
            printf("Iteration %d: Distance = %e, Delta = %e\n", iter, newDistance,
                   oldDistance - newDistance);
         fflush(stdout);
      }
   } while ((oldDistance - newDistance) > thresh);

   if (trace & T_CLUSTERS)
      printf("Cluster 1: Score %e, Occ %e\t Cluster 2: Score %e, Occ %e\n",
             ch1->clusterScore,ch1->clustAcc, ch2->clusterScore,ch2->clustAcc);

   CreateChildNodes(parent->list, ch1, ch2, vSize);

}

static void GetTreeVector(RegNode **nVec, RegNode *t) 
{  
  RNode *n;
  int i;

  if (t->numChild>0) {
    for (i=1;i<=t->numChild;i++) 
      GetTreeVector(nVec,t->child[i]);     
  }
  n = GetRNode(t);
  nVec[n->nodeIndex] = t;
}

void PrintRegTree(FILE *f, RegTree *t, int nNodes, char* rname, char* bname) 
{  
   RegNode **tVec;
   RNode *n;
   int i,j;

   tVec = (RegNode **) New(&gstack, nNodes*sizeof(RegNode *));
   --tVec;
   GetTreeVector(tVec, t->root);
   fprintf(f,"~r %s\n",ReWriteString(rname,NULL,DBL_QUOTE));
   fprintf(f,"<BASECLASS>~b %s\n",ReWriteString(bname,NULL,DBL_QUOTE));
   for (i=1;i<=nNodes;i++) {
     n = GetRNode(tVec[i]);
     if (tVec[i]->numChild > 0) {
       fprintf(f, "<NODE> %d 2 ", n->nodeIndex);
       for (j=1;j<=tVec[i]->numChild;j++) {
	 n = GetRNode(tVec[i]->child[j]);
	 n = GetRNode(tVec[n->nodeIndex]);
	 fprintf(f, "%d ", n->nodeIndex);
	 t->numNodes++;
       }
       fprintf(f,"\n");
     } else {
       t->numTNodes++;
       /* always one base class */
       fprintf(f, "<TNODE> %d 1 %d\n",n->nodeIndex,t->numTNodes);
     }
   }
}

void PrintBaseClass(FILE *f, RegTree *t, int nNodes, char *bname) 
{  
  CoList *c;
  RNode *n;
  int i, class=0;
  RegNode **tVec;
  
  tVec = (RegNode **) New(&gstack, nNodes*sizeof(RegNode *));
  --tVec;
  GetTreeVector(tVec, t->root);
  fprintf(f,"~b %s\n",ReWriteString(bname,NULL,DBL_QUOTE));
  fprintf(f,"<MMFIDMASK> %s\n",mmfIdMask);
  fprintf(f,"<PARAMETERS> MIXBASE\n");
  fprintf(f,"<NUMCLASSES> %d\n",t->numTNodes);
  for (i=1;i<=nNodes;i++) {
    /* print out the information from the baseclasses */
    if (tVec[i]->numChild == 0) {
      class++;
      n = GetRNode(tVec[i]);
      fprintf(f,"<CLASS> %d {", class);
      for (c = n->list; c != NULL; c= c->next) {
	if (c!=n->list) fprintf(f,",");
	if (hset->swidth[0] == 1) /* single stream case */
	  fprintf(f,"%s.state[%d].mix[%d]", c->hmmName, c->state, c->mix);
	else 
	  fprintf(f,"%s.state[%d].stream[%d].mix[%d]", c->hmmName, c->state, c->stream, c->mix);

      }
      fprintf(f,"}\n");
    }
  }
}

RegNode *FindBestTerminal(RegNode *t, float *score, int vSize, RegNode *best) 
{
  RNode *n;
  float nodeScore;
  int i;

  if (t != NULL) {
    if (t->numChild>0) {
      for (i=1;i<=t->numChild;i++)
	best = FindBestTerminal(t->child[i], score, vSize, best);
    } else {
      n = GetRNode(t);
      nodeScore = n->clusterScore;
      if ((*score < nodeScore) && (n->nComponents > vSize*3)) {
	*score = nodeScore;
	best = t;
      }
    }
  }
  return(best);
}


int BuildRegClusters(RegNode *rtree, int vSize, int nTerminals, 
                     Boolean nonSpeechNode) 
{
  RegNode *t;
  RNode *ch1, *ch2, *n, *r;   /* temporary children */  
  float score;
  int k, index;

  ch1 = CreateRegTreeNode(NULL, vSize);
  ch2 = CreateRegTreeNode(NULL, vSize);

  /* the artificial first split of speech/non-speech has been made */
  if (nonSpeechNode)
    index = 4;
  else
    index = 2;

  k = index/2;
  for (; k < nTerminals; k++) { 
    
    score = 0.0;
    /* find the best (worst scoring) terminal to split 
       based on score and number of examples */
    t = FindBestTerminal(rtree, &score, vSize, rtree);  
    /* check to see if there are no more nodes to split */
    if ((k > 1 || score == 0.0) && t == rtree) {
      printf("No more nodes to split..\n");
      fflush(stdout);
      break;
    }
    r = GetRNode(t);
    if (trace & T_BID) {
      printf("Splitting Node %d, score %e\n", r->nodeIndex, score);
      fflush(stdout);
    }

    /* copy parent node distribution to children */
    CopyVector(r->aveMean, ch1->aveMean);
    CopyVector(r->aveMean, ch2->aveMean);
    CopyVector(r->aveCovar, ch1->aveCovar);
    CopyVector(r->aveCovar, ch2->aveCovar);
    
    PerturbMean(ch1->aveMean, ch1->aveCovar, 0.2);
    PerturbMean(ch2->aveMean, ch2->aveCovar, -0.2);
    
    ClusterChildren(r, ch1, ch2, vSize);
  
    /* Currently always build a binary regression tree */
    t->numChild = 2;
    t->child = (RegNode **)New(&tmpHeap,3*sizeof(RegNode *));
    /* now create new left child node and store the clusters */
    t->child[1] = (RegNode *) New(&tmpHeap, sizeof(RegNode));
    n = CreateRegTreeNode(ch1->list, vSize);
    t->child[1]->info = n;
    t->child[1]->numChild = 0;
    t->child[1]->child = NULL;

    CopyVector(ch1->aveMean, n->aveMean);
    CopyVector(ch1->aveCovar, n->aveCovar);
    n->clusterScore = ch1->clusterScore;
    n->clustAcc     = ch1->clustAcc;
    n->nComponents  = ch1->nComponents;
    n->nodeIndex    = index++;

    /* now create new right child nodes and store the clusters */
    t->child[2] = (RegNode *) New(&tmpHeap, sizeof(RegNode));
    n = CreateRegTreeNode(ch2->list, vSize);
    t->child[2]->info = n;
    t->child[2]->numChild = 0;
    t->child[2]->child = NULL;

    CopyVector(ch2->aveMean, n->aveMean);
    CopyVector(ch2->aveCovar, n->aveCovar);
    n->clusterScore = ch2->clusterScore;
    n->clustAcc     = ch2->clustAcc;
    n->nComponents  = ch2->nComponents;
    n->nodeIndex    = index++;
  }
  return(index-1);   
} 

void RegClassesCommand(void) 
{
  char ch;
  char buf[MAXSTRLEN], fname[MAXSTRLEN], bname[MAXSTRLEN], tname[MAXSTRLEN];
  char macroname[MAXSTRLEN], fname2[MAXSTRLEN];
  RegTree *regTree;
  char type = 'p';             /* type of items must be p (pdfs) */
  ILink ilist=NULL;            /* list of items that are non-speech sounds */
  int nTerminals, vSize, nNodes;   
  FILE *f;
  Boolean isPipe;

  nTerminals = ChkedInt("Maximum number of terminal nodes is 256",0,256);
  ChkedAlpha("RC regression trees identifier",buf);         
  if (trace & T_BID) {
    printf("\nRC %d %s\n Building regression tree with %d terminals\n",
	   nTerminals, buf, nTerminals);
    printf("Creating regression class tree with ident %s.tree and baseclass %s.base\n",
	   buf, buf);
    fflush(stdout);
  }
  do {
    ch = GetCh(&source);
    if (ch=='\n') break;
  }
  while(ch!=EOF && isspace((int) ch));
  UnGetCh(ch,&source);
  if (ch != '\n')
    PItemList(&ilist,&type,hset,&source,trace&T_ITM);

  regTree = InitRegTree(hset, &vSize, ilist);
  if (ilist != NULL) 
    nNodes = BuildRegClusters(regTree->root, vSize, nTerminals, TRUE);
  else
    nNodes = BuildRegClusters(regTree->root, vSize, nTerminals, FALSE);

  /* now store the baseclasses and regression classes */
  MakeFN(buf,newDir,"tree",fname);
  /* ar527: sometimes an output filter does exist
  if ((f=FOpen(fname,NoOFilter,&isPipe)) == NULL){
    HError(999,"RC: Cannot create output file %s",fname);
  }*/
  if ((f=FOpen(fname,HMMDefOFilter,&isPipe)) == NULL){
    HError(999,"RC: Cannot create output file %s",fname);
  }
  MakeFN(buf,NULL,"tree",tname);
  MakeFN(buf,NULL,"base",bname);
  PrintRegTree(f,regTree,nNodes,tname,bname);
  FClose(f,isPipe);  
  MakeFN(buf,newDir,"base",fname2);
  /* ar527: sometimes an output filter does exist
  if ((f=FOpen(fname2,NoOFilter,&isPipe)) == NULL){
    HError(999,"RC: Cannot create output file %s",bname);
  }*/
  if ((f=FOpen(fname2,HMMDefOFilter,&isPipe)) == NULL){
    HError(999,"RC: Cannot create output file %s",bname);
  }
  PrintBaseClass(f,regTree,nNodes,bname); 
  FClose(f,isPipe);  

  /* Create the macros so they can be stored with the models */
  LoadBaseClass(hset,NameOf(bname,macroname),fname2);  
  LoadRegTree(hset,NameOf(fname,macroname),fname);  

  /* 
     In practice the above commands are wasteful, but useful to check
     macronames. Turn off model storage instead.
  */
  saveHMMSet = FALSE;
}


/* ----------------- InputXForm Command -------------------- */

void InputXFormCommand()
{
   char fn[256], macroname[256];
   InputXForm *xf;

   ChkedAlpha("XF input Xform file name",fn); 
   if (trace & T_BID) {
      printf("\nXF %s\n Setting HMMSet input XForm\n", fn);
      fflush(stdout);
   }
   xf = LoadInputXForm(hset, NameOf(fn,macroname), fn);
   xf->nUse++;
   hset->xf = xf;
}


/* ----------------- ReOrderFeatures Command -------------------- */

void ReOrderFeaturesCommand()
{
   int i,j,vSize,nr,nc;
   char *mac="varFloor1"; /* single stream assumed */
   MLink m;
   LabId id;
   Vector tmpm,tmpv,mean,var;
   Matrix tmpxf,xform;
   IntVec frv;
   HMMScanState hss;

   vSize = hset->vecSize;
   frv = CreateIntVec(&gstack,vSize);
   for (i=1; i<=vSize; i++)
      frv[i] = ChkedInt("Feature index",1,vSize);
   tmpm = CreateVector(&gstack,vSize);
   tmpv = CreateVector(&gstack,vSize);
   NewHMMScan(hset,&hss);
   if (!(hss.isCont || (hss.hset->hsKind == TIEDHS))) {
         HError(2690,"ReOrderFeaturesCommand: only continuous feature systems supported");
   }
   while(GoNextMix(&hss,FALSE)) {
      if (hss.mp->ckind != DIAGC)
         HError(2690,"ReOrderFeaturesCommand: CovKind not DIAGC");
      mean = hss.mp->mean;
      var = hss.mp->cov.var;
      for (i=1; i<=vSize; i++) {
	 tmpm[i] = mean[frv[i]];
	 tmpv[i] = var[frv[i]];
      }
      CopyVector(tmpm,mean);
      CopyVector(tmpv,var);
   }
   EndHMMScan(&hss);
   id = GetLabId(mac,FALSE);
   if (id != NULL  && (m=FindMacroName(hset,'v',id)) != NULL) {
      var = (Vector)m->structure;
      for (i=1; i<=vSize; i++)
	 tmpv[i] = var[frv[i]];
      CopyVector(tmpv,var);
   } else
      HError(2623,"ReOrderFeaturesCommand: variance floor macro %s not found",mac);
   if (hset->xf != NULL) {
      xform = hset->xf->xform->xform[1];
      nc = NumCols(xform);
      nr = hset->xf->xform->vecSize;
      if (nr != vSize)
	 HError(2623,"ReOrderFeaturesCommand: Num rows %d in InputXForm does not match the vecsize %d",nr,vSize);
      tmpxf = CreateMatrix(&gstack,nr,nc);
      for (i=1; i<=nr; i++)
	 for (j=1; j<=nc; j++)
	    tmpxf[i][j] = xform[frv[i]][j];
      CopyMatrix(tmpxf,xform);
      if (trace & T_BID) {
	 printf(" RF: Transform re-ordered according to %d, %d, %d, ...\n",frv[1],frv[2],frv[3]);
	 fflush(stdout);
      }
   }
   if (trace & T_BID) {
      printf(" RF: Parameters re-ordered according to %d, %d, %d, ...\n",frv[1],frv[2],frv[3]);
      fflush(stdout);
   }
   FreeIntVec(&gstack,frv);
}


/* -------------------- Initialisation --------------------- */
 
void Initialise(char *hmmListFn)
{
  
   CreateHeap(&questHeap,"Question Heap",MSTAK,1,1.0,8000,16000);
   CreateHeap(&tmpHeap,"Temporary Heap",MSTAK,1,1.0,40000,400000);

   if(MakeHMMSet(&hSet,hmmListFn)<SUCCESS)
      HError(2628,"Initialise: MakeHMMSet failed");
   if (noAlias) ZapAliases(); 
   if(LoadHMMSet(&hSet,hmmDir,hmmExt)<SUCCESS)
      HError(2628,"Initialise: LoadHMMSet failed");

   maxStates = MaxStatesInSet(hset);
   maxMixes = MaxMixInSet(hset);

   if (trace != 0) {
      printf("HHEd\n");
      printf(" %d/%d Models Loaded [%d states max, %d mixes max]\n",
             hset->numLogHMM,hset->numPhyHMM,maxStates,maxMixes);
      fflush(stdout);
   }

}

/* -------------------- Top Level of Editing ---------------- */


static int nCmds = 56;	/* cz277 */

static char *cmdmap[] = {"AT","RT","SS","CL","CO","JO","MU","TI","UF","NC",
                         "TC","UT","MT","SH","SU","SW","SK",
                         "RC",
                         "RO","RM","RN","RP",
                         "LS","QS","TB","TR","AU","GQ","MD","ST","LT",
                         "MM","DP","HK","FC","FA","FV","XF","PS","PR", 
                         "AV", "SV", "AM", "SM", "AF", "IL", "CF", "CD", "EL", "CA", "EF", "CP", "DL", "CM", "CH", "SL", "" };

typedef enum           { AT=1, RT , SS , CL , CO , JO , MU , TI , UF , NC ,
                         TC , UT , MT , SH , SU , SW , SK ,
                         RC ,
                         RO , RM , RN , RP ,
                         LS , QS , TB , TR , AU , GQ , MD , ST , LT ,
                         MM , DP , HK , FC , FA , FV, XF, PS, PR, 
                         AV , SV , AM , SM , AF , IL, CF, CD, EL, CA, EF, CP, DL, CM, CH, SL }
cmdNum;

/* CmdIndex: return index 1..N of given command */
int CmdIndex(char *s)
{
   int i;
   
   for (i=1; i<=nCmds; i++) {
      if (strcmp(cmdmap[i-1],s) == 0) 
         return i;
   }
   return 0;
}

/* DoEdit: edit the loaded HMM Defs using given command file */
void DoEdit(char * editFn)
{
   int prevCommand = 0,c1,c2;
   char cmds[] = "  ";
  
   /* Open the Edit Script File */
   if(InitSource(editFn,&source,NoFilter)<SUCCESS)
      HError(2610,"DoEdit: Can't open file %s", editFn);
      
   /* Apply each Command */
   for (;;) {
      SkipWhiteSpace(&source);
      if( (c1 = GetCh(&source)) == EOF) break;
      if( (c2 = GetCh(&source)) == EOF) break;
      cmds[0] = c1; cmds[1] = c2;
      if (!isspace(GetCh(&source))) 
         HError(2650,"DoEdit: White space expected after command %s",cmds);
      thisCommand = CmdIndex(cmds);
      if ((thisCommand != lastCommand && prevCommand != TR &&
           thisCommand != SH && thisCommand != ST) || thisCommand == TR)
         trace = cmdTrace;
      switch (thisCommand) {
      case AT: EditTransMat(TRUE); break;
      case RT: EditTransMat(FALSE); break;
      case SS: SplitStreamCommand(FALSE); break;
      case CL: CloneCommand(); break;
      case CM: ConcatenateCommand(); break;
      case CO: CompactCommand(); break;
      case JO: JoinSizeCommand(); break;
      case MU: MixUpCommand(); break;
      case MD: MixDownCommand(); break;
      case TI: TieCommand(); break;
      case NC: ClusterCommand(TRUE); break;
      case TC: ClusterCommand(FALSE); break;
      case UT: UntieCommand(); break;
      case MT: MakeTriCommand(); break;
      case SH: ShowHMMSet(); break;
      case SU: SplitStreamCommand(TRUE); break;
      case SW: SetStreamWidthCommand(); break;
      case SK: SetSampKindCommand(); break;
      case HK: SetHSetKindCommand(); break;
      case RC: RegClassesCommand(); break;
      case RM: RemMeansCommand(); break;
      case RN: RenameHMMSetIdCommand(); break;
      case RP: ReOrderFeaturesCommand(); break;
      case RO: RemOutliersCommand(); break;
      case LS: LoadStatsCommand(); break;
      case QS: QuestionCommand(); break;
      case TB: TreeBuildCommand(); break;
      case TR: SetTraceCommand(); break;
      case AU: AddUnseenCommand(); break;
      case ST: ShowTreesCommand(); break;
      case LT: LoadTreesCommand(); break;
      case MM: MakeIntoMacrosCommand(); break;
      case DP: DuplicateCommand(); break;
      case XF: InputXFormCommand(); break;
      case UF: UseCommand(); break;
      case FA: FloorAverageCommand(); break;
      case FC: FullCovarCommand(); break;
      case FV: FloorVectorCommand(); break;
      case PS: PowerSizeCommand(); break;
      case PR: ProjectCommand(); break;
      case AV: AddOneNVecBundle("AddOneNVecBundle", NULL, hset, FALSE); break;
      case SV: SetOneNVecBundle(hset); break;
      case AM: AddOneNMatBundle("AddOneNMatBundle", NULL, hset, FALSE); break;
      case SM: SetOneNMatBundle(hset); break;
      case AF: AddOneFeaMix("AddOneFeaMix", NULL, FALSE); break;   	
      case IL: InsertOneANNLayer(); break;			/* done */
      case CF: ChangeOneANNLayerInput(); break;			/* done */
      case CD: ChangeOneANNLayerDims(); break;			/* done */
      case EL: EraseOneANNLayer(); break;			/* done */
      case DL: DeleteOneANNLayer(); break;			/* done */
      case CA: ChangeOneANNLayerActFun(); break;		/* done */ 
      case EF: ExtendOneFeaMix(); break;			/* done */
      case CP: CopyANNParameters(); break;			/* done */
      /*case RS: ReplaceHMMState(); break;*/			/* not tested yet */
      case CH: ConnectANNtoHMMs(); break;
      case SL: ShowANNLayerStats(); break;			/* done */
      default: 
         HError(2650,"DoEdit: Command %s not recognised",cmds);
      }
      if (thisCommand != TR && thisCommand != SH && thisCommand != ST)
         lastCommand = thisCommand;
      prevCommand = thisCommand;
   }
   CloseSource(&source);

   if (saveHMMSet) {
      /* Save the Edited HMM Files */
      if (trace & T_BID) {
         printf("\nSaving new HMM files ...\n");
         fflush(stdout);
      }
      /*FixAllGConsts(hset); 
      badGC=FALSE;
      PurgeMacros(hset);*/
      /* cz277 - ANN */
      if (hset->annSet != NULL) {
          CheckANNConsistency(hset);
          FindANNCycles(hset);
          ResetDrvContext(hset);
          SetDrvContext(hset);
          InitXYBatch(hset);
      }
      FixAllGConsts(hset);         /* in case any bad gConsts around */
      badGC=FALSE;
      PurgeMacros(hset);
      if (mmfFn!=NULL)
         SaveInOneFile(hset,mmfFn);
      if (nmfFn != NULL) {
          if (SaveANNDefs(hset, nmfFn, inBinary) < SUCCESS) 
              HError(9999, "DoEdit: SaveANNDefs failed");
      }
      /* cz277 - 1007 */
      if (newDir != NULL || mmfFn != NULL) {
          if (SaveHMMSet(&hSet, newDir, newExt, NULL, inBinary) < SUCCESS)
              HError(2611, "DoEdit: SaveHMMSet failed");
      }
   }
   /* reset auxSet when exit */
   if (auxSet != NULL && hset != auxSet)
       ResetHMMSet(auxSet);

   if (trace & T_BID) {
      printf("Edit Complete\n");
      fflush(stdout);
   }

}


/* ----------------------------------------------------------- */
/*                        END:  HHEd.c                         */
/* ----------------------------------------------------------- */

