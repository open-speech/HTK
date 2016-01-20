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
/*         2002-2003  Cambridge University                     */
/*                    Engineering Department                   */
/*                                                             */
/*   Use of this software is governed by a License Agreement   */
/*    ** See the file License for the Conditions of Use  **    */
/*    **     This banner notice must not be removed      **    */
/*                                                             */
/* ----------------------------------------------------------- */
/*         File: HLVNet.c Network handling for HTK LV Decoder  */
/* ----------------------------------------------------------- */

char *hlvnet_version = "!HVER!HLVNet:   3.4.1 [GE 12/03/09]";
char *hlvnet_vc_id = "$Id: HLVNet.c,v 1.1.1.1 2006/10/11 09:54:55 jal58 Exp $";

#include "HShell.h"
#include "HMem.h"
#include "HMath.h"
#include "HWave.h"
#include "HLabel.h"
#include "HAudio.h"
#include "HParm.h"
#include "HDict.h"
#include "HModel.h"
/* all the above are necessary just to include HModel.h */
/* #### sort out  include file dependencies! */

#include "config.h"

#include "HLVLM.h"      /* for LMId */
#include "HLVNet.h"

#include <assert.h>


#define LIST_BLOCKSIZE 70

#undef DEBUG_LABEL_NET

/* ----------------------------- Trace Flags ------------------------- */

#define T_TOP 0001         /* top level Trace  */
#define T_NET 0002         /* network summary*/
#define T_NETCON 0004    /* network construction  */

static int trace=0;
static ConfParam *cParm[MAXGLOBS];      /* config parameters */
static int nParm = 0;

/* -------------------------- Global Variables etc ---------------------- */

static MemHeap tnetHeap;                /* used for temporary data in net creation */

/* --------------------------- Initialisation ---------------------- */

/* EXPORT->InitLVNet: register module & set configuration parameters */
void InitLVNet(void)
{
   int i;
   
   Register(hlvnet_version,hlvnet_vc_id);
   nParm = GetConfig("HLVNET", TRUE, cParm, MAXGLOBS);
   if (nParm>0){
      if (GetConfInt(cParm,nParm,"TRACE",&i)) trace = i;
   }

   CreateHeap (&tnetHeap, "Net temp heap", MSTAK, 1, 0,100000, 800000);
}

/* --------------------------- the real code  ---------------------- */

TLexNode *NewTLexNodeMod (MemHeap *heap, TLexNet *net, int layerId, HLink hmm)
{
   TLexNode *ln;

   assert (heap);
   ln = (TLexNode *) New (heap, sizeof (TLexNode));
   ln->nlinks = 0;
   ln->link = NULL;
   ln->ln = NULL;
   ln->loWE = ln->hiWE = ln->lmlaIdx = 0;

   ln->chain = net->root;
   net->root = ln;

   ln->layerId = layerId;
   ++net->nNodesLayer[layerId];

   ln->pron = NULL;
   ln->hmm = hmm;
   ln->type = LN_MODEL;

   return ln;
}

TLexNode *NewTLexNodeWe (MemHeap *heap, TLexNet *net, int layerId, Pron pron)
{
   TLexNode *ln;

   assert (heap);
   ln = (TLexNode *) New (heap, sizeof (TLexNode));
   ln->nlinks = 0;
   ln->link = NULL;
   ln->ln = NULL;
   ln->loWE = ln->hiWE = ln->lmlaIdx = 0;

   ln->chain = net->root;
   net->root = ln;

   ln->layerId = layerId;
   ++net->nNodesLayer[layerId];

   ln->hmm = NULL;
   ln->pron = pron;
   ln->type = LN_WORDEND;

   return ln;
}

TLexNode *NewTLexNodeCon (MemHeap *heap, TLexNet *net, int layerId, LabId lc, LabId rc)
{
   TLexNode *ln;

   assert (heap);
   ln = (TLexNode *) New (heap, sizeof (TLexNode));
   ln->nlinks = 0;
   ln->link = NULL;
   ln->ln = NULL;
   ln->loWE = ln->hiWE = ln->lmlaIdx = 0;

   ln->chain = net->root;
   net->root = ln;

   ln->layerId = layerId;
   ++net->nNodesLayer[layerId];

   ln->type = LN_CON;
   ln->lc = lc;
   ln->rc = rc;

   return ln;
}



/* from HNet.c */
/* Binary search to find elem in n element array */
static int BSearch (Ptr elem, int n, Ptr *array)
{
   int l,u,c;

   l=1;u=n;
   while(l<=u) {
      c=(l+u)/2;
      if (array[c]==elem) return(c);
      else if (array[c]<elem) l=c+1; 
      else u=c-1;
   }
   return(0);
}

/* from HNet.c */
/* Binary search to find elem in n element array */
static int BAddSearch (Ptr elem, int *np, Ptr **ap)
{
   Ptr *array;
   int l,u,c;

   array=*ap;
   l=1;u=*np;
   while(l<=u) {
      c=(l+u)/2;
      if (array[c]==elem) return(c);
      else if (array[c]<elem) l=c+1; 
      else u=c-1;
   }
   if (((*np+1) % LIST_BLOCKSIZE)==0) {
      Ptr *newId;

      newId=(Ptr *) New(&gcheap,sizeof(Ptr)*(*np+1+LIST_BLOCKSIZE));
      for (c=1;c<=*np;c++)
         newId[c]=array[c];
      Dispose(&gcheap,array);
      *ap=array=newId;
   }
   for (c=1;c<=*np;c++)
      if (elem<array[c]) break;
   for (u=(*np)++;u>=c;u--)
      array[u+1]=array[u];
   array[c]=elem;
   return(c);
}

#define HASH1(p1,s)  (((int)(p1))%(s))
#define HASH2(p1,p2,s)  (( HASH1(p1,s) + HASH1(p2,s)) % (s))

TLexConNode *FindAddTLCN (MemHeap *heap, TLexNet *net, int layerId, int *n, TLexConNode *lcnHashTab[], LabId lc, LabId rc)
{
   TLexConNode *lcn;
   unsigned int hash;

   hash = HASH2(lc, rc, LEX_CON_HASH_SIZE);

   lcn = lcnHashTab[hash];
   while (lcn && !((lcn->lc == lc) && (lcn->rc == rc)))
      lcn = lcn->next;

#if 0
   for ( ; lcn; lcn = lcn->next) {              /* traverse chain */
      if ((lcn->lc == lc) && (lcn->rc == rc)) 
         break;
   }
#endif

   if (!lcn) {          /* not found */
      lcn = (TLexConNode *) NewTLexNodeCon (heap, net, layerId, lc, rc);

      lcn->next = lcnHashTab[hash];
      lcnHashTab[hash] = lcn;
      ++(*n);
   }
   return lcn;
}

/* add TLexNode to hashtable
   only hmm is used for hashing and comparison, 
   i.e. no two node with same hmm, but different types are possible
*/
TLexNode *FindAddTLexNode (MemHeap *heap, TLexNet *net, int layerId, int *n, TLexNode *lnHashTab[], LexNodeType type , HLink hmm)
{
   TLexNode *ln;
   unsigned int hash;

   hash = HASH1 (hmm, LEX_MOD_HASH_SIZE);

   ln = lnHashTab[hash];
   while (ln && !((ln->hmm == hmm)))
      ln = ln->next;

   if (!ln) {           /* not found */
      if (!heap)
         return NULL;
      ln = NewTLexNodeMod (heap, net, layerId, hmm);
      ln->type = type;          /*#### do we ever use anything but LN_MODEL? */

      ln->next = lnHashTab[hash];
      lnHashTab[hash] = ln;
      ++(*n);
   }
   return ln;
}



#define BUFLEN 100

HLink FindTriphone (HMMSet *hset, LabId a, LabId b, LabId c)
{
   char buf[BUFLEN];
   LabId triLabId;
   MLink triMLink;
   
   /*   if (snprintf (buf, BUFLEN, "%s-%s+%s", a->name, b->name, c->name) == -1)  */
   if (sprintf (buf, "%s-%s+%s", a->name, b->name, c->name) == -1) 
      HError (9999, "HLVNet: model names too long");
   
   triLabId = GetLabId (buf, FALSE);
   if (!triLabId)
      HError (9999, "HLVNet: no model label for phone (%s)", buf);
   
   triMLink = FindMacroName (hset, 'l', triLabId);
   if (!triMLink)
      HError (9999, "HLVNet: no model macro for phone (%s)", buf);
   
   return ((HLink) triMLink->structure);
}


/*  scan vocabulary pronunciations for phone sets A, B, AB, YZ
    sets A and B are small (~50) and stored in arays
    sets AB and YZ are larger (~750) and are stored in hash tables.
*/
void CollectPhoneStats (MemHeap *heap, TLexNet *net)
{
   int i, j;
   Word word;
   Pron pron;
   LabId sil, z, p;

   for (i = 0; i < VHASHSIZE; i++)
      for (word = net->voc->wtab[i]; word ; word = word->next)
         if (word->aux == (Ptr) 1 && (word->wordName != net->startId && word->wordName != net->endId))
            for (pron = word->pron; pron ; pron = pron->next) {
               if (pron->nphones < 1)
                  HError (9999, "CollectPhoneStats: pronunciation of '%s' is empty", word->wordName->name);

               if (pron->aux == (Ptr) 1) {
                  BAddSearch ((Ptr) pron->phones[0], &net->nlexA, ((Ptr **) &net->lexA));
                  BAddSearch ((Ptr) pron->phones[pron->nphones - 1], &net->nlexZ, ((Ptr **) &net->lexZ));

                  if (pron->nphones >= 2) {
                     /* add to AB and YZ hashes */
                     FindAddTLCN (heap, net, LAYER_AB, &net->nlexAB, net->lexABhash, 
                                  pron->phones[0], pron->phones[1]);
                     FindAddTLCN (heap, net, LAYER_YZ, &net->nlexYZ, net->lexYZhash, 
                                  pron->phones[pron->nphones - 2], pron->phones[pron->nphones - 1]);
                  }
                  else {
                     BAddSearch ((Ptr) pron->phones[0], &net->nlexP, ((Ptr **) &net->lexP));
                     /*#### need to add ZP node in YZ list for single phone word */
                     /*#### here only collect phones used in single phone words */
                  }
               }
            }
   
   /* add 'sil' to A and Z lists */
   sil = GetLabId ("sil", FALSE);
   if (!sil)
      HError (9999, "cannot find 'sil' model.");

   BAddSearch ((Ptr) sil, &net->nlexA, ((Ptr **) &net->lexA));
   BAddSearch ((Ptr) sil, &net->nlexZ, ((Ptr **) &net->lexZ));


   /* for each phone P occuring in a single phone word 
        for each word end context Z
          add node ZP to SA and YZ layers
   */
   if (trace & T_NETCON)
      printf ("adding extra nodes for single phone words:\n");

   for (i = 1; i <= net->nlexP; ++i) {
      if (trace & T_NETCON)
         printf ("P='%s'  ", net->lexP[i]->name);
      p = net->lexP[i];
      for (j = 1; j <= net->nlexZ; ++j) {
         if (trace & T_NETCON)
            printf ("z='%s' ", net->lexZ[j]->name);
         z = net->lexZ[j];

         FindAddTLCN (heap, net, LAYER_SA, &net->nlexSA, net->lexSAhash, z, p);
         FindAddTLCN (heap, net, LAYER_YZ, &net->nlexYZ, net->lexYZhash, z, p);
      }
      if (trace & T_NETCON)
         printf ("\n");
   }

   if (trace & T_NETCON) { 
      TLexNode *lcn;

      /* debug: print lexA, lexZ, lexAB & lexYZ*/
      printf ("nlexA = %d   ", net->nlexA);
      for (i = 1; i <= net->nlexA; ++i)
         printf ("%s ", net->lexA[i]->name);
      printf ("\n");
      
      printf ("nlexZ = %d   ", net->nlexZ);
      for (i = 1; i <= net->nlexZ; ++i)
         printf ("%s ", net->lexZ[i]->name);
      printf ("\n");
      
      printf ("nlexAB = %d   ", net->nlexAB);
      for (i = 0; i < LEX_CON_HASH_SIZE; ++i)
         for (lcn = net->lexABhash[i]; lcn; lcn = lcn->next)
            printf ("%s-%s  ", lcn->lc->name, lcn->rc->name);
      printf ("\n");
      
      printf ("nlexYZ = %d   ", net->nlexYZ);
      for (i = 0; i < LEX_CON_HASH_SIZE; ++i)
         for (lcn = net->lexYZhash[i]; lcn; lcn = lcn->next)
            printf ("%s-%s  ", lcn->lc->name, lcn->rc->name);
      printf ("\n");
   }

}

/* add link from TLexNode start to end, if it doesn't exist already 
*/
void AddLink (MemHeap *heap, TLexNode *start, TLexNode *end)
{
   TLexLink *ll;

   assert (start);
   assert (end);
   /* check if link exists already */
   /*#### GE: make check optional? */
   for (ll = start->link; ll; ll = ll->next)
      if (ll->end == end)
         break;

   if (!ll) {
      ll = (TLexLink *) New (heap, sizeof (TLexLink));
      ll->start = start;
      ll->end = end;
      
      ++start->nlinks;
      ll->next = start->link;
      start->link = ll;
   }
}

/* Find link from node ln to a node corresponding to model hmm
 */
TLexLink *FindHMMLink (TLexNode *ln, HLink hmm)
{
   TLexLink *ll;

   for (ll = ln->link; ll; ll = ll->next)
      if (ll->end->hmm == hmm)
         break;

   return ll;
}

/* Create initial phone (A) layer of z-a+b nodes 
*/
void CreateAnodes (MemHeap *heap, TLexNet *net)
{     
   int i,j;
   TLexConNode *lcnAB, *lcnZA;
   TLexNode *lnZAB;
   LabId z, a, b;
   HLink hmm;
   
   /* create word initial (A) nodes */
   /*   for each AB node */
   for (i = 0; i < LEX_CON_HASH_SIZE; ++i) 
      for (lcnAB = net->lexABhash[i]; lcnAB; lcnAB = lcnAB->next) {
         a = lcnAB->lc;
         b = lcnAB->rc;
         /* for each Z node */
         for (j = 1; j <= net->nlexZ; ++j) {
            z = net->lexZ[j];

            /* create Node z-a+b */
            hmm = FindTriphone (net->hset, z, a, b);
            lnZAB = FindAddTLexNode (heap, net, LAYER_A, &net->nNodeA, net->nodeAhash, LN_MODEL, hmm);

#ifdef DEBUG_LABEL_NET          /*#### expensive name lookup for dot graph! */
            lnZAB->lc = FindMacroStruct (net->hset, 'h', hmm)->id;
#else 
            lnZAB->lc = NULL;
#endif 
            /* connect to ZA node */
            lcnZA = FindAddTLCN (heap, net, LAYER_SA, &net->nlexSA, net->lexSAhash, z, a);
            AddLink (heap, lcnZA, lnZAB);

            /* connect to AB node */
            AddLink (heap, lnZAB, lcnAB);
         }
      }

   if (trace & T_NETCON) {
      TLexNode *lcn;
      printf ("nlexSA = %d   ", net->nlexSA);
      for (i = 0; i < LEX_CON_HASH_SIZE; ++i)
         for (lcn = net->lexSAhash[i]; lcn; lcn = lcn->next)
            printf ("%s-%s  ", lcn->lc->name, lcn->rc->name);
      printf ("\n");
   }

   if (trace & T_NETCON) {
      printf ("nNodeA = %d   ", net->nNodeA);
      for (i = 0; i < LEX_MOD_HASH_SIZE; ++i)
         for (lnZAB = net->nodeAhash[i]; lnZAB; lnZAB = lnZAB->next)
            printf ("zab_hmm %p nlinks %d\n", lnZAB->hmm, lnZAB->nlinks);
   }
}


/* Create final phone (Z) layer of y-z+a nodes 
*/
void CreateZnodes (MemHeap *heap, TLexNet *net)
{
   int i,j;
   TLexConNode *lcnYZ, *lcnZA;
   TLexNode *lnYZA;
   LabId y, z, a;
   HLink hmm;


   /* create word final (Z) nodes */
   /*   for each YZ node */
   for (i = 0; i < LEX_CON_HASH_SIZE; ++i) 
      for (lcnYZ = net->lexYZhash[i]; lcnYZ; lcnYZ = lcnYZ->next) {
         y = lcnYZ->lc;
         z = lcnYZ->rc;
         /* for each A phone */
         for (j = 1; j <= net->nlexA; ++j) {
            a = net->lexA[j];
            /* create Node y-z+a */
            hmm = FindTriphone (net->hset, y, z, a);
            lnYZA = FindAddTLexNode (heap, net, LAYER_Z, &net->nNodeZ, net->nodeZhash, LN_MODEL, hmm);

#ifdef DEBUG_LABEL_NET          /*#### expensive name lookup for dot graph! */
            lnYZA->lc = FindMacroStruct (net->hset, 'h', hmm)->id;
#else 
            lnYZA->lc = NULL;
#endif 
            /* connect to YZ node */
            AddLink (heap, lcnYZ, lnYZA);
            /* connect to ZS node */
            lcnZA = FindAddTLCN (heap, net, LAYER_ZS, &net->nlexZS, net->lexZShash, z, a);
            AddLink (heap, lnYZA, lcnZA);
         }
      }
   if (trace & T_NETCON) {
      TLexNode *lcn;
      printf ("nlexZS = %d   ", net->nlexZS);
      for (i = 0; i < LEX_CON_HASH_SIZE; ++i)
         for (lcn = net->lexZShash[i]; lcn; lcn = lcn->next)
            printf ("%s-%s  ", lcn->lc->name, lcn->rc->name);
      printf ("\n");
   }

   if (trace & T_NETCON) {
      printf ("nNodeZ = %d   ", net->nNodeZ);
      for (i = 0; i < LEX_MOD_HASH_SIZE; ++i)
         for (lnYZA = net->nodeZhash[i]; lnYZA; lnYZA = lnYZA->next)
            printf ("yza_hmm %p nlinks %d\n", lnYZA->hmm, lnYZA->nlinks);
   }
}

/* get HLink from LabId
   #### GE: this should really be in HModel
 */
HLink FindHMM (HMMSet *hset, LabId id)
{
   MLink ml;
   
   assert (id);
   ml = FindMacroName (hset, 'l', id);
   if (!ml)
      HError (9999, "cannot find model for label '%s'", id->name);
   assert (ml->structure);
   return ((HLink) ml->structure);
}

/* Create sil/sp model nodes between ZS and SA nodes 
 */
void CreateSILnodes (MemHeap *heap, TLexNet *net)
{
   int i, j;
   TLexNode *lcnZS, *lcnSA, *ln;
   LabId z, s, a, sil, sp;
   HLink hmmSIL, hmmSP;

   /* find sil & sp models */
   sil = GetLabId ("sil", FALSE);
   if (!sil)
      HError (9999, "cannot find 'sil' model.");
   hmmSIL = FindHMM (net->hset, sil);
   sp = GetLabId ("sp", FALSE);
   if (!sp)
      HError (9999, "cannot find 'sp' model.");
   hmmSP = FindHMM (net->hset, sp);

   for (i = 0; i < LEX_CON_HASH_SIZE; ++i)
      for (lcnZS = net->lexZShash[i]; lcnZS; lcnZS = lcnZS->next) {
         z = lcnZS->lc;
         s = lcnZS->rc;
         /*         printf ("ZS node %p %s-%s i %d\n", lcnZS, z->name, s->name, i); */
         ln = NewTLexNodeMod (heap, net, LAYER_SIL, (s==sil) ? hmmSIL : hmmSP);

         ln->next = net->nodeSIL;
         net->nodeSIL = ln;
         ++net->nNodeSIL;
         
         AddLink (heap, lcnZS, ln);
         if (s == sil) {        /* sil node */
            ln->lc = sil;

            /* connect sil node to all sil-A nodes */
            for (j = 1; j <= net->nlexA; ++j) {
               a = net->lexA[j];
               if (a != sil) {          /* list of A contxts includes sil! */
                  lcnSA = FindAddTLCN (heap, net, LAYER_SA, &net->nlexSA, net->lexSAhash, s, a);
                  /*  printf ("  sil SA node %s-%s\n", lcnSA->lc->name, lcnSA->rc->name); */
                  AddLink (heap, ln, lcnSA);
               }
            }
         }
         else {         /* sp node */
            ln->lc = sp;

            /* connect sp node to corresponding SA node (SA==ZS) */
            lcnSA = FindAddTLCN (NULL, net, LAYER_SA, &net->nlexSA, net->lexSAhash, z, s);
            /*  printf ("  sp  SA node %s-%s\n", lcnSA->lc->name, lcnSA->rc->name); */
            AddLink (heap, ln, lcnSA);
         }
      }
}


/* Handle1PhonePron

     create network structure for single phone words
     The actual phones are put in the Z layer.
     duplicate wordend for each preceeding context Z

     ZP ----> WE ---> ZP ----> z-p+  ---> PS
     SA               YZ       y-z+a ---> ZS    

*/
void Handle1PhonePron (MemHeap *heap, TLexNet *net, Pron pron)
{
   LabId p;            /* the single phone */
   LabId z;
   int i;
   TLexNode *zp_sa, *zp_yz, *we;
   PronId pronid;
   int lmlaIdx;

   assert (pron->nphones == 1);

   if (trace & T_NETCON)
      printf ("handle single phone pron for '%s'\n", pron->word->wordName->name);

   p = pron->phones[0];

   pronid = ++net->nPronIds;
   lmlaIdx = ++net->lmlaCount;

   pron->aux = (Ptr) (int) pronid;

   /* for each word end phone Z
        create a WE node and link to appropriate nodes in SA & YZ layers
        zp_sa ---> WE ---> zp_yz
   */
   for (i = 1; i <= net->nlexZ; ++i) {
      if (trace & T_NETCON)
         printf ("Z node '%s' P node '%s'\n", net->lexZ[i]->name, p->name);
      z = net->lexZ[i];

      zp_sa = FindAddTLCN (NULL, net, LAYER_SA, &net->nlexSA, net->lexSAhash, z, p);
      zp_yz = FindAddTLCN (NULL, net, LAYER_YZ, &net->nlexYZ, net->lexYZhash, z, p);

      /* create word end node */
      we = NewTLexNodeWe (heap, net, LAYER_WE, pron);
      
      we->loWE = we->hiWE = pronid;
      we->lmlaIdx = lmlaIdx;

      we->lc = pron->word->wordName;
      we->next = net->nodeBY; /*#### link into BY list? */
      net->nodeBY = we;
      ++net->nNodeBY;

      AddLink (heap, zp_sa, we);
      AddLink (heap, we, zp_yz);
   }

}

/* CreateBYnodes

     Create the forward tree from phone B to phone Y, 
     linking the appropriate AB and YZ nodes.
*/
void CreateBYnodes (MemHeap *heap, TLexNet *net)
{
   int i,p;
   HLink hmm;
   TLexNode *prevln, *ln;
   TLexLink *ll;
   Word word;
   Pron pron;
   int nshared = 0;

   /* Create tree B -- Y */

   /* for each pron with 2 or more phones */
   for (i = 0; i < VHASHSIZE; i++)
      for (word = net->voc->wtab[i]; word ; word = word->next)
         if (word->aux == (Ptr) 1 && (word->wordName != net->startId && word->wordName != net->endId))
            for (pron = word->pron; pron ; pron = pron->next) {
               if (pron->aux == (Ptr) 1) {
                  if (pron->nphones >= 2) {
               
                     /* find AB node */
                     prevln = FindAddTLCN (NULL, net, LAYER_AB, &net->nlexAB, net->lexABhash, 
                                           pron->phones[0], pron->phones[1]);
               
                     /* add models for phones B -- Y */
                     for (p = 1; p < pron->nphones - 1; ++p) {
                        hmm = FindTriphone (net->hset, pron->phones[p-1], pron->phones[p], pron->phones[p+1]);
                        /* search in prevln's successors */
                        
                        ll = FindHMMLink (prevln, hmm);
                        if (ll) {
                           prevln = ll->end;
                           ++nshared;
                        }
                        else {             /* model not found -> create a new one */
                           ln = NewTLexNodeMod (heap, net, LAYER_BY, hmm);
                           
                           ln->next = net->nodeBY;
                           net->nodeBY = ln;
                           ++net->nNodeBY;
                           
                           AddLink (heap, prevln, ln);   /* guaranteed to be a new link! */
                           
#ifdef DEBUG_LABEL_NET          /*#### expensive name lookup for dot graph! */
                           ln->lc = FindMacroStruct (net->hset, 'h', hmm)->id;
#else 
                           ln->lc = NULL;
#endif 
                           prevln = ln;
                        }
                     }
                     /* create word end node */
                     ln = NewTLexNodeWe (heap, net, LAYER_WE, pron);
                     
                     ln->lc = pron->word->wordName;
                     ln->next = net->nodeBY;
                     net->nodeBY = ln;
                     ++net->nNodeBY;
                     
                     AddLink (heap, prevln, ln);   /* guaranteed to be a NEW link! */
                     prevln = ln;
                     
                     /* find YZ node */
                     ln = FindAddTLCN (NULL, net, LAYER_YZ, &net->nlexYZ, net->lexYZhash, 
                                       pron->phones[pron->nphones - 2], pron->phones[pron->nphones - 1]);
                     /* find LexNode and connect prevln to it */
                     AddLink (heap, prevln, ln);
                  }
                  else {
                     /*  printf ("one- or two-phone word (%s) ignored\n", word->wordName->name); */
                  Handle1PhonePron (heap, net, pron);
                  }
               }
            }
   if (trace & T_NETCON)
      printf ("nodes shared in prefix tree: %d\n", nshared);
}


/* CreateBoundary

     create one model and one word end node for a given boundary (start or end) labid.
     return WE node with lnWE->next pointin to model node
*/
TLexNode *CreateBoundary (MemHeap *heap, TLexNet *tnet, LabId labid, int modLayer, int weLayer)
{
   Word w;
   TLexNode *lnMod, *lnWe;

   w = GetWord (tnet->voc, labid, FALSE);
   if (!w)
      HError (9999, "HLVNet: cannot find START/ENDWORD '%s'", labid->name);
   if (w->nprons != 1 || w->pron->nphones != 1)
      HError (9999, "HLVNet: only one pronuinciation with one model allowed for START/ENDWORD '%s'", 
              labid->name);
   
   /* create model node */
   lnMod = NewTLexNodeMod (heap, tnet, modLayer, FindHMM (tnet->hset, w->pron->phones[0]));

   lnMod->next = NULL;
   lnMod->lc = w->pron->phones[0];

   /* create word end node */
   lnWe = NewTLexNodeWe (heap, tnet, weLayer, w->pron);

   lnWe->lc = w->pron->word->wordName;
   lnWe->next = lnMod;
   
   return (lnWe);
}

/* CreateStartEnd 

     create model and word end nodes for STARTWORD and ENDWORD and connect them 
     to the appropriate places.

*/
void CreateStartEnd (MemHeap *heap, TLexNet *tnet)
{
   TLexNode *lnWe, *lnMod, *lnTime;
   TLexNode *ln;
   int i;

   /* STARTWORD */
   /* the two layers must be before SA; do NOT use ZS (spSkipLayer) */   
   lnWe = CreateBoundary (heap, tnet, tnet->startId, LAYER_Z, LAYER_SIL); 
   lnMod = lnWe->next;
   lnWe->next = NULL;

   tnet->start = lnMod;

   AddLink (heap, lnMod, lnWe);

   lnWe->loWE = lnWe->hiWE = ++tnet->nPronIds;
   lnWe->lmlaIdx = ++tnet->lmlaCount;
   lnWe->pron->aux = (Ptr) lnWe->loWE;

   /* connect start WE -> all matching SA nodes */
   /* loop over all SA nodes */
   for (i = 0; i < LEX_CON_HASH_SIZE; ++i)
      for (ln = tnet->lexSAhash[i]; ln; ln = ln->next)
         if (ln->lc == lnMod->lc) {         /* matching left context? */
            AddLink (heap, lnWe, ln);
         }


   /* ENDWORD */
   /*   put null node (lnTime) into LAYER_SA to get updated end time
        in UpdateWordEndHyp() really only for HVite compatible times.
        this node is also used to detect the path to the Sentend silence for 
        the pronprob handling in HLVRec!  */
   
   lnWe = CreateBoundary (heap, tnet, tnet->endId, LAYER_A, LAYER_AB);  
   lnMod = lnWe->next;
   lnWe->next = NULL;
   AddLink (heap, lnMod, lnWe);
   
   lnWe->loWE = lnWe->hiWE = ++tnet->nPronIds;
   lnWe->lmlaIdx = ++tnet->lmlaCount;
   lnWe->pron->aux = (Ptr) lnWe->loWE;
   
   lnTime = NewTLexNodeCon (heap, tnet, LAYER_SA, lnWe->lc, lnWe->lc);
   AddLink (heap, lnTime, lnMod);
   
   tnet->end = lnWe;
   

   /* connect start all matching ZS nodes -> LN time  */
   /* loop over all ZS nodes */
   for (i = 0; i < LEX_CON_HASH_SIZE; ++i)
      for (ln = tnet->lexZShash[i]; ln; ln = ln->next)
         if (ln->rc == lnMod->lc) {         /* matching right context? */
            AddLink (heap, ln, lnTime);
         }

   /* for -/sp/sil dicts add an extra sp and sil model leading to SENT_END */
   if (tnet->silDict) {
      LabId sil, sp;
      HLink hmmSIL, hmmSP;

      /* find sil & sp models */
      sil = GetLabId ("sil", FALSE);
      if (!sil)
         HError (9999, "cannot find 'sil' model.");
      hmmSIL = FindHMM (tnet->hset, sil);
      sp = GetLabId ("sp", FALSE);
      if (!sp)
         HError (9999, "cannot find 'sp' model.");
      hmmSP = FindHMM (tnet->hset, sp);
      
      
      tnet->lnSEsp = NewTLexNodeMod (heap, tnet, LAYER_SIL, hmmSP);
      tnet->lnSEsil = NewTLexNodeMod (heap, tnet, LAYER_SIL, hmmSIL);

      AddLink (heap, tnet->lnSEsp, lnTime);
      AddLink (heap, tnet->lnSEsil, lnTime);
   }

}


/*  TraverseTree

      number all wordend nodes below ln, starting with Id start and return
      highest Id. In every node store lo and hi Ids of WE nodes
      in subtree below it.

*/
int TraverseTree (TLexNode *ln, int start, int *lmlaCount)
{
   TLexLink *ll;
   int curHi;
   static int depth = 0;

   ++depth;
   
#if 0
   for (i = 0; i < depth; ++i)
      printf(" ");
   printf ("TT %p\n", ln);
#endif

   assert (ln->loWE == 0);
   assert (ln->hiWE == 0); 
   /* WE? then bottom out, get new id, store it with ln and return */
   if (ln->type == LN_WORDEND) {
      ln->loWE = ln->hiWE = start;
      /*# store ID in ln->pron */

      ln->pron->aux = (Ptr) ln->loWE;
      /*       printf("LMLA %p %d %d  %d\n", ln, ln->loWE, ln->hiWE, *lmlaCount); */
      --depth;
      return start;
   }

   ln->loWE = start;
   curHi = start - 1;
   for (ll = ln->link; ll; ll = ll->next) {
      start = curHi + 1;
      curHi = TraverseTree (ll->end, start, lmlaCount);
   }
   ln->hiWE = curHi;
   /* I could probably replace curHi by ln->hiWE in the above */

   
   /* unique successors don't get a slot in the LMlaTree */
   assert (ln->nlinks >= 1);
   if (ln->nlinks > 1) {
      int nl=0;
      for (ll = ln->link; ll; ll = ll->next) {
         /* assign new lmlaIds to successor node */
         ll->end->lmlaIdx = ++*lmlaCount;
         ++nl;
         assert (!((ln->loWE==ll->end->loWE) && (ln->hiWE==ll->end->hiWE)));
      }
      assert (nl == ln->nlinks);
      /*     *lmlaCount += ln->nlinks; */
   }
   else {
      /* LA tree compression: no LM handling required for the 
         unique successor of this node */
      assert ((ln->loWE==ln->link->end->loWE) && (ln->hiWE==ln->link->end->hiWE));
      ln->link->end->lmlaIdx = 0;
   }

   /*    printf("LMLA %p %d %d  %d\n", ln, ln->loWE, ln->hiWE, *lmlaCount);*/

   return curHi;
}

/* AssignWEIds

*/
void AssignWEIds(TLexNet *tnet)
{
   int i;
   TLexNode *ln;
   int start, highest;

   /* assign PronIds to wordend nodes reachable from AB nodes.
      this excludes:
       - single phone prons (cf. Handle1PhonePron)
       - start/end words  (cf. CreateStartEnd)
   */

   highest = tnet->nPronIds;
   for (i = 0; i < LEX_CON_HASH_SIZE; ++i)
      for (ln = tnet->lexABhash[i]; ln; ln = ln->next) {
         start = highest + 1;

         /*          ln->loWE = start; */
         ln->lmlaIdx = ++tnet->lmlaCount;
         highest = TraverseTree (ln, start, &tnet->lmlaCount);
         ln->hiWE = highest;


         if (trace & T_NETCON)
            printf ("AB %s-%s  loWE %d hiWE %d  lmlaIdx %d\n", ln->lc->name, ln->rc->name, 
                    ln->loWE, ln->hiWE, ln->lmlaIdx);
      }

   /* one A node can have multiple successors. */
   /* create complex lmla nodes for them in CreateCompLMLA() */


   /*# handle single phone prons */

   /*# handle <s> and </s> */

   /*# sanity check for prons without id */

   printf ("lmla count %d\n", tnet->lmlaCount );
   printf ("nprons %d\n", highest);

   if (sizeof(PronId) < 4 || sizeof (LMId) < 4) {
      if (highest > (1UL << (8 * sizeof (PronId))) ||
          highest > (1UL << (8 * sizeof (LMId))))
         HError (9999, "AssignWEIds: too many pronunciations for PronId/LMId type. Recompile with type int");
   }
}


static void CreateCompLMLA (MemHeap *heap, LMlaTree *laTree, TLexNet *tnet)
{
   int i, nComp;
   TLexNode *ln;

   /* first handle simple case and count complex cases */
   nComp = 0;
   for (i = 0; i < LEX_MOD_HASH_SIZE; ++i)
      for (ln = tnet->nodeAhash[i]; ln; ln = ln->next) {
         if (!ln->link->next) {      /* unique successor node */
            /* copy lo&hi to A nodes from AB nodes */
            ln->lmlaIdx = ln->link->end->lmlaIdx;
            ln->loWE = ln->link->end->loWE;
            ln->hiWE = ln->link->end->hiWE;
         }
         else {
            ++nComp;
         }
      }

   /* alloc space for comp nodes */
   laTree->nCompNodes = nComp;

   laTree->compNode = (CompLMlaNode *) New (heap, nComp * sizeof (CompLMlaNode));

   /* fill in info */
   for (i = 0; i < LEX_MOD_HASH_SIZE; ++i)
      for (ln = tnet->nodeAhash[i]; ln; ln = ln->next) {
         if (!ln->link->next) {      /* simple case */
         }
         else {
            /* construct complex LMLA node */
            TLexLink *tll;
            int j, nfoll = 0;
            CompLMlaNode *compNode;
            
            for (tll = ln->link; tll; tll = tll->next)
               ++nfoll;
#if 0
            printf ("nfoll %d\n", nfoll);
#endif            
            /* create CompLMLANode */
            ln->lmlaIdx = ++tnet->lmlaCount;

            compNode = &laTree->compNode[ln->lmlaIdx - laTree->nNodes];
            compNode->n = nfoll;
            compNode->lmlaIdx = New (heap, nfoll * sizeof (int));
            
            for (tll = ln->link, j = 0; tll; tll = tll->next, ++j)
               compNode->lmlaIdx[j] = tll->end->lmlaIdx;
         }
      }

   assert (tnet->lmlaCount+1 - laTree->nNodes == laTree->nCompNodes); /* extra entry for lmlaIdx=0 */
}


/*

*/
static void InitLMlaTree(LexNet *net, TLexNet *tnet)
{
   LMlaTree *laTree;

   /* simple nodes */

   laTree = (LMlaTree *) New (net->heap, sizeof(LMlaTree));
   net->laTree = laTree;
   laTree->nNodes = tnet->lmlaCount + 1;        /* extra entry for lmlaIdx=0 */

   laTree->node = (LMlaNode *) New (net->heap, laTree->nNodes * sizeof (LMlaNode));


   /* complex nodes */
#if 1
   CreateCompLMLA (net->heap, laTree, tnet);
#endif
}


/* ConvertTLex2Lex

     convert the large, verbose temp lex structure into the compact LexNet structure
     nodes are ordered in layers
 */
LexNet *ConvertTLex2Lex (MemHeap *heap, TLexNet *tnet)
{
   int i, nn, nl, nlTotal = 0;
   LexNet *net;
   TLexNode *tln;
   LexNode *ln;
   TLexLink *tll;
   LexNode **foll;
   LexNode *layerCur[NLAYERS];        /* pointer to next free LN in layer */

   if (trace &T_NET) {
      printf ("number of nodes:\n");
      printf (" A model nodes: %d\n", tnet->nNodeA);
      printf (" AB context nodes: %d\n", tnet->nlexAB);
      printf (" B-Y model nodes: %d\n", tnet->nNodeBY);
      printf (" YZ context nodes: %d\n", tnet->nlexYZ);
      printf (" Z model nodes: %d\n", tnet->nNodeZ);
      printf (" ZS context nodes: %d\n", tnet->nlexZS);
      printf (" SIL model nodes: %d\n", tnet->nNodeSIL);
      printf (" SA context nodes: %d\n", tnet->nlexSA);
      printf ("total: %d\n", tnet->nNodeA + tnet->nlexAB + tnet->nNodeBY + tnet->nlexYZ +
              tnet->nNodeZ + tnet->nlexZS + tnet->nlexSA + tnet->nNodeSIL);
   }

   nn = 0;
   for (i = 0; i < NLAYERS; ++i) {
      nn += tnet->nNodesLayer[i];
      if (trace & T_NET)
         printf ("layer %d contains %d nodes\n", i, tnet->nNodesLayer[i]);
   }
   if (trace & T_NET)
      printf ("total: %d nodes\n", nn);
   
   if (tnet->silDict)
      /* 7 special models: <s> Mod/WE,  </s> Time, </s> Mod/WE, </s> SP, </s> SIL */
      assert (nn == tnet->nNodeA + tnet->nlexAB + tnet->nNodeBY + tnet->nlexYZ +
              tnet->nNodeZ + tnet->nlexZS + tnet->nlexSA + tnet->nNodeSIL + 7);
   else
      /* 5 special models: <s> Mod/WE,  </s> Time, </s> Mod/WE */
      assert (nn == tnet->nNodeA + tnet->nlexAB + tnet->nNodeBY + tnet->nlexYZ +
              tnet->nNodeZ + tnet->nlexZS + tnet->nlexSA + tnet->nNodeSIL + 5);
   

   /* alloc and init LexNet */
   net = (LexNet *) New (heap, sizeof (LexNet));
   net->heap = heap;
   net->voc = tnet->voc;
   net->hset = tnet->hset;
   net->nLayers = NLAYERS;      /* keep HLVRec general without ugly constants! */
   net->layerStart = (LexNode **) New (heap, net->nLayers * sizeof (LexNode *));

   net->nNodes = nn;
   net->node = (LexNode *) New (heap, net->nNodes * sizeof (LexNode));

   /* pronId to pron mapping */ 
   net->pronlist = (Pron *) New (heap, (net->voc->nprons + 1) * sizeof (Pron));
   net->pronlist[0] = NULL;

   /* initialise pointers to start of layers */
   ln = net->node;
   for (i = 0; i < net->nLayers; ++i) {
      layerCur[i] = ln;
      net->layerStart[i] = ln;
      ln += tnet->nNodesLayer[i];
   }

   /* initialise TLexNode -> LexNode pointers */
   ln = net->node;
   for (tln = tnet->root; tln; tln = tln->chain) {
      /* take node from appropriate layer memory block */
      tln->ln = layerCur[tln->layerId];
      ++layerCur[tln->layerId];
   }
   

#if 1
   /* check whether all layer memory blocks are full (as they should be) */
   for (i = 0; i < net->nLayers-1; ++i) {
      assert (net->layerStart[i+1] == layerCur[i]);
   }
#endif

   InitLMlaTree(net, tnet);


   /* the real conversion follows here: */
   net->start = tnet->start->ln;
   net->end = tnet->end->ln;
   
   /* copy info and convert links */
   for (tln = tnet->root; tln; tln = tln->chain) {
      ln = tln->ln;
      ln->type = (unsigned short) tln->type;

      switch (tln->type) {
      case LN_MODEL:
         ln->data.hmm = tln->hmm;
         break;
      case LN_CON:
         ln->data.hmm = NULL;
         break;
      case LN_WORDEND:
         assert (tln->loWE == tln->hiWE);
         ln->data.pron = tln->loWE;
         net->pronlist[tln->loWE] = tln->pron;
         break;
      default:
         HError (9999, "HLVNet: unknown node type %d\n", tln->type);
         break;
      }

      if (tnet->silDict) {
         net->lnSEsp = tnet->lnSEsp->ln;
         net->lnSEsil = tnet->lnSEsil->ln;
      }

      ln->lmlaIdx = tln->lmlaIdx;
      assert(ln->lmlaIdx < net->laTree->nNodes + net->laTree->nCompNodes);
      /*       printf ("LMLA idx %d lo %d hi %d\n", tln->lmlaIdx, tln->loWE, tln->hiWE); */

      /* only handle simple case here -- complex one has already been done in CreateCompLMLA() */
      if (ln->lmlaIdx < net->laTree->nNodes) {  
         net->laTree->node[ln->lmlaIdx].loWE = tln->loWE;
         net->laTree->node[ln->lmlaIdx].hiWE = tln->hiWE;
      }

      ln->nfoll = tln->nlinks;
      ln->foll = (LexNode **) New (heap, ln->nfoll * sizeof (LexNode *));

      /* convert linked list of TLexLinks into array of pointers to
         end nodes of these links */
      nl=0;
      for (tll = tln->link, foll = ln->foll; tll; tll = tll->next, ++foll) {
         assert (tll->start == tln);
         *foll = tll->end->ln;
         ++nl;
      }
      assert (nl == ln->nfoll);
      nlTotal += nl;

      ++ln;
   }

   if (trace & T_NET)
      printf ("converted %d links\n", nlTotal);

   return (net);
}


void WriteTLex (TLexNet *net, char *fn)
{
   int i;
   TLexNode *ln;
   TLexLink *ll;
   FILE *dotFile;
   Boolean isPipe;

   dotFile = FOpen (fn, NoOFilter, &isPipe);

   fprintf (dotFile, "digraph G {\n");
   fprintf (dotFile, "rankdir=LR;\n");
   fprintf (dotFile, "size=\"7,11\";\n");
   fprintf (dotFile, "{ A -> AB -> YZ -> Z -> ZS -> SA -> A2; }\n");
   

   /* A model nodes */
   fprintf (dotFile, "{ rank=same;\n");
   fprintf (dotFile, "A;\n");
   for (i = 0; i < LEX_MOD_HASH_SIZE; ++i)
      for (ln = net->nodeAhash[i]; ln; ln = ln->next) {
         fprintf (dotFile, "n%p [label=\"%s\"];\n", ln, ln->lc ? ln->lc->name : "A?");
      }
   fprintf (dotFile, "}\n");

   /* links from A */
   for (i = 0; i < LEX_MOD_HASH_SIZE; ++i)
      for (ln = net->nodeAhash[i]; ln; ln = ln->next) {
         for (ll = ln->link; ll; ll = ll->next) {
            assert (ll->start == ln);
            fprintf (dotFile, "n%p -> n%p;\n", ll->start, ll->end);
         }
      }

   /* AB context nodes */
   fprintf (dotFile, "{ rank=same;\n");
   fprintf (dotFile, "AB [shape=box];\n");
   for (i = 0; i < LEX_CON_HASH_SIZE; ++i)
      for (ln = net->lexABhash[i]; ln; ln = ln->next)
         fprintf (dotFile, "n%p [label=\"%s-%s\", shape=box];\n", ln, ln->lc ? ln->lc->name : "A?", 
                  ln->rc ? ln->rc->name : "B?");
   fprintf (dotFile, "}\n");
   
   /* links from AB nodes */
   for (i = 0; i < LEX_CON_HASH_SIZE; ++i)
      for (ln = net->lexABhash[i]; ln; ln = ln->next) {
         for (ll = ln->link; ll; ll = ll->next) {
            assert (ll->start == ln);
            fprintf (dotFile, "n%p -> n%p;\n", ll->start, ll->end);
         }
      }

   /* BY model nodes & links */
   for (ln = net->nodeBY; ln; ln = ln->next) {
      if (ln->type == LN_WORDEND) 
         fprintf (dotFile, "n%p [label=\"%s\", shape=diamond];\n", ln, ln->lc ? ln->lc->name : "BY?");
      else
         fprintf (dotFile, "n%p [label=\"%s\"];\n", ln, ln->lc ? ln->lc->name : "BY?");
      for (ll = ln->link; ll; ll = ll->next) {
         assert (ll->start == ln);
         fprintf (dotFile, "n%p -> n%p;\n", ll->start, ll->end);
      }
   }


   /* YZ context nodes */
   fprintf (dotFile, "{ rank=same;\n");
   fprintf (dotFile, "YZ [shape=box];\n");
   for (i = 0; i < LEX_CON_HASH_SIZE; ++i)
      for (ln = net->lexYZhash[i]; ln; ln = ln->next)
         fprintf (dotFile, "n%p [label=\"%s-%s\", shape=box];\n", ln, ln->lc ? ln->lc->name : "Y?",
                  ln->rc ? ln->rc->name : "Z?");
   fprintf (dotFile, "}\n");


   /* links from YZ nodes */
   for (i = 0; i < LEX_CON_HASH_SIZE; ++i)
      for (ln = net->lexYZhash[i]; ln; ln = ln->next) {
         for (ll = ln->link; ll; ll = ll->next) {
            assert (ll->start == ln);
            fprintf (dotFile, "n%p -> n%p;\n", ll->start, ll->end);
         }
      }

   /* Z model nodes & links */
   fprintf (dotFile, "Z;\n");
   for (i = 0; i < LEX_MOD_HASH_SIZE; ++i)
      for (ln = net->nodeZhash[i]; ln; ln = ln->next) {
         fprintf (dotFile, "n%p [label=\"%s\"];\n", ln, ln->lc ? ln->lc->name : "Z?");
         for (ll = ln->link; ll; ll = ll->next) {
            assert (ll->start == ln);
            fprintf (dotFile, "n%p -> n%p;\n", ll->start, ll->end);
         }
      }

   /* ZS context nodes */
   fprintf (dotFile, "{ rank=same;\n");
   fprintf (dotFile, "ZS [shape=box];\n");
   for (i = 0; i < LEX_CON_HASH_SIZE; ++i)
      for (ln = net->lexZShash[i]; ln; ln = ln->next)
         fprintf (dotFile, "n%p [label=\"%s-%s\", shape=box];\n", ln, ln->lc ? ln->lc->name : "Z?",
                  ln->rc ? ln->rc->name : "S?");
   fprintf (dotFile, "}\n");
   
   /* links from ZS nodes */
   for (i = 0; i < LEX_CON_HASH_SIZE; ++i)
      for (ln = net->lexZShash[i]; ln; ln = ln->next)
         for (ll = ln->link; ll; ll = ll->next) {
            assert (ll->start == ln);
            fprintf (dotFile, "n%p -> n%p;\n", ll->start, ll->end);
         }

   /* SIL model nodes */
   fprintf (dotFile, "{ rank=same;\n");
   /*    fprintf (dotFile, "SIL ;\n"); */
   for (ln = net->nodeSIL; ln; ln = ln->next)
      fprintf (dotFile, "n%p [label=\"%s\"];\n", ln, ln->lc ? ln->lc->name : "SIL?");
   fprintf (dotFile, "}\n");

   /* links from SIL model nodes */
   for (ln = net->nodeSIL; ln; ln = ln->next)
      for (ll = ln->link; ll; ll = ll->next) {
         assert (ll->start == ln);
         fprintf (dotFile, "n%p -> n%p;\n", ll->start, ll->end);
      }
   
   
   /* SA context nodes */
   fprintf (dotFile, "{ rank=same;\n");
   fprintf (dotFile, "SA [shape=box];\n");
   for (i = 0; i < LEX_CON_HASH_SIZE; ++i)
      for (ln = net->lexSAhash[i]; ln; ln = ln->next)
         fprintf (dotFile, "n%p [label=\"%s-%s\", shape=box];\n", ln, ln->lc ? ln->lc->name : "S?",
                  ln->rc ? ln->rc->name : "A?");
   fprintf (dotFile, "}\n");
   
#if 1
   /* links from SA nodes */
   for (i = 0; i < LEX_CON_HASH_SIZE; ++i)
      for (ln = net->lexSAhash[i]; ln; ln = ln->next)
         for (ll = ln->link; ll; ll = ll->next) {
            assert (ll->start == ln);
            if (ll->end->type == LN_WORDEND)
               fprintf (dotFile, "n%p -> n%p;\n", ll->start, ll->end);
            else
               fprintf (dotFile, "n%p -> n2%p;\n", ll->start, ll->end);
         }
   /* copy of A nodes */
   fprintf (dotFile, "{ rank=same;\n");
   fprintf (dotFile, "A2 [label=A];\n");
   for (i = 0; i < LEX_MOD_HASH_SIZE; ++i)
      for (ln = net->nodeAhash[i]; ln; ln = ln->next) {
         fprintf (dotFile, "n2%p [label=\"%s\"];\n", ln, ln->lc ? ln->lc->name : "A?");
      }
   fprintf (dotFile, "}\n");
#endif

   fprintf (dotFile, "}\n");
   FClose (dotFile, FALSE);
}

/* Create the Lexicon Network based on the vocabulary and model set
*/
LexNet *CreateLexNet (MemHeap *heap, Vocab *voc, HMMSet *hset, 
                      char *startWord, char *endWord, Boolean silDict)
{
   int i;
   TLexNet *tnet;
   LexNet *net;

   tnet = (TLexNet *) New (&tnetHeap, sizeof (TLexNet));
   tnet->heap = &tnetHeap;
   tnet->root = NULL;
   tnet->nlexA = tnet->nlexZ = tnet->nlexP = tnet->nlexAB = tnet->nlexYZ = tnet->nlexZS = tnet->nlexSA = 0;
   tnet->nNodeA = tnet->nNodeZ = 0;
   tnet->lexA = (LabId *) New (&gcheap, LIST_BLOCKSIZE * sizeof(LabId));
   tnet->lexZ = (LabId *) New (&gcheap, LIST_BLOCKSIZE * sizeof(LabId));
   tnet->lexP = (LabId *) New (&gcheap, LIST_BLOCKSIZE * sizeof(LabId));
   for (i = 0; i < LEX_CON_HASH_SIZE; ++i)
      tnet->lexABhash[i] = tnet->lexYZhash[i] = 
         tnet->lexZShash[i] = tnet->lexSAhash[i] = NULL;
   for (i = 0; i < LEX_MOD_HASH_SIZE; ++i)
      tnet->nodeAhash[i] = tnet->nodeZhash[i] = NULL;
   tnet->nNodeBY = 0;
   tnet->nodeBY = NULL;
   tnet->nNodeSIL = 0;
   tnet->nodeSIL = NULL;
   tnet->lmlaCount = 0;
   tnet->nPronIds = 0;

   tnet->voc = voc;
   tnet->hset = hset;
   tnet->silDict = silDict;
   /* init root node */
   tnet->root = NULL;

   tnet->startId = GetLabId (startWord, FALSE);
   if (!tnet->startId) 
      HError (9999, "HLVNet: cannot find STARTWORD '%s'\n", startWord);
   tnet->endId = GetLabId (endWord, FALSE);
   if (!tnet->endId) 
      HError (9999, "HLVNet: cannot find ENDWORD '%s'\n", endWord);


   for (i = 0; i < NLAYERS; ++i)
      tnet->nNodesLayer[i] = 0;
   
#if 0
   {
   MLink ml;
   for (i = 0; i < MACHASHSIZE; i++)
      for (ml = hset->mtab[i]; ml != NULL; ml = ml->next)
	 /* 	 if (ml->type == 'l') */
	 printf ("macro type: %c  name: %s  ptr: %x nUse: %d\n", 
		 ml->type, ml->id->name, ml->structure,
		 ((HLink) ml->structure)->nUse);
   }
#endif


   /* init phone sets A, B, AB, YZ 
      and create LexNodes for AB and YZ */
   CollectPhoneStats (tnet->heap, tnet);

   /* Create initial phone (A) layer of z-a+b nodes
      also creates SA nodes*/
   CreateAnodes (tnet->heap, tnet);

   /* Create final phone (Z) layer of y-z+a nodes 
      also creates ZS nodes */
   CreateZnodes (tnet->heap, tnet);

   /* Create silence (sil/sp) nodes connecting ZS and SA nodes */
   CreateSILnodes (tnet->heap, tnet);

   /* Create prefix tree nodes (B -- Y) */
   CreateBYnodes (tnet->heap, tnet);

   /* Create sentence start and end nodes */
   CreateStartEnd (heap, tnet);

#if 0
   /* Output TLexNet in dot format */
   WriteTLex (tnet, "lex.dot");
#endif      




#if 0           /* debug code: print contents of A layer */
   {
      int i;
      TLexNode *ln;
      TLexLink *ll;
      
      printf ("nNodeA = %d   \n", tnet->nNodeA);
      for (i = 0; i < LEX_MOD_HASH_SIZE; ++i)
         for (ln = tnet->nodeAhash[i]; ln; ln = ln->next) {
            printf ("a_hmm %s nlinks %d ", FindMacroStruct (tnet->hset, 'h', ln->hmm)->id->name,
                    ln->nlinks);
            for (ll = ln->link; ll; ll = ll->next) {
               printf (" %s-%s ", ll->end->lc->name, ll->end->rc->name);
            }
            printf ("\n");
         }
      
   }
#endif

#if 0
   printf ("nlexA   %6d\n", tnet->nlexA);
   printf ("nlexZ   %6d\n\n", tnet->nlexZ);

   printf ("Nodes:\n");
   printf ("nNodeA   %6d\n", tnet->nNodeA);
   printf ("nlexAB   %6d\n", tnet->nlexAB);
   printf ("nNodeBY  %6d\n", tnet->nNodeBY);
   printf ("nlexYZ   %6d\n", tnet->nlexYZ);
   printf ("nNodeZ   %6d\n", tnet->nNodeZ);
   printf ("nlexZS   %6d\n", tnet->nlexZS);
   printf ("nlexSA   %6d\n", tnet->nlexSA);
   printf ("total    %6d\n", tnet->nNodeA + tnet->nlexAB + tnet->nNodeBY + 
           tnet->nlexYZ + tnet->nNodeZ + tnet->nlexZS + tnet->nlexSA);
#endif

   AssignWEIds(tnet);

   /* convert TLexNet to more compact LexNet */
   net = ConvertTLex2Lex(heap, tnet);

   /* all tokens pass through SA directly before (i.e. with no time diff) the first 
      model of a new word. Update time and score in last weHyp of token in this layer */
   net->wordEndLayerId = LAYER_SA;

   net->startPron = (PronId) (int) GetWord (voc, tnet->startId, FALSE)->pron->aux;
   net->endPron = (PronId) (int) GetWord (voc, tnet->endId, FALSE)->pron->aux;

   /* add pronprobs in ZS nodes and (if S==sp) propagate - variant bypassing sp model. */
   net->spSkipLayer = LAYER_ZS;
   { 
      LabId spLab;
      spLab = GetLabId ("sp", FALSE);
      if (!spLab)
         HError (9999, "cannot find 'sp' model.");
      net->hmmSP = FindHMM (net->hset, spLab);
   }
   net->silDict = silDict;

   /* get rid of temporary data structures */
   ResetHeap (&tnetHeap);

   return net;
}


/* -------------- vocab handling --------------- */

Boolean CompareBasePron (Pron b, Pron p)
{
   int i;

   if (p->nphones != b->nphones + 1)
      return FALSE;
   
   for (i = 0; i < b->nphones; ++i)
      if (p->phones[i] != b->phones[i])
         return FALSE;
   
   return TRUE;
}

/*
   ConvertSilDict

     sort the prons into -/sp/sil order, i.e p->next = p_sp; p_sp->next = p_sil
     and mark - prons.
*/
void ConvertSilDict (Vocab *voc, LabId spLab, LabId silLab, 
                     LabId startLab, LabId endLab)
{
   int i, j, nPron, maxnPron;
   Word word;
   Pron p, *pSort;
   LabId l;

   maxnPron = 15;
   pSort = (Pron *) New (&gcheap, maxnPron * sizeof (Pron));

   /* iterate over all words */
   for (i = 0; i < VHASHSIZE; i++)
      for (word = voc->wtab[i]; word ; word = word->next) {
         /* skip START/END words */
         if (word->wordName == startLab || word->wordName == endLab || word->nprons == 0)
            continue;

         if ((word->nprons % 3) != 0)
            HError (9999, "ConvertSilDict: word '%s' does not have -/sp/sil variants",
                    word->wordName->name);

         if (word->nprons  > maxnPron) {
            Dispose (&gcheap, pSort);
            maxnPron = word->nprons ;
            pSort = (Pron *) New (&gcheap, maxnPron * sizeof (Pron));
         }
         nPron = 0;
         for (j = 0; j < maxnPron; ++j)
            pSort[j] = NULL;

         /* find - variants */
         for (p = word->pron; p; p = p->next) {
            l = p->phones[p->nphones - 1];
            if (l == spLab) 
               p->aux = (Ptr) 2;
            else if (l == silLab)
               p->aux = (Ptr) 3;
            else {
               p->aux = (Ptr) 1;        /* mark pron for CreateLexNet !! */
               pSort[nPron] = p;
               pSort[nPron+1] = pSort[nPron+2] = NULL;
               nPron += 3;
               assert (nPron <= maxnPron);
            }
         }
         assert (nPron == word->nprons);

         /* sort sp/sil variants */
         for (p = word->pron; p; p = p->next) {
            if (p->aux > (Ptr) 1) {   /* ignore - variant */
               for (j = 0; j < nPron; j += 3) {
                  if (CompareBasePron (pSort[j], p)) {
                     assert (j + (((int) p->aux) - 1) <= nPron);
                     pSort [j + (((int) p->aux) - 1)] = p;
                     p->aux = (Ptr) 0;  /* unmark pron for CreateLexNet !! */
                     break;     /* next p */
                  }
               }
            }
         }

         /* create linked list */
         for (j = 0; j < nPron-1; ++j) {
            pSort[j]->next = pSort[j+1];
            if (!pSort[j])
               HError (9999, "ConvertSilDict: word '%s' does not have -/sp/sil variants",
                       word->wordName->name);
         }
         if (!pSort[nPron - 1])
            HError (9999, "ConvertSilDict: word '%s' does not have -/sp/sil variants",
                    word->wordName->name);
         pSort[nPron-1]->next = NULL;
         
         word->pron = pSort[0];
      } /* next word */
   
   Dispose (&gcheap, pSort);
}

/*
   MarkAllProns

     mark all prons for inclusion in LexNet
*/
void MarkAllProns (Vocab *voc)
{
   int i;
   Word word;
   Pron pron;

   for (i = 0; i < VHASHSIZE; i++)
      for (word = voc->wtab[i]; word ; word = word->next) {
         /*          word->aux  = (Ptr) 1; */
         for (pron = word->pron; pron ; pron = pron->next) {
            pron->aux  = (Ptr) 1;
         }
      }
}

void MarkAllWords (Vocab *voc)
{
   int i;
   Word word;

   for (i = 0; i < VHASHSIZE; i++)
      for (word = voc->wtab[i]; word ; word = word->next) {
         word->aux  = (Ptr) 1;
      }
}

void UnMarkAllWords (Vocab *voc)
{
   int i;
   Word word;

   for (i = 0; i < VHASHSIZE; i++)
      for (word = voc->wtab[i]; word ; word = word->next) {
         word->aux  = (Ptr) 0;
      }
}

void MarkAllWordsfromLat (Vocab *voc, Lattice *lat, Boolean silDict)
{
   int i;
   LNode *ln;
   Word word;
   Pron pron;

   for (i = 0; i < lat->nn; ++i) {
      ln = &lat->lnodes[i];
      word = ln->word;

      if (word) {
         word->aux  = (Ptr) 1;
         if (silDict) { /* -/sp/sil style dict: only mark - variants */
            if ((word->nprons % 3) == 0) {   /* exclude !SENT_START/END */
               for (pron = word->pron; pron ; pron = pron->next) {
                  pron->aux  = (Ptr) 1;
                  pron = pron->next;       /* sp variant */
                  pron->aux  = (Ptr) 0;
                  pron = pron->next;       /* sil variant */
                  pron->aux  = (Ptr) 0;
               }
            }
         }
         else {         /* lvx-style dict: mark all prons */
            for (pron = word->pron; pron ; pron = pron->next) {
               pron->aux  = (Ptr) 1;
            }
         }
      }
   }
}

/**********************************************************************/
/*   State based network expansion for arbitrary m-phones             */

#define DIR_FORW 0
#define DIR_BACKW 1

typedef struct _STLexLink STLexLink;
typedef struct _STLexNode STLexNode;

struct _STLexLink {
   STLexNode *node[2];  /* numbered left to right */
   STLexLink *next[2];
   Pron we;
};

typedef enum _STLexNodeType {
  STLN_WORDEND, STLN_CON, STLN_MODEL
} STLexNodeType;

struct _STLexNode {
   STLexNodeType type;
   union {
      LabId monoId;
      Pron pron;
   } data;
   STLexLink *link[2];
};

int stlNodeN = 0;
int stlLinkN = 0;

STLexNode *NewSTLNode (MemHeap *heap)
{
   STLexNode *n;
   int d;

   ++stlNodeN;
   n = New (heap, sizeof (STLexNode));
   n->data.monoId = NULL;
   for (d = 0; d < 2; ++d)
      n->link[d] = NULL;

   n->type = STLN_MODEL;

   return n;
}

STLexLink *NewSTLLink (MemHeap *heap)
{
   STLexLink *n;
   int d;

   ++stlLinkN;
   n = New (heap, sizeof (STLexLink));

   n->we = NULL;
   for (d = 0; d < 2; ++d) {
      n->node[d] = NULL;
      n->next[d] = NULL;
   }

   return n;
}

STLexLink *AddSTLexLink (MemHeap *heap, STLexNode *start, STLexNode *end)
{
   STLexLink *l;

   l = NewSTLLink (heap);
   l->node[0] = start;
   l->node[1] = end;

   l->next[DIR_FORW] = start->link[DIR_FORW];
   start->link[DIR_FORW] = l;

   l->next[DIR_BACKW] = end->link[DIR_BACKW];
   end->link[DIR_BACKW] = l;

   return l;
}


STLexNode *AddMonoPron_S (MemHeap *heap, STLexNode *root, int n, LabId *p)
{
   int i;
   STLexNode *cur;
   STLexLink *l;

   cur = root;
   for (i = 0; i < n; ++i) {
      /* try to find phone p[i] */
      for (l = cur->link[DIR_FORW]; l; l = l->next[DIR_FORW]) {
         if (l->node[1]->data.monoId == p[i])
            break;
      }
      if (l) {
         cur = l->node[1];
      }
      else {    /* can't find p[i], add it */
         STLexNode *newNode;

         newNode = NewSTLNode (heap);
         newNode->data.monoId = p[i];
         AddSTLexLink (heap, cur, newNode);
         cur = newNode;
      }
   }

   return cur;
}

void FindContexts_s (HMMSet *hset, STLexNode *n)
{
   int i;
   STLexLink *back, *forw, *l;
   STLexNode *left, *right;

   if (n->data.monoId) {   /* not root or final ?*/

      /* deal with n*/
      for (back = n->link[DIR_BACKW]; back; back = back->next[DIR_BACKW]) {
         left = back->node[0];
         for (forw = n->link[DIR_FORW]; forw; forw = forw->next[DIR_FORW]) {
            right = forw->node[1];
            
            printf ("%s-%s+%s\n",
                    left->data.monoId ? left->data.monoId->name : "???",
                    n->data.monoId->name,
                    right->data.monoId ? right->data.monoId->name : "???");

            if (left->data.monoId && right->data.monoId) {
               HLink hmm;
               hmm = FindTriphone (hset, left->data.monoId, n->data.monoId, right->data.monoId);
               for (i = 2; i < hmm->numStates; ++i) {
                  printf (" %4d ", hmm->svec[i].info->sIdx);
               }
               printf ("\n");
            }
         }
      }
   }
   
   /* recurse on children */
   for (l = n->link[DIR_FORW]; l; l = l->next[DIR_FORW]) {
      FindContexts_s (hset, l->node[1]);
   }
}

int comp_stll (const void *v1, const void *v2) {
   STLexLink *l1, *l2;
   
   l1 = (STLexLink *) v1;
   l2 = (STLexLink *) v2;
   
   return ((int) (l1->node[0]->data.monoId - l2->node[0]->data.monoId));
};


LexNet *CreateLexNet_S (MemHeap *heap, Vocab *voc, HMMSet *hset, char *startWord, char *endWord)
{
   int p;
   Word word;
   Pron pron;
   LabId startId, endId;
   STLexNode *root, *final, *last;
   STLexLink *l;

   /* find start and end Ids */
   startId = GetLabId (startWord, FALSE);
   if (!startId) 
      HError (9999, "HLVNet: cannot find STARTWORD '%s'\n", startWord);
   endId = GetLabId (endWord, FALSE);
   if (!endId) 
      HError (9999, "HLVNet: cannot find ENDWORD '%s'\n", endWord);


   /* init root node */
   root = NewSTLNode (heap);
   final = NewSTLNode (heap);

   /* build a monophone tree */
   for (p = 0; p < VHASHSIZE; p++)
      for (word = voc->wtab[p]; word ; word = word->next)
         if (word->aux == (Ptr) 1 && (word->wordName != startId && word->wordName != endId))
            for (pron = word->pron; pron ; pron = pron->next) {
               if (pron->aux == (Ptr) 1) {

                  last = AddMonoPron_S (heap, root, pron->nphones, pron->phones);

#if 0
                  /* alloc word end node */
                  weLN = NewSTLNode (heap);
                  weLN->type = STLN_WORDEND;
                  weLN->data.pron = pron;

                  /* link last phone via word end to final node */
                  AddSTLexLink (heap, last, weLN);
                  AddSTLexLink (heap, weLN, final);
#endif 
                  l = AddSTLexLink (heap, last, final);
                  l->we = pron;
               }
            }

   /* now we have a monophone tree hanging of 'root' with word end nodes at 
      the leaves which are connected to 'final' */


   /* move back word ends as far as possible */
   {
      STLexLink *weL, *l;
      Pron we;

      for (weL = final->link[DIR_BACKW]; weL; weL = weL->next[DIR_BACKW]) {
         we = weL->we;
         /* keep going backward as long as l is the only link */
         for (l = weL;        /* l = weLN->link[DIR_BACKW]; */
              !l->next[DIR_FORW] && l->node[0]->link[DIR_FORW] == l;
              l = l->node[0]->link[DIR_BACKW])
            ;
         
         /* tag link with wordend */
         if (l->we) {
            /* add to linked list */
         }
         else {
            l->we = we;
         }
      }
   }
   
   /*#### merge suffixes */
   {
      int i, nLinks;
      STLexLink **links;

      nLinks = 0;
      for (l = final->link[DIR_BACKW]; l; l = l->next[DIR_BACKW])
         ++nLinks;

      links = (STLexLink **) New (&gcheap, nLinks * sizeof (STLexLink *));

      for (i = 0, l = final->link[DIR_BACKW]; l; l = l->next[DIR_BACKW], ++i)
         links[i] = l;

      qsort (links, nLinks, sizeof (STLexLink *), comp_stll);

      for (i = 0; i < nLinks; ++i) {
         
      }
      
      root->data.monoId = final->data.monoId = GetLabId ("sil", FALSE);
   }
#if 0
   /* expand contexts */
   {
      FindContexts_s (hset, root);
      
   }
#endif

   return NULL;
}





/*  CC-mode style info for emacs
 Local Variables:
 c-file-style: "htk"
 End:
*/
