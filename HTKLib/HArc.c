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
/* ----------------------------------------------------------- */
/*           Copyright: Cambridge University                   */
/*                      Engineering Department                 */
/*            2002-2015 Cambridge, Cambridgeshire UK           */
/*                      http://www.eng.cam.ac.uk               */
/*                                                             */
/*   Use of this software is governed by a License Agreement   */
/*    ** See the file License for the Conditions of Use  **    */
/*    **     This banner notice must not be removed      **    */
/*                                                             */
/* ----------------------------------------------------------- */
/*           File: HArc.c   Forward backward routines          */
/* ----------------------------------------------------------- */

char *arc_version = "!HVER!HArc:   3.5.0 [CUED 12/10/15]";
char *arc_vc_id = "$Id: HArc.c,v 1.2 2015/10/12 12:07:24 cz277 Exp $";


/*
  Performs forward/backward alignment
*/


#include "HShell.h"     /* HMM ToolKit Modules */
#include "HMem.h"
#include "HMath.h"
#include "HSigP.h"
#include "HWave.h"
#include "HAudio.h"
#include "HParm.h"
#include "HLabel.h"
#include "HANNet.h"
#include "HModel.h"
#include "HUtil.h"
#include "HDict.h"
#include "HNet.h"
#include "HArc.h"



/* Trace Flags */

#define T_ARC  01    /* Arcs. [dp] */
#define T_ARC2 02   /* Arcs, fuller. */


static int trace=1;

static ConfParam *cParm[MAXGLOBS];  /* config parameters */
static int nParm = 0;

/*  LMSCALE and WDPEN here can override ones in lattice. Perhaps delete this.  */
static float LMSCALE = 0;
static Boolean IsLMScale = FALSE;
static float WDPEN = 0;
static Boolean IsWdPen = FALSE;
static float FRAMEDUR = 0; 
/*static int debug=0;*/


#define MAX(a,b) ((a)>(b) ? (a):(b))
#define MIN(a,b) ((a)<(b) ? (a):(b))


Boolean StackInitialised=FALSE;
static MemHeap tempArcStack;                  /* for temporary structures. */
/* cz277 - mtload */
static MemHeap arcstak;


/* -------------------------- Arc support routines.----------------------- */

/* ArcTrans  *ArcTransExists(Arc *start, Arc *end){
   ArcTrans *at;
   for(at=start->follTrans;at;at=at->start_foll)
   if(at->end==end)
   return at;
   return NULL;
   } */

void AddArcTrans(MemHeap *mem,  HArc *start, HArc *end, float lmlike){
   /*Adds a transition with probability "lmlike" scaled by "lmScale".  The reason those two are given separately
     (instead of their product) is that it makes a difference when two transitions are added. */

   ArcTrans *at;

   at = New(mem, sizeof(ArcTrans));
   at->start = start;
   at->end = end;
   at->lmlike = lmlike;
   if (isnan(lmlike)) HError(1, "lmlike isnan..");
   /*insert it into the ll attached to start->follTrans.*/
   at->start_foll = start->follTrans;
   if(start->follTrans) start->follTrans->start_prec = at;
   start->follTrans=at;
   at->start_prec = NULL;
    
   /*insert it into the ll attached to end->precTrans.*/
   at->end_foll = end->precTrans;
   if(end->precTrans) end->precTrans->end_prec = at;
   end->precTrans=at;
   at->end_prec = NULL;
}


typedef struct _ArcList{
   HArc *h;
   struct _ArcList *t;
} ArcList;

ArcList *ConsArcList(MemHeap *mem, HArc *h, ArcList *t){
   ArcList *ans = New(mem, sizeof(ArcList));
   ans->h = h;
   ans ->t = t;
   return ans;
}


float Depth(ArcInfo *aInfo, Boolean All){
   float maxTime=0.0, totalTime=0.0;
   HArc *arc;
   for(arc=aInfo->start;arc;arc=arc->foll){
      if(All || arc->id != GHOST_ARC){
         if(arc->t_end > maxTime) maxTime = arc->t_end;
         totalTime += arc->t_end - arc->t_start;
      }
   }
   return totalTime / maxTime;
}


int TimeToNFrames(float time, ArcInfo *aInfo){
   float fans; int ans;
   fans =  time/aInfo->framedur;
   ans = (int)(fans+0.5); 

   if(fabs(ans-fans) > 0.1) { 
       HError(1, "There is a problem with lattice frame length.  Set ARC:FRAMEDUR to frame length in seconds (e.g 0.01)");
   }

   return ans;
}

HArc *CreateArc(MemHeap *mem, Lattice *lat, LArc *la, int start_time, int pos, int arcId, HArc *lastArc, float insPen, float lmScale, HMMSet *hset, ArcInfo *aInfo){ /*Creates an arc.*/
   MLink macroName;
   LNode *ln;
   ArcList *al;
   HArc *ans = (HArc*)New(mem, sizeof(HArc)), *arc;
   /*First create the arc.*/
   ans->id = arcId;
   ans->pos = pos;
   ans->parentLarc = la;
   ans->t_start = start_time + 1;
   ans->t_end   = start_time + TimeToNFrames((float)la->lAlign[pos].dur, aInfo);
   ans->mpe = NULL;

   if(ans->t_end < start_time) HError (1, "Error- bad times in lattice");

   ans->follTrans = ans->precTrans = NULL;
   ans->phone = la->lAlign[pos].label; /*The phone in context.*/
   if(pos==0) ans->word = la->end->word->wordName;
   else ans->word = NULL;

   if((macroName=FindMacroName(hset,'l',ans->phone))==NULL)
      HError(2321,"CreateArc: Unknown align label %s",ans->phone->name); 
   ans->hmm = (HLink)macroName->structure;  /* Note-- "hmm" has to be set at this stage because HMM addresses are compared for purposes of
                                               sorting arcs and setting the calcArc field later to decide which ones need to have
                                               forward-backward done. */
   ans->calcArc = NULL;
  
   /*Next add the to- transitions (if there are any.)*/
   ln=la->start;
   if(pos==0) /*First phone of the word-- add connecting arcs to the last phones of preceding words.*/
      for(al=(ArcList*)ln->hook;al;al=al->t){
         arc = al->h; 
         AddArcTrans(mem, arc, ans, la->prlike*lat->prscale + la->lmlike*lmScale + insPen);
         if(ans->t_start != arc->t_end+1) HError(1, "CreateArc: mismatched times.");
      }
   else{ /*Add the transition between this and the previous phone. */
      AddArcTrans(mem, lastArc, ans, 0);
      if(ans->t_end < lastArc->t_end)  HError(1, "Times don't match in lattice.");
   }    

   /*Now for the from-transitions...*/
   /*All we need to do is annotate the ending LNode with this arc.
     Any necessary transitions will be added later. */
   ln = la->end;
   if(pos==la->nAlign-1)/*if at end of word,*/
      ln->hook = (ArcList*) ConsArcList(&tempArcStack, ans, (ArcList*)ln->hook); 
   /* create from temporary memory,  freed when ArcFromLat finishes. */
  
   return ans;
}
/*Note that the arcs which go to the start and end of the file do not have any
  to- and from- transitions respectively.*/


 


void PrintArc(FILE *f, HArc *a){
   ArcTrans *at;
   fprintf(f, "Arc{ id=%d, pos=%d, parentLarc=%p, t_start=%f, t_end=%f",
           a->id, a->pos, a->parentLarc, (float)a->t_start, (float)a->t_end);
   if(a->prec && a->t_end > a->prec->t_end){
      printf("(>%f by %E)\n", (float)a->prec->t_end, (float)(a->t_end - a->prec->t_end));
      if(a->t_end + 0.0 == a->prec->t_end){ printf("****\n"); }
   }
   else printf("\n");

   fprintf(f, "follTrans ="); 
   fprintf(f, "{");
   for(at=a->follTrans;at;at=at->start_foll){ fprintf(f, "%d/%f ", at->end->id, at->lmlike); }
   fprintf(f, "}");
   fprintf(f, ",precTrans ="); 
   fprintf(f, "{");
   for(at=a->precTrans;at;at=at->end_foll){ fprintf(f, "%d/%f ", at->start->id, at->lmlike); }


   fprintf(f, "}");
   fprintf(f, "\nphone=%s, foll=%d, prev=%d\n}\n", a->phone->name, a->foll?a->foll->id:-1,
           a->prec?a->prec->id:-1);
}

void PrintArcs(FILE *f, HArc *a){
   fprintf(f, "\n****PrintArcs:****\n");
   for(;a;a=a->foll)
      PrintArc(f, a);
   fprintf(f, "*********************\n\n");
}

void PrintArcInfo(FILE *f, ArcInfo *ai){
   fprintf(f, "ArcInfo{\n");
   fprintf(f, " nLats=%d\n mem=%s\n", ai->nLats, ai->mem->name);
   fprintf(f, " nArcs=%d\n lmScale=%f insPen=%f\n", ai->nArcs, ai->lmScale, ai->insPen);
   PrintArcs(f, ai->start);
}
	 


void ZeroHooks(Lattice *lat){ /*Checks that the hooks are zero.*/
   int n;
   for(n=0;n<lat->nn;n++)
      lat->lnodes[n].hook = NULL;
}



int arc_compare( const void* a , const void* b )
{
   if ( (*(HArc**)a)->t_end > (*(HArc**)b)->t_end  || 
        ( (*(HArc**)a)->t_end == (*(HArc**)b)->t_end  && (*(HArc**)a)->t_start > (*(HArc**)b)->t_start ) ||
        ( (*(HArc**)a)->t_end == (*(HArc**)b)->t_end  && (*(HArc**)a)->t_start == (*(HArc**)b)->t_start &&   (*(HArc**)a)->hmm > (*(HArc**)b)->hmm))

      return 1;
   else
      return -1;;
}

void SortArcs( ArcInfo *aInfo )
{
   int q;

   /* cz277 - mtload */
   if(!StackInitialised) {
      CreateHeap(&tempArcStack, "tempArcStore", MSTAK, 1, 0.5, 10000, 100000);
      StackInitialised = TRUE;
   }

   /* cz277 - mtload */
   /*HArc   **arclist = New( &gstack , aInfo->nArcs * sizeof(HArc*) );*/
   HArc **arclist = New(&tempArcStack, aInfo->nArcs * sizeof(HArc*));

   HArc   *a , *prec = NULL ;  
   HArc   **al = arclist , **ale = arclist + aInfo->nArcs ;
   int   id = 1 ;

   for ( a = aInfo->start ; al != ale ; *(al++) = a , a = a->foll );

   qsort( arclist , aInfo->nArcs , sizeof(HArc*) , arc_compare ); 

   for ( al = arclist ; al != ale ; ++al )
      {
         (*al)->prec = prec ; prec = *al ; 
         (*al)->foll = *(al+1) ; 
         (*al)->id = id++ ;
      }

   /* prec should point to last arc */
   prec->foll   = NULL;
   aInfo->start = *arclist ;
   aInfo->end   = prec;
   /*
     for ( al = arclist ; al != ale ; ++al )
     {
     printf("%p ( prec %10p foll %10p ) %-5d %-12f %-12f \n",
     *al , (*al)->prec , (*al)->foll , (*al)->id , (*al)->t_end , (*al)->t_start );
     }
     printf("%p %p %p\n" , prec , aInfo->start , aInfo->end ); fflush(stdout);
   */
   q=0;
   for(a=aInfo->start;a;a=a->foll)
      a->id = ++q;
   /* cz277 - mtload */
   /*Dispose( &gstack , arclist );*/
   /*Dispose(&gcheap, arclist);*/
   Dispose(&tempArcStack, arclist);

}


Boolean BackTransitions(ArcInfo *aInfo){ /* a check, should never happen */
   HArc *a;
   ArcTrans *at;
   for(a=aInfo->start;a;a=a->foll)
      for(at = a->follTrans; at; at=at->start_foll)
         if(at->end->id <= a->id)
            return TRUE;
   return FALSE;
}




void FixLatTimes(Lattice *lat){ /*Makes it so that the sum of phone lengths equals the length of each word.*/
   /*This is to fix a bug encountered in the lattices produced for quinphone hmm sets. */
   /* ... and to test whether the lattice lacks phone marking.. */
   /*Can kill this code after a suitable interval when the broken lattices are got rid of. */

   int larcid;
   int seg;
   float dur, new_time;
   for(larcid=0;larcid<lat->na;larcid++){
      float sumdur=0,start_time,end_time;
      start_time = lat->larcs[larcid].start->time;
      end_time =  lat->larcs[larcid].end->time;
      dur = end_time-start_time;
      for(seg=0;seg<lat->larcs[larcid].nAlign;seg++){
         sumdur += lat->larcs[larcid].lAlign[seg].dur;
      }
      if(dur!=sumdur){
         if(lat->larcs[larcid].nAlign==0 && dur !=0) HError(1, "Lattice appears not to have phone markings.  Need to use recogniser to add phone markings to lattice.");
         new_time = lat->larcs[larcid].lAlign[lat->larcs[larcid].nAlign-2].dur + (dur-sumdur);
         if(new_time > 0 && lat->larcs[larcid].nAlign >= 2)
            lat->larcs[larcid].lAlign[lat->larcs[larcid].nAlign-2].dur = new_time;
         else /*scale.*/
            for(seg=0;seg<lat->larcs[larcid].nAlign;seg++){
               lat->larcs[larcid].lAlign[seg].dur *= dur/sumdur;
            }
      }
   }
}


/* -------------------------- Creates the arcs from the lattice. ----------------------- */

/* cz277 - cuda fblat */
#ifdef CUDA
void InitAcousticDev(Acoustic *ac, AcousticDev *acDev) {
    int i, j, T1, Nq1, size;
    StreamElem *ste;
    AcousticDev acHost;
    int *indexes;
    NFloat *transp;

    Nq1 = ac->Nq + 1;
    T1 = ac->t_end - ac->t_start + 1 + 1;
    /* Nq */
    acHost.Nq = ac->Nq;
    acHost.t_start = ac->t_start;
    acHost.t_end = ac->t_end;
    acHost.aclike = ac->aclike;
    acHost.locc = ac->locc;
    acHost.SP = ac->SP;
    
    /* indexes */
    size = sizeof(int) * Nq1;
    indexes = (int *) New(&arcstak, size); 
    for (i = 2; i < ac->Nq; ++i) {
        indexes[i] = ac->hmm->svec[i].info->pdf[1].targetIdx;
    }
    DevNew((void **) &acHost.indexes, size);
    SyncHost2Dev(indexes, acHost.indexes, size); 
    /* transp */
    size = sizeof(NFloat) * Nq1 * Nq1;
    transp = (NFloat *) New(&arcstak, size);
    for (i = 1; i <= ac->Nq; ++i) 
        for (j = 1; j <= ac->Nq; ++j) 
            transp[i * Nq1 + j] = ac->hmm->transP[i][j];
    DevNew((void **) &acHost.transp, size);
    SyncHost2Dev(transp, acHost.transp, size);
    /* alphaPlus */
    size = sizeof(NFloat) * T1 * Nq1;
    acHost.alphaPlus = NULL;
    if (ac->alphaPlus != NULL) {
        DevNew((void **) &acHost.alphaPlus, size);
    }
    /* betaPlus */
    acHost.betaPlus = NULL;
    if (ac->betaPlus != NULL) {
        DevNew((void **) &acHost.betaPlus, size);
    }
    /* otprob */
    acHost.otprob = NULL;
    if (ac->otprob != NULL) {
        DevNew((void **) &acHost.otprob, size);
    }

    /*Dispose(&arcstak, transp);*/
    Dispose(&arcstak, indexes);

    /* transfer */
    SyncHost2Dev(&acHost, acDev, sizeof(AcousticDev));
}
#endif


/* cz277 - cuda fblat */
#ifdef CUDA
void ClearAcousticDev(AcousticDev *acDev) {
    int i, T1, Nq1, size;
    AcousticDev acHost;

    SyncDev2Host(acDev, &acHost, sizeof(AcousticDev));
    T1 = acHost.t_end - acHost.t_start + 1 + 1;
    Nq1 = acHost.Nq + 1;
    /* indexes */
    size = sizeof(int) * Nq1;
    DevDispose(acHost.indexes, size);
    /* transp */
    size = sizeof(NFloat) * Nq1 * Nq1;
    DevDispose(acHost.transp, size);
    /* alphaPlus */
    size = sizeof(NFloat) * T1 * Nq1;
    if (acHost.alphaPlus != NULL) {
        DevDispose(acHost.alphaPlus, size);
    }
    /* betaPlus */
    if (acHost.betaPlus != NULL) {
        DevDispose(acHost.betaPlus, size);
    }
    /* otprob */
    if (acHost.otprob != NULL) {
        DevDispose(acHost.otprob, size);
    }

}
#endif


/* cz277 - cuda fblat */
#ifdef CUDA
void SyncAcousticDev2HostBeta(AcousticDev *acDev, Acoustic *ac, MemHeap *mem) {
    int i, j, t, T1, Nq1, size, base;
    NFloat *betaPlus, *otprob;
    AcousticDev acHost;

    SyncDev2Host(acDev, &acHost, sizeof(AcousticDev));
    T1 = acHost.t_end - acHost.t_start + 1 + 1;
    Nq1 = acHost.Nq + 1;
    /* aclike */
    ac->aclike = acHost.aclike;

    /* betaPlus (not necessary?) */
    if (acHost.SP == FALSE) {
        size = sizeof(NFloat) * T1 * Nq1;
        betaPlus = (NFloat *) New(&arcstak, size);
        SyncDev2Host(acHost.betaPlus, betaPlus, size);
        for (i = ac->t_start; i <= ac->t_end; ++i) {
            t = i - ac->t_start + 1;
            base = t * Nq1;
            for (j = 1; j <= ac->Nq; ++j)
                ac->betaPlus[i][j] = betaPlus[base + j];
        }
        /* otprob (not necessary?) */
        otprob = (NFloat *) New(&arcstak, size);
        SyncDev2Host(acHost.otprob, otprob, size);
        for (i = ac->t_start; i <= ac->t_end; ++i) {
            t = i - ac->t_start + 1;
            base = t * Nq1;
            for (j = 2; j < ac->Nq; ++j) {
                if (ac->otprob[i][j][0] == NULL)
                    ac->otprob[i][j][0] = (float *) New(mem, sizeof(float) * 1);
                ac->otprob[i][j][0][0] = (float) otprob[base + j];
            }
        }  

        /* Dispose(&arcstak, otprob); */
        Dispose(&arcstak, betaPlus); 
    }
}
#endif


/* cz277 - cuda fblat */
#ifdef CUDA
void SyncAcousticHost2DevAlpha(Acoustic *ac, AcousticDev *acDev) {
    NFloat locc;

    locc = ac->locc;
    /* locc */
    SyncHost2Dev(&locc, &acDev->locc, sizeof(float));
}
#endif


/* cz277 - cuda fblat */
#ifdef CUDA
void SyncAcousticDev2HostAlpha(AcousticDev *acDev, Acoustic *ac) {
    int size, i, j, t, T1, Nq1, base;
    AcousticDev acHost;
    NFloat *alphaPlus;

    SyncDev2Host(acDev, &acHost, sizeof(AcousticDev));
    T1 = acHost.t_end - acHost.t_start + 1 + 1;
    Nq1 = acHost.Nq + 1;
    /* alphaPlus */
    size = sizeof(NFloat) * T1 * Nq1;
    alphaPlus = (NFloat *) New(&arcstak, size);
    SyncDev2Host(acHost.alphaPlus, alphaPlus, size);
    for (i = ac->t_start; i <= ac->t_end; ++i) {
        t = i - ac->t_start + 1;
        base = t * Nq1;
        for (j = 1; j <= ac->Nq; ++j)
            ac->alphaPlus[i][j] = alphaPlus[base + j];
    }
    Dispose(&arcstak, alphaPlus);
    
}
#endif


void ArcFromLat(ArcInfo *aInfo, HMMSet *hset){
   Lattice *lat;
   int larcid,seg;
   int start_time;
   HArc *arc;
   int l;
   float framedur;

   /* cz277 - mtload */
   if(!StackInitialised) {
      CreateHeap(&tempArcStack, "tempArcStore", MSTAK, 1, 0.5, 10000, 100000);
      StackInitialised = TRUE;
   }

   aInfo->start=aInfo->end=0;
   aInfo->nArcs=0;
   if(IsLMScale)
      aInfo->lmScale = LMSCALE;
   else /* If none is specified use the one from the lattice. */
      aInfo->lmScale = aInfo->lat[0]->lmscale;
  
   /* Lattices actually never specify frame duration.  Usual source is config, FRAMEDUR. */
   /* if(lat->framedur != 0) framedur = lat->framedur / 10000000;   */
   if (FRAMEDUR) framedur = FRAMEDUR; /* config. */
   else framedur = 0.01;
   aInfo->framedur = framedur;

   if(!IsWdPen)
      aInfo->insPen = aInfo->lat[0]->wdpenalty;
   else aInfo->insPen = WDPEN;

   /*  Add all the arcs in the lattice into the Arc structure. */
   for(l=0;l<aInfo->nLats;l++) /* this handles multiple lattices so e.g. the correct lattice can be added to the recognition lattice
                                  in case it happens to be absent. */
      {
         lat = aInfo->lat[l];
         ZeroHooks(lat);
         FixLatTimes(lat); /* this can be deleted at some point. */
         for(larcid=0;larcid<lat->na;larcid++){
            start_time = TimeToNFrames(lat->larcs[larcid].start->time, aInfo);
      
            for(seg=0;seg<lat->larcs[larcid].nAlign;seg++){
               arc = CreateArc(aInfo->mem, lat, lat->larcs+larcid, start_time, seg, ++aInfo->nArcs, (seg==0?NULL:aInfo->end), aInfo->insPen, aInfo->lmScale, hset, aInfo);
               /* this creates the phone arc and also the transitions between them. (confusingly, phone 
                  arcs are actually nodes in the datastructure used here, and transitions are arcs)  */
	
               /*Insert into the linked list of arcs:*/
               if(!aInfo->start)  aInfo->start=arc;
               arc->prec = aInfo->end;
               arc->foll = NULL;
               if(aInfo->end) aInfo->end->foll = arc;
               aInfo->end = arc;
               start_time = aInfo->end->t_end;
            }
         }
      }
  

   SortArcs(aInfo);  /* Sort the arcs in order of start & end time
                        [ & in such a way that ones with identical HMMs are adjacent.] */

   if(BackTransitions(aInfo)){ /* A check-- should never be true if properly sorted */
      HError(1, "Error: file %s: back transitions exist.\n", (aInfo->lat[0]->utterance?aInfo->lat[0]->utterance:"[unknown]"));
   }


   /* Pool sets of identical models with identical start & end times. */
   for(arc=aInfo->start; arc; arc=arc->foll){
      HArc *arc2;
      if((arc2=arc->prec) != NULL && arc2->t_end==arc->t_end && arc2->t_start==arc->t_start && arc2->hmm==arc->hmm){
         if(arc2->calcArc == NULL) arc->calcArc = arc2;
         else arc->calcArc = arc2->calcArc;
      }
   }


   {  /*  Set up the structures which relate to the computation of model likelihoods. */
      int q=0,Q; MLink macroName;
      int T=0,t;
      for(arc=aInfo->start;arc;arc=arc->foll){
         if(arc->calcArc) arc->id = GHOST_ARC;
         else{ arc->id = ++q; }
         T=MAX(T,arc->t_end);
      }
      aInfo->Q = Q = q; /* num unique arcs. */
      aInfo->T = T; 
      aInfo->ac = New(aInfo->mem, sizeof(Acoustic) * (q+1)); 

      q=0;
      for(arc=aInfo->start;arc;arc=arc->foll){
         if(!arc->calcArc){  /* if this is one of the 'calculated' arcs */
            Acoustic *ac = arc->ac = aInfo->ac + ++q;

            ac->myArc=arc;
            ac->t_start=arc->t_start; ac->t_end=arc->t_end;
	
            if((macroName=FindMacroName(hset,'l',arc->phone))==NULL)
               HError(2321,"CreateArc: Unknown align label %s",arc->phone->name); 
            ac->hmm = (HLink)macroName->structure;  

            ac->Nq = ac->hmm->numStates; /* could be either HMM, same structure. */
            if(ac->t_start == ac->t_end+1){ 
               ac->SP=TRUE; 
               ac->alphat=ac->alphat1=NULL;ac->betaPlus=NULL;ac->otprob=NULL;
               /* cz277 - cuda fblat */
               ac->alphaPlus = NULL;
            } else {
               int j,s,SS,S = hset->swidth[0]; /* probably just 1. */
	       StreamElem *ste;

	       SS=(S==1)?1:S+1;
               ac->SP=FALSE;
               ac->alphat = CreateDVector(aInfo->mem, ac->Nq);
               ac->alphat1 = CreateDVector(aInfo->mem, ac->Nq);
               ac->betaPlus = ((DVector*)New(aInfo->mem, sizeof(DVector)*(ac->t_end-ac->t_start+1)))-ac->t_start;
               ac->otprob = ((float****)New(aInfo->mem, sizeof(float***)*(ac->t_end-ac->t_start+1)))-ac->t_start;
               /* cz277 - cuda fblat */
               ac->alphaPlus = ((DVector *) New(aInfo->mem, sizeof(DVector) * (ac->t_end - ac->t_start + 1))) - ac->t_start; 

               for(t=ac->t_start;t<=ac->t_end;t++){
                  ac->betaPlus[t] = CreateDVector(aInfo->mem,ac->Nq);
		  ac->otprob[t] = ((float***)New(aInfo->mem,(ac->Nq-2)*sizeof(float **)))-2;
                  /* cz277 - cuda fblat */
                  ac->alphaPlus[t] = CreateDVector(aInfo->mem, ac->Nq);

		  for(j=2;j<ac->Nq;j++){
                     ac->otprob[t][j] = (float**)New(aInfo->mem,SS*sizeof(float*)); /*2..Nq-1*/
		     ste = ac->hmm->svec[j].info->pdf+1;
		     if (S==1) {
		        ac->otprob[t][j][0] = NULL;
		     } else {
		        ac->otprob[t][j][0] = (float*)New(aInfo->mem,sizeof(float));
			ac->otprob[t][j][0][0] = LZERO;
			for (s=1;s<=S;s++,ste++)
			  ac->otprob[t][j][s] = NULL;
		     }
                  }
               }	
            }
         }
      }
      for(arc=aInfo->start;arc;arc=arc->foll)  if(arc->calcArc) arc->ac = arc->calcArc->ac;

      /* Set up arrays qLo[t] and qHi[t] which are used to speed up iterations over q. */
      aInfo->qLo = (int*)New(aInfo->mem, sizeof(int)*(aInfo->T+1));  /* For efficiency later, work out min & max q active at each time t. */
      aInfo->qHi = (int*)New(aInfo->mem, sizeof(int)*(aInfo->T+1));
      for(t=1;t<=T;t++){  aInfo->qLo[t] = T+1; aInfo->qHi[t] = -1;  }
      for(q=1;q<=Q;q++){
         for(t=aInfo->ac[q].t_start;t<=MAX(aInfo->ac[q].t_end,aInfo->ac[q].t_start);t++){ 
            /* the MAX above is to handle tee models, so they will be in beam at t==t_start. */
            aInfo->qLo[t] = MIN(aInfo->qLo[t], q);
            aInfo->qHi[t] = MAX(aInfo->qHi[t], q);
         }
      } 
   }

    /* cz277 - cuda fblat */
#ifdef CUDA
    if (aInfo->FBLatCUDA == TRUE) {
        /* acDev */
        DevNew((void **) &aInfo->acDev, sizeof(AcousticDev) * (aInfo->Q + 1));
        for (l = 1; l <= aInfo->Q; ++l) {
            InitAcousticDev(&aInfo->ac[l], &aInfo->acDev[l]);
        }
        /* qLoDev & qHiDev */
        DevNew((void **) &aInfo->qLoDev, sizeof(int) * (aInfo->T + 1));
        DevNew((void **) &aInfo->qHiDev, sizeof(int) * (aInfo->T + 1));
        SyncHost2Dev(aInfo->qLo, aInfo->qLoDev, sizeof(int) * (aInfo->T + 1));
        SyncHost2Dev(aInfo->qHi, aInfo->qHiDev, sizeof(int) * (aInfo->T + 1)); 
    }
#endif

   /* cz277 - arcs */
   /*if(trace&T_ARC && debug++ < 100){*/
   if(trace&T_ARC) {
      printf("\t\t[HArc:] %d arcs, depth %f, depth[reduced] %f\n", aInfo->nArcs, Depth(aInfo,TRUE), Depth(aInfo,FALSE));
      if(trace&T_ARC2)
         PrintArcs(stdout, aInfo->start);
   }
  
   ResetHeap(&tempArcStack);
}

void AttachMPEInfo(ArcInfo *aInfo){  /* attach the "mpe" structure to the arcs. */
   HArc *a;
   for(a=aInfo->start;a;a=a->foll){
      if(a->mpe!=NULL) HError(1, "a->mpe != NULL, possible memory or coding error");
      a->mpe = New(aInfo->mem, sizeof(MPEStruct));
   }
}


Boolean SameArcs(LArc *a1, LArc *a2){ /*word arcs have same times & phones */
   int seg;
   if(a1->nAlign!=a2->nAlign) return FALSE;
   if(a1->start->time!=a2->start->time) return FALSE;
   for(seg=0;seg<a1->nAlign;seg++) 
      if (a1->lAlign[seg].dur != a2->lAlign[seg].dur || 
          a1->lAlign[seg].label != a2->lAlign[seg].label) return FALSE;
   return TRUE; 
}

Boolean LatInLatRec(LNode *n1, LNode *n2){

   if(n1->foll==NULL && n2->foll==NULL){
      return TRUE; /*Path found.*/
   }

   if(n1->pred && n1->pred->parc != NULL) {
       return TRUE; /*This is to take the case where the numerator could have choices of arcs (multiple prons).
                                                         Where the arcs re-join, just check the last one re-joining.  This makes sure we don't take exponential time.*/
   }

   if(n1->foll==NULL || n2->foll==NULL) {
      return FALSE;
   }
   else{
      LNode *newn1;
      LArc *la;
      LArc *n1FollArc;
    
      for(n1FollArc=n1->foll;n1FollArc;n1FollArc=n1FollArc->farc){ /*For each following arc.  This could make the time exponential, were it not for
                                                                     the code above (if(n1->pred->parc!=NULL) return TRUE). */
         Boolean ThisIsOK = FALSE;
         newn1 = n1FollArc->end;
      
         for(la = n2->foll;la;la=la->farc){
            LNode *newn2 = la->end;
            if(newn2->word == newn1->word  &&
               SameArcs(n1FollArc, la))
               if(LatInLatRec(newn1, newn2)) ThisIsOK = TRUE; /*We've found a path to the end (or to re-join point).  --> this num pron is OK.*/
         }
         if (!ThisIsOK) {
             return FALSE; /*This num pron was not there.*/
         }
      }
      return TRUE;
   }
}

Boolean LatInLat(Lattice *numLat, Lattice *denLat){
   return LatInLatRec(numLat->lnodes+0, denLat->lnodes+0);
}



/* ------------------------------------ Initialisation ------------------------------------ */

/* EXPORT->InitArc: initialise configuration parameters */
void InitArc(void)
{
   int i;
   double f;

   Register(arc_version,arc_vc_id);
   nParm = GetConfig("HARC", TRUE, cParm, MAXGLOBS);
   if (nParm>0){
      if (GetConfInt(cParm,nParm,"TRACE",&i)) trace = i;                          /*   Set TRACE=1 for minimal tracing, 3 for fuller tracing.  */
      if (GetConfFlt(cParm,nParm,"LMSCALE",&f)){ LMSCALE = f; IsLMScale = TRUE; } /*   Overrides lattice-specified one.  */
      if (GetConfFlt(cParm,nParm,"FRAMEDUR",&f)){ FRAMEDUR = f; }                 /*   Important.  Frame duration in seconds.  If != 0.01, specify it. */
      if (GetConfFlt(cParm,nParm,"WDPEN",&f)){ WDPEN = f; IsWdPen = TRUE; }       /*   Overrides lattice-specified one.  */
   }
   /* cz277 - mtload */
   CreateHeap(&arcstak, "HArc Stack", MSTAK, 1, 0.0, 100000, ULONG_MAX);
}

/* ------------------------- End of HArc.c ------------------------- */

