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
/*             File: HMath.c   Math support module             */
/* ----------------------------------------------------------- */

char *hmath_version = "!HVER!HMath:   3.5.0 [CUED 12/10/15]";
char *hmath_vc_id = "$Id: HMath.c,v 1.2 2015/10/12 12:07:24 cz277 Exp $";

/*
   This library provides math support in the following three areas
   
      a) Vector/Matrix Operators
      b) Log Arithmetic
      c) Random Deviates
      
   It relies on the basic vector and matrix types defined by HMem.
   The separation of functionality betwee HMem and HMath is such
   that no routine in this module requires any knowledge of the
   hidden fields in these types.  Thus, if a change of representation
   is required, it should only be necessary to change HMem.
*/

#include "HShell.h"        /* HTK Libraries */
#include "HMem.h"
#include "HMath.h"
#include "config.h"

#ifdef CUDA
#include "HCUDA.h"
#endif

#ifdef MKL
#include "mkl_lapacke.h"
#endif

/* ----------------------------- Trace Flags ------------------------- */

static int trace = 0;

#define T_DIM   0002        /* Matrix/Vector dimension checking */
/* cz277 - 1004 */
#define MAXSVDITER 30
#define SIGN(a, b) (b > 0.0 ? fabs(a): -fabs(a))
#define MAX(a, b) ((a)>(b)? (a): (b))
#define MIN(a, b) ((a)<(b)? (a): (b))

/* -------------------- Configuration Parameters --------------------- */

static ConfParam *cParm[MAXGLOBS];       /* config parameters */
static int numParm = 0;
/* from mjfg - cz277 141022 */
#define MAXMATRIXSIZE 400

/* cz277 - ANN */
#ifdef MKL
static int nMKLThreads = 1;             /* the number of threads used by CPU LAPACK/BLAS kernels (MKL) */
static char *nMKLThreadEnvVar = "";
#endif
static NMatrix *tmpNMat = NULL;         /* the pointer to the temp matrix */
static NVector *tmpNVec = NULL;
static int tmpRowNum = 1;               /* the row number of the temp matrix*/
static int tmpColNum = 1;               /* the column number of the temp matrix */

/* ------------------ Vector Oriented Routines ----------------------- */

/*
   ShortVecs and IntVecs are pointers to arrays of short/int.  Vectors
   are pointers to arrays of float (ie float*).  All indexing is
   v[1..n].  The size is stored in v[0].
*/

/* EXPORT->ZeroShortVec: Zero the elements of v */
void ZeroShortVec(ShortVec v)
{
   int i,n;
   
   n=ShortVecSize(v);
   for (i=1;i<=n;i++) v[i]=0;
}

/* EXPORT->ZeroIntVec: Zero the elements of v */
void ZeroIntVec(IntVec v)
{
   int i,n;
   
   n=IntVecSize(v);
   for (i=1;i<=n;i++) v[i]=0;
}

/* EXPORT->ZeroVector: Zero the elements of v */
void ZeroVector(Vector v)
{
   int i,n;
   
   n=VectorSize(v);
   for (i=1;i<=n;i++) v[i]=0.0;
}

/* EXPORT->ZeroDVector: Zero the elements of v */
void ZeroDVector(DVector v)
{
   int i,n;
   
   n=DVectorSize(v);
   for (i=1;i<=n;i++) v[i]=0.0;
}

/* EXPORT->CopyShortVec: copy v1 into v2 */
void CopyShortVec(ShortVec v1, ShortVec v2)
{
   int i,size; 
   
   size = ShortVecSize(v1);
   if (size != ShortVecSize(v2))
      HError(5270,"CopyShortVec: sizes differ %d vs %d",
             size,ShortVecSize(v2));
   for (i=1; i<=size; i++) 
      v2[i] = v1[i];
}

/* EXPORT->CopyIntVec: copy v1 into v2 */
void CopyIntVec(IntVec v1, IntVec v2)
{
   int i,size; 
   
   size = IntVecSize(v1);
   if (size != IntVecSize(v2))
      HError(5270,"CopyIntVec: sizes differ %d vs %d",
             size,IntVecSize(v2));
   for (i=1; i<=size; i++) 
      v2[i] = v1[i];
}

/* EXPORT->CopyVector: copy v1 into v2 */
void CopyVector(Vector v1, Vector v2)
{
   int i,size; 
   
   size = VectorSize(v1);
   if (size != VectorSize(v2))
      HError(5270,"CopyVector: sizes differ %d vs %d",
             size,VectorSize(v2));
   for (i=1; i<=size; i++) 
      v2[i] = v1[i];
}

/* EXPORT->CopyDVector: copy v1 into v2 */
void CopyDVector(DVector v1, DVector v2)
{
   int i,size; 
   
   size = DVectorSize(v1);
   if (size != DVectorSize(v2))
      HError(5270,"CopyDVector: sizes differ %d vs %d",
             size,DVectorSize(v2));
   for (i=1; i<=size; i++) 
      v2[i] = v1[i];
}

/* EXPORT->ReadShortVec: read vector from src in ascii or binary */ 
Boolean ReadShortVec(Source *src, ShortVec v, Boolean binary)
{
   return ReadShort(src,v+1,ShortVecSize(v),binary);
}

/* EXPORT->ReadIntVec: read vector from src in ascii or binary */ 
Boolean ReadIntVec(Source *src, IntVec v, Boolean binary)
{
   return ReadInt(src,v+1,IntVecSize(v),binary);
}

/* EXPORT->ReadVector: read vector from src in ascii or binary */ 
Boolean ReadVector(Source *src, Vector v, Boolean binary)
{
   return ReadFloat(src,v+1,VectorSize(v),binary);
}

/* EXPORT->WriteShortVec: write vector v to stream f */
void WriteShortVec(FILE *f, ShortVec v, Boolean binary)
{
   WriteShort(f,v+1,ShortVecSize(v),binary);
   if (!binary) fputc('\n',f);
}

/* EXPORT->WriteIntVec: write vector v to stream f */
void WriteIntVec(FILE *f, IntVec v, Boolean binary)
{
   WriteInt(f,v+1,IntVecSize(v),binary);
   if (!binary) fputc('\n',f);
}

/* EXPORT->WriteVector: write vector v to stream f */
void WriteVector(FILE *f, Vector v, Boolean binary)
{
   WriteFloat(f,v+1,VectorSize(v),binary);
   if (!binary) fputc('\n',f);
}

/* Export->ShowShortVec: show the short vector v preceded by title */
void ShowShortVec(char * title, ShortVec v,int maxTerms)
{
   int i, size, maxi;
   
   size = maxi = ShortVecSize(v);
   if (maxi>maxTerms) maxi=maxTerms;
   printf("%s\n   ",title);   
   for (i=1;i<=maxi;i++)  printf("%3d ",v[i]);
   if (maxi<size) printf("...");
   printf("\n");
}

/* Export->ShowIntVec: show the int vector v preceded by title */
void ShowIntVec(char * title, IntVec v,int maxTerms)
{
   int i, size, maxi;
   
   size = maxi = IntVecSize(v);
   if (maxi>maxTerms) maxi=maxTerms;
   printf("%s\n   ",title);   
   for (i=1;i<=maxi;i++)  printf("%5d ",v[i]);
   if (maxi<size) printf("...");
   printf("\n");
}

/* Export->ShowVector: show the vector v preceded by title */
void ShowVector(char * title,Vector v,int maxTerms)
{
   int i, size, maxi;
   
   size = maxi = VectorSize(v);
   if (maxi>maxTerms) maxi=maxTerms;
   printf("%s\n   ",title);   
   for (i=1;i<=maxi;i++)  printf("%8.2f ",v[i]);
   if (maxi<size) printf("...");
   printf("\n");
}

/* Export->ShowDVector: show the vector v preceded by title */
void ShowDVector(char * title, DVector v,int maxTerms)
{
   int i, size, maxi;
   
   size = maxi = DVectorSize(v);
   if (maxi>maxTerms) maxi=maxTerms;
   printf("%s\n   ",title);   
   for (i=1;i<=maxi;i++)  printf("%10.4f ",v[i]);
   if (maxi<size) printf("...");
   printf("\n");
}

/* ------------------ Matrix Oriented Routines ----------------------- */

/*
  Matrices are pointers to an array of vectors (ie float**).  Matrices
  are indexed m[1..r][1..c].  The rows of matrix are genuine vectors
  and can be treated as such except that matrices must be disposed of
  in their entirety.  If the rows of a matrix are substituted by
  other vectors then it should not be disposed.  TriMats are indexed
  m[1..r][1..i] where i is the row number ie only the lower triangle
  is stored.
*/

/* EXPORT->ZeroMatrix: Zero the elements of m */
void ZeroMatrix(Matrix m)
{
   int i,j,nr,nc;
   
   nr=NumRows(m); nc=NumCols(m);
   for (i=1;i<=nr;i++)
      for (j=1;j<=nc;j++) m[i][j]=0.0;
}

/* EXPORT->ZeroDMatrix: Zero the elements of m */
void ZeroDMatrix(DMatrix m)
{
   int i,j,nr,nc;
   
   nr=NumDRows(m); nc=DVectorSize(m[1]);
   for (i=1;i<=nr;i++)
      for (j=1;j<=nc;j++) m[i][j]=0.0;
}

/* EXPORT->ZeroTriMat: Zero the elements of m */
void ZeroTriMat(TriMat m)
{
   int i,j,size;
   
   size = TriMatSize(m);
   for (i=1;i<=size;i++)
      for (j=1;j<=i;j++) m[i][j]=0.0;
}

/* EXPORT->CopyMatrix: copy matrix m1 to m2 */
void CopyMatrix(Matrix m1, Matrix m2)
{
   int i,nrows;
   
   nrows = NumRows(m1);
   if (nrows != NumRows(m2))
      HError(5270,"CopyMatrix: row sizes differ %d vs %d",
             nrows,NumRows(m2));
   for (i=1; i<=nrows; i++)
      CopyVector(m1[i],m2[i]);
}

/* EXPORT->CopyDMatrix: copy matrix m1 to m2 */
void CopyDMatrix(DMatrix m1, DMatrix m2)
{
   int i,nrows;
   
   nrows = NumDRows(m1);
   if (nrows != NumDRows(m2))
      HError(5270,"CopyDMatrix: row sizes differ %d vs %d",
             nrows,NumDRows(m2));
   for (i=1; i<=nrows; i++)
      CopyDVector(m1[i],m2[i]);
}

/* EXPORT->CopyTriMat: copy triangular matrix m1 to m2 */
void CopyTriMat(TriMat m1, TriMat m2)
{
   int i,size;
   
   size = TriMatSize(m1);
   if (size != TriMatSize(m2))
      HError(5270,"CopyTriMat: sizes differ %d vs %d",
             size,TriMatSize(m2));
   for (i=1; i<=size; i++)
      CopyVector(m1[i],m2[i]);
}

/* EXPORT->Mat2DMat: convert matrix m1 to double matrix m2 */
void Mat2DMat(Matrix m1,  DMatrix m2)
{
   int i,j,nrows,ncols;

   nrows = NumRows(m1);
   if (nrows != NumDRows(m2))
      HError(5270,"Mat2DMat: row sizes differ %d vs %d",
             nrows,NumDRows(m2));
   ncols = NumCols(m1);
   if (ncols != NumDCols(m2))
      HError(5270,"Mat2DMat: col sizes differ %d vs %d",
             ncols,NumDCols(m2));
   for (i=1; i<=nrows; i++)
      for (j=1; j<=ncols; j++) 
         m2[i][j] = m1[i][j];
}

/* EXPORT->DMat2Mat: convert double matrix m1 to matrix m2 */
void DMat2Mat(DMatrix m1, Matrix m2)
{
   int i,j,nrows,ncols;

   nrows = NumDRows(m1);
   if (nrows != NumRows(m2))
      HError(5270,"DMat2Mat: row sizes differ %d vs %d",
             nrows,NumRows(m2));
   ncols = NumDCols(m1);
   if (ncols != NumCols(m2))
      HError(5270,"DMat2Mat: col sizes differ %d vs %d",
             ncols,NumCols(m2));
   for (i=1; i<=nrows; i++)
      for (j=1; j<=ncols; j++) 
         m2[i][j] = m1[i][j];
}

/* EXPORT->Mat2Tri: convert matrix m1 to tri matrix m2 */
void Mat2Tri (Matrix m1,  TriMat m2)
{
   int i,j,nrows,ncols;

   nrows = NumRows(m1); ncols = NumCols(m1);
   if (nrows != ncols)
      HError(5270,"Mat2Tri: source matrix not square %d vs %d",
             nrows,ncols);   
   if (ncols != TriMatSize(m2))
      HError(5270,"Mat2Tri: sizes differ %d vs %d",
             ncols,TriMatSize(m2));
   for (i=1; i<=nrows; i++)
      for (j=1; j<=i; j++) 
         m2[i][j] = m1[i][j];
}

/* EXPORT->Tri2Mat: convert tri matrix m1 to matrix m2 */
void Tri2Mat (TriMat m1, Matrix m2)
{
   int i,j,nrows,ncols;

   nrows = NumRows(m2); ncols = NumCols(m2);
   if (nrows != ncols)
      HError(5270,"Tri2Mat: target matrix not square %d vs %d",
             nrows,ncols);   
   if (ncols != TriMatSize(m1))
      HError(5270,"Tri2Mat: sizes differ %d vs %d",
             TriMatSize(m1),ncols);
   for (i=1; i<=nrows; i++)
      for (j=1; j<=i; j++) {
         m2[i][j] = m1[i][j];
         if (i!=j) m2[j][i] = m1[i][j];
      }
}

/* EXPORT->ReadMatrix: read matrix from source into m */
Boolean ReadMatrix(Source *src, Matrix m, Boolean binary)
{
   int i,nrows;
   
   nrows = NumRows(m);
   for (i=1; i<=nrows; i++)
      if (!ReadVector(src,m[i],binary)) 
         return FALSE;
   return TRUE;
}

/* EXPORT->ReadTriMat: read symmetric matrix in lower triangular
                       form from source into m */
Boolean ReadTriMat(Source *src, TriMat m, Boolean binary)
{
   int i,j,size;
   
   size = TriMatSize(m);
   for (j=1; j<=size; j++) {
      for (i=j; i<=size; i++)
         if (!ReadFloat(src,&(m[i][j]),1,binary))
            return FALSE;
   }
   return TRUE;
}

/* EXPORT->WriteMatrix: write matrix to f */
void WriteMatrix(FILE *f, Matrix m, Boolean binary)
{
   int i,nrows;
   
   nrows = NumRows(m);
   for (i=1; i<=nrows; i++)
      WriteVector(f,m[i],binary);
}

/* EXPORT->WriteTriMat: write symmetric matrix to stream f in
                        upper triangular form */
void WriteTriMat(FILE *f, TriMat m, Boolean binary)
{
   int i,j,size;
   
   size = TriMatSize(m);
   for (j=1; j<=size; j++) {
      for (i=j; i<=size; i++)
         WriteFloat(f,&(m[i][j]),1,binary);
      if (!binary) fputc('\n',f);
   }
}

/* Export->ShowMatrix: show the matrix m preceded by title */
void ShowMatrix(char * title,Matrix m,int maxCols,int maxRows)
{
   int i,j;
   int maxi,maxj,nrows,ncols;
   
   maxi = nrows = NumRows(m);
   if (maxi>maxRows) maxi = maxRows;
   maxj = ncols = NumCols(m);
   if (maxj>maxCols) maxj = maxCols;
   printf("%s\n",title);
   for (i=1;i<=maxi;i++) {
      printf("   ");
      for (j=1;j<=maxj;j++)
         printf("%8.2f ",m[i][j]);
      if (maxj<ncols) printf("...");
      printf("\n");
   }
   if (maxi<nrows)
      printf("   ...\n");
}

/* Export->ShowDMatrix: show the matrix m preceded by title */
void ShowDMatrix(char * title,DMatrix m,int maxCols,int maxRows)
{
   int i,j;
   int maxi,maxj,nrows,ncols;
   
   maxi = nrows = NumDRows(m);
   if (maxi>maxRows) maxi = maxRows;
   maxj = ncols = DVectorSize(m[1]);
   if (maxj>maxCols) maxj = maxCols;
   printf("%s\n",title);
   for (i=1;i<=maxi;i++) {
      printf("   ");
      for (j=1;j<=maxj;j++)
         printf("%10.4f ",m[i][j]);
      if (maxj<ncols) printf("...");
      printf("\n");
   }
   if (maxi<nrows)
      printf("   ...\n");
}

/* Export->ShowTriMat: show the matrix m preceded by title */
void ShowTriMat(char * title,TriMat m,int maxCols,int maxRows)
{
   int i,j;
   int maxi,maxj,size;
   
   size = TriMatSize(m);
   maxi = size;
   if (maxi>maxRows) maxi = maxRows;
   printf("%s\n",title);
   for (i=1;i<=maxi;i++) {
      printf("   ");
      maxj = i;
      if (maxj>maxCols) maxj = maxCols;
      for (j=1;j<=maxj;j++)
         printf("%8.2f ",m[i][j]);
      if (maxj<i) printf("...");
      printf("\n");
   }
   if (maxi<size)
      printf("   ...\n");
}

/* -------------------- Matrix Operations ---------------------- */

/* Choleski: Place lower triangular choleski factor of A in L.*/
/*           Return FALSE if matrix singular or not +definite */
static Boolean Choleski(TriMat A, DMatrix L)
{
   int size,i,j,k;
   double sum;

   size = TriMatSize(A);
   for (i=1; i<=size; i++)
      for (j=1; j<=i; j++) {
         sum=A[i][j];
         for (k=1; k<j; k++)
            sum -= (L[i][k]*L[j][k]);
         if ((i==j)&&(sum<=0.0)) 
            return FALSE;
         else if (i==j)
            sum = sqrt(sum);
         else if (L[j][j]==0.0)
            return FALSE;
         else
            sum /= L[j][j];
         L[i][j] = sum;
      }
   for (i=1; i<=size; i++) 
      for (j=i+1; j<=size; j++) 
         L[i][j] = 0.0;
   return TRUE;
}

/* MSolve: solve Ly=e^i and L^t x = y, where e^i is a unit vector */
static void MSolve(DMatrix L, int i, DVector x, DVector y)
{
   int nr,j,k;
   double sum;
   
   nr=NumDRows(L);
   for (j=1; j<i; j++) y[j] = 0.0;  /* forward sub */
   y[i] = 1.0/L[i][i];
   for (j=i+1; j<=nr; j++){
      sum = 0.0;
      for (k=i; k<j; k++)
         sum -= L[j][k]*y[k];
      y[j] = sum /L[j][j];
   }
   x[nr] = y[nr]/L[nr][nr];         /* backward sub */
   for (j=nr-1; j>=1; j--){
      sum = y[j];
      for (k=j+1; k<=nr; k++)
         sum -= L[k][j]*x[k];
      x[j] = sum / L[j][j];
   }
}

/* EXPORT->CovInvert: puts inverse of c in invc, returns log(Det(c)) */
/*          Note that c must be positive definite */
LogFloat CovInvert(TriMat c, Matrix invc)
{
   DMatrix l;     /* Lower Tri Choleski Matrix */
   DVector x,y;   /* for f/b substitution */
   LogFloat ldet = 0.0;
   int i,j,n;
   Boolean isTri;
   
   n = TriMatSize(c); isTri = IsTriMat(invc);
   l = CreateDMatrix(&gstack,n,n);
   x = CreateDVector(&gstack,n);
   y = CreateDVector(&gstack,n);
   if (Choleski(c,l)){
      for (j=1; j<=n; j++){
         MSolve(l,j,x,y);
         for (i=isTri?j:1; i<=n; i++)
            invc[i][j] = x[i];
         ldet += log(l[j][j]);
      }
   } else
      HError(5220,"CovInvert: [%f ...] not invertible",c[1][1]);
   FreeDMatrix(&gstack,l);    /* cut back stack to entry state */
   return 2.0*ldet;
}

/* EXPORT->CovDet: Returns log(Det(c)), c must be positive definite */
LogFloat CovDet(TriMat c)
{
   DMatrix l;  /* Lower Tri Choleski Matrix */
   LogFloat ldet = 0.0;
   int j,n;
   
   n = TriMatSize(c);
   l = CreateDMatrix(&gstack,n,n);
   if (Choleski(c,l)){
      for (j=1; j<=n; j++)
         ldet += log(l[j][j]);
   } else
      HError(5220,"CovDet: [%f ...] not invertible",c[1][1]);
   FreeDMatrix(&gstack,l);
   return 2.0*ldet;
}

/* Quadratic prod of a full square matrix C and an arbitry full matrix transform A */
void LinTranQuaProd(Matrix Prod, Matrix A, Matrix C)
{
   int i,j,k;
   float tempElem;
   Matrix tempMatrix_A_mult_C;
   
   if (NumRows(C) != NumCols(C)){
      HError(5221,"HMath: LinTranQuaProd: Matrix C is not square!\n");
   }
   else {
      tempMatrix_A_mult_C = CreateMatrix(&gstack,NumRows(A),NumCols(C));
      ZeroMatrix(tempMatrix_A_mult_C);
      
      for (i=1;i<=NumRows(tempMatrix_A_mult_C);i++){
         for (j=1;j<=NumCols(tempMatrix_A_mult_C);j++){
            tempElem = 0.0;
            for (k=1;k<=NumCols(A);k++){
               tempElem += A[i][k]*C[j][k];
            }
            tempMatrix_A_mult_C[i][j] = tempElem;
         }
      }
      
      for (i=1;i<=NumRows(Prod);i++){
         for (j=1;j<=i;j++){
            tempElem = 0.0;
            for (k=1;k<=NumCols(tempMatrix_A_mult_C);k++){
               tempElem += tempMatrix_A_mult_C[i][k]*A[j][k];
            }
            Prod[i][j] = tempElem;
         }
      }
      
      for (i=1;i<=NumRows(Prod);i++){
         for (j=1;j<i;j++){
            Prod[j][i] = Prod[i][j];
         }
      }    
      
      FreeMatrix(&gstack,tempMatrix_A_mult_C);
   }
}


/* ------------- Singular Value Decomposition --------------- */
/**************************************************************************
 **
 ** Copyright (C) 1993 David E. Steward & Zbigniew Leyk, all rights reserved.
 **
 **                          Meschach Library
 ** 
 ** This Meschach Library is provided "as is" without any express 
 ** or implied warranty of any kind with respect to this software. 
 ** In particular the authors shall not be liable for any direct, 
 ** indirect, special, incidental or consequential damages arising 
 ** in any way from use of the software.
** 
** Everyone is granted permission to copy, modify and redistribute this
** Meschach Library, provided:
**  1.  All copies contain this copyright notice.
**  2.  All modified copies shall carry a notice stating who
**      made the last modification and the date of such modification.
**  3.  No charge is made for this software or works derived from it.  
**      This clause shall not be construed as constraining other software
**      distributed on the same medium as this software, nor is a
**      distribution fee considered a charge.
**
**  Modifications made to conform with HTK formats by 
**  Daniel Kershaw, Entropic Ltd, Cambridge, England.
**
***************************************************************************/
#define MACHEPS 2.22045e-16
#define FZERO 1.0e-6
#define sgn(x)  ((x) >= 0 ? 1 : -1)
#define minab(a,b) ((a) > (b) ? (b) : (a))
#define MAX_STACK       100

/* Givens -- returns c,s parameters for Givens rotation to
   eliminate y in the vector [ x y ]' */
static void Givens(double x, double y, double *c, double *s)
{
   double norm;
  
   norm = sqrt(x*x+y*y);
   if ( norm == 0.0 ) {
      *c = 1.0;
      *s = 0.0;       
   }       /* identity */
   else {
      *c = x/norm;
      *s = y/norm;
   }
}


/* RotRows -- premultiply mat by givens rotation described by c,s */
static void RotRows(DMatrix M, int i, int k, 
                    double c, double s)
{
   int   j, n;
   double temp;
  
   n = NumDRows(M);
  
   if (i > n || k > n)
      HError(1, "RotRows: Index tooo big i=%d k=%d\n", i, k);
  
  
   for ( j=1; j<=n; j++ ) {
      temp = c*M[i][j] + s*M[k][j];
      M[k][j] = -s*M[i][j] + c*M[k][j];
      M[i][j] = temp;
   }
  
}

/* FixSVD -- fix minor details about SVD make singular values non-negative
   -- sort singular values in decreasing order */
static void FixSVD(DVector d, DMatrix U, DMatrix V)
{
  
   int  i, j, n;
  
   n = DVectorSize(d);


   /* make singular values non-negative */
   for (i = 1; i <= n; i++) {
      if ( d[i] < 0.0 ) {
         d[i] = - d[i];
         for ( j = 1; j <= NumDRows(U); j++ )
            U[i][j] = - U[i][j];
      }
   }

   return;

#if 0 /* #### ge: what is this code after return supposed to do here? */
   {
      int  k, l, r, stack[MAX_STACK], sp;
      double tmp, v;

   /* sort singular values */
   sp = -1;
   l = 1;       
   r = n;
   for ( ; ; ) {
      while ( r >= l ) {
         /* i = partition(d->ve,l,r) */
         v = d[r];
         i = l-1;           
         j = r;
         for ( ; ; ) {
            /* inequalities are "backwards" for **decreasing** order */
            while ( d[++i] > v );
            while ( d[--j] < v );
            if ( i >= j )
               break;
            /* swap entries in d->ve */
            tmp = d[i];   
            d[i] = d[j];
            d[j] = tmp;
            /* swap rows of U & V as well */
            for ( k = 1; k <= DVectorSize(U[1]); k++ ) {
               tmp = U[i][k];
               U[i][k] = U[j][k];
               U[j][k] = tmp;
            }
            for ( k = 1; k <= DVectorSize(V[1]); k++ ) {
               tmp = V[i][k];
               V[i][k] = V[j][k];
               V[j][k] = tmp;
            }
         }
         tmp = d[i];
         d[i] = d[r];
         d[r] = tmp;
         for ( k = 1; k <= DVectorSize(U[1]); k++ ) {
            tmp = U[i][k];
            U[i][k] = U[r][k];
            U[r][k] = tmp;
         }
         for ( k = 1; k <= DVectorSize(V[1]); k++ ) {
            tmp = V[i][k];
            V[i][k] = V[r][k];
            V[r][k] = tmp;
         }
         /* end i = partition(...) */
         if ( i - l > r - i ) {
            stack[++sp] = l;    
            stack[++sp] = i-1;
            l = i+1;    
         }
         else {
            stack[++sp] = i+1;
            stack[++sp] = r;
            r = i-1;    
         }
      }
      if ( sp < 0 )
         break;
      r = stack[sp--];
      l = stack[sp--];
   }
   }
#endif
}

/* BiSvd -- svd of a bidiagonal m x n matrix represented by d (diagonal) and
   f (super-diagonals) */
static void BiSVD(DVector d, DVector f, DMatrix U, DMatrix V)
{
   int i, j, n;
   int i_min, i_max, split;
   double c, s, shift, size, z;
   double d_tmp, diff, t11, t12, t22;

   if ( ! d || ! f )
      HError(1,"BiSVD: Vectors are null!");
   if ( DVectorSize(d) != DVectorSize(f) + 1 )
      HError(1, "BiSVD: Error with the vector sizes!");
 
   n = DVectorSize(d);
  
   if ( ( U && DVectorSize(U[1]) < n ) || ( V && NumDRows(V) < n ) )
      HError(1, "BiSVD: Error Matrix sizes!");
   if ( ( U && NumDRows(U) != DVectorSize(U[1])) || 
        ( V && NumDRows(V) != DVectorSize(V[1])) )
      HError(1, "BiSVD: One of the matrices must be square");
  
  
   if ( n == 1 )
      return;
    
   s = 0.0;
   for ( i = 1; i <= n; i++)
      s += d[i]*d[i];
   size = sqrt(s);
   s = 0.0;
   for ( i = 1; i < n; i++)
      s += f[i]*f[i];
   size += sqrt(s);
   s = 0.0;
  
   i_min = 1;
   while ( i_min <= n ) {   /* outer while loop */
      /* find i_max to suit;
         submatrix i_min..i_max should be irreducible */
      i_max = n;
      for ( i = i_min; i < n; i++ )
         if ( d[i] == 0.0 || f[i] == 0.0 ) {
            i_max = i;
            if ( f[i] != 0.0 ) {
               /* have to ``chase'' f[i] element out of matrix */
               z = f[i];
               f[i] = 0.0;
               for ( j = i; j < n && z != 0.0; j++ ) {
                  Givens(d[j+1],z, &c, &s);
                  s = -s;
                  d[j+1] =  c*d[j+1] - s*z;
                  if ( j+1 < n ) {
                     z      = s*f[j+1];
                     f[j+1] = c*f[j+1];
                  }
                  RotRows(U,i,j+1,c,s);
               }
            }
            break;
         }    
  

      if ( i_max <= i_min ) {
         i_min = i_max + 1;
         continue;
      }

      split = FALSE;
      while ( ! split ) {
         /* compute shift */
         t11 = d[i_max-1]*d[i_max-1] +
            (i_max > i_min+1 ? f[i_max-2]*f[i_max-2] : 0.0);
         t12 = d[i_max-1]*f[i_max-1];
         t22 = d[i_max]*d[i_max] + f[i_max-1]*f[i_max-1];
         /* use e-val of [[t11,t12],[t12,t22]] matrix
            closest to t22 */
         diff = (t11-t22)/2;
         shift = t22 - t12*t12/(diff +
                                sgn(diff)*sqrt(diff*diff+t12*t12));
      
         /* initial Givens' rotation */
         Givens(d[i_min]*d[i_min]-shift,
                d[i_min]*f[i_min], &c, &s);
      
         /* do initial Givens' rotations */
         d_tmp      = c*d[i_min] + s*f[i_min];
         f[i_min]   = c*f[i_min] - s*d[i_min];
         d[i_min]   = d_tmp;
         z          = s*d[i_min+1];
         d[i_min+1] = c*d[i_min+1];
         RotRows(V,i_min,i_min+1,c,s);

         /* 2nd Givens' rotation */
         Givens(d[i_min],z, &c, &s);
         d[i_min]   = c*d[i_min] + s*z;
         d_tmp      = c*d[i_min+1] - s*f[i_min];
         f[i_min]   = s*d[i_min+1] + c*f[i_min];
         d[i_min+1] = d_tmp;
         if ( i_min+1 < i_max ) {
            z          = s*f[i_min+1];
            f[i_min+1] = c*f[i_min+1];
         }
         RotRows(U,i_min,i_min+1,c,s);
      
         for ( i = i_min+1; i < i_max; i++ ) {
            /* get Givens' rotation for zeroing z */
            Givens(f[i-1],z, &c, &s);
            f[i-1] = c*f[i-1] + s*z;
            d_tmp  = c*d[i] + s*f[i];
            f[i]   = c*f[i] - s*d[i];
            d[i]   = d_tmp;
            z      = s*d[i+1];
            d[i+1] = c*d[i+1];
            RotRows(V,i,i+1,c,s);

            /* get 2nd Givens' rotation */
            Givens(d[i],z, &c, &s);
            d[i]   = c*d[i] + s*z;
            d_tmp  = c*d[i+1] - s*f[i];
            f[i]   = c*f[i] + s*d[i+1];
            d[i+1] = d_tmp;
            if ( i+1 < i_max ) {
               z      = s*f[i+1];
               f[i+1] = c*f[i+1];
            }
            RotRows(U,i,i+1,c,s);
         }
         /* should matrix be split? */
         for ( i = i_min; i < i_max; i++ )
            if ( fabs(f[i]) <
                 MACHEPS*(fabs(d[i])+fabs(d[i+1])) )
               {
                  split = TRUE;
                  f[i] = 0.0;
               }
            else if ( fabs(d[i]) < MACHEPS*size )
               {
                  split = TRUE;
                  d[i] = 0.0;
               }
      }
   }
}

/* HholdVec -- calulates Householder vector to eliminate all entries after the
   i0 entry of the vector vec. It is returned as out. May be in-situ */
static void HholdVec(DVector tmp, int i0, int size,
                     double *beta, double *newval)
{
   int i;
   double norm = 0.0;

   for (i = i0; i <= size; i++) {
      norm += tmp[i]*tmp[i];
   }
   norm = sqrt(norm);

   if ( norm <= 0.0 ) {
      *beta = 0.0;
   }
   else {
      *beta = 1.0/(norm * (norm+fabs(tmp[i0])));
      if ( tmp[i0] > 0.0 )
         *newval = -norm;
      else
         *newval = norm;
      tmp[i0] -= *newval;
   }

}


/* HholdTrRows -- transform a matrix by a Householder vector by rows
   starting at row i0 from column j0 -- in-situ */
static void HholdTrRows(DMatrix M, int i0, int j0, DVector hh, double beta)
{
   double ip, scale;
   int i, j;
   int m,n;

   m = NumDRows(M);
   n = DVectorSize(M[1]);

   if ( M==NULL || hh==NULL )
      HError(1,"HholdTrRows: matrix or vector is NULL!");
   if ( DVectorSize(hh) != n )
      HError(1,"HholdTrRows: hh vector size must = number of M columns");
   if ( i0 > m+1 || j0 > n+1 )
      HError(1,"HholdTrRows: Bounds matrix/vec size error i=%d j=%d m=%d n=%d",
             i0, j0, m, n);
  
   if ( beta != 0.0 ) {
      /* for each row ... */
      for ( i = i0; i <= m; i++ )
         {  
            /* compute inner product */
            /* ip = __ip__(&(M->me[i][j0]),&(hh->ve[j0]),(int)(M->n-j0));*/
            ip = 0.0;
            for ( j = j0; j <= n; j++ )
               ip += M[i][j]*hh[j];
            scale = beta*ip;
            if ( scale == 0.0 )
               continue;
            /* __mltadd__(&(M->me[i][j0]),&(hh->ve[j0]),-scale,
               (int)(M->n-j0)); */
            for ( j = j0; j <= n; j++ )
               M[i][j] -= scale*hh[j];
         }
   }
}

/* HholdTrCols -- transform a matrix by a Householder vector by columns
   starting at row i0 from column j0 -- in-situ */
static void HholdTrCols(DMatrix M, int i0, int j0, 
                        DVector hh, double beta, DVector w)
{
   int i, j;
   int n;

   n = NumDRows(M);

   if ( M==NULL || hh==NULL )
      HError(1,"HholdTrCols: matrix or vector is NULL!");
   if ( DVectorSize(hh) != n )
      HError(1,"HholdTrCols: hh vector size must = number of M columns");
   if ( i0 > n+1 || j0 > n+1 )
      HError(1,"HholdTrCols: Bounds matrix/vec size error i=%d j=%d n=%d",
             i0, j0, n);

   ZeroDVector(w);

   if ( beta != 0.0 ) {

      for ( i = i0; i <= n; i++ )
         if ( hh[i] != 0.0 )
            for ( j = j0; j <= n; j++ )
               w[j] += M[i][j]*hh[i];

      for ( i = i0; i <= n; i++ )
         if ( hh[i] != 0.0 )
            for ( j = j0; j <= n; j++ )
               M[i][j] -= w[j]*beta*hh[i];

   }
}


/* copy a row from a matrix into  a vector */
static void CopyDRow(DMatrix M, int k, DVector v) 
{
   int i, size;
   DVector w;

   if (v == NULL)
      HError(1, "CopyDRow: Vector is NULL");

   size = DVectorSize(v);
   w = M[k];

   for (i = 1; i <= size; i++)
      v[i] = w[i];
}

/* copy a column from a matrix into  a vector */
static void CopyDColumn(DMatrix M, int k, DVector v) 
{
   int i, size;

   if (v == NULL)
      HError(1, "CopyDColumn: Vector is NULL");

   size = DVectorSize(v);
   for (i = 1; i <= size; i++)
      v[i] = M[i][k];
}

/* BiFactor -- perform preliminary factorisation for bisvd
   -- updates U and/or V, which ever is not NULL */
static void BiFactor(DMatrix A, DMatrix U, DMatrix V)
{
   int n, k;
   DVector tmp1, tmp2, tmp3;
   double beta;

   n = NumDRows(A);

   tmp1 = CreateDVector(&gstack, n);
   tmp2 = CreateDVector(&gstack, n);
   tmp3 = CreateDVector(&gstack, n);

   for ( k = 1; k <= n; k++ ) {
      CopyDColumn(A,k,tmp1);
      HholdVec(tmp1,k,n,&beta,&(A[k][k]));
      HholdTrCols(A,k,k+1,tmp1,beta,tmp3);
      if ( U )
         HholdTrCols(U,k,1,tmp1,beta,tmp3);
      if ( k+1 > n )
         continue;
      CopyDRow(A,k,tmp2);
      HholdVec(tmp2,k+1,n,&beta,&(A[k][k+1]));
      HholdTrRows(A,k+1,k+1,tmp2,beta);
      if ( V )
         HholdTrCols(V,k+1,1,tmp2,beta,tmp3);
   }

   FreeDVector(&gstack, tmp1);
}

/* mat_id -- set A to being closest to identity matrix as possible
        -- i.e. A[i][j] == 1 if i == j and 0 otherwise */
static void InitIdentity(DMatrix A) 
{
   int     i, size;
  
   ZeroDMatrix(A);
   size = minab(NumDRows(A), DVectorSize(A[1]));
   for ( i = 1; i <= size; i++ )
      A[i][i] = 1.0;
}

/* EXPORT->SVD: Calculate the decompostion of matrix A.
   NOTE: on return that U and V hold U' and V' respectively! */
void SVD(DMatrix A, DMatrix U, DMatrix V, DVector d)
{
   DVector f=NULL;
   int i, n;
   DMatrix A_tmp;

   /* do initial size checks! */
   if ( A == NULL )
      HError(1,"svd: Matrix A is null");

   n = NumDRows(A);

   if (U == NULL || V == NULL || d == NULL)
      HError(1, "SVD: The svd matrices and vector must be initialised b4 call");
 
   A_tmp = CreateDMatrix(&gstack, n, n);

   CopyDMatrix(A, A_tmp);
   InitIdentity(U);
   InitIdentity(V);
   f = CreateDVector(&gstack,n-1);

   BiFactor(A_tmp,U,V);
   for ( i = 1; i <= n; i++ ) {
      d[i] = A_tmp[i][i];
      if ( i+1 <= n )
         f[i] = A_tmp[i][i+1];
   }

   BiSVD(d,f,U,V);
   FixSVD(d,U,V);
   FreeDMatrix(&gstack, A_tmp);
}

/* EXPORT->InvSVD: Inverted Singular Value Decomposition (calls SVD)
   and inverse of A is returned in Result */
void InvSVD(DMatrix A, DMatrix U, DVector W, DMatrix V, DMatrix Result)
{
   int m, n, i, j, k;
   double wmax, wmin;
   Boolean isSmall = FALSE;
   DMatrix tmp1;

   m = NumDRows(U);
   n = DVectorSize(U[1]);

   if (m != n)
      HError(1, "InvSVD: Matrix inversion only for symmetric matrices!\n");

   SVD(A, U, V, W);
   /* NOTE U and V actually now hold U' and V' ! */

   tmp1 = CreateDMatrix(&gstack,m, n);

   wmax = 0.0;
   for (k = 1; k <= n; k ++)
      if (W[k] > wmax)
         wmax = W[k];
   wmin = wmax * 1.0e-8;
   for (k = 1; k <= n; k ++)
      if (W[k] < wmin) {
         /* A component of the diag matrix 'w' of the SVD of 'a'
            was smaller than 1.0e-6 and consequently set to zero. */
         if (trace>0) {
            printf("%d (%e) ", k, W[k]); 
            fflush(stdout);
         }
         W[k] = 0.0;
         isSmall = TRUE;
      }
   if (trace>0 && isSmall) {
      printf("\n"); 
      fflush(stdout);
   }
   /* tmp1 will be the product of matrix v and the diagonal 
      matrix of singular values stored in vector w. tmp1 is then
      multiplied by the transpose of matrix u to produce the 
      inverse which is returned */
   for (j = 1; j <= m; j++)
      for (k = 1; k <= n; k ++)
         if (W[k] > 0.0)
            /* Only non-zero elements of diag matrix w are 
               used to compute the inverse. */
            tmp1[j][k] = V[k][j] / W[k];
         else
            tmp1[j][k] = 0.0;

   ZeroDMatrix(Result);
   for (i=1;i<=m;i++)
      for (j=1;j<=m;j++)
         for (k=1;k<=n;k++)
            Result[i][j] += tmp1[i][k] * U[k][j];
   FreeDMatrix(&gstack,tmp1);
}

/* LUDecompose: perform LU decomposition on Matrix a, the permutation
       of the rows is returned in perm and sign is returned as +/-1
       depending on whether there was an even/odd number of row 
       interchanges */
static Boolean LUDecompose(Matrix a, int *perm, int *sign)
{
   int i,imax,j,k,n;
   double scale,sum,xx,yy;
   Vector vv,tmp;
   
   n = NumRows(a); imax = 0;
   vv = CreateVector(&gstack,n);
   *sign = 1;
   for (i=1; i<=n; i++) {
      scale = 0.0;
      for (j=1; j<=n; j++)
         if ((xx = fabs(a[i][j])) > scale )
            scale = xx;
      if (scale == 0.0) {
         HError(-1,"LUDecompose: Matrix is Singular");
	 return(FALSE);
      }
      vv[i] = 1.0/scale;
   }
   for (j=1; j<=n; j++) {
      for (i=1; i<j; i++) {
         sum = a[i][j];
         for (k=1; k<i; k++) sum -= a[i][k]*a[k][j];
         a[i][j]=sum;
      }
      scale=0.0;
      for (i=j; i<=n; i++) {
         sum = a[i][j];
         for (k=1; k<j; k++) sum -= a[i][k]*a[k][j];
         a[i][j]=sum;
         if ( (yy=vv[i]*fabs(sum)) >=scale) {
            scale = yy; imax = i;
         }
      }
      if (j != imax) {
         tmp = a[imax]; a[imax] = a[j]; a[j] = tmp;
         *sign = -(*sign);
         vv[imax]=vv[j];
      }
      perm[j]=imax;
      if (a[j][j] == 0.0) {
         HError(-1,"LUDecompose: Matrix is Singular");
	 return(FALSE);
      }
      if (j != n) {
         yy = 1.0/a[j][j];
         for (i=j+1; i<=n;i++) a[i][j] *= yy;
      }
   }
   FreeVector(&gstack,vv);
   return(TRUE);
}


/* EXPORT->MatDet: determinant of a matrix */
float MatDet(Matrix c)
{
   Matrix a;
   float det;
   int n,perm[1600],i,sign;
   
   n=NumRows(c);
   a=CreateMatrix(&gstack,n,n);
   CopyMatrix(c,a);                /* Make a copy of c */
   LUDecompose(a,perm,&sign);      /* Do LU Decomposition */
   det = sign;                     /* Calc Det(c) */
   for (i=1; i<=n; i++) {
      det *= a[i][i];
   }
   FreeMatrix(&gstack,a);
   return det;
}


/* DLUDecompose: perform LU decomposition on Matrix a, the permutation
       of the rows is returned in perm and sign is returned as +/-1
       depending on whether there was an even/odd number of row 
       interchanges */
static Boolean DLUDecompose(DMatrix a, int *perm, int *sign)
{
   int i,imax,j,k,n;
   double scale,sum,xx,yy;
   DVector vv,tmp;
   
   n = NumDRows(a); imax = 0;
   vv = CreateDVector(&gstack,n);
   *sign = 1;
   for (i=1; i<=n; i++) {
      scale = 0.0;
      for (j=1; j<=n; j++)
         if ((xx = fabs(a[i][j])) > scale )
            scale = xx;
      if (scale == 0.0) {
         HError(-1,"LUDecompose: Matrix is Singular");
         return(FALSE);
      }
      vv[i] = 1.0/scale;
   }
   for (j=1; j<=n; j++) {
      for (i=1; i<j; i++) {
         sum = a[i][j];
         for (k=1; k<i; k++) sum -= a[i][k]*a[k][j];
         a[i][j]=sum;
      }
      scale=0.0;
      for (i=j; i<=n; i++) {
         sum = a[i][j];
         for (k=1; k<j; k++) sum -= a[i][k]*a[k][j];
         a[i][j]=sum;
         if ( (yy=vv[i]*fabs(sum)) >=scale) {
            scale = yy; imax = i;
         }
      }
      if (j != imax) {
         tmp = a[imax]; a[imax] = a[j]; a[j] = tmp;
         *sign = -(*sign);
         vv[imax]=vv[j];
      }
      perm[j]=imax;
      if (a[j][j] == 0.0) {
         HError(-1,"LUDecompose: Matrix is Singular");
         return(FALSE);
      }
      if (j != n) {
         yy = 1.0/a[j][j];
         for (i=j+1; i<=n;i++) a[i][j] *= yy;
      }
   }
   FreeDVector(&gstack,vv);
   return(TRUE);
}


/* EXPORT->DMatDet: determinant of a double matrix */
double DMatDet(DMatrix c)
{
   DMatrix a;
   double det;
   int n,perm[1600],i,sign;
   
   n=NumDRows(c);
   a=CreateDMatrix(&gstack,n,n);
   CopyDMatrix(c,a);                /* Make a copy of c */
   DLUDecompose(a,perm,&sign);      /* Do LU Decomposition */
   det = sign;                     /* Calc Det(c) */
   for (i=1; i<=n; i++) {
      det *= a[i][i];
   }
   FreeDMatrix(&gstack,a);
   return det;
}



/* LinSolve: solve the set of linear equations Ax = b, returning
        the result x in  b */
static void LinSolve(Matrix a, int *perm, float *b)
{
   int i,ii=0,ip,j,n;
   double sum;
   
   n=NumRows(a);
   for (i=1;i<=n;i++) {
      ip=perm[i]; sum=b[ip]; b[ip]=b[i];
      if (ii)
         for (j=ii;j<=i-1;j++) sum -=a[i][j]*b[j];
      else
         if (sum) ii=i;
      b[i]=sum;
   }
   for (i=n; i>=1; i--) {
      sum=b[i];
      for (j=i+1; j<=n; j++)
         sum -=a[i][j]*b[j];
      b[i]=sum/a[i][i];
   }
}        


/* EXPORT-> MatInvert: puts inverse of c in invc, returns Det(c) */
float MatInvert(Matrix c, Matrix invc)
{
   Matrix a;
   float col[MAXMATRIXSIZE];
   float det;
   int sign;
   int n,i,j,perm[MAXMATRIXSIZE];
   
   n=NumRows(c);
   /* from mjfg - cz277 - 141022 */   
   if (n > MAXMATRIXSIZE)
      HError (5221, "Matrix too large");

   a=CreateMatrix(&gstack,n,n);
   CopyMatrix(c,a);           /* Make a copy of c */
   LUDecompose(a,perm,&sign);      /* Do LU Decomposition */
   for (j=1; j<=n; j++) {     /* Invert matrix */
      for (i=1; i<=n; i++)
         col[i]=0.0;
      col[j]=1.0;
      LinSolve(a,perm,col);
      for (i=1; i<=n; i++)
         invc[i][j] = col[i];
   }  
   det = sign;                /* Calc log(det(c)) */
   for (i=1; i<=n; i++) {
      det *= a[i][i];
   }
   FreeMatrix(&gstack,a);
   return det;
}                
 
/* DLinSolve: solve the set of linear equations Ax = b, returning
        the result x in  b */
static void DLinSolve(DMatrix a, int *perm, double *b)
{
   int i,ii=0,ip,j,n;
   double sum;
   
   n=NumDRows(a);
   for (i=1;i<=n;i++) {
      ip=perm[i]; sum=b[ip]; b[ip]=b[i];
      if (ii)
         for (j=ii;j<=i-1;j++) sum -=a[i][j]*b[j];
      else
         if (sum) ii=i;
      b[i]=sum;
   }
   for (i=n; i>=1; i--) {
      sum=b[i];
      for (j=i+1; j<=n; j++)
         sum -=a[i][j]*b[j];
      b[i]=sum/a[i][i];
   }
}       

/* Inverting a double matrix */
double DMatInvert(DMatrix c, DMatrix invc)
{
   DMatrix a;
   double col[MAXMATRIXSIZE];
   double det;
   int sign;
   int n,i,j,perm[MAXMATRIXSIZE];
   
   n=NumDRows(c);
   /* from mjfg - cz277 141022 */
   if (n > MAXMATRIXSIZE)
      HError (5221, "Matrix too large");

   a=CreateDMatrix(&gstack,n,n);
   CopyDMatrix(c,a);           /* Make a copy of c */
   DLUDecompose(a,perm,&sign);      /* Do LU Decomposition */
   for (j=1; j<=n; j++) {     /* Invert matrix */
      for (i=1; i<=n; i++)
         col[i]=0.0;
      col[j]=1.0;
      DLinSolve(a,perm,col);
      for (i=1; i<=n; i++)
         invc[i][j] = col[i];
   }  
   det = sign;                /* Calc log(det(c)) */
   for (i=1; i<=n; i++) {
      det *= a[i][i];
   }
   FreeDMatrix(&gstack,a);
   return det;
}

/* EXPORT-> DMatCofact: generates the cofactors of row r of matrix c */
double DMatCofact(DMatrix c, int r, DVector cofact)
{
   DMatrix a;
   double col[MAXMATRIXSIZE];
   double det;
   int sign;
   int n,i,perm[MAXMATRIXSIZE];
   
   n=NumDRows(c);
   /* from mjfg, cz277 - 141022 */
   if (n > MAXMATRIXSIZE)
      HError (5221, "Matrix too large");

   a=CreateDMatrix(&gstack,n,n);
   CopyDMatrix(c,a);                      /* Make a copy of c */
   if (! DLUDecompose(a,perm,&sign))      /* Do LU Decomposition */
     return 0;
   det = sign;                         /* Calc det(c) */
   for (i=1; i<=n; i++) {
      det *= a[i][i];
   }
   for (i=1; i<=n; i++)
     col[i]=0.0;
   col[r]=1.0;
   DLinSolve(a,perm,col);
   for (i=1; i<=n; i++)
     cofact[i] = col[i]*det;
   FreeDMatrix(&gstack,a);
   return det;
}

/* EXPORT-> MatCofact: generates the cofactors of row r of matrix c */
double MatCofact(Matrix c, int r, Vector cofact)
{
   DMatrix a;
   DMatrix b;
   double col[MAXMATRIXSIZE];
   float det;
   int sign;
   int n,i,perm[MAXMATRIXSIZE];
 
   n=NumRows(c);
   /* from mjfg, cz277 - 141022 */
   if (n > MAXMATRIXSIZE)
      HError (5221, "Matrix too large");

   a=CreateDMatrix(&gstack,n,n);
   b=CreateDMatrix(&gstack,n,n);
   Mat2DMat(c,b);
   CopyDMatrix(b,a);                      /* Make a copy of c */
   if (! DLUDecompose(a,perm,&sign))      /* Do LU Decomposition */
     return 0;
   det = sign;                         /* Calc det(c) */
   for (i=1; i<=n; i++) {
      det *= a[i][i];
   }
   for (i=1; i<=n; i++)
     col[i]=0.0;
   col[r]=1.0;
   DLinSolve(a,perm,col);
   for (i=1; i<=n; i++)
     cofact[i] = col[i]*det;
   
   FreeDMatrix(&gstack,b);
   FreeDMatrix(&gstack,a);
   return det;
}

/* -------------------- Log Arithmetic ---------------------- */

/*
  The types LogFloat and LogDouble are used for representing
  real numbers on a log scale.  LZERO is used for log(0) 
  in log arithmetic, any log real value <= LSMALL is 
  considered to be zero.
*/

static LogDouble minLogExp;

/* EXPORT->LAdd: Return sum x + y on log scale, 
                sum < LSMALL is floored to LZERO */
LogDouble LAdd(LogDouble x, LogDouble y)
{
   LogDouble temp,diff,z;
   
   if (x<y) {
      temp = x; x = y; y = temp;
   }
   diff = y-x;
   if (diff<minLogExp) 
      return  (x<LSMALL)?LZERO:x;
   else {
      z = exp(diff);
      return x+log(1.0+z);
   }
}

/* EXPORT->LSub: Return diff x - y on log scale, 
                 diff < LSMALL is floored to LZERO */
LogDouble LSub(LogDouble x, LogDouble y)
{
   LogDouble diff,z;
   
   if (x<y)    
      HError(5271,"LSub: result -ve");
   diff = y-x;
   if (diff<minLogExp) 
      return  (x<LSMALL)?LZERO:x;
   else {
      z = 1.0 - exp(diff);
      return (z<MINLARG) ? LZERO : x+log(z);
   }
}

/* EXPORT->L2F: Convert log(x) to double, result is
                floored to 0.0 if x < LSMALL */
double   L2F(LogDouble x)
{
   return (x<LSMALL) ? 0.0 : exp(x);
}

/* -------------------- Random Numbers ---------------------- */


#ifdef UNIX
/* Prototype for C Library functions drand48 and srand48 */
double drand48(void);
void srand48(long);
#define RANDF() drand48()
#define SRAND(x) srand48(x)
#else
/* if not unix use ANSI C defaults */
#define RANDF() ((float)rand()/RAND_MAX)
#define SRAND(x) srand(x)
#endif

/* EXPORT->RandInit: Initialise random number generators 
           if seed is -ve, then system clock is used */
void RandInit(int seed)
{
   if (seed<0) seed = (int)time(NULL)%257;
   SRAND(seed);
}

/* EXPORT->RandomValue:  */
float RandomValue(void)
{
   return RANDF();
}

/* EXPORT->GaussDeviate: random number with a N(mu,sigma^2) distribution */
float GaussDeviate(float mu, float sigma)
{
   double fac,r,v1,v2,x;
   static int gaussSaved = 0; /* GaussDeviate generates numbers in pairs */
   static float gaussSave;    /* 2nd of pair is remembered here */


   if (gaussSaved) {
      x = gaussSave; gaussSaved = 0;
   }
   else {
      do {
         v1 = 2.0*(float)RANDF() - 1.0;
         v2 = 2.0*(float)RANDF() - 1.0;
         r = v1*v1 + v2*v2;
      }
      while (r>=1.0);
      fac = sqrt(-2.0*log(r)/r);
      gaussSaved = 1;
      gaussSave = v1*fac;
      x = v2*fac;
   }
   return x*sigma+mu;
}

/* from xl207, cz277 - gau */
float GaussInv(float p)
{  
  HError(999, "Gaussianisation is disabled");
  return 0.0;
}

/* from xl207, cz277 - gau */
float CumGauss(float x, float mean, float var)
{
   HError(999, "Gaussianisation is disabled");
   return 0.0;
}

/* --------------------- Initialisation ---------------------- */

/* EXPORT->InitMath: initialise this module */
void InitMath(void)
{
   int i;
#ifdef MKL
   ConfParam *cpVal;
#endif

   Register(hmath_version,hmath_vc_id);
   RandInit(-1);
   minLogExp = -log(-LZERO);
   numParm = GetConfig("HMATH", TRUE, cParm, MAXGLOBS);
   if (numParm>0){
      if (GetConfInt(cParm,numParm,"TRACE",&i)) trace = i;
/* cz277 - ANN */
#ifdef MKL
      if (GetConfAny(cParm, numParm, "NMKLTHREADS", &cpVal)) {
          if (cpVal->kind == IntCKind) 
              nMKLThreads = cpVal->val.i;
          else if (cpVal->kind == StrCKind) 
              nMKLThreadEnvVar = CopyString(&gcheap, cpVal->val.s); 
          else 
              HError(5222, "InitMath: Unknown NMKLTHREADS value kind");
      }
#endif
   }
}

/* cz277 - ANN */
/* --------------------- ANN related math kernels --------------------- */

/* cz277 - 151020 */
#ifdef MKL
void StartMKL() {
    char *envVar = NULL;
    int maxMKLThreads = 1;
    MKLVersion version;

    /* print the MKL version */
    mkl_get_version(&version);
    printf("MKL version %d.%d.%d (build %s)\n", version.MajorVersion, version.MinorVersion, version.UpdateVersion, version.Build);
    printf("%s\n", version.Platform);
    /*printf("Optimised for %s\n", version.Processor);*/

    /* set and print the thread number */
    maxMKLThreads = mkl_get_max_threads();
    if (nMKLThreads > maxMKLThreads) {
        HError(-5223, "StartMKL: Thread number %d set by NMKLTHREADS > maximum thread number %d, reset to %d", nMKLThreads, maxMKLThreads, maxMKLThreads);
        nMKLThreads = maxMKLThreads;
    }
    if (strcmp(nMKLThreadEnvVar, "") != 0) {
        envVar = getenv(nMKLThreadEnvVar);
        if (envVar == NULL) {
            HError(-5223, "StartMKL: Environment variable %s not defined, reset to use 1 thread\n", envVar);
            nMKLThreads = 1;
        }
        else {
            nMKLThreads = atoi(envVar);
        }
    } 
    if (nMKLThreads > 0) {
        mkl_set_dynamic(0);
        mkl_set_num_threads(nMKLThreads);
        printf("MKL thread number = %d\n", nMKLThreads);
    }
    else {
        mkl_set_dynamic(maxMKLThreads);
        printf("MKL dynamic threading control (upto %d threads)\n", maxMKLThreads);
    }
    printf("\n");
}
#endif


void RegisterTmpNMat(int nrows, int ncols) {
    if (nrows > tmpRowNum)
        tmpRowNum = nrows;
    if (ncols > tmpColNum)
        tmpColNum = ncols;
}

void CreateTmpNMat(MemHeap *heap) {
    if (tmpRowNum != 0 && tmpColNum != 0) {
        tmpNMat = CreateNMatrix(heap, tmpRowNum, tmpColNum);
        tmpNVec = (NVector *) New(heap, sizeof(NVector));
        memset(tmpNVec, 0, sizeof(NVector));
        tmpNVec->vecLen = tmpNMat->rowNum * tmpNMat->colNum;
        tmpNVec->vecElems = tmpNMat->matElems;
#ifdef CUDA
        tmpNVec->devElems = tmpNMat->devElems;
#endif
    }
}

NMatrix *GetTmpNMat(void) {
    return tmpNMat;
}

NVector *GetTmpNVec(void) {
    return tmpNVec;
}

void FreeTmpNMat(MemHeap *heap) {
    if (tmpNMat != NULL) {
        FreeNMatrix(heap, tmpNMat);
        tmpNMat = NULL;
        tmpRowNum = 0;
        tmpColNum = 0;
    }
}

static inline void CopyNSegmentCPU(NFloat *srcPtr, int segLen, NFloat *dstPtr) {
   memcpy(dstPtr, srcPtr, segLen * sizeof(NFloat));
}

#ifdef MKL
static inline void CopyNSegmentMKL(NFloat *srcPtr, int segLen, NFloat *dstPtr) {
    #ifdef DOUBLEANN
    cblas_dcopy(segLen, srcPtr, 1, dstPtr, 1);
    #else
    cblas_scopy(segLen, srcPtr, 1, dstPtr, 1);
    #endif
}
#endif

void CopyNSegment(NMatrix *srcMat, int srcOff, int segLen, NMatrix *dstMat, int dstOff) {
    if (trace & T_DIM) {
        if (!(srcOff >= 0 && segLen >= 0 && (srcOff + segLen) < srcMat->rowNum * srcMat->colNum))
            HError(5221, "CopyNSegment: Illegal source matrix offset or segment length");
        if (!(dstOff >= 0 && (dstOff + segLen) < dstMat->rowNum * dstMat->colNum))
            HError(5221, "CopyNSegment: Illegal destinate matrix offset");
    }
#ifdef CUDA
    CopyNSegmentCUDA(srcMat->devElems + srcOff, segLen, dstMat->devElems + dstOff);
#else
    #ifdef MKL
    CopyNSegmentMKL(srcMat->matElems + srcOff, segLen, dstMat->matElems + dstOff);
    #else
    CopyNSegmentCPU(srcMat->matElems + srcOff, segLen, dstMat->matElems + dstOff);
    #endif
#endif
}

void CopyNVectorSegment(NVector *srcVec, int srcOff, int segLen, NVector *dstVec, int dstOff) {
    if (trace & T_DIM) {
        if (!(srcOff >= 0 && segLen >= 0 && (srcOff + segLen) < srcVec->vecLen))
            HError(5221, "CopyNVectorSegment: Illegal source vector offset or segment length");
        if (!(dstOff >= 0 && (dstOff + segLen) < dstVec->vecLen))
            HError(5221, "CopyNVectorSegment: Illegal destinate vector offset");
    }
#ifdef CUDA
    CopyNSegmentCUDA(srcVec->devElems + srcOff, segLen, dstVec->devElems + dstOff);
#else
    #ifdef MKL
    CopyNSegmentMKL(srcVec->vecElems + srcOff, segLen, dstVec->vecElems + dstOff);
    #else
    CopyNSegmentCPU(srcVec->vecElems + srcOff, segLen, dstVec->vecElems + dstOff);
    #endif
#endif
}

/*typedef struct _SRank {
    int idx;
    NFloat val;
} SRank;

static int cmpfuncSRank(const void *a, const void *b) {
    return (((SRank *)b)->val - ((SRank *)a)->val);
}

static void SortOrthVecsCPU(NFloat *U, int nrows, int ncols, NFloat *d, NFloat *V) {
    int i, j, k;
    NFloat *temp;
    SRank *slist;

    temp = (NFloat *) New(&gstack, sizeof(NFloat) * MAX(nrows, ncols) * MAX(nrows, ncols));
    slist = (SRank *) New(&gstack, ncols);
    for (i = 0; i < ncols; ++i) {
        slist[i].idx = i;
        slist[i].val = d[i];
    }
    qsort(slist, ncols, sizeof(SRank), cmpfuncSRank);

    CopyNSegmentCPU(U, nrows * ncols, temp);
    for (i = 0; i < ncols; ++i) {
        j = slist[i].idx;
        if (i != j) {
            for (k = 0; k < nrows; ++k) {
                U[k * ncols + i] = temp[k * ncols + j];
            }
        }
    } 
    for (i = 0; i < ncols; ++i) {
        d[i] = slist[i].val;
    }
    CopyNSegmentCPU(V, ncols * ncols, temp);
    for (i = 0; i < ncols; ++i) {
        j = slist[i].idx;
        if (i != j) {
            memcpy(&V[i * ncols], &temp[j * ncols], sizeof(NFloat) * ncols);
        }
    }

    Dispose(&gstack, temp);
}

static double SVDanyPythag(NFloat a, NFloat b) {
    double absa, absb, result;
 
    absa = fabs(a);
    absb = fabs(b);
    if (absa > absb) {
        result = absa * sqrt(pow(absb / absa, 2) + 1.0);
    }
    else if (absb > 0.0) {
        result = absb * sqrt(pow(absa / absb, 2) + 1.0);
    }
    else {
        result = 0.0;
    }

    return result;    
}

static void SVDanyCPU(NFloat *A, int nrows, int ncols, NFloat *U, NFloat *d, NFloat *Vt) {
    int flag, i, its, j, jj, k, l, nm;
    double anorm, c, f, g, h, s, scale, x, y, z, *rv1;

    if (A != U) {
        CopyNSegmentCPU(A, nrows * ncols, U);
    }
    rv1 = (double *) New(&gstack, sizeof(double) * ncols);

    g = scale = anorm = 0.0;
    for(i = 0; i < ncols; ++i) {
        l = i + 1;
        rv1[i] = scale * g;
        g = s = scale = 0.0;
        if (i < nrows) {
            for(k = i; k < nrows; ++k) 
                scale += fabs(U[k * ncols + i]);
        if (scale) {
	    for(k = i; k < nrows; ++k) {
	        U[k * ncols + i] /= scale;		
	        s += U[k * ncols + i] * U[k * ncols + i];
	    }
	    f = U[i * ncols + i];
	    g = -SIGN(sqrt(s), f);
	    h = f * g - s;
	    U[i * ncols + i] = f - g;
	    for (j = l; j < ncols; ++j) {
	        for (s = 0.0, k = i; k < nrows; ++k) 
                    s += U[k * ncols + i] * U[k * ncols + j];
	        f = s / h;
	        for (k = i; k < nrows; ++k) 
                    U[k * ncols + j] += f * U[k * ncols + i];
	    }
	    for (k = i; k < nrows; ++k) 
                U[k * ncols + i] *= scale;
            }
        }
        d[i] = scale * g;
        g = s = scale = 0.0;
        if (i < nrows && i != ncols-1) {
            for (k = l; k < ncols; ++k) 
                scale += fabs(U[i * ncols + k]);
            if (scale) {
	         for(k = l; k < ncols; ++k) {
	             U[i * ncols + k] /= scale;
	             s += U[i * ncols + k] * U[i * ncols + k];
	         }
	         f = U[i * ncols + l];
	         g = -SIGN(sqrt(s), f);
	         h = f * g - s;
	         U[i * ncols + l] = f - g;
	         for (k = l; k < ncols; ++k) 
                     rv1[k] = U[i * ncols + k] / h;
	         for (j = l; j < nrows; ++j) {
	             for (s = 0.0, k = l; k < ncols; ++k) 
                         s += U[j * ncols + k] * U[i * ncols + k];
	             for (k = l; k < ncols; ++k) 
                         U[j * ncols + k] += s * rv1[k];
	         }
	         for (k = l; k < ncols; ++k) 
                     U[i * ncols + k] *= scale;
            }
        }
        anorm = MAX(anorm, (fabs(d[i]) + fabs(rv1[i])));
    }

    for (i = ncols - 1; i >= 0; --i) {
        if(i < ncols - 1) {
            if(g) {
	        for (j = l; j < ncols; ++j)
	            Vt[j * ncols + i] = (U[i * ncols + j] / U[i * ncols + l]) / g;
	        for (j = l; j < ncols; ++j) {
	            for (s = 0.0, k = l; k < ncols; ++k) 
                        s += U[i * ncols + k] * Vt[k * ncols + j];
	            for (k = l; k < ncols; ++k) 
                        Vt[k * ncols + j] += s * Vt[k * ncols + i];
	        }
            }
            for (j = l; j < ncols; ++j) 
                Vt[i * ncols + j] = Vt[j * ncols + i] = 0.0;
        }
        Vt[i * ncols + i] = 1.0;
        g = rv1[i];
        l = i;
    }
    for (i = MIN(nrows, ncols) - 1; i >= 0; --i) {
        l = i + 1;
        g = d[i];
        for (j = l; j < ncols; ++j) 
            U[i * ncols + j] = 0.0;
        if (g) {
            g = 1.0 / g;
            for (j = l; j < ncols; ++j) {
	        for (s = 0.0, k = l; k < nrows; ++k) 
                    s += U[k * ncols + i] * U[k * ncols + j];
	        f = (s / U[i * ncols + i]) * g;
	        for (k = i; k < nrows; ++k) 
                    U[k * ncols + j] += f * U[k * ncols + i];
            }
            for (j = i; j < nrows; ++j) 
                U[j * ncols + i] *= g;
        }
        else {
            for (j = i; j < nrows; ++j) 
                U[j * ncols + i] = 0.0;
        }
        U[i * ncols + i] += 1.0;
    }

    for (k = ncols - 1; k >= 0; --k) {
        for (its = 0; its < MAXSVDITER; ++its) {
            flag = 1;
            for (l = k; l >= 0; --l) {
	        nm = l - 1;
	        if ((fabs(rv1[l]) + anorm) == anorm) {
	            flag = 0;
	            break;
	        }
	        if ((fabs(d[nm]) + anorm) == anorm) 
                    break;
            }
            if (flag) {
	        c = 0.0;
	        s = 1.0;
	        for (i = l;i <= k; ++i) {
	            f = s * rv1[i];
	            rv1[i] = c * rv1[i];
	            if ((fabs(f) + anorm) == anorm) 
                        break;
	            g = d[i];
	            h = SVDanyPythag(f, g);
	            d[i] = h;
	            h = 1.0 / h;
	            c = g * h;
	            s = -f * h;
	            for (j = 0; j < nrows; ++j) {
	                y = U[j * ncols + nm];
	                z = U[j * ncols + i];
	                U[j * ncols + nm] = y * c + z * s;
	                U[j * ncols + i] = z * c - y * s;
	            }
	        }
            }
            z = d[k];
            if (l == k) {
	        if (z < 0.0) {
	            d[k] = -z;
	            for (j = 0; j < ncols; ++j) 
                        Vt[j * ncols + k] = -Vt[j * ncols + k];
	        }
	        break;
            }
            if (its == MAXSVDITER) 
                HError(999, "Fail to converge in %d iterations", MAXSVDITER);
            x = d[l];
            nm = k - 1;
            y = d[nm];
            g = rv1[nm];
            h = rv1[k];
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
            g = SVDanyPythag(f, 1.0);
            f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;
            c = s = 1.0;
            for (j = l; j <= nm; ++j) {
	       i = j + 1;
	       g = rv1[i];
	       y = d[i];
	       h = s * g;
	       g = c * g;
	       z = SVDanyPythag(f, h);
	       rv1[j] = z;
	       c = f / z;
	       s = h / z;
	       f = x * c + g * s;
	       g = g * c - x * s;
	       h = y * s;
	       y *= c;
	       for(jj = 0; jj < ncols; ++jj) {
	           x = Vt[jj * ncols + j];
	           z = Vt[jj * ncols + i];
	           Vt[jj * ncols + j] = x * c + z * s;
	           Vt[jj * ncols + i] = z * c - x * s;
	       }
	       z = SVDanyPythag(f, h);
	       d[j] = z;
	       if (z) {
	           z = 1.0 / z;
	           c = f * z;
	           s = h * z;
	       }
	       f = c * g + s * y;
	       x = c * y - s * g;
	       for (jj = 0; jj < nrows; ++jj) {
	           y = U[jj * ncols + j];
	           z = U[jj * ncols + i];
	           U[jj * ncols + j] = y * c + z * s;
	           U[jj * ncols + i] = z * c - y * s;
	       }
           }
           rv1[l] = 0.0;
           rv1[k] = f;
           d[k] = x;
       }
   }
  
   Dispose(&gstack, rv1);
  
}

void NanySVD(NMatrix *A, NMatrix *U, NVector *d, NMatrix *Vt) {
    
#ifdef CUDA
    SyncNMatrixDev2Host(A);
#endif
    if (trace & T_DIM) {
        if (!(A->rowNum == U->rowNum))
            HError(9999, "NMatirxSVD: Inconsistent row dimensions of A and U");
        if (!(A->colNum == U->colNum == d->vecLen == Vt->rowNum == Vt->colNum))
            HError(9999, "NMatirxSVD: Inconsistent column dimensions of A, U, d, V");
        
    }
    SVDanyCPU(A->matElems, A->rowNum, A->colNum, U->matElems, d->vecElems, Vt->matElems);
    SortOrthVecsCPU(U->matElems, U->rowNum, U->colNum, d->vecElems, V->matElems);

}*/

#ifdef MKL
void NSVDanyMKL(NFloat *A, int nrows, int ncols, NFloat *U, NFloat *d, NFloat *Vt) {
    int ret;

#ifdef DOUBLEANN
    ret = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'A', nrows, ncols, A, ncols, d, U, nrows, Vt, ncols);
#else
    ret = LAPACKE_sgesdd(LAPACK_ROW_MAJOR, 'A', nrows, ncols, A, ncols, d, U, nrows, Vt, ncols);
#endif
    
    /*superb = CreateVector(&gstack, MIN(nrows, ncols) - 1);
#ifdef DOUBLEANN
    ret = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', nrows, ncols, A, ncols, d, U, nrows, Vt, ncols, superb);
#else
    ret = LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'A', 'A', nrows, ncols, A, ncols, d, U, nrows, Vt, ncols, superb);
#endif
    Dispose(&gstack, superb);*/

    if (ret > 0) 
        HError(5224, "NSVDanyMKL: LAPACK_?gesvd did not converge");
}
#endif

void NanySVD(NMatrix *A, NMatrix *U, NVector *d, NMatrix *Vt) {

#ifdef CUDA
    HError(5201, "NanySVD: GPU based SVD decomposition not implemented yet");
#else 
    #ifdef MKL
    NSVDanyMKL(A->matElems, A->rowNum, A->colNum, U->matElems, d->vecElems, Vt->matElems);
    #else 
    HError(5201, "NanySVD: CPU based SVD decomposition not implemented yet");
    #endif
#endif

}

static inline void AddNSegmentCPU(NFloat *srcPtr, int segLen, NFloat *dstPtr) {
    int i;

    for (i = 0; i < segLen; ++i) {
        dstPtr[i] += srcPtr[i];
    }
}

#ifdef MKL
static inline void AddNSegmentMKL(NFloat *srcPtr, int segLen, NFloat *dstPtr) {
    #ifdef DOUBLEANN
    vdAdd(segLen, srcPtr, dstPtr, dstPtr);
    #else
    vsAdd(segLen, srcPtr, dstPtr, dstPtr);
    #endif
}
#endif

void AddNSegment(NMatrix *srcMat, int srcOff, int segLen, NMatrix *dstMat, int dstOff) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcOff >= 0 && segLen >= 0 && (srcOff + segLen) < srcMat->rowNum * srcMat->colNum))
            HError(5221, "CopyNSegment: Illegal source matrix offset or segment length");
        if (!(dstOff >= 0 && (dstOff + segLen) < dstMat->rowNum * dstMat->colNum))
            HError(5221, "CopyNSegment: Illegal destinate matrix offset");
    }
#ifdef CUDA
    AddNSegmentCUDA(srcMat->devElems + srcOff, segLen, dstMat->devElems + dstOff);
#else
    #ifdef MKL
    AddNSegmentMKL(srcMat->matElems + srcOff, segLen, dstMat->matElems + dstOff);
    #else
    AddNSegmentCPU(srcMat->matElems + srcOff, segLen, dstMat->matElems + dstOff);
    #endif
#endif
}

void AddNMatrix(NMatrix *srcMat, int row, int col, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "AddNMatrix: Matrix row number inconsistent");
        if (!(col > 0 && col <= srcMat->colNum && col <= dstMat->colNum))
            HError(5221, "AddNMatrix: Matrix column number inconsistent");
    }
#ifdef CUDA
    AddNSegmentCUDA(srcMat->devElems, row * col, dstMat->devElems);
#else
    #ifdef MKL
    AddNSegmentMKL(srcMat->matElems, row * col, dstMat->matElems);
    #else
    AddNSegmentCPU(srcMat->matElems, row * col, dstMat->matElems);
    #endif
#endif
}

void AddNVector(NVector *srcVec, int len, NVector *dstVec) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(len > 0 && len <= srcVec->vecLen && len <= dstVec->vecLen))
            HError(5221, "AddNVector: Vector length inconsistent");
    }
#ifdef CUDA
    AddNSegmentCUDA(srcVec->devElems, len, dstVec->devElems);
#else
    #ifdef MKL
    AddNSegmentMKL(srcVec->vecElems, len, dstVec->vecElems);
    #else
    AddNSegmentCPU(srcVec->vecElems, len, dstVec->vecElems);
    #endif
#endif
}

/* cz277 - l2 fix */
static void AddScaledNSegmentCPU(NFloat *srcPtr, int segLen, NFloat scale, NFloat *dstPtr) {
    int i;

    for (i = 0; i < segLen; ++i) {
        dstPtr[i] += scale * srcPtr[i];
    }
}

/* cz277 - l2 fix */
#ifdef MKL
static void AddScaledNSegmentMKL(NFloat *srcPtr, int segLen, NFloat scale, NFloat *dstPtr) {

#ifdef DOUBLEANN
    cblas_daxpy(segLen, scale, srcPtr, 1, dstPtr, 1);
#else
    cblas_saxpy(segLen, scale, srcPtr, 1, dstPtr, 1);
#endif
}
#endif

/* cz277 - l2 fix */
void AddScaledNMatrix(NMatrix *srcMat, int row, int col, NFloat scale, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "AddScaledNMatrix: Matrix row number inconsistent");
        if (!(col > 0 && col <= srcMat->colNum && col <= dstMat->colNum))
            HError(5221, "AddScaledNMatrix: Matrix column number inconsistent");
    }
#ifdef CUDA
    AddScaledNSegmentCUDA(srcMat->devElems, row * col, scale, dstMat->devElems);
#else
    #ifdef MKL
    AddScaledNSegmentMKL(srcMat->matElems, row * col, scale, dstMat->matElems);
    #else
    AddScaledNSegmentCPU(srcMat->matElems, row * col, scale, dstMat->matElems);
    #endif
#endif
}

/* cz277 - l2 fix */
void AddScaledNVector(NVector *srcVec, int len, NFloat scale, NVector *dstVec) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(len > 0 && len <= srcVec->vecLen && len <= dstVec->vecLen))
            HError(5221, "AddScaledNVector: Vector length inconsistent");
    }
#ifdef CUDA
    AddScaledNSegmentCUDA(srcVec->devElems, len, scale, dstVec->devElems);
#else
    #ifdef MKL
    AddScaledNSegmentMKL(srcVec->vecElems, len, scale, dstVec->vecElems);
    #else
    AddScaledNSegmentCPU(srcVec->vecElems, len, scale, dstVec->vecElems);
    #endif
#endif
}

#ifdef MKL
static inline void ScaleNSegmentMKL(int segLen, NFloat scale, NFloat *valPtr) {
    #ifdef DOUBLEANN
    cblas_dscal(segLen, scale, valPtr, 1);
    #else
    cblas_sscal(segLen, scale, valPtr, 1);
    #endif
}
#endif

static inline void ScaleNSegmentCPU(int segLen, NFloat scale, NFloat *valPtr) {
    int i;

    for (i = 0; i < segLen; ++i)
        valPtr[i] *= scale;
}

void ScaleNMatrix(NFloat scale, int row, int col, NMatrix *valMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(row * col <= valMat->rowNum * valMat->colNum))
            HError(5221, "ScaleNMatrix: Matrix dimensions inconsistent");
    }
#ifdef CUDA
    ScaleNSegmentCUDA(row * col, scale, valMat->devElems);
#else
    #ifdef MKL
    ScaleNSegmentMKL(row * col, scale, valMat->matElems);
    #else
    ScaleNSegmentCPU(row * col, scale, valMat->matElems);
    #endif
#endif

}

void ScaleNVector(NFloat scale, int len, NVector *valVec) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(len <= valVec->vecLen))
            HError(5221, "ScaleNVector: Vector lengths inconsistent");
    }
#ifdef CUDA
    ScaleNSegmentCUDA(len, scale, valVec->devElems);
#else
    #ifdef MKL
    ScaleNSegmentMKL(len, scale, valVec->vecElems);
    #else
    ScaleNSegmentCPU(len, scale, valVec->vecElems);
    #endif
#endif
}

#ifdef MKL
static inline void ScaledSelfAddNSegmentMKL(NFloat *rhPtr, int segLen, NFloat scale, NFloat *lhPtr) {
    #ifdef DOUBLEANN
    cblas_dscal(segLen, scale, lhPtr, 1);
    vdAdd(segLen, rhPtr, lhPtr, lhPtr);
    #else
    cblas_sscal(segLen, scale, lhPtr, 1);
    vsAdd(segLen, rhPtr, lhPtr, lhPtr);
    #endif
}
#endif

static inline void ScaledSelfAddNSegmentCPU(NFloat *rhPtr, int segLen, NFloat scale, NFloat *lhPtr) {
    int i;

    for (i = 0; i < segLen; ++i) 
        lhPtr[i] = scale * lhPtr[i] + rhPtr[i];
}

void ScaledSelfAddNVector(NVector *rhVec, int len, NFloat scale, NVector *lhVec) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(len <= rhVec->vecLen && len <= lhVec->vecLen))
            HError(5221, "ScaledSelfAddNVector: Vector lengths inconsistent");
    }
#ifdef CUDA
    ScaledSelfAddNSegmentCUDA(rhVec->devElems, len, scale, lhVec->devElems);
#else
    #ifdef MKL
    ScaledSelfAddNSegmentMKL(rhVec->vecElems, len, scale, lhVec->vecElems);
    #else
    ScaledSelfAddNSegmentCPU(rhVec->vecElems, len, scale, lhVec->vecElems);
    #endif
#endif
}

void ScaledSelfAddNMatrix(NMatrix *rhMat, int row, int col, NFloat scale, NMatrix *lhMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(row > 0 && row <= rhMat->rowNum && row <= lhMat->colNum))
            HError(5221, "ScaledSelfAddNMatrix: Matrix row number inconsistent");
        if (!(col > 0 && col == rhMat->colNum && col == lhMat->colNum))
            HError(5221, "ScaledSelfAddNMatrix: Matrix column number inconsistent");
    }

#ifdef CUDA
    ScaledSelfAddNSegmentCUDA(rhMat->devElems, row * col, scale, lhMat->devElems);
#else
    #ifdef MKL
    ScaledSelfAddNSegmentMKL(rhMat->matElems, row * col, scale, lhMat->matElems);
    #else
    ScaledSelfAddNSegmentCPU(rhMat->matElems, row * col, scale, lhMat->matElems);
    #endif
#endif
}


static inline void DupNSegmentCPU(NFloat *srcPtr, int segLen, NFloat *dstPtr, int times) {
    int i, off;
    
    for (i = 0, off = 0; i < times; ++i, off += segLen) {
        memcpy(dstPtr + off, srcPtr, segLen * sizeof(NFloat));
    }
}

#ifdef MKL
static inline void DupNSegmentMKL(NFloat *srcPtr, int segLen, NFloat *dstPtr, int times) {
    int i, off;

    for (i = 0, off = 0; i < times; ++i, off += segLen) {
    #ifdef DOUBLEANN
        cblas_dcopy(segLen, srcPtr, 1, dstPtr + off, 1);
    #else
        cblas_scopy(segLen, srcPtr, 1, dstPtr + off, 1);
    #endif
    }
}
#endif

void DupNVector(NVector *srcVec, NMatrix *dstMat, int times) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(times > 0))
            HError(5221, "DupNVector: Times should be positive integer");
        if (srcVec->vecLen * times < dstMat->rowNum * dstMat->colNum)
            HError(5221, "DupNVector: Too many duplicate times");
    }
#ifdef CUDA
    DupNSegmentCUDA(srcVec->devElems, srcVec->vecLen, dstMat->devElems, times);
#else
    #ifdef MKL
    DupNSegmentMKL(srcVec->vecElems, srcVec->vecLen, dstMat->matElems, times);
    #else
    DupNSegmentCPU(srcVec->vecElems, srcVec->vecLen, dstMat->matElems, times);
    #endif
#endif
}

static inline void SubNSegmentCPU(NFloat *lhPtr, NFloat *rhPtr, int segLen, NFloat *resPtr) {
    int i;

    for (i = 0; i < segLen; ++i) {
        resPtr[i] = lhPtr[i] - rhPtr[i];
    }
}

#ifdef MKL
static inline void SubNSegmentMKL(NFloat *lhPtr, NFloat *rhPtr, int segLen, NFloat *resPtr) {
    #ifdef DOUBLEANN
    vdSub(segLen, lhPtr, rhPtr, resPtr);
    #else
    vsSub(segLen, lhPtr, rhPtr, resPtr);
    #endif
}
#endif

void SubNMatrix(NMatrix *lhMat, NMatrix *rhMat, int row, int col, NMatrix *resMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(row > 0 && row <= lhMat->rowNum && row <= rhMat->rowNum && row <= resMat->rowNum)) {
            HError(5221, "SubNMatrix: Matrix row dimension out of range");
        }
        if (!(col == lhMat->colNum && col == rhMat->colNum && col == resMat->colNum)) {
            HError(5221, "SubNMatrix: Inconsistent matrix col dimensions");
        }
    }
#ifdef CUDA
    SubNSegmentCUDA(lhMat->devElems, rhMat->devElems, row * col, resMat->devElems);
#else
    #ifdef MKL
    SubNSegmentMKL(lhMat->matElems, rhMat->matElems, row * col, resMat->matElems);
    #else
    SubNSegmentCPU(lhMat->matElems, rhMat->matElems, row * col, resMat->matElems);
    #endif
#endif
}

static inline void MulNSegmentCPU(NFloat *lhPtr, NFloat *rhPtr, int segLen, NFloat *resPtr) {
    int i;

    for (i = 0; i < segLen; ++i) {
        resPtr[i] = lhPtr[i] * rhPtr[i];
    }
}

#ifdef MKL
static inline void MulNSegmentMKL(NFloat *lhPtr, NFloat *rhPtr, int segLen, NFloat *resPtr) {
    #ifdef DOUBLEANN
    vdMul(segLen, lhPtr, rhPtr, resPtr);
    #else
    vsMul(segLen, lhPtr, rhPtr, resPtr);
    #endif
}
#endif

void MulNMatrix(NMatrix *lhMat, NMatrix *rhMat, int row, int col, NMatrix *resMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(row > 0 && row <= lhMat->rowNum && row <= rhMat->rowNum && row <= resMat->rowNum)) {
            HError(5221, "MulNMatrix: Matrix row dimension out of range");
        }
        if (!(col == lhMat->colNum && col == rhMat->colNum && col == resMat->colNum)) {
            HError(5221, "MulNMatrix: Inconsistent matrix col dimensions");
        }
    }
#ifdef CUDA
    MulNSegmentCUDA(lhMat->devElems, rhMat->devElems, row * col, resMat->devElems);
#else
    #ifdef MKL
    MulNSegmentMKL(lhMat->matElems, rhMat->matElems, row * col, resMat->matElems);
    #else
    MulNSegmentCPU(lhMat->matElems, rhMat->matElems, row * col, resMat->matElems);
    #endif
#endif
}

void MulNVector(NVector *lhVec, NVector *rhVec, int len, NVector *resVec) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(len > 0 && len <= lhVec->vecLen && len <= rhVec->vecLen && len <= resVec->vecLen))
            HError(5221, "MulNVector: Vector lengths inconsistent");
    }
#ifdef CUDA
    MulNSegmentCUDA(lhVec->devElems, rhVec->devElems, len, resVec->devElems);
#else
    #ifdef MKL
    MulNSegmentMKL(lhVec->vecElems, rhVec->vecElems, len, resVec->vecElems);
    #else
    MulNSegmentCPU(lhVec->vecElems, rhVec->vecElems, len, resVec->vecElems);
    #endif
#endif
}

/* cz277 - laf */
static inline void ApplyAffineActCPU(NFloat *srcPtr, int row, int col, NFloat *scalePtr, NFloat *shiftPtr, NFloat *dstPtr) {
    int i, j;

    for (i = 0; i < row; ++i) {
        for (j = 0; j < col; ++j) {
            dstPtr[i * col + j] = scalePtr[j] * srcPtr[i * col + j] + shiftPtr[j];
        }
    }
}

/* cz277 - laf */
#ifdef MKL
static inline void ApplyAffineActMKL(NFloat *srcPtr, int row, int col, NFloat *scalePtr, NFloat *shiftPtr, NFloat *dstPtr) {
    int i;
    NFloat *curSrc, *curDst;

    for (i = 0, curSrc = srcPtr, curDst = dstPtr; i < row; ++i, curSrc += col, curDst += col) {
#ifdef DOUBLEANN
        vdMul(col, curSrc, scalePtr, curDst);
        vdAdd(col, curSrc, shiftPtr, curDst);
#else
        vsMul(col, curSrc, scalePtr, curDst);
        vsAdd(col, curSrc, shiftPtr, curDst);
#endif
    }
}
#endif

/* cz277 - laf */
void ApplyAffineAct(NMatrix *srcMat, int row, int col, NVector *scaleVec, NVector *shiftVec, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && dstMat->colNum == col && scaleVec->vecLen == col && shiftVec->vecLen == col))
            HError(5221, "ApplyAffineAct: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "ApplyAffineAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyAffineActCUDA(srcMat->devElems, row, col, scaleVec->devElems, shiftVec->devElems, dstMat->devElems); 
#else
    #ifdef MKL
    ApplyAffineActMKL(srcMat->matElems, row, col, scaleVec->vecElems, shiftVec->vecElems, dstMat->matElems);
    #else
    ApplyAffineActCPU(srcMat->matElems, row, col, scaleVec->vecElems, shiftVec->vecElems, dstMat->matElems);
    #endif
#endif

}

/* cz277 - pact */
static inline void ApplyDAffineActCPU(NFloat *srcPtr, int row, int col, NFloat *scalePtr, NFloat *shiftPtr, NFloat *dstPtr) {
    int i;

    for (i = 0; i < row; ++i) {
        memcpy(&dstPtr[i * col], scalePtr, col * sizeof(NFloat));
    }
}

/* cz277 - pact */
void ApplyDAffineAct(NMatrix *srcMat, int row, int col, NVector *scaleVec, NVector *shiftVec, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && dstMat->colNum == col && scaleVec->vecLen == col && shiftVec->vecLen == col))
            HError(5221, "ApplyDAffineAct: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "ApplyDAffineAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyDAffineActCUDA(srcMat->devElems, row, col, scaleVec->devElems, shiftVec->devElems, dstMat->devElems);
#else
    ApplyDAffineActCPU(srcMat->matElems, row, col, scaleVec->vecElems, shiftVec->vecElems, dstMat->matElems);
#endif

}

/* cz277 - pact */
static inline void ApplyTrAffineActCPU(NFloat *errPtr, NFloat *actPtr, int row, int col, NFloat *scalePtr, NFloat *shiftPtr, Boolean accFlag, NFloat *dScalePtr, NFloat *dShiftPtr) {
    int i, j;

    if (accFlag == FALSE) {
        memset(dScalePtr, 0, col * sizeof(NFloat));
        memset(dShiftPtr, 0, col * sizeof(NFloat));
    }
    for (i = 0; i < row; ++i) {
        for (j = 0; j < col; ++j) {
             dScalePtr[j] += errPtr[i * col + j] * actPtr[i * col + j];
             dShiftPtr[j] += errPtr[i * col + j];
        }
    }
}

/* cz277 - pact */
void ApplyTrAffineAct(NMatrix *errMat, NMatrix *actMat, int row, int col, NVector *scaleVec, NVector *shiftVec, Boolean accFlag, NVector *dScaleVec, NVector *dShiftVec) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(errMat->colNum == col && actMat->colNum == col && scaleVec->vecLen == col && shiftVec->vecLen == col && dScaleVec->vecLen == col && dShiftVec->vecLen == col))
            HError(5221, "ApplyDTrAffineAct: Input column number incompatible");
    }
#ifdef CUDA
    ApplyTrAffineActCUDA(errMat->devElems, actMat->devElems, row, col, scaleVec->devElems, shiftVec->devElems, accFlag, dScaleVec->devElems, dShiftVec->devElems);
#else
    ApplyTrAffineActCPU(errMat->matElems, actMat->matElems, row, col, scaleVec->vecElems, shiftVec->vecElems, accFlag, dScaleVec->vecElems, dShiftVec->vecElems);
#endif
}

/* cz277 - laf */
static inline void AccMeanNSegmentCPU(NFloat *valPtr, int row, int col, NFloat tSamp, NFloat *meanPtr) {
    int i, j;

    for (i = 0; i < row; ++i) {
        for (j = 0; j < col; ++j) {
            meanPtr[j] += valPtr[i * col + j] / tSamp;
        }
    }
}

/* cz277 - laf */
void AccMeanNVector(NMatrix *valMat, int row, int col, NFloat tSamp, NVector *meanVec) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(valMat->colNum == col && meanVec->vecLen == col))
            HError(5221, "ApplyInitMeanAMVNAct: Input column number incompatible");
        /*if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(9999, "ApplyDAMVNAct: Input row number out of range");*/
    }
#ifdef CUDA
    AccMeanNSegmentCUDA(valMat->devElems, row, col, tSamp, meanVec->devElems);
#else
    AccMeanNSegmentCPU(valMat->matElems, row, col, tSamp, meanVec->vecElems);
#endif
}

/* cz277 - laf */
static inline void AccVarianceNSegmentCPU(NFloat *valPtr, int row, int col, NFloat tSamp, NFloat *meanPtr, NFloat *varPtr) {
    int i, j;

    for (i = 0; i < row; ++i) {
        for (j = 0; j < col; ++j) {
            varPtr[j] += pow(valPtr[i * col + j] - meanPtr[j], 2.0) / tSamp;
        }
    }
}

/* cz277 - laf */
void AccVarianceNVector(NMatrix *srcMat, int row, int col, NFloat tSamp, NVector *meanVec, NVector *varVec) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && meanVec->vecLen == col && varVec->vecLen == col))
            HError(5221, "AccVarianceNVector: Input column number incompatible");
        /*if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(9999, "ApplyDAMVNAct: Input row number out of range");*/
    }
#ifdef CUDA
    AccVarianceNSegmentCUDA(srcMat->devElems, row, col, tSamp, meanVec->devElems, varVec->devElems);
#else
    AccVarianceNSegmentCPU(srcMat->matElems, row, col, tSamp, meanVec->vecElems, varVec->vecElems);
#endif
}

void ApplyHermiteAct(NMatrix *srcMat, int row, int col, NVector *parmVec, NMatrix*dstMat) {
    HError(5201, "Unimplemented method!");
}

static inline void ApplyReLUActCPU(NFloat *srcPtr, int len, NFloat scale, NFloat *dstPtr) {
    int i;

    /* len = row * col; */
    for (i = 0; i < len; ++i) {
        if (srcPtr != dstPtr && srcPtr[i] > 0)
            dstPtr[i] = srcPtr[i];
        if (srcPtr[i] < 0)
            dstPtr[i] = srcPtr[i] * scale;
    }
}

void ApplyReLUAct(NMatrix *srcMat, int row, int col, NFloat scale, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && dstMat->colNum == col))
            HError(5221, "ApplyReLUAct: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "ApplyReLUAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyReLUActCUDA(srcMat->devElems, row * col, scale, dstMat->devElems);
#else
    ApplyReLUActCPU(srcMat->matElems, row * col, scale, dstMat->matElems);
#endif
}

static inline void ApplyDReLUActCPU(NFloat *srcPtr, int len, NFloat scale, NFloat *dstPtr) {
    int i;

    for (i = 0; i < len; ++i) {
        if (srcPtr[i] > 0)
            dstPtr[i] = 1.0;
        else
            dstPtr[i] = scale;
    }
}

void ApplyDReLUAct(NMatrix *srcMat, int row, int col, NFloat scale, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && dstMat->colNum == col))
            HError(5221, "ApplyDReLUAct: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "ApplyDReLUAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyDReLUActCUDA(srcMat->devElems, row * col, scale, dstMat->devElems);
#else
    ApplyDReLUActCPU(srcMat->matElems, row * col, scale, dstMat->matElems);
#endif
}

void ApplyLHUCSigmoidActCPU(NFloat *srcPtr, int row, int col, NFloat *rolePtr, NFloat *dstPtr) {
    int i, j, pos;
    NFloat floatVal, lhucVal;

    for (i = 0; i < col; ++i) {
        floatVal = -1.0 * rolePtr[i];
        CHKNFLTEXPE(floatVal)
        lhucVal = 2.0 / (1.0 + exp(floatVal));
        for (j = 0; j < row; ++j) {
            pos = j * col + i;
            floatVal = -1.0 * srcPtr[pos];
            CHKNFLTEXPE(floatVal)
            dstPtr[pos] = lhucVal * 1.0 / (1.0 + exp(floatVal));
        }
    }
}

void ApplyLHUCSigmoidAct(NMatrix *srcMat, int row, int col, NVector *roleVec, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && roleVec->vecLen == col && dstMat->colNum == col))
            HError(5221, "ApplyLHUCSigmoidAct: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "ApplyLHUCSigmoidAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyLHUCSigmoidActCUDA(srcMat->devElems, row, col, roleVec->devElems, dstMat->devElems);
#else
    ApplyLHUCSigmoidActCPU(srcMat->matElems, row, col, roleVec->vecElems, dstMat->matElems);
#endif
}

void ApplyDLHUCSigmoidActCPU(NFloat *srcPtr, int row, int col, NFloat *rolePtr, NFloat *dstPtr) {
    int i, j, pos;
    NFloat floatVal, lhucVal;

    for (i = 0; i < col; ++i) {
        floatVal = -1.0 * rolePtr[i];
        CHKNFLTEXPE(floatVal)
        lhucVal = 2.0 / (1.0 + exp(floatVal));    
        for (j = 0; j < row; ++j) {
            pos = j * col + i;
            floatVal = srcPtr[pos] / lhucVal;
            dstPtr[pos] = srcPtr[pos] * (1 - floatVal);
        }
    }
}

void ApplyDLHUCSigmoidAct(NMatrix *srcMat, int row, int col, NVector *roleVec, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && roleVec->vecLen == col && dstMat->colNum == col))
            HError(5221, "ApplyDLHUCSigmoidAct: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "ApplyDLHUCSigmoidAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyDLHUCSigmoidActCUDA(srcMat->devElems, row, col, roleVec->devElems, dstMat->devElems);
#else
    ApplyDLHUCSigmoidActCPU(srcMat->matElems, row, col, roleVec->vecElems, dstMat->matElems);
#endif
}

void ApplyTrLHUCSigmoidActCPU(NFloat *errPtr, NFloat *actPtr, int row, int col, NFloat *rolePtr, Boolean accFlag, NFloat *dRolePtr) {
    int i, j, pos;
    NFloat floatVal;
    
    if (accFlag == FALSE)
        memset(dRolePtr, 0, col * sizeof(NFloat));
    for (i = 0; i < col; ++i) {
        floatVal = -1.0 * rolePtr[i];
        CHKNFLTEXPE(floatVal)
        /*lhucVal = 2.0 / (1.0 + exp(floatVal)); * 0.5 */
        floatVal = 0.5 * 2.0 / (1.0 + exp(floatVal));
        for (j = 0; j < row; ++j) {
            pos = j * col + i;
            dRolePtr[i] += errPtr[pos] * actPtr[pos] * (1.0 - floatVal);
        }
    }
}

void ApplyTrLHUCSigmoidAct(NMatrix *errMat, NMatrix *actMat, int row, int col, NVector *roleVec, Boolean accFlag, NVector *dRoleVec) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(errMat->colNum == col && actMat->colNum == col && roleVec->vecLen == col && dRoleVec->vecLen == col))
            HError(5221, "ApplyTrLHUCSigmoidAct: Input column number incompatible");
        if (!(row > 0 && row <= errMat->rowNum && row <= actMat->rowNum))
            HError(5221, "ApplyTrLHUCSigmoidAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyTrLHUCSigmoidActCUDA(errMat->devElems, actMat->devElems, row, col, roleVec->devElems, accFlag, dRoleVec->devElems);
#else
    ApplyTrLHUCSigmoidActCPU(errMat->matElems, actMat->matElems, row, col, roleVec->vecElems, accFlag, dRoleVec->vecElems);
#endif
}

static inline void ApplyDLinearActCPU(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int i;

    for (i = 0; i < len; ++i) {
        dstPtr[i] = 1.0;
    }
}

void ApplyDLinearAct(NMatrix *srcMat, int row, int col, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && dstMat->colNum == col))
            HError(5221, "ApplyDLinearLAct: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "ApplyDLinearAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyDLinearActCUDA(srcMat->devElems, row * col, dstMat->devElems);
#else
    ApplyDLinearActCPU(srcMat->matElems, row * col, dstMat->matElems);
#endif
}

/* cz277 - pact */
static inline void ApplyPSigmoidActCPU(NFloat *srcPtr, int row, int col, NFloat *etaPtr, NFloat *dstPtr) {
    int i, j, pos;
    NFloat floatVal;

    /* len = row * col; */
    for (i = 0; i < row; ++i) {
        for (j = 0; j < col; ++j) {
            pos = i * col + j;
            floatVal = (-1.0) * srcPtr[pos];
            CHKNFLTEXPE(floatVal)
            dstPtr[pos] = etaPtr[j] / (1.0 + exp(floatVal));
        }
    }
}

/* cz277 - pact */
void ApplyPSigmoidAct(NMatrix *srcMat, int row, int col, NVector *etaVec, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && etaVec->vecLen == col && dstMat->colNum == col))
            HError(5221, "ApplyPSigmoidAct: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "ApplyPSigmoidAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyPSigmoidActCUDA(srcMat->devElems, row, col, etaVec->devElems, dstMat->devElems);
#else
    ApplyPSigmoidActCPU(srcMat->matElems, row, col, etaVec->vecElems, dstMat->matElems);
#endif
}

/* cz277 - pact */
static inline void ApplyDPSigmoidActCPU(NFloat *srcPtr, int row, int col, NFloat *etaPtr, NFloat *dstPtr) {
    int i, j, pos;

    for (i = 0; i < row; ++i) {
        for (j = 0; j < col; ++j) {
            pos = i * col + j;
            if (etaPtr[j] != 0.0)
                dstPtr[pos] = srcPtr[pos] * (1.0 - srcPtr[pos] / etaPtr[j]);
            else
                dstPtr[pos] = 0.0;
        }
    }
}

/* cz277 - pact */
void ApplyDPSigmoidAct(NMatrix *srcMat, int row, int col, NVector *etaVec, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && etaVec->vecLen == col && dstMat->colNum == col))
            HError(5221, "ApplyDPSigmoidAct: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "ApplyDPSigmoidAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyDPSigmoidActCUDA(srcMat->devElems, row, col, etaVec->devElems, dstMat->devElems);
#else
    ApplyDPSigmoidActCPU(srcMat->matElems, row, col, etaVec->vecElems, dstMat->matElems);
#endif
}

/* cz277 - pact */
static inline void ApplyTrPSigmoidActCPU(NFloat *errPtr, NFloat *srcPtr, NFloat *etaPtr, int row, int col, Boolean accFlag, NFloat *dEtaPtr) {
    int i, j, pos;

    if (accFlag == FALSE) 
        memset(dEtaPtr, 0, col * sizeof(NFloat));
    for (i = 0, pos = 0; i < row; ++i) {
        for (j = 0; j < col; ++j, ++pos) {
            if (etaPtr[j] != 0.0)
                dEtaPtr[j] += errPtr[pos] * srcPtr[pos] / etaPtr[j];
        }
    }
}

/* cz277 - pact */
void ApplyTrPSigmoidAct(NMatrix *errMat, NMatrix *srcMat, NVector *etaVec, int row, int col, Boolean accFlag, NVector *dEtaVec) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(errMat->colNum == col && srcMat->colNum == col && etaVec->vecLen == col && dEtaVec->vecLen == col))
            HError(5221, "ApplyTrPSigmoidAct: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= errMat->rowNum))
            HError(5221, "ApplyTrPSigmoidAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyTrPSigmoidActCUDA(errMat->devElems, srcMat->devElems, etaVec->devElems, row, col, accFlag, dEtaVec->devElems);
#else
    ApplyTrPSigmoidActCPU(errMat->matElems, srcMat->matElems, etaVec->vecElems, row, col, accFlag, dEtaVec->vecElems);
#endif
}


/* cz277 - pact */
static inline void ApplyParmSigmoidActCPU(NFloat *srcPtr, int row, int col, NFloat *etaPtr, NFloat *gammaPtr, NFloat *thetaPtr, NFloat *dstPtr) {
    int i, j, pos;
    NFloat floatVal;

    for (i = 0; i < row; ++i) {
        for (j = 0; j < col; ++j) {
            pos = i * col + j;
            floatVal = (-1.0) * gammaPtr[j] * srcPtr[pos] + thetaPtr[j];
            CHKNFLTEXPE(floatVal)
            dstPtr[pos] = etaPtr[j] / (1.0 + exp(floatVal));
        }
    }
}

/* cz277 - pact */
void ApplyParmSigmoidAct(NMatrix *srcMat, int row, int col, NVector *etaVec, NVector *gammaVec, NVector *thetaVec, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && etaVec->vecLen == col && gammaVec->vecLen == col && thetaVec->vecLen == col && dstMat->colNum == col))
            HError(5221, "ApplyParmSigmoidAct: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "ApplyParmCSigmoidAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyParmSigmoidActCUDA(srcMat->devElems, row, col, etaVec->devElems, gammaVec->devElems, thetaVec->devElems, dstMat->devElems);
#else
    ApplyParmSigmoidActCPU(srcMat->matElems, row, col, etaVec->vecElems, gammaVec->vecElems, thetaVec->vecElems, dstMat->matElems);
#endif
}

/* cz277 - pact */
static inline void ApplyDParmSigmoidActCPU(NFloat *srcPtr, int row, int col, NFloat *etaPtr, NFloat *gammaPtr, NFloat *thetaPtr, NFloat *dstPtr) {
    int i, j, pos;

    for (i = 0; i < row; ++i) {
        for (j = 0; j < col; ++j) {
            pos = i * col + j;
            if (etaPtr[j] != 0.0)
                dstPtr[pos] = gammaPtr[j] * srcPtr[pos] * (1.0 - srcPtr[pos] / etaPtr[j]);
            else
                dstPtr[pos] = 0.0;
        }
    }
}

/* cz277 - pact */
void ApplyDParmSigmoidAct(NMatrix *srcMat, int row, int col, NVector *etaVec, NVector *gammaVec, NVector *thetaVec, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && etaVec->vecLen == col && gammaVec->vecLen == col && thetaVec->vecLen == col && dstMat->colNum == col))
            HError(5221, "ApplyDParmSigmoidAct: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "ApplyDParmSigmoidAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyDParmSigmoidActCUDA(srcMat->devElems, row, col, etaVec->devElems, gammaVec->devElems, thetaVec->devElems, dstMat->devElems);
#else
    ApplyDParmSigmoidActCPU(srcMat->matElems, row, col, etaVec->vecElems, gammaVec->vecElems, thetaVec->vecElems, dstMat->matElems);
#endif
}

/* cz277 - pact */
static inline void ApplyTrParmSigmoidActCPU(NFloat *errPtr, NFloat *inpPtr, int row, int col, NFloat *etaPtr, NFloat *gammaPtr, NFloat *thetaPtr, Boolean accFlag, NFloat *dEtaPtr, NFloat *dGammaPtr, NFloat *dThetaPtr) {
    int i, j, pos;
    NFloat floatVal, expVal, fracVal;

    if (accFlag == FALSE) {
        memset(dEtaPtr, 0, col * sizeof(NFloat));
        memset(dGammaPtr, 0, col * sizeof(NFloat));
        memset(dThetaPtr, 0, col * sizeof(NFloat));
    }
    for (i = 0, pos = 0; i < row; ++i) {
        for (j = 0; j < col; ++j, ++pos) {
            floatVal = (-1.0) * gammaPtr[j] * inpPtr[pos] + thetaPtr[j];
            CHKNFLTEXPE(floatVal)
            expVal = exp(floatVal);
            fracVal = 1.0 / (1.0 + expVal);
            dEtaPtr[j] += errPtr[pos] * fracVal;
            if (etaPtr[j] != 0.0) {
                dGammaPtr[j] += errPtr[pos] * inpPtr[pos] * etaPtr[j] * fracVal * (1.0 - fracVal);
                dThetaPtr[j] -= errPtr[pos] * etaPtr[j] * fracVal * (1.0 - fracVal);
            }
        }
    }
}

/* cz277 - pact */
void ApplyTrParmSigmoidAct(NMatrix *errMat, NMatrix *inpMat, int row, int col, NVector *etaVec, NVector *gammaVec, NVector *thetaVec, Boolean accFlag, NVector *dEtaVec, NVector *dGammaVec, NVector *dThetaVec) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(errMat->colNum == col && inpMat->colNum == col && etaVec->vecLen == col && gammaVec->vecLen == col && thetaVec->vecLen == col && dEtaVec->vecLen == col && dGammaVec->vecLen == col && dThetaVec->vecLen == col))
            HError(5221, "ApplyTrParmSigmoidAct: Input column number incompatible");
        if (!(row > 0 && row <= inpMat->rowNum && row <= errMat->rowNum))
            HError(5221, "ApplyTrParmSigmoidAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyTrParmSigmoidActCUDA(errMat->devElems, inpMat->devElems, row, col, etaVec->devElems, gammaVec->devElems, thetaVec->devElems, accFlag, dEtaVec->devElems, dGammaVec->devElems, dThetaVec->devElems);
#else
    ApplyTrParmSigmoidActCPU(errMat->matElems, inpMat->matElems, row, col, etaVec->vecElems, gammaVec->vecElems, thetaVec->vecElems, accFlag, dEtaVec->vecElems, dGammaVec->vecElems, dThetaVec->vecElems);
#endif
}

/* cz277 - laf */
/*static inline void ApplyPABCSoftReLUActCPU(NFloat *srcPtr, int row, int col, NFloat *alphaPtr, NFloat *betaPtr, NFloat *gammaPtr, NFloat *dstPtr) {
    int i, j, pos;
    NFloat floatVal;

    for (i = 0; i < row; ++i) {
        for (j = 0; j < col; ++j) {
            pos = i * col + j;
            floatVal = betaPtr[j] * srcPtr[pos] + gammaPtr[j];
            CHKNFLTEXPE(floatVal)
            dstPtr[pos] = alphaPtr[j] * log(1.0 + exp(floatVal));
        }
    }
}*/

/* cz277 - laf */
/*void ApplyPABCSoftReLUAct(NMatrix *srcMat, int row, int col, NVector *alphaVec, NVector *betaVec, NVector *gammaVec, NMatrix *dstMat) {
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && alphaVec->vecLen == col && betaVec->vecLen == col && gammaVec->vecLen == col && dstMat->colNum == col))
            HError(9999, "ApplyPABCSoftReLUAct: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(9999, "ApplyPABCSoftReLUAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyPABCSoftReLUActCUDA(srcMat->devElems, row, col, alphaVec->devElems, betaVec->devElems, gammaVec->devElems, dstMat->devElems);
#else
    ApplyPABCSoftReLUActCPU(srcMat->matElems, row, col, alphaVec->vecElems, betaVec->vecElems, gammaVec->vecElems, dstMat->matElems);
#endif
}*/

/* cz277 - laf */
/*static inline void ApplyDPABCSoftReLUActCPU(NFloat *srcPtr, int row, int col, NFloat *alphaPtr, NFloat *betaPtr, NFloat *dstPtr) {
    int i, j, pos;
    NFloat floatVal;

    for (i = 0; i < row; ++i) {
        for (j = 0; j < col; ++j) {
            pos = i * col + j;
            if (alphaPtr[j] == 0.0)
                dstPtr[pos] = 0.0;
            else {
                floatVal = -1.0 / alphaPtr[j] * srcPtr[pos];
                CHKNFLTEXPE(floatVal)
                dstPtr[pos] = alphaPtr[j] * betaPtr[j] * (1 - exp(floatVal));
            }

        }
    }
}*/

/* cz277 - laf */
/*void ApplyDPABCSoftReLUAct(NMatrix *srcMat, int row, int col, NVector *alphaVec, NVector *betaVec, NMatrix *dstMat) {
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && alphaVec->vecLen == col && betaVec->vecLen == col && dstMat->colNum == col))
            HError(9999, "ApplyDPABSigmoidAct: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(9999, "ApplyDPABSigmoidAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyDPABCSoftReLUActCUDA(srcMat->devElems, row, col, alphaVec->devElems, betaVec->devElems, dstMat->devElems);
#else
    ApplyDPABCSoftReLUActCPU(srcMat->matElems, row, col, alphaVec->vecElems, betaVec->vecElems, dstMat->matElems);
#endif
}*/


/* cz277 - laf */
static inline void ApplyParmReLUActCPU(NFloat *inpPtr, int row, int col, NFloat *posPtr, NFloat *negPtr, NFloat *dstPtr) {
    int i, j, pos;

    /* len = row * col; */
    for (i = 0; i < row; ++i) {
        for (j = 0; j < col; ++j) {
            pos = i * col + j;
            if (inpPtr[pos] > 0)
                dstPtr[pos] = posPtr[j] * inpPtr[pos];
            else
                dstPtr[pos] = negPtr[j] * inpPtr[pos];
        }
    }
}

/* cz277 - laf */
void ApplyParmReLUAct(NMatrix *srcMat, int row, int col, NVector *posVec, NVector *negVec, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && posVec->vecLen == col && negVec->vecLen && dstMat->colNum == col))
            HError(5221, "ApplyParmReLUAct: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "ApplyParmReLUAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyParmReLUActCUDA(srcMat->devElems, row, col, posVec->devElems, negVec->devElems, dstMat->devElems);
#else
    ApplyParmReLUActCPU(srcMat->matElems, row, col, posVec->vecElems, negVec->vecElems, dstMat->matElems);
#endif
}

/* cz277 - laf */
static inline void ApplyDParmReLUActCPU(NFloat *inpPtr, int row, int col, NFloat *posPtr, NFloat *negPtr, NFloat *dstPtr) {
    int i, j, pos;

    /* len = row * col; */
    for (i = 0; i < row; ++i) {
        for (j = 0; j < col; ++j) {
            pos = i * col + j;
            if (inpPtr[pos] > 0)
                dstPtr[pos] = posPtr[j];
            else
                dstPtr[pos] = negPtr[j];
        }
    }
}

/* cz277 - laf */
void ApplyDParmReLUAct(NMatrix *inpMat, int row, int col, NVector *posVec, NVector *negVec, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(inpMat->colNum == col && posVec->vecLen == col && negVec->vecLen && dstMat->colNum == col))
            HError(5221, "ApplyDParmReLUAct: Input column number incompatible");
        if (!(row > 0 && row <= inpMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "ApplyDParmReLUAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyDParmReLUActCUDA(inpMat->devElems, row, col, posVec->devElems, negVec->devElems, dstMat->devElems);
#else
    ApplyDParmReLUActCPU(inpMat->matElems, row, col, posVec->vecElems, negVec->vecElems, dstMat->matElems);
#endif
}

static inline void ApplyTrParmReLUActCPU(NFloat *errPtr, NFloat *inpPtr, int row, int col, Boolean accFlag, NFloat *dPosPtr, NFloat *dNegPtr) {
    int i, j, pos;

    if (accFlag == FALSE) {
        memset(dPosPtr, 0, col * sizeof(NFloat));
        memset(dNegPtr, 0, col * sizeof(NFloat));
    }
    for (i = 0, pos = 0; i < row; ++i) {
        for (j = 0; j < col; ++j, ++pos) {
            if (inpPtr[pos] > 0.0)
                dPosPtr[j] += errPtr[pos] * inpPtr[pos];
            else
                dNegPtr[j] += errPtr[pos] * inpPtr[pos];
        }
    }
}

/* cz277 - pact */
void ApplyTrParmReLUAct(NMatrix *errMat, NMatrix *inpMat, int row, int col, Boolean accFlag, NVector *dPosVec, NVector *dNegVec) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(errMat->colNum == col && inpMat->colNum == col && dPosVec->vecLen == col && dNegVec->vecLen == col))
            HError(5221, "ApplyTrParmReLUAct: Input column number incompatible");
        if (!(row > 0 && row <= inpMat->rowNum && row <= errMat->rowNum))
            HError(5221, "ApplyTrParmReLUAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyTrParmReLUActCUDA(errMat->devElems, inpMat->devElems, row, col, accFlag, dPosVec->devElems, dNegVec->devElems);
#else
    ApplyTrParmReLUActCPU(errMat->matElems, inpMat->matElems, row, col, accFlag, dPosVec->vecElems, dNegVec->vecElems);
#endif
}

/* cz277 - pact */
static inline void ApplyPReLUActCPU(NFloat *srcPtr, int row, int col, NFloat *scalePtr, NFloat *dstPtr) {
    int i, j, pos;

    for (i = 0; i < row; ++i) {
        for (j = 0; j < col; ++j) {
            pos = i * col + j;
            if (srcPtr[pos] > 0)
                dstPtr[pos] = scalePtr[j] * srcPtr[pos];
            else
                dstPtr[pos] = 0.0;
        }
    }
}

/* cz277 - pact */
void ApplyPReLUAct(NMatrix *srcMat, int row, int col, NVector *scaleVec, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && scaleVec->vecLen == col && dstMat->colNum == col))
            HError(5221, "ApplyPReLUAct: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "ApplyPReLUAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyPReLUActCUDA(srcMat->devElems, row, col, scaleVec->devElems, dstMat->devElems);
#else
    ApplyPReLUActCPU(srcMat->matElems, row, col, scaleVec->vecElems, dstMat->matElems);
#endif
}

/* cz277 - pact */
static inline void ApplyDPReLUActCPU(NFloat *srcPtr, int row, int col, NFloat *scalePtr, NFloat *dstPtr) {
    int i, j, pos;

    for (i = 0; i < row; ++i) {
        for (j = 0; j < col; ++j) {
            pos = i * col + j;
            if (scalePtr[j] != 0.0 && srcPtr[pos] / scalePtr[j] > 0.0)
                dstPtr[pos] = scalePtr[j];
            else
                dstPtr[pos] = 0.0;
        }
    }
}

/* cz277 - pact */
void ApplyDPReLUAct(NMatrix *srcMat, int row, int col, NVector *scaleVec, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && scaleVec->vecLen == col && dstMat->colNum == col))
            HError(5221, "ApplyDPReLUAct: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "ApplyPDReLUAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyDPReLUActCUDA(srcMat->devElems, row, col, scaleVec->devElems, dstMat->devElems);
#else
    ApplyDPReLUActCPU(srcMat->matElems, row, col, scaleVec->vecElems, dstMat->matElems);
#endif
}

/* cz277 - pact */
static inline void ApplyTrPReLUActCPU(NFloat *errPtr, NFloat *srcPtr, int row, int col, NFloat *scalePtr, Boolean accFlag, NFloat *dScalePtr) {
    int i, j, pos;
    NFloat act;

    if (accFlag == FALSE)
        memset(dScalePtr, 0, col * sizeof(NFloat));
    for (i = 0, pos = 0; i < row; ++i) {
        for (j = 0; j < col; ++j, ++pos) {
            if (scalePtr[j] != 0.0) {
                act = srcPtr[pos] / scalePtr[j];
                if (act > 0.0) 
                    dScalePtr[j] += errPtr[pos] * act;
            }
        }
    }
}

/* cz277 - pact */
void ApplyTrPReLUAct(NMatrix *errMat, NMatrix *srcMat, int row, int col, NVector *scaleVec, Boolean accFlag, NVector *dScaleVec) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(errMat->colNum == col && srcMat->colNum == col && scaleVec->vecLen == col && dScaleVec->vecLen == col))
            HError(5221, "ApplyDPReLUAct: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= errMat->rowNum))
            HError(5221, "ApplyDPReLUAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyTrPReLUActCUDA(errMat->devElems, srcMat->devElems, row, col, scaleVec->devElems, accFlag, dScaleVec->devElems);
#else
    ApplyTrPReLUActCPU(errMat->matElems, srcMat->matElems, row, col, scaleVec->vecElems, accFlag, dScaleVec->vecElems);
#endif
}

static inline void ApplyDSoftReLActCPU(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int i;
    NFloat expVal;

    for (i = 0; i < len; ++i) {
        expVal = srcPtr[i];
        CHKNFLTEXPE(expVal)
        expVal = exp(expVal);
        dstPtr[i] = 1.0 - 1.0 / expVal;
    }
}

#ifdef MKL
static inline void ApplyDSoftReLActMKL(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int i;

    for (i = 0; i < len; ++i) {
        if (srcPtr != dstPtr) {
            dstPtr[i] = srcPtr[i];
        }
        CHKNFLTEXPE(dstPtr[i])
    }
    #ifdef DOUBLEANN
    vdExp(len, dstPtr, dstPtr);
    vdLinearFrac(len, dstPtr, dstPtr, 1.0, -1.0, 1.0, 0.0, dstPtr);
    #else
    vsExp(len, dstPtr, dstPtr);
    vsLinearFrac(len, dstPtr, dstPtr, 1.0, -1.0, 1.0, 0.0, dstPtr);
    #endif
}
#endif

void ApplyDSoftReLAct(NMatrix *srcMat, int row, int col, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && dstMat->colNum == col))
            HError(5221, "ApplyDSoftReLAct: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "ApplyDSoftReLAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyDSoftReLActCUDA(srcMat->devElems, row * col, dstMat->devElems);
#else
    #ifdef MKL
    ApplyDSoftReLActMKL(srcMat->matElems, row * col, dstMat->matElems);
    #else
    ApplyDSoftReLActCPU(srcMat->matElems, row * col, dstMat->matElems);
    #endif
#endif
}

static inline void ApplySoftReLActCPU(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int i;
    NFloat expVal;

    for (i = 0; i < len; ++i) {
        expVal = srcPtr[i];
	CHKNFLTEXPE(expVal)
        expVal = exp(expVal);
        dstPtr[i] = log(1.0 + expVal);
    }
}

#ifdef MKL
static inline void ApplySoftReLActMKL(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int i;

    for (i = 0; i < len; ++i) {
        if (srcPtr != dstPtr) {
            dstPtr[i] = srcPtr[i];
        }
        CHKNFLTEXPE(dstPtr[i])
    }

    #ifdef DOUBLEANN
    vdExp(len, dstPtr, dstPtr);
    vdLog1p(len, dstPtr, dstPtr);
    #else
    vsExp(len, dstPtr, dstPtr);
    vsLog1p(len, dstPtr, dstPtr);
    #endif
}
#endif

void ApplySoftReLAct(NMatrix *srcMat, int row, int col, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && dstMat->colNum == col))
            HError(5221, "ApplySoftReLAct: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "ApplySoftReLAct: Input row number out of range");
    }
#ifdef CUDA
    ApplySoftReLActCUDA(srcMat->devElems, row * col, dstMat->devElems);
#else
    #ifdef MKL
    ApplySoftReLActMKL(srcMat->matElems, row * col, dstMat->matElems);
    #else
    ApplySoftReLActCPU(srcMat->matElems, row * col, dstMat->matElems);
    #endif
#endif
}

static inline void ApplySigmoidActCPU(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int i;
    float floatVal;

    /* len = row * col */
    for (i = 0; i < len; ++i) {
        floatVal = -1.0 * srcPtr[i];
        CHKNFLTEXPE(floatVal)
        dstPtr[i] = 1.0 / (1.0 + exp(floatVal));
        /*dstPtr[i] = 1.0 / (1.0 + exp(-1.0 * srcPtr[i]));*/
    }
}

#ifdef MKL
/*static inline void ApplySigmoidActMKL(NFloat *srcPtr, int row, int col, NFloat *dstPtr) {
    int len;

    len = row * col;
    #ifdef DOUBLEANN
    vdExp(len, srcPtr, dstPtr);
    vdLinearFrac(len, dstPtr, dstPtr, 1.0, 0.0, 1.0, 1.0, dstPtr);
    #else
    vsExp(len, srcPtr, dstPtr);
    vsLinearFrac(len, dstPtr, dstPtr, 1.0, 0.0, 1.0, 1.0, dstPtr);
    #endif
}*/
static inline void ApplySigmoidActMKL(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int i;

    /*len = row * col;*/
    for (i = 0; i < len; ++i) {
        dstPtr[i] = -1.0 * srcPtr[i];
        CHKNFLTEXPE(dstPtr[i])
    }
    #ifdef DOUBLEANN
    vdExp(len, dstPtr, dstPtr);
    vdLinearFrac(len, dstPtr, dstPtr, 0.0, 1.0, 1.0, 1.0, dstPtr);
    #else
    vsExp(len, dstPtr, dstPtr);
    vsLinearFrac(len, dstPtr, dstPtr, 0.0, 1.0, 1.0, 1.0, dstPtr);
    #endif
}
#endif

void ApplySigmoidAct(NMatrix *srcMat, int row, int col, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && dstMat->colNum == col))
            HError(5221, "ApplySigmoidAct: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "ApplySigmoidAct: Input row number out of range");
    }
#ifdef CUDA
    ApplySigmoidActCUDA(srcMat->devElems, row * col, dstMat->devElems);
#else
    #ifdef MKL
    ApplySigmoidActMKL(srcMat->matElems, row * col, dstMat->matElems);
    #else
    ApplySigmoidActCPU(srcMat->matElems, row * col, dstMat->matElems);
    #endif
#endif
}

static inline void ApplyDSigmoidActCPU(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int i;

    /*len = row * col;*/
    for (i = 0; i < len; ++i)
        dstPtr[i] = (1 - srcPtr[i]) * srcPtr[i];
}

#ifdef MKL
static inline void ApplyDSigmoidActMKL(NFloat *srcPtr, int len, NFloat *dstPtr) {
    /*int len;

    len = row * col;*/
    #ifdef DOUBLEANN
    vdSqr(len, srcPtr, tmpNMat->matElems);
    vdSub(len, srcPtr, tmpNMat->matElems, dstPtr);
    #else
    vsSqr(len, srcPtr, tmpNMat->matElems);
    vsSub(len, srcPtr, tmpNMat->matElems, dstPtr);
    #endif
}
#endif

void ApplyDSigmoidAct(NMatrix *srcMat, int row, int col, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && dstMat->colNum == col))
            HError(5221, "ApplyDSigmoidAct: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "ApplyDSigmoidAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyDSigmoidActCUDA(srcMat->devElems, row * col, dstMat->devElems);
#else
    #ifdef MKL
    ApplyDSigmoidActMKL(srcMat->matElems, row * col, dstMat->matElems);
    #else
    ApplyDSigmoidActCPU(srcMat->matElems, row * col, dstMat->matElems);
    #endif
#endif
}

static inline void ApplyTanHActCPU(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int i;
    float floatVal;

    /* len = row * col */
    for (i = 0; i < len; ++i) {
        floatVal = srcPtr[i];
        CHKNFLTEXPE(floatVal)
        floatVal = exp(floatVal);
        dstPtr[i] = (floatVal - 1.0 / floatVal) / (floatVal + 1.0 / floatVal);
    }
}
    
#ifdef MKL
static inline void ApplyTanHActMKL(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int i;
    
    /*len = row * col;*/
    for (i = 0; i < len; ++i) {
        if (dstPtr != srcPtr) {
            dstPtr[i] = srcPtr[i];
        }
        CHKNFLTEXPE(dstPtr[i])
    }
    #ifdef DOUBLEANN
    vdTanh(len, dstPtr, dstPtr);
    #else
    vsTanh(len, dstPtr, dstPtr);
    #endif
}
#endif

void ApplyTanHAct(NMatrix *srcMat, int row, int col, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && dstMat->colNum == col))
            HError(5221, "ApplyTanHAct: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "ApplyTanHAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyTanHActCUDA(srcMat->devElems, row * col, dstMat->devElems);
#else
    #ifdef MKL
    ApplyTanHActMKL(srcMat->matElems, row * col, dstMat->matElems);
    #else
    ApplyTanHActCPU(srcMat->matElems, row * col, dstMat->matElems);
    #endif
#endif
}

static inline void ApplyDTanHActCPU(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int i;

    /*len = row * col;*/
    for (i = 0; i < len; ++i)
        dstPtr[i] = 1 - pow(srcPtr[i], 2);
}

#ifdef MKL
static inline void ApplyDTanHActMKL(NFloat *srcPtr, int len, NFloat *dstPtr) {
    /*int len;

    len = row * col;*/
    #ifdef DOUBLEANN
    vdPowx(len, srcPtr, 2, dstPtr);
    vdLinearFrac(len, dstPtr, dstPtr, -1.0, 1.0, 0.0, 1.0, dstPtr);
    #else
    vsPowx(len, srcPtr, 2, dstPtr);
    vsLinearFrac(len, dstPtr, dstPtr, -1.0, 1.0, 0.0, 1.0, dstPtr);
    #endif
}
#endif

void ApplyDTanHAct(NMatrix *srcMat, int row, int col, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && dstMat->colNum == col))
            HError(5221, "ApplyDTanHAct: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "ApplyDTanHAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyDTanHActCUDA(srcMat->devElems, row * col, dstMat->devElems);
#else
    #ifdef MKL
    ApplyDTanHActMKL(srcMat->matElems, row * col, dstMat->matElems);
    #else
    ApplyDTanHActCPU(srcMat->matElems, row * col, dstMat->matElems);
    #endif
#endif
}

static inline void ApplySoftmaxActCPU(NFloat *srcPtr, int row, int col, NFloat *dstPtr) {
    int i, j, off;
    NFloat sumval, maxval;

    for (i = 0, off = 0; i < row; ++i, off += col) {
    #ifdef DOUBLEANN
        maxval = MINFLTEXPE;
    #else
        maxval = MINDBLEXPE;
    #endif
        sumval = 0.0;
        
        for (j = 0; j < col; ++j) {
            dstPtr[off + j] = srcPtr[off + j];
            CHKNFLTEXPE(dstPtr[off + j])
            if (dstPtr[off + j] > maxval)
                maxval = dstPtr[off + j];
        }
        for (j = 0; j < col; ++j) {
            dstPtr[off + j] = exp(dstPtr[off + j] - maxval);
            sumval += dstPtr[off + j];
        }
        for (j = 0; j < col; ++j) {
            dstPtr[off + j] /= sumval;
        }
    }
}

#ifdef MKL
static inline void ApplySoftmaxActMKL(NFloat *srcPtr, int row, int col, NFloat *dstPtr) {
    int i, j, off;
    NFloat sumval, maxval;

    for (i = 0, off = 0; i < row; ++i, off += col) {
#ifdef DOUBLEANN
        maxval = MINFLTEXPE;
#else
        maxval = MINDBLEXPE;
#endif
        for (j = 0; j < col; ++j) {
            dstPtr[off + j] = srcPtr[off + j];
            CHKNFLTEXPE(dstPtr[off + j])
            if (dstPtr[off + j] > maxval)
                maxval = dstPtr[off + j];
        }
        for (j = 0; j < col; ++j) {
            dstPtr[off + j] -= maxval;
        }
    #ifdef DOUBLEANN
        vdExp(col, &dstPtr[off], &dstPtr[off]);
        sumval = cblas_dasum(col, &dstPtr[off], 1);
        cblas_dscal(col, 1.0 / sumval, &dstPtr[off], 1);
    #else
        vsExp(col, &dstPtr[off], &dstPtr[off]);
        sumval = cblas_sasum(col, &dstPtr[off], 1);
        cblas_sscal(col, 1.0 / sumval, &dstPtr[off], 1);
    #endif
    }
}
#endif

void ApplySoftmaxAct(NMatrix *srcMat, int row, int col, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && dstMat->colNum == col))
            HError(5221, "ApplySoftmaxAct: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "ApplySoftmaxAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyRedSoftmaxActCUDA(srcMat->devElems, row, col, dstMat->devElems);
#else
    #ifdef MKL
    ApplySoftmaxActMKL(srcMat->matElems, row, col, dstMat->matElems);
    #else
    ApplySoftmaxActCPU(srcMat->matElems, row, col, dstMat->matElems);
    #endif
#endif
}

static inline void ApplySoftSignActCPU(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int i;

    /*len = row * col;*/
    for (i = 0; i < len; ++i)
        dstPtr[i] = srcPtr[i] / (1 + abs(srcPtr[i]));
}

void ApplySoftSignAct(NMatrix *srcMat, int row, int col, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && dstMat->colNum == col))
            HError(5221, "ApplySoftSignAct: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "ApplySoftSignAct: Input row number out of range");
    }
#ifdef CUDA
    ApplySoftSignActCUDA(srcMat->devElems, row * col, dstMat->devElems);
#else
    ApplySoftSignActCPU(srcMat->matElems, row * col, dstMat->matElems);
#endif
}

/* or 1.7159 * tanh(0.666666 * x) */
/*static inline void ApplyTanHActCPU(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int i;
    NFloat expVal;

    for (i = 0; i < len; ++i) {
        expVal = srcPtr[i];
        CHKNFLTEXPE(expVal)
        expVal = exp(expVal);
        dstPtr[i] = (expVal - 1 / expVal) / (expVal + 1 / expVal);
    }
}

#ifdef MKL
static inline void ApplyTanHActMKL(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int i;

    for (i = 0; i < len; ++i) {
        if (dstPtr != srcPtr) {
            dstPtr[i] = srcPtr[i];
        }
        CHKNFLTEXPE(dstPtr[i])
    }
    #ifdef DOUBLEANN
    vdTanh(len, dstPtr, dstPtr);
    #else
    vsTanh(len, dstPtr, dstPtr);
    #endif
}
#endif

void ApplyTanHAct(NMatrix *srcMat, int row, int col, NMatrix *dstMat) {
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && dstMat->colNum == col))
            HError(9999, "ApplyTanHAct: Input column number incompatible");    
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(9999, "ApplyTanHAct: Input row number out of range");
    }
#ifdef CUDA
    ApplyTanHActCUDA(srcMat->devElems, row * col, dstMat->devElems);
#else
    #ifdef MKL
    ApplyTanHActMKL(srcMat->matElems, row * col, dstMat->matElems);
    #else
    ApplyTanHActCPU(srcMat->matElems, row * col, dstMat->matElems);
    #endif
#endif
}
*/

static inline void ApplyLogTransCPU(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int i;

    /*len = row * col;*/
    for (i = 0; i < len; ++i) {
        if (srcPtr[i] <= 0) {   /* srcPtr[i] < 0? */
            dstPtr[i] = LZERO;
        }
        else {
            dstPtr[i] = log(srcPtr[i]);
            if (dstPtr[i] < LSMALL) {
                dstPtr[i] = LSMALL;
            }
        }
    }
}

/*#ifdef MKL
static inline void ApplyLogTransMKL(NFloat *srcPtr, int row, int col, NFloat *dstPtr) {
    int i, len;
    
    len = row * col;
    for (i = 0; i < len; ++i) {
        if (srcPtr[i] <= 0) {
            srcPtr[i] = 1E-20;
        }
    }
    #ifdef DOUBLEANN
    vdLn(len, srcPtr, dstPtr);
    #else
    vsLn(len, srcPtr, dstPtr);
    #endif
}
#endif*/

void ApplyLogTrans(NMatrix *srcMat, int row, int col, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col && dstMat->colNum == col))
            HError(5221, "ApplyLogTrans: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "ApplyLogTrans: Input row number out of range");
    }
#ifdef CUDA
    ApplyLogTransCUDA(srcMat->devElems, row * col, dstMat->devElems);
#else
    ApplyLogTransCPU(srcMat->matElems, row * col, dstMat->matElems);
#endif
}

static inline void SumNMatrixByColCPU(NFloat *srcPtr, int row, int col, Boolean accFlag, NFloat *dstPtr) {
    NFloat *res_end_b8p = dstPtr + (col & ~7);
    NFloat *res_end_p = dstPtr + col;
    NFloat *res_p = dstPtr;
    NFloat *in_p = srcPtr;
    int i;

    if (accFlag == FALSE) 
        memset(res_p, 0, col * sizeof(NFloat));

    while (res_p != res_end_b8p) {
        res_p[0] += in_p[0];
        res_p[1] += in_p[1];
        res_p[2] += in_p[2];
        res_p[3] += in_p[3];
        res_p[4] += in_p[4];
        res_p[5] += in_p[5];
        res_p[6] += in_p[6];
        res_p[7] += in_p[7];
        res_p += 8;
        in_p += 8;
    }
    while (res_p != res_end_p) 
        (*res_p++) += (*in_p++);

    for (i = 1; i != row; ++i) {
        res_p = dstPtr;
        while (res_p != res_end_b8p) {
            res_p[0] += in_p[0];
            res_p[1] += in_p[1];
            res_p[2] += in_p[2];
            res_p[3] += in_p[3];
            res_p[4] += in_p[4];
            res_p[5] += in_p[5];
            res_p[6] += in_p[6];
            res_p[7] += in_p[7];
            res_p += 8;
            in_p += 8;
        }
        while (res_p != res_end_p) 
            (*res_p++) += (*in_p++);
    }
}

void SumNMatrixByCol(NMatrix *srcMat, int row, int col, Boolean accFlag, NVector *dstVec) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->rowNum >= row))
            HError(5221, "SumMatrixByCol: Row number inconsistent");
        if (!(srcMat->colNum == col && dstVec->vecLen == col))
            HError(5221, "SumMatrixByCol: Column number inconsistent");
    }
#ifdef CUDA
    RedSumNMatrixByColCUDA(srcMat->devElems, row, col, accFlag, dstVec->devElems);
#else
    SumNMatrixByColCPU(srcMat->matElems, row, col, accFlag, dstVec->vecElems);
#endif

}

static inline void SquaredNSegmentCPU(NFloat *srcPtr, int segLen, NFloat *dstPtr) {
    int i;

    for (i = 0; i < segLen; ++i) {
        dstPtr[i] = pow(srcPtr[i], 2);
    }
}

#ifdef MKL
static inline void SquaredNSegmentMKL(NFloat *srcPtr, int segLen, NFloat *dstPtr) {
    #ifdef DOUBLEANN
    vdPowx(segLen, srcPtr, 2, dstPtr);
    #else
    vsPowx(segLen, srcPtr, 2, dstPtr);
    #endif
}
#endif

void SquaredNMatrix(NMatrix *srcMat, int row, int col, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->rowNum >= row && dstMat->rowNum >= row))
            HError(5221, "SquaredNMatrix: Incompatible matrix row number");
        if (!(srcMat->colNum >= col && dstMat->colNum >= col))
            HError(5221, "SquaredNMatrix: Incompatible matrix column number");
    }

#ifdef CUDA
    SquaredNSegmentCUDA(srcMat->devElems, row * col, dstMat->devElems);
#else
    #ifdef MKL
    SquaredNSegmentMKL(srcMat->matElems, row * col, dstMat->matElems);
    #else
    SquaredNSegmentCPU(srcMat->matElems, row * col, dstMat->matElems);
    #endif
#endif 
}

void SquaredNVector(NVector *srcVec, int len, NVector *dstVec) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcVec->vecLen >= len && dstVec->vecLen >= len))
            HError(5221, "SquaredNMatrix: Incompatible vector lengths");
    }

#ifdef CUDA
    SquaredNSegmentCUDA(srcVec->devElems, len, dstVec->devElems);
#else
    #ifdef MKL
    SquaredNSegmentMKL(srcVec->vecElems, len, dstVec->vecElems);
    #else
    SquaredNSegmentCPU(srcVec->vecElems, len, dstVec->vecElems);
    #endif
#endif 
}

static inline void CompAdaGradNSegmentCPU(NFloat eta, int K, int segLen, NFloat *ssgSeg, NFloat *nlrSeg) {
    int i;

    for (i = 0; i < segLen; ++i) {
        nlrSeg[i] = eta / sqrt(K + ssgSeg[i]);
    }
}

void CompAdaGradNVector(NFloat eta, int K, NVector *ssgVec, NVector *nlrVec) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(ssgVec->vecLen == nlrVec->vecLen))
            HError(5221, "CompAdaGradNVector: Vector length inconsistent");
    }
#ifdef CUDA
    CompAdaGradNSegmentCUDA(eta, K, nlrVec->vecLen, ssgVec->devElems, nlrVec->devElems);
#else
    CompAdaGradNSegmentCPU(eta, K, nlrVec->vecLen, ssgVec->vecElems, nlrVec->vecElems);
#endif
}

void CompAdaGradNMatrix(NFloat eta, int K, NMatrix *ssgMat, NMatrix *nlrMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(ssgMat->rowNum == nlrMat->rowNum)) 
            HError(5221, "CompAdaGradNMatrix: Matrix row inconsistent");
        if (!(ssgMat->colNum == nlrMat->colNum))
            HError(5221, "CompAdaGradNMatrix: Matrix column inconsistent");
    }
#ifdef CUDA
    CompAdaGradNSegmentCUDA(eta, K, nlrMat->rowNum * nlrMat->colNum, ssgMat->devElems, nlrMat->devElems);
#else
    CompAdaGradNSegmentCPU(eta, K, nlrMat->rowNum * nlrMat->colNum, ssgMat->matElems, nlrMat->matElems);
#endif
}

static inline void FindMaxElementCPU(NFloat *srcPtr, int row, int col, IntVec resVec) {
    int maxIdx, i, j;
    NFloat maxVal;

    for (i = 0; i < row; ++i) {
        maxIdx = 0;
        maxVal = srcPtr[i * col + 0];
        for (j = 1; j < col; ++j) {
            if (maxVal < srcPtr[i * col + j]) {
                maxIdx = j;
                maxVal = srcPtr[i * col + j];
            }
        }
        resVec[i + 1] = maxIdx;
    }
}


/*#ifdef MKL
static inline void FindMaxElementMKL(NFloat *srcPtr, int row, int col, IntVec resVec) {
    int i, maxIdx;

    for (i = 0; i < row; ++i) {
#ifdef DOUBLEANN
        maxIdx = cblas_idamax(col, &srcPtr[i * col], 1);
#else
        maxIdx = cblas_isamax(col, &srcPtr[i * col], 1);
#endif
        resVec[i + 1] = maxIdx;
    }
}
#endif*/

/* TODO: GPU support */
void FindMaxElement(NMatrix *srcMat, int row, int col, IntVec resVec) {
#ifdef CUDA
    int i;
#endif

    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->colNum == col))
            HError(5221, "FindMaxElement: Input column number incompatible");
        if (!(row > 0 && row <= srcMat->rowNum && row <= IntVecSize(resVec)))
            HError(5221, "FindMaxElement: Input row number out of range");
    }

#ifdef CUDA
    FindMaxElementCUDA(srcMat->devElems, row, col, tmpNMat->devElems);
    SyncNMatrixDev2Host(tmpNMat);
    for (i = 0; i < row; ++i) 
        resVec[i + 1] = (int) tmpNMat->matElems[i];
#else
    FindMaxElementCPU(srcMat->matElems, row, col, resVec);
#endif
    
}

static inline void HNBlasNNgemmCPU(int m, int n, int k, NFloat alpha, NFloat *A, NFloat *B, NFloat beta, NFloat *C) {
    int i, j, l;

    /*for (i = 0; i < m; ++i) {
        for (l = 0; l < k; ++l) {
            C[i * k + l] *= beta;
            for (j = 0; j < n; ++j) {
                C[i * k + l] += alpha * A[i * n + j] * B[j * k + l];
            }
        }
    }*/
    for (i = 0; i < n; ++i) {
        for (j = 0; j < m; ++j) {
            C[i * m + j] *= beta;
            for (l = 0; l < k; ++l) {
                C[i * m + j] += alpha * A[l * m + j] * B[i * k + l];
            }
        }
    }
}

#ifdef MKL
static inline void HNBlasNNgemmMKL(int m, int n, int k, NFloat alpha, NFloat *A, NFloat *B, NFloat beta, NFloat *C) {
    #ifdef DOUBLEANN
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, m, B, k, beta, C, m);
    #else
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, m, B, k, beta, C, m);
    #endif
}
#endif

/* do C[m * k] = a * A[m * n] * B[n * k] + b * C[m * k] */
void HNBlasNNgemm(int m, int n, int k, NFloat alpha, NMatrix *A, NMatrix *B, NFloat beta, NMatrix *C) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(m > 0 && m <= A->rowNum && m <= C->rowNum))
            HError(5221, "HNBlasNNgemm: First input dimension out of range");
        if (!(n > 0 && n <= A->colNum && n <= B->rowNum))
            HError(5221, "HNBlasNNgemm: Second input dimension out of range");
        if (!(k > 0 && k <= B->colNum && k <= C->colNum))
            HError(5221, "HNBlasNNgemm: Third input dimension out of range");
    }
#ifdef CUDA
    HNBlasNNgemmCUDA(m, n, k, alpha, A->devElems, B->devElems, beta, C->devElems);
#else
    #ifdef MKL
    HNBlasNNgemmMKL(m, n, k, alpha, A->matElems, B->matElems, beta, C->matElems);
    #else
    HNBlasNNgemmCPU(m, n, k, alpha, A->matElems, B->matElems, beta, C->matElems);
    #endif
#endif
}

static inline void HNBlasNTgemmCPU(int m, int n, int k, NFloat alpha, NFloat *A, NFloat *B, NFloat beta, NFloat *C) {
    int i, j, l;
    /*for (i = 0; i < m; ++i) {
        for (l = 0; l < k; ++l) {
            C[i * k + l] *= beta;
            for (j = 0; j < n; ++j) {
                C[i * k + l] += alpha * A[i * n + j] * B[l * n + j];
            }
        }
    }*/
    for (i = 0; i < n; ++i) {
        for (j = 0; j < m; ++j) {
            C[i * m + j] *= beta;
            for (l = 0; l < k; ++l) {
                C[i * m + j] += alpha * A[l * m + j] * B[l * n + i];
            }
        }
    }
}

#ifdef MKL
static inline void HNBlasNTgemmMKL(int m, int n, int k, NFloat alpha, NFloat *A, NFloat *B, NFloat beta, NFloat *C) {
    #ifdef DOUBLEANN
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, A, m, B, n, beta, C, m);
    #else
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, A, m, B, n, beta, C, m);
    #endif
}
#endif

/* do C[m * k] = a * A[m * n] * B[k * n]^T + b * C[m * k] */
void HNBlasNTgemm(int m, int n, int k, NFloat alpha, NMatrix *A, NMatrix *B, NFloat beta, NMatrix *C) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(m > 0 && m <= A->rowNum && m <= C->rowNum))
            HError(5221, "HNBlasNTgemm: First input dimension out of range");
        if (!(n > 0 && n <= A->colNum && n <= B->colNum))
            HError(5221, "HNBlasNTgemm: Second input dimension out of range");
        if (!(k > 0 && k <= B->rowNum && k <= C->colNum))
            HError(5221, "HNBlasNTgemm: Third input dimension out of range");
    }
#ifdef CUDA
    HNBlasNTgemmCUDA(m, n, k, alpha, A->devElems, B->devElems, beta, C->devElems);
#else
    #ifdef MKL
    HNBlasNTgemmMKL(m, n, k, alpha, A->matElems, B->matElems, beta, C->matElems);
    #else
    HNBlasNTgemmCPU(m, n, k, alpha, A->matElems, B->matElems, beta, C->matElems);
    #endif
#endif

}

static inline void HNBlasTNgemmCPU(int m, int n, int k, NFloat alpha, NFloat *A, NFloat *B, NFloat beta, NFloat *C) {
    int i, j, l;

    for (i = 0; i < n; ++i) {
        for (j = 0; j < m; ++j) {
            C[i * m + j] *= beta;
            for (l = 0; l < k; ++l) {
                C[i * m + j] += alpha * A[j * k + l] * B[i * k + l];
            }
        }
    }
}

#ifdef MKL
static inline void HNBlasTNgemmMKL(int m, int n, int k, NFloat alpha, NFloat *A, NFloat *B, NFloat beta, NFloat *C) {
    #ifdef DOUBLEANN
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n, k, alpha, A, k, B, k, beta, C, m);
    #else
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n, k, alpha, A, k, B, k, beta, C, m);
    #endif
}
#endif

/* do C[m * k] = a * A[n * m]^T * B[n * k] + b * C[m * k] */
void HNBlasTNgemm(int m, int n, int k, NFloat alpha, NMatrix *A, NMatrix *B, NFloat beta, NMatrix *C) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(m > 0 && m <= A->colNum && m <= C->rowNum))
            HError(5221, "HNBlasTNgemm: First input dimension out of range");
        if (!(n > 0 && n <= A->rowNum && n <= B->rowNum))
            HError(5221, "HNBlasTNgemm: Second input dimension out of range");
        if (!(k > 0 && k <= B->colNum && k <= C->colNum))
            HError(5221, "HNBlasTNgemm: Third input dimension out of range");
    }
#ifdef CUDA
    HNBlasTNgemmCUDA(m, n, k, alpha, A->devElems, B->devElems, beta, C->devElems);
#else
    #ifdef MKL
    HNBlasTNgemmMKL(m, n, k, alpha, A->matElems, B->matElems, beta, C->matElems);
    #else
    HNBlasTNgemmCPU(m, n, k, alpha, A->matElems, B->matElems, beta, C->matElems);
    #endif
#endif

}

void SetNSegmentCPU(NFloat val, NFloat *segPtr, int segLen) {
    int i;
 
    for (i = 0; i < segLen; ++i)
        segPtr[i] = val;
}

void SetNVector(NFloat val, NVector *vec) {
#ifdef CUDA
    SetNSegmentCUDA(val, vec->devElems, vec->vecLen);
#else
    SetNSegmentCPU(val, vec->vecElems, vec->vecLen);
#endif
}

void SetNMatrix(NFloat val, NMatrix *mat, int nrows) {
    int len;

    len = nrows * mat->colNum;
#ifdef CUDA
    SetNSegmentCUDA(val, mat->devElems, len);
#else
    SetNSegmentCPU(val, mat->matElems, len);
#endif
}

void SetNMatrixSegment(NFloat val, NMatrix *mat, int off, int len) {

#ifdef CUDA
    SetNSegmentCUDA(val, mat->devElems + off, len);
#else
    SetNSegmentCPU(val, mat->matElems + off, len);
#endif
}

void SetNVectorSegment(NFloat val, NVector *vec, int off, int len) {
#ifdef CUDA
    SetNSegmentCUDA(val, vec->devElems + off, len);
#else
    SetNSegmentCPU(val, vec->vecElems + off, len);
#endif
}

static void ClearNSegmentCPU(NFloat *segPtr, int segLen) {
    memset(segPtr, 0, segLen * sizeof(NFloat));
}

void ClearNVector(NVector *vec) {

#ifdef CUDA
    ClearNSegmentCUDA(vec->devElems, vec->vecLen);
#else
    ClearNSegmentCPU(vec->vecElems, vec->vecLen);
#endif
}

void ClearNMatrix(NMatrix *mat, int nrows) {
    int len;

    len = nrows * mat->colNum;
#ifdef CUDA
    ClearNSegmentCUDA(mat->devElems, len);
#else
    ClearNSegmentCPU(mat->matElems, len);
#endif
}

void ClearNMatrixSegment(NMatrix *mat, int off, int len) {

#ifdef CUDA
    ClearNSegmentCUDA(mat->devElems + off, len);
#else
    ClearNSegmentCPU(mat->matElems + off, len);
#endif
}

void ClearNVectorSegment(NVector *vec, int off, int len) {

#ifdef CUDA
    ClearNSegmentCUDA(vec->devElems + off, len);
#else
    ClearNSegmentCPU(vec->vecElems + off, len);
#endif
}

/*void ClearNVector(NVector *vec) {

    ClearNSegmentCPU(vec->vecElems, vec->vecLen);
#ifdef CUDA
    SyncHost2Dev(vec->vecElems, vec->devElems, vec->vecLen * sizeof(NFloat));
#endif
}

void ClearNMatrix(NMatrix *mat, int nrows) {
    int len;

    len = nrows * mat->colNum;
    ClearNSegmentCPU(mat->matElems, len);
#ifdef CUDA
    SyncHost2Dev(mat->matElems, mat->devElems, len * sizeof(NFloat));
#endif
}

void ClearNMatrixSegment(NMatrix *mat, int off, int len) {

    ClearNSegmentCPU(mat->matElems + off, len);
#ifdef CUDA
    SyncHost2Dev(mat->matElems + off, mat->devElems + off, len * sizeof(NFloat));
#endif
}

void ClearNVectorSegment(NVector *vec, int off, int len) {

    ClearNSegmentCPU(vec->vecElems + off, len);
#ifdef CUDA
    SyncHost2Dev(vec->vecElems + off, vec->devElems + off, len * sizeof(NFloat));
#endif
}*/

/*void SetNVector(NFloat val, NVector *vec) {

#ifdef CUDA
    SyncNVectorDev2Host(vec);
#endif
    SetNSegmentCPU(val, vec->vecElems, vec->vecLen);
#ifdef CUDA
    SyncNVectorHost2Dev(vec);
#endif
}

void SetNMatrix(NFloat val, NMatrix *mat, int nrows) {

#ifdef CUDA
    SyncNMatrixDev2Host(mat);
#endif
    SetNSegmentCPU(val, mat->matElems, nrows * mat->colNum);
#ifdef CUDA
    SyncNMatrixHost2Dev(mat);
#endif
}

void SetNMatrixSegment(NFloat val, NMatrix *mat, int off, int len) {

#ifdef CUDA
    SyncNMatrixDev2Host(mat);
#endif
    SetNSegmentCPU(val, mat->matElems + off, len);
#ifdef CUDA
    SyncNMatrixHost2Dev(mat);
#endif
}

void SetNVectorSegment(NFloat val, NVector *vec, int off, int len) {

#ifdef CUDA
    SyncNVectorDev2Host(vec);
#endif
    SetNSegmentCPU(val, vec->vecElems + off, len);
#ifdef CUDA
    SyncNVectorHost2Dev(vec);
#endif
}*/

/* cz277 - laf */
void RandNSegmentGaussian(NFloat mu, NFloat sigma, int segLen, NFloat *segPtr) {
    int i;

    if (sigma <= 0.0)
        HError(5225, "RandNSegmentGauss: Standard deviation need to be positive");
    for (i = 0; i < segLen; ++i) {
        segPtr[i] = GaussDeviate(mu, sigma);
    }
}

void RandNSegmentUniform(NFloat lower, NFloat upper, int segLen, NFloat *segPtr) {
    int i;
    NFloat range;

    range = upper - lower;
    if (range < 0)
        HError(5225, "RandNSegment: Random range should be positive");
    for (i = 0; i < segLen; ++i) {
        segPtr[i] = RandomValue();
        if (range > 0) {	/* range == 0 means no operation here */
            segPtr[i] *= range;
            segPtr[i] += lower;
        }
    }
}

/* cz277 - 0 mask */
void RandMaskNSegment(NFloat prob, NFloat mask, int segLen, NFloat *segPtr) {
    int i;

    if (prob < 0 || prob > 1)
        HError(5225, "RandMaskNSegment: Mask probability shoudl be in [0, 1]");
    for (i = 0; i < segLen; ++i) {
        if (RandomValue() < prob)
            segPtr[i] = mask;
    }
}

static inline void CalXENTCriterionMKL(NFloat *refSeg, NFloat *hypSeg, int segLen) {
    HError(5201, "CalXENTCriterionMKL: Unimplemented method");
}

static inline void CalXENTCriterionCPU(NFloat *refSeg, NFloat *hypSeg, int segLen) {
    int i;
    NFloat tn, yn, sum;

    sum = 0.0;
    for (i = 0; i < segLen; ++i) {
        tn = refSeg[i];
        yn = hypSeg[i];
        if (tn == 0.0)
            sum += 0.0;
        else if (yn == 0.0)
            sum += tn * LZERO;
        else
            sum += (-1.0) * tn * log(yn / tn);
    }
    tmpNMat->matElems[0] = sum;
}

NFloat CalXENTCriterion(NMatrix *refMat, NMatrix *hypMat, int rowNum) {
    int segLen;

    /* safety check */
    if (trace & T_DIM) {
        if (!(refMat->colNum == hypMat->colNum)) 
            HError(5221, "CalXENTCriterion: Column number should be consistent");
        if (!(refMat->rowNum >= rowNum && hypMat->rowNum >= rowNum))
            HError(5221, "CalXENTCriterion: Row number out of range");
    }
    segLen = rowNum * refMat->colNum;
#ifdef CUDA
    CalXENTCriterionCUDA(refMat->devElems, hypMat->devElems, segLen, tmpNMat->devElems);
    SyncNMatrixDev2Host(tmpNMat);
#else
    /*#ifdef MKL
    CalXENTCriterionMKL(refMat->matElems, hypMat->matElems, segLen);
    #else
    CalXENTCriterionCPU(refMat->matElems, hypMat->matElems, segLen);
    #endif*/
    CalXENTCriterionCPU(refMat->matElems, hypMat->matElems, segLen);
#endif
    return tmpNMat->matElems[0];
}

static inline void CalMMSECriterionMKL(NFloat *refSeg, NFloat *hypSeg, int segLen) {
    HError(5201, "CalXENTCriterionMKL: Unimplemented method");
}

static inline void CalMMSECriterionCPU(NFloat *refSeg, NFloat *hypSeg, int segLen) {
    int i;

    tmpNMat->matElems[0] = 0.0;
    for (i = 0; i < segLen; ++i) 
        tmpNMat->matElems[0] += pow(hypSeg[i] - refSeg[i], 2.0);
}

NFloat CalMMSECriterion(NMatrix *refMat, NMatrix *hypMat, int rowNum) {
    int segLen;

    /* safety check */
    if (trace & T_DIM) {
        if (!(refMat->colNum == hypMat->colNum)) 
            HError(5221, "CalMMSECriterion: Column number should be consistent");
        if (!(refMat->rowNum >= rowNum && hypMat->rowNum >= rowNum))
            HError(5221, "CalMMSECriterion: Row number out of range");
    }
    segLen = rowNum * refMat->colNum;
#ifdef CUDA
    CalMMSECriterionCUDA(refMat->devElems, hypMat->devElems, segLen, tmpNMat->devElems);
    SyncNMatrixDev2Host(tmpNMat);
#else
    /*#ifdef MKL
    CalMMSECriterionMKL(refMat->matElems, hypMat->matElems, segLen);
    #else
    CalMMSECriterionCPU(refMat->matElems, hypMat->matElems, segLen);
    #endif*/
    CalMMSECriterionCPU(refMat->matElems, hypMat->matElems, segLen);
#endif
    return tmpNMat->matElems[0] / refMat->colNum;
}

static inline void AddNSegmentTargetPenCPU(NFloat *srcSeg, NFloat *penSeg, int row, int col, NFloat *dstSeg) {
    int i, j;

    for (i = 0; i < row; ++i) {
        for (j = 0; j < col; ++j) {
            dstSeg[i * col + j] = srcSeg[i * col + j] + penSeg[j];
        }
    }
}

#ifdef MKL
static inline void AddNSegmentTargetPenMKL(NFloat *srcSeg, NFloat *penSeg, int row, int col, NFloat *dstSeg) {
    int i;

    for (i = 0; i < row; ++i) {
#ifdef DOUBLEANN
        vdAdd(col, srcSeg + i * col, penSeg, dstSeg + i * col);
#else
        vsAdd(col, srcSeg + i * col, penSeg, dstSeg + i * col);
#endif
    }
}
#endif

void AddNVectorTargetPen(NMatrix *srcMat, NVector *penVec, int nrows, NMatrix *dstMat) {
    
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->colNum == penVec->vecLen && dstMat->colNum == penVec->vecLen)) 
            HError(5221, "AddNVectorTargetPen: Column number should equal to the vector length");
        if (!(nrows <= srcMat->rowNum && nrows <= dstMat->rowNum))
            HError(5221, "AddNVectorTargetPen: Row number out of range");
    }    
#ifdef CUDA
    AddNSegmentTargetPenCUDA(srcMat->devElems, penVec->devElems, nrows, penVec->vecLen, dstMat->devElems);
#else
    #ifdef MKL
    AddNSegmentTargetPenMKL(srcMat->matElems, penVec->vecElems, nrows, penVec->vecLen, dstMat->matElems);
    #else
    AddNSegmentTargetPenCPU(srcMat->matElems, penVec->vecElems, nrows, penVec->vecLen, dstMat->matElems);
    #endif
#endif
}

static inline void ShiftNSegmentValsCPU(NFloat *srcSeg, int segLen, NFloat shiftVal, NFloat *dstSeg) {
    int i;

    for (i = 0; i < segLen; ++i) {
        dstSeg[i] = srcSeg[i] + shiftVal;
    }
}

/* cz277 - semi */
/*  */
void ShiftNMatrixVals(NMatrix *srcMat, int row, int col, NFloat shiftVal, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->rowNum >= row && dstMat->rowNum >= row))
            HError(5221, "ShiftNMatrixVals: Incompatible matrix row number");
        if (!(srcMat->colNum >= col && dstMat->colNum >= col))
            HError(5221, "ShiftNMatrixVals: Incompatible matrix column number");
    }
#ifdef CUDA
    ShiftNSegmentValsCUDA(srcMat->devElems, row * col, shiftVal, dstMat->devElems);
#else
    ShiftNSegmentValsCPU(srcMat->matElems, row * col, shiftVal, dstMat->matElems);
#endif
}

/* cz277 - semi */
/*  */
void ShiftNVectorVals(NVector *srcVec, int len, NFloat shiftVal, NVector *dstVec) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(srcVec->vecLen >= len && dstVec->vecLen >= len))
            HError(5221, "ShiftNVectorVals: Incompatible vector lengths");
    }
#ifdef CUDA
    ShiftNSegmentValsCUDA(srcVec->devElems, len, shiftVal, dstVec->devElems);
#else
    ShiftNSegmentValsCPU(srcVec->vecElems, len, shiftVal, dstVec->vecElems);
#endif
}

static void CopyPartialNSegmentCPU(int minRow, int minCol, NFloat *srcPtr, int srcCol, NFloat *dstPtr, int dstCol) {
    int i;
    
    for (i = 0; i < minRow; ++i) {
        memcpy(dstPtr + i * dstCol, srcPtr + i * srcCol, sizeof(NFloat) * minCol);
    }
}

void CopyPartialNSegment(int minRow, int minCol, NFloat *srcPtr, int srcCol, NFloat *dstPtr, int dstCol) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(minCol > srcCol || minCol > dstCol))
            HError(5221, "CopyPartialNSegment: minCol should be smaller than srcCol and dstCol");
    }    

#ifdef CUDA
    CopyPartialNSegmentCUDA(minRow, minCol, srcPtr, srcCol, dstPtr, dstCol);
#else
    CopyPartialNSegmentCPU(minRow, minCol, srcPtr, srcCol, dstPtr, dstCol);
#endif
}

/* cz277 - gradlim */
static void ClipNSegmentValsCPU(NFloat* srcSeg, int len, NFloat upperLim, NFloat lowerLim, NFloat *dstSeg) {
    int i;

    for (i = 0; i < len; ++i) {
        if (srcSeg[i] > upperLim)
            dstSeg[i] = upperLim;
        else if (srcSeg[i] < lowerLim)
            dstSeg[i] = lowerLim;
        else if (srcSeg != dstSeg)
            dstSeg[i] = srcSeg[i];
    }
}

/* cz277 - gradlim */
void ClipNMatrixVals(NMatrix* srcMat, int row, int col, NFloat upperLim, NFloat lowerLim, NMatrix *dstMat) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(col > 0 && col <= srcMat->colNum && col <= dstMat->colNum))
            HError(5221, "ClipNMatrixVals: Input matrix column number out of range");
        if (!(row > 0 && row <= srcMat->rowNum && row <= dstMat->rowNum))
            HError(5221, "ClipNMatrixVals: Input matrix row number out of range");
        if (!(upperLim >= lowerLim))
            HError(5225, "ClipNMatrixVals: Invalid input value limits");
    }
#ifdef CUDA
    ClipNSegmentValsCUDA(srcMat->devElems, row * col, upperLim, lowerLim, dstMat->devElems);
#else 
    ClipNSegmentValsCPU(srcMat->matElems, row * col, upperLim, lowerLim, dstMat->matElems);
#endif

}

/* cz277 - gradlim */
void ClipNVectorVals(NVector* srcVec, int len, NFloat upperLim, NFloat lowerLim, NVector *dstVec) {
    /* safety check */
    if (trace & T_DIM) {
        if (!(len > 0 && len <= srcVec->vecLen && len <= dstVec->vecLen))
            HError(5221, "ClipNVectorVals: Input vector length out of range");
        if (!(upperLim >= lowerLim))
            HError(5225, "ClipNVectorVals: Invalid input value limits");
    }
#ifdef CUDA
    ClipNSegmentValsCUDA(srcVec->devElems, len, upperLim, lowerLim, dstVec->devElems);
#else
    ClipNSegmentValsCPU(srcVec->vecElems, len, upperLim, lowerLim, dstVec->vecElems);
#endif

}

/* cz277 - max norm */
static void CalNMatrixL2NormByRowCPU(NFloat *matPtr, int row, int col, NFloat *normPtr) {
    int i, j;

    for (i = 0; i < row; ++i) {
        normPtr[i] = 0.0;
        for (j = 0; j < col; ++j) 
            normPtr[i] += pow(matPtr[i * col + j], 2.0);
        normPtr[i] = sqrt(normPtr[i]);
    }
}

/* cz277 - max norm */
void CalNMatrixL2NormByRow(NMatrix *srcMat, NVector *normVec) {

    /* safety check */
    if (trace & T_DIM) {
        if (!(srcMat->rowNum <= normVec->vecLen))
            HError(5221, "CalNMatrixL2NormByRow: Norm vector dimension too small");
    }
#ifdef CUDA
    CalNMatrixL2NormByRowCUDA(srcMat->devElems, srcMat->rowNum, srcMat->colNum, normVec->devElems);
#else
    CalNMatrixL2NormByRowCPU(srcMat->matElems, srcMat->rowNum, srcMat->colNum, normVec->vecElems);
#endif
}

void CalNVectorL2Norm(NVector *srcVec, NFloat *normVal) {
 
#ifdef CUDA
    NVector *tmpNVec = GetTmpNVec();
    CalNMatrixL2NormByRowCUDA(srcVec->devElems, 1, srcVec->vecLen, tmpNVec->devElems);
    SyncNVectorDev2Host(tmpNVec);
    normVal[0] = tmpNVec->vecElems[0];
#else
    CalNMatrixL2NormByRowCPU(srcVec->vecElems, 1, srcVec->vecLen, normVal);
#endif
}

static void DivideNMatrixByRowCPU(NFloat *srcPtr, int row, int col, NFloat *normPtr, NFloat *dstPtr) {
    int i, j;

    for (i = 0; i < row; ++i)
        for (j = 0; j < col; ++j)
            dstPtr[i * col + j] = srcPtr[i * col + j] / normPtr[i];
}

void DivideNMatrixByRow(NMatrix *srcMat, NVector *normVec, NMatrix *dstMat) {
    if (trace & T_DIM) {
        if (!(srcMat->rowNum <= normVec->vecLen))
            HError(5221, "DivideNMatrixByRow: Norm vector dimension too small");
        if (!(srcMat->rowNum == dstMat->rowNum && srcMat->colNum >= dstMat->colNum))
            HError(5221, "DivideNMatrixByRow: Inconsistent matrix dimensions");
    }
#ifdef CUDA
    DivideNMatrixByRowCUDA(srcMat->devElems, srcMat->rowNum, srcMat->colNum, normVec->devElems, dstMat->devElems);
#else
    DivideNMatrixByRowCPU(srcMat->matElems, srcMat->rowNum, srcMat->colNum, normVec->vecElems, dstMat->matElems);
#endif
 
}

/* cz277 - xform */
int ClipInt(int min, int max, int val) {
    if (val < min)
        return min;
    else if (val > max)
        return max;
    else
        return val;
}

/* ------------------------- End of HMath.c ------------------------- */
