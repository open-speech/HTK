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
/*                 File: HMath.h   Math Support                */
/* ----------------------------------------------------------- */

/* !HVER!HMath:   3.5.0 [CUED 12/10/15] */

#ifndef _HMATH_H_
#define _HMATH_H_


#ifdef WIN32
#include <float.h>
#define isnan _isnan
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef PI
#undef PI                /* PI is defined in Linux */
#endif
#define PI   3.14159265358979
#define TPI  6.28318530717959     /* PI*2 */
#define LZERO  (-1.0E10)   /* ~log(0) */
#define LSMALL (-0.5E10)   /* log values < LSMALL are set to LZERO */
#define MINEARG (-708.3)   /* lowest exp() arg  = log(MINLARG) */
#define MINLARG 2.45E-308  /* lowest log() arg  = exp(MINEARG) */
/* cz277 - ANN */
/*#define MAXFLTEXPE (88.722839052)*/           /* FLT_MAX, 3.402823466e+38 */
/*#define MINFLTEXPE (-87.3365447504)*/         /* FLT_MIN, 1.175494351e–38 */
/*#define MAXDBLEXPE (709.78271289338399678)*/  /* DBL_MAX, 1.7976931348623158e+308 */
/*#define MINDBLEXPE (-708.39641853226410622)*/ /* DBL_MIN, 2.2250738585072014e–308 */
#define MAXFLTEXPE (87.3365447504)           
#define MINFLTEXPE (-87.3365447504)         
#define MAXDBLEXPE (708.39641853226410622) 
#define MINDBLEXPE (-708.39641853226410622) 

#define RELUNEGSCALE 0.01
/*#define RELUNEGSCALE 0.0*/
#define PLRELUNEGSCALE 0.25	/* cz277 - laf */

#ifdef DOUBLEANN
#define CHKNFLTEXPE(x) if((x)<MINDBLEXPE) (x)=MINDBLEXPE; else if((x)>MAXDBLEXPE) (x)=MAXDBLEXPE;
#else
#define CHKNFLTEXPE(x) if((x)<MINFLTEXPE) (x)=MINFLTEXPE; else if((x)>MAXFLTEXPE) (x)=MAXFLTEXPE;
#endif

/* cz277 - 1004 */
/*#define MAX(a, b) (a > b? a: b)
#define MIN(a, b) (a < b? a: b)*/


/* NOTE: On some machines it may be necessary to reduce the
         values of MINEARG and MINLARG
*/

typedef float  LogFloat;   /* types just to signal log values */
typedef double LogDouble;

typedef enum {  /* Various forms of covariance matrix */
   DIAGC,         /* diagonal covariance */
   INVDIAGC,      /* inverse diagonal covariance */
   FULLC,         /* inverse full rank covariance */
   XFORMC,        /* arbitrary rectangular transform */
   LLTC,          /* L' part of Choleski decomposition */
   NULLC,         /* none - implies Euclidean in distance metrics */
   NUMCKIND       /* DON'T TOUCH -- always leave as final element */
} CovKind;

typedef union {
   SVector var;         /* if DIAGC or INVDIAGC */
   STriMat inv;         /* if FULLC or LLTC */
   SMatrix xform;       /* if XFORMC */
} Covariance;


/* ------------------------------------------------------------------- */

void InitMath(void);
/*
   Initialise the module
*/

/* ------------------ Vector Oriented Routines ----------------------- */

void ZeroShortVec(ShortVec v);
void ZeroIntVec(IntVec v);
void ZeroVector(Vector v);
void ZeroDVector(DVector v);
/*
   Zero the elements of v
*/

void CopyShortVec(ShortVec v1, ShortVec v2);
void CopyIntVec(IntVec v1, IntVec v2);
void CopyVector(Vector v1, Vector v2);
void CopyDVector(DVector v1, DVector v2);
/*
   Copy v1 into v2; sizes must be the same
*/

Boolean ReadShortVec(Source *src, ShortVec v, Boolean binary);
Boolean ReadIntVec(Source *src, IntVec v, Boolean binary);
Boolean ReadVector(Source *src, Vector v, Boolean binary);
/*
   Read vector v from source in ascii or binary
*/

void WriteShortVec(FILE *f, ShortVec v, Boolean binary);
void WriteIntVec(FILE *f, IntVec v, Boolean binary);
void WriteVector(FILE *f, Vector v, Boolean binary);
/*
   Write vector v to stream f in ascii or binary
*/

void ShowShortVec(char * title, ShortVec v,int maxTerms);
void ShowIntVec(char * title, IntVec v,int maxTerms);
void ShowVector(char * title,Vector v,int maxTerms);
void ShowDVector(char * title,DVector v,int maxTerms);
/*
   Print the title followed by upto maxTerms elements of v
*/

/* Quadratic prod of a full square matrix C and an arbitry full matrix transform A */
void LinTranQuaProd(Matrix Prod, Matrix A, Matrix C);

/* ------------------ Matrix Oriented Routines ----------------------- */

void ZeroMatrix(Matrix m);
void ZeroDMatrix(DMatrix m);
void ZeroTriMat(TriMat m);
/*
   Zero the elements of m
*/

void CopyMatrix (Matrix m1,  Matrix m2);
void CopyDMatrix(DMatrix m1, DMatrix m2);
void CopyTriMat (TriMat m1,  TriMat m2);
/*
   Copy matrix m1 to m2 which must have identical dimensions
*/

void Mat2DMat(Matrix m1,  DMatrix m2);
void DMat2Mat(DMatrix m1, Matrix m2);
void Mat2Tri (Matrix m1,  TriMat m2);
void Tri2Mat (TriMat m1,  Matrix m2);
/*
   Convert matrix format from m1 to m2 which must have identical 
   dimensions
*/

Boolean ReadMatrix(Source *src, Matrix m, Boolean binary);
Boolean ReadTriMat(Source *src, TriMat m, Boolean binary);
/*
   Read matrix from source into m using ascii or binary.
   TriMat version expects m to be in upper triangular form
   but converts to lower triangular form internally.
*/
   
void WriteMatrix(FILE *f, Matrix m, Boolean binary);
void WriteTriMat(FILE *f, TriMat m, Boolean binary);
/*
    Write matrix to stream in ascii or binary.  TriMat version 
    writes m in upper triangular form even though it is stored
    in lower triangular form!
*/

void ShowMatrix (char * title,Matrix m, int maxCols,int maxRows);
void ShowDMatrix(char * title,DMatrix m,int maxCols,int maxRows);
void ShowTriMat (char * title,TriMat m, int maxCols,int maxRows);
/*
   Print the title followed by upto maxCols elements of upto
   maxRows rows of m.
*/

/* ------------------- Linear Algebra Routines ----------------------- */

LogFloat CovInvert(TriMat c, Matrix invc);
/*
   Computes inverse of c in invc and returns the log of Det(c),
   c must be positive definite.
*/

LogFloat CovDet(TriMat c);
/*
   Returns log of Det(c), c must be positive definite.
*/

/* EXPORT->MatDet: determinant of a matrix */
float MatDet(Matrix c);

/* EXPORT->DMatDet: determinant of a double matrix */
double DMatDet(DMatrix c);

/* EXPORT-> MatInvert: puts inverse of c in invc, returns Det(c) */
  float MatInvert(Matrix c, Matrix invc);
  double DMatInvert(DMatrix c, DMatrix invc);
 
/* DMatCofact: generates the cofactors of row r of doublematrix c */
double DMatCofact(DMatrix c, int r, DVector cofact);

/* MatCofact: generates the cofactors of row r of doublematrix c */
double MatCofact(Matrix c, int r, Vector cofact);

/* ------------- Singular Value Decomposition Routines --------------- */

void SVD(DMatrix A, DMatrix U,  DMatrix V, DVector d);
/* 
   Singular Value Decomposition (based on MESCHACH)
   A is m x n ,  U is m x n,  W is diag N x 1, V is n x n
*/

void InvSVD(DMatrix A, DMatrix U, DVector W, DMatrix V, DMatrix Result);
/* 
   Inverted Singular Value Decomposition (calls SVD)
   A is m x n ,  U is m x n,  W is diag N x 1, V is n x n, Result is m x n 
*/

/* ------------------- Log Arithmetic Routines ----------------------- */

LogDouble LAdd(LogDouble x, LogDouble y);
/*
   Return x+y where x and y are stored as logs, 
   sum < LSMALL is floored to LZERO 
*/

LogDouble LSub(LogDouble x, LogDouble y);
/*
   Return x-y where x and y are stored as logs, 
   diff < LSMALL is floored to LZERO 
*/

double L2F(LogDouble x);
/*
   Convert log(x) to real, result is floored to 0.0 if x < LSMALL 
*/

/* ------------------- Random Number Routines ------------------------ */

void RandInit(int seed);
/* 
   Initialise random number generators, if seed is -ve, then system 
   clock is used.  RandInit(-1) is called by InitMath.
*/

float RandomValue(void);
/*
   Return a random number in range 0.0->1.0 with uniform distribution
*/

float GaussDeviate(float mu, float sigma);
/*
   Return a random number with a N(mu,sigma) distribution
*/

/* from xl207, cz277 - gau */
float GaussInv(float p);
float CumGauss(float x, float mean, float var);

/* cz277 - ANN */
/* --------------------- ANN related math kernels --------------------- */

/* cz277 - 151020 */
#ifdef MKL
void StartMKL(void);
#endif

void RegisterTmpNMat(int nrows, int ncols);
void CreateTmpNMat(MemHeap *heap);
NMatrix *GetTmpNMat(void);
NVector *GetTmpNVec(void);
void FreeTmpNMat(MemHeap *heap);

void CopyNSegment(NMatrix *srcMat, int srcOff, int segLen, NMatrix *dstMat, int dstOff);
void CopyNVectorSegment(NVector *srcVec, int srcOff, int segLen, NVector *dstVec, int dstOff);
void AddNSegment(NMatrix *srcMat, int srcOff, int segLen, NMatrix *dstMat, int dstOff);
void AddNMatrix(NMatrix *srcMat, int row, int col, NMatrix *dstMat);
void AddNVector(NVector *srcVec, int len, NVector *dstVec);
void DupNVector(NVector *srcVec, NMatrix *dstMat, int times);
void SubNMatrix(NMatrix *lhMat, NMatrix *rhMat, int row, int col, NMatrix *resMat);
void MulNMatrix(NMatrix *lhMat, NMatrix *rhMat, int row, int col, NMatrix *resMat);
void MulNVector(NVector *lhVec, NVector *rhVec, int len, NVector *resVec);
void ScaleNMatrix(NFloat scale, int row, int col, NMatrix *valMat);
void ScaleNVector(NFloat scale, int len, NVector *valVec);
void ScaledSelfAddNVector(NVector *rhVec, int len, NFloat scale, NVector *lhVec);
void ScaledSelfAddNMatrix(NMatrix *rhMat, int row, int col, NFloat scale, NMatrix *lhMat);
/*void SumNMatrixByCol(NMatrix *srcMat, int row, int col, NFloat alpha, NFloat beta, NVector *dstVec);*/
void SumNMatrixByCol(NMatrix *srcMat, int row, int col, Boolean accFlag, NVector *dstVec);
void SquaredNMatrix(NMatrix *srcMat, int row, int col, NMatrix *dstMat);
void SquaredNVector(NVector *srcVec, int len, NVector *dstVec);
void CompAdaGradNVector(NFloat eta, int K, NVector *ssgVec, NVector *nlrVec);
void CompAdaGradNMatrix(NFloat eta, int K, NMatrix *ssgMat, NMatrix *nlrMat);
/* cz277 - laf */
void ApplyAffineAct(NMatrix *srcMat, int row, int col, NVector *scaleVec, NVector *shiftVec, NMatrix *dstMat);
void ApplyDAffineAct(NMatrix *srcMat, int row, int col, NVector *scaleVec, NVector *shiftVec, NMatrix *dstMat); 
void ApplyTrAffineAct(NMatrix *errMat, NMatrix *actMat, int row, int col, NVector *scaleVec, NVector *shiftVec, Boolean accFlag, NVector *dScaleVec, NVector *dShiftVec);
void AccMeanNVector(NMatrix *valMat, int row, int col, NFloat tSamp, NVector *meanVec);
void AccVarianceNVector(NMatrix *srcMat, int row, int col, NFloat tSamp, NVector *meanVec, NVector *varVec);
void ApplyLHUCSigmoidAct(NMatrix *srcMat, int row, int col, NVector *roleVec, NMatrix *dstMat);
void ApplyDLHUCSigmoidAct(NMatrix *srcMat, int row, int col, NVector *roleVec, NMatrix *dstMat);
void ApplyTrLHUCSigmoidAct(NMatrix *errMat, NMatrix *actMat, int row, int col, NVector *roleVec, Boolean accFlag, NVector *dRoleVec);
void ApplyPReLUAct(NMatrix *srcMat, int row, int col, NVector *scaleVec, NMatrix *dstMat);
void ApplyDPReLUAct(NMatrix *srcMat, int row, int col, NVector *scaleVec, NMatrix *dstMat);
void ApplyTrPReLUAct(NMatrix *errMat, NMatrix *srcMat, int row, int col, NVector *scaleVec, Boolean accFlag, NVector *dScaleVec);
void ApplyParmReLUAct(NMatrix *srcMat, int row, int col, NVector *posVec, NVector *negVec, NMatrix *dstMat);
void ApplyDParmReLUAct(NMatrix *inpMat, int row, int col, NVector *posVec, NVector *negVec, NMatrix *dstMat);
void ApplyTrParmReLUAct(NMatrix *errMat, NMatrix *inpMat, int row, int col, Boolean accFlag, NVector *dPosVec, NVector *dNegVec);
void ApplyPSigmoidAct(NMatrix *srcMat, int row, int col, NVector *etaVec, NMatrix *dstMat);
void ApplyDPSigmoidAct(NMatrix *srcMat, int row, int col, NVector *etaVec, NMatrix *dstMat);
void ApplyTrPSigmoidAct(NMatrix *errMat, NMatrix *srcMat, NVector *etaVec, int row, int col, Boolean accFlag, NVector *dEtaVec);
void ApplyParmSigmoidAct(NMatrix *srcMat, int row, int col, NVector *etaVec, NVector *gammaVec, NVector *thetaVec, NMatrix *dstMat);
void ApplyDParmSigmoidAct(NMatrix *srcMat, int row, int col, NVector *etaVec, NVector *gammaVec, NVector *thetaVec, NMatrix *dstMat);
void ApplyTrParmSigmoidAct(NMatrix *errMat, NMatrix *inpMat, int row, int col, NVector *etaVec, NVector *gammaVec, NVector *thetaVec, Boolean accFlag, NVector *dEtaVec, NVector *dGammaVec, NVector *dThetaVec);


void ApplyHermiteAct(NMatrix *srcMat, int row, int col, NVector *parmVec, NMatrix*dstMat);
void ApplyReLUAct(NMatrix *srcMat, int row, int col, NFloat scale, NMatrix *dstMat);
void ApplyDReLUAct(NMatrix *srcMat, int row, int col, NFloat scale, NMatrix *dstMat);
void ApplyDLinearAct(NMatrix *srcMat, int row, int col, NMatrix *dstMat);
void ApplySigmoidAct(NMatrix *srcMat, int row, int col, NMatrix *dstMat);
void ApplyDSigmoidAct(NMatrix *srcMat, int row, int col, NMatrix *dstMat);
void ApplyTanHAct(NMatrix *srcMat, int row, int col, NMatrix *dstMat);
void ApplyDTanHAct(NMatrix *srcMat, int row, int col, NMatrix *dstMat);
void ApplySoftmaxAct(NMatrix *srcMat, int row, int col, NMatrix *dstMat);
void ApplyDSoftmaxAct(NMatrix *srcMat, int row, int col, NMatrix *dstMat);
void ApplySoftReLAct(NMatrix *srcMat, int row, int col, NMatrix *dstMat);
void ApplyDSoftReLAct(NMatrix *srcMat, int row, int col, NMatrix *dstMat);
void ApplySoftSignAct(NMatrix *srcMat, int row, int col, NMatrix *dstMat);
void ApplyLogTrans(NMatrix *srcMat, int row, int col, NMatrix *dstMat);
void FindMaxElement(NMatrix *srcMat, int row, int col, IntVec resVec);
void HNBlasNNgemm(int m, int n, int k, NFloat alpha, NMatrix *A, NMatrix *B, NFloat beta, NMatrix *C);
void HNBlasNTgemm(int m, int n, int k, NFloat alpha, NMatrix *A, NMatrix *B, NFloat beta, NMatrix *C);
void HNBlasTNgemm(int m, int n, int k, NFloat alpha, NMatrix *A, NMatrix *B, NFloat beta, NMatrix *C);
/*void SetNSegment(NFloat val, NFloat *segPtr, int segLen);*/
void RandNSegmentGaussian(NFloat mu, NFloat sigma, int segLen, NFloat *segPtr);	/* cz277 - laf */
void RandNSegmentUniform(NFloat lower, NFloat upper, int segLen, NFloat *segPtr);
/* cz277 - 0 mask */
void RandMaskNSegment(NFloat prob, NFloat mask, int segLen, NFloat *segPtr);

NFloat CalXENTCriterion(NMatrix *refMat, NMatrix *hypMat, int rowNum);
NFloat CalMMSECriterion(NMatrix *refMat, NMatrix *hypMat, int rowNum);
void AddNVectorTargetPen(NMatrix *srcMat, NVector *penVec, int nrows, NMatrix *dstMat);
/* cz277 - semi */
void ShiftNMatrixVals(NMatrix *srcMat, int row, int col, NFloat shiftVal, NMatrix *dstMat);
void ShiftNVectorVals(NVector *srcVec, int len, NFloat shiftVal, NVector *dstVec);

void SetNSegmentCPU(NFloat val, NFloat *segPtr, int segLen);
void SetNVector(NFloat val, NVector *vec);
void SetNMatrix(NFloat val, NMatrix *mat, int nrows);
void SetNMatrixSegment(NFloat val, NMatrix *mat, int off, int len);
void SetNVectorSegment(NFloat val, NVector *vec, int off, int len);
void ClearNVector(NVector *vec);
void ClearNMatrix(NMatrix *mat, int nrows);
void ClearNMatrixSegment(NMatrix *mat, int off, int len);
void ClearNVectorSegment(NVector *vec, int off, int len);
void CopyPartialNSegment(int minRow, int minCol, NFloat *srcPtr, int srcCol, NFloat *dstPtr, int dstCol);
void NanySVD(NMatrix *A, NMatrix *U, NVector *d, NMatrix *Vt);
/* cz277 - l2 fix */
void AddScaledNMatrix(NMatrix *srcMat, int row, int col, NFloat scale, NMatrix *dstMat);
void AddScaledNVector(NVector *srcVec, int len, NFloat scale, NVector *dstVec);

/* cz277 - gradlim */
void ClipNMatrixVals(NMatrix* srcMat, int row, int col, NFloat upperLim, NFloat lowerLim, NMatrix *dstMat);
void ClipNVectorVals(NVector* srcVec, int len, NFloat upperLim, NFloat lowerLim, NVector *dstVec);

/* cz277 - max norm */
void CalNMatrixL2NormByRow(NMatrix *srcMat, NVector *normVec);
void CalNVectorL2Norm(NVector *srcVec, NFloat *normVal);
void DivideNMatrixByRow(NMatrix *srcMat, NVector *normVec, NMatrix *dstMat);

/* cz277 - xform */
int ClipInt(int min, int max, int val);

#ifdef __cplusplus
}
#endif

#endif  /* _HMATH_H_ */

/* ------------------------- End of HMath.h -------------------------- */

