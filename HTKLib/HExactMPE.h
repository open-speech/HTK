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
/*         File: HExactMPE.h  MPE implementation (exact)       */
/* ----------------------------------------------------------- */

/* !HVER!HExactMPE:   3.5.0 [CUED 12/10/15] */

/* A (rather long) routine called from HFBLat.c, relating to the 
   exact implementation of MPE.
*/
   

#define SUPPORT_EXACT_CORRECTNESS


void InitExactMPE(void); /* set configs. */


#ifdef SUPPORT_EXACT_CORRECTNESS
void DoExactCorrectness(FBLatInfo *fbInfo, Lattice *lat);   
#endif

/* ------------------- End of HExactMPE.h -------------------- */

