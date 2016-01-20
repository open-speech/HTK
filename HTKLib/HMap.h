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
/*               File: HMap.c  - MAP Model Updates             */
/* ----------------------------------------------------------- */

/* !HVER!HMap:   3.5.0 [CUED 12/10/15] */


void InitMap(void);
/* 
   Initialise configuration variables for the MAP adaptation
   library module.
*/

void MAPUpdateModels(HMMSet *hset, UPDSet uflags);
/*
  Using the accumulates obtained using FB perform Gauvain
  and Lee MAP update on the model parameters specified
  by uflags. 

  Note:
  1) MAP transition updates nit supported.
  2) TIEDHS model kind MAP updates not supported
*/

/* ------------------------- End of HMap.h ------------------------- */

