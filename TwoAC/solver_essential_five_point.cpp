
// Copyright (c) 2010 libmv authors.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

// Copyright (c) 2012, 2013 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "solver_essential_five_point.h"
#include <iostream>
namespace TwoAC {

Mat FivePointsNullspaceBasis(const AffineCorrespondences &x) {
  Eigen::Matrix<double,9, 9> A;
  A.setZero();  // Make A square until Eigen supports rectangular SVD.
  EncodeEpipolarEquation(x, &A);
  Eigen::JacobiSVD<Eigen::Matrix<double,9, 9> > svd(A,Eigen::ComputeFullV);
  return svd.matrixV().topRightCorner<9,4>();
}

// In the following code, polynomials are expressed as vectors containing
// their coeficients in the basis of monomials:
//
//  [xxx xxy xyy yyy xxz xyz yyz xzz yzz zzz xx xy yy xz yz zz x y z 1]
//
// Note that there is an error in Stewenius' paper.  In equation (9) they
// propose to use the basis:
//
//  [xxx xxy xxz xyy xyz xzz yyy yyz yzz zzz xx xy xz yy yz zz x y z 1]
//
// But this is not the basis used in the rest of the paper, neither in
// the code they provide.  I (pau) have spend 4 hours debugging and
// reverse engineering their code to find the problem. :(
enum
{
	coef_xxx,
	coef_xxy,
	coef_xyy,
	coef_yyy,
	coef_xxz,
	coef_xyz,
	coef_yyz,
	coef_xzz,
	coef_yzz,
	coef_zzz,
	coef_xx,
	coef_xy,
	coef_yy,
	coef_xz,
	coef_yz,
	coef_zz,
	coef_x,
	coef_y,
	coef_z,
	coef_1
};

/**
* @brief Multiply two polynomials of degree 1.
* @param a Polynomial a1 + a2 x + a3 y + a4 z
* @param b Polynomial b1 + b2 x + b3 y + b4 z
* @return Product of a and b :
* res = a1 b1 +
(a1b2 + b1a2) x +
(a1b3 + b1a3) y +
(a1b4 + b1a4) z +
(a2b3 + b2a3) xy +
(a2b4 + b2a4) xz +
(a3b4 + b3a4) yz +
a2b2 x^2 +
a3b3 y^2 +
a4b4 z^2
* @note Ordering is defined as follow :
* [xxx xxy xyy yyy xxz xyz yyz xzz yzz zzz xx xy yy xz yz zz x y z 1]
*/
Vec o1(const Vec &a, const Vec &b) {
  Vec res = Vec::Zero(20);

  res(coef_xx) = a(coef_x) * b(coef_x);
  res(coef_xy) = a(coef_x) * b(coef_y)
               + a(coef_y) * b(coef_x);
  res(coef_xz) = a(coef_x) * b(coef_z)
               + a(coef_z) * b(coef_x);
  res(coef_yy) = a(coef_y) * b(coef_y);
  res(coef_yz) = a(coef_y) * b(coef_z)
               + a(coef_z) * b(coef_y);
  res(coef_zz) = a(coef_z) * b(coef_z);
  res(coef_x)  = a(coef_x) * b(coef_1)
               + a(coef_1) * b(coef_x);
  res(coef_y)  = a(coef_y) * b(coef_1)
               + a(coef_1) * b(coef_y);
  res(coef_z)  = a(coef_z) * b(coef_1)
               + a(coef_1) * b(coef_z);
  res(coef_1)  = a(coef_1) * b(coef_1);

  return res;
}

/**
* @brief Multiply two polynomials of degree 2
* @param a Polynomial a1 + a2 x + a3 y + a4 z + a5 x^2 + a6 y^2 + a7 z^2
* @param b Polynomial b1 + b2 x + b3 y + b4 z + b5 x^2 + b6 y^2 + b7 z^2
* Product of a and b
* @note Ordering is defined as follow :
* [xxx xxy xyy yyy xxz xyz yyz xzz yzz zzz xx xy yy xz yz zz x y z 1]
*/
Vec o2(const Vec &a, const Vec &b) {
  Vec res(20);

  res(coef_xxx) = a(coef_xx) * b(coef_x);
  res(coef_xxy) = a(coef_xx) * b(coef_y)
                + a(coef_xy) * b(coef_x);
  res(coef_xxz) = a(coef_xx) * b(coef_z)
                + a(coef_xz) * b(coef_x);
  res(coef_xyy) = a(coef_xy) * b(coef_y)
                + a(coef_yy) * b(coef_x);
  res(coef_xyz) = a(coef_xy) * b(coef_z)
                + a(coef_yz) * b(coef_x)
                + a(coef_xz) * b(coef_y);
  res(coef_xzz) = a(coef_xz) * b(coef_z)
                + a(coef_zz) * b(coef_x);
  res(coef_yyy) = a(coef_yy) * b(coef_y);
  res(coef_yyz) = a(coef_yy) * b(coef_z)
                + a(coef_yz) * b(coef_y);
  res(coef_yzz) = a(coef_yz) * b(coef_z)
                + a(coef_zz) * b(coef_y);
  res(coef_zzz) = a(coef_zz) * b(coef_z);
  res(coef_xx)  = a(coef_xx) * b(coef_1)
                + a(coef_x)  * b(coef_x);
  res(coef_xy)  = a(coef_xy) * b(coef_1)
                + a(coef_x)  * b(coef_y)
                + a(coef_y)  * b(coef_x);
  res(coef_xz)  = a(coef_xz) * b(coef_1)
                + a(coef_x)  * b(coef_z)
                + a(coef_z)  * b(coef_x);
  res(coef_yy)  = a(coef_yy) * b(coef_1)
                + a(coef_y)  * b(coef_y);
  res(coef_yz)  = a(coef_yz) * b(coef_1)
                + a(coef_y)  * b(coef_z)
                + a(coef_z)  * b(coef_y);
  res(coef_zz)  = a(coef_zz) * b(coef_1)
                + a(coef_z)  * b(coef_z);
  res(coef_x)   = a(coef_x)  * b(coef_1)
                + a(coef_1)  * b(coef_x);
  res(coef_y)   = a(coef_y)  * b(coef_1)
                + a(coef_1)  * b(coef_y);
  res(coef_z)   = a(coef_z)  * b(coef_1)
                + a(coef_1)  * b(coef_z);
  res(coef_1)   = a(coef_1)  * b(coef_1);

  return res;
}

Mat FivePointsPolynomialConstraints(const Mat &E_basis) {
  // Build the polynomial form of E (equation (8) in Stewenius et al. [1])
  Vec E[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      E[i][j] = Vec::Zero(20);
      E[i][j](coef_x) = E_basis(3 * i + j, 0);
      E[i][j](coef_y) = E_basis(3 * i + j, 1);
      E[i][j](coef_z) = E_basis(3 * i + j, 2);
      E[i][j](coef_1) = E_basis(3 * i + j, 3);
    }
  }

  // The constraint matrix.
  Mat M(10, 20);
  int mrow = 0;

  // Determinant constraint det(E) = 0; equation (19) of Nister [2].
  M.row(mrow++) = o2(o1(E[0][1], E[1][2]) - o1(E[0][2], E[1][1]), E[2][0]) +
                  o2(o1(E[0][2], E[1][0]) - o1(E[0][0], E[1][2]), E[2][1]) +
                  o2(o1(E[0][0], E[1][1]) - o1(E[0][1], E[1][0]), E[2][2]);

  // Cubic singular values constraint.
  // Equation (20).
  Vec EET[3][3];
  for (int i = 0; i < 3; ++i) {    // Since EET is symmetric, we only compute
    for (int j = 0; j < 3; ++j) {  // its upper triangular part.
      if (i <= j) {
        EET[i][j] = o1(E[i][0], E[j][0])
                  + o1(E[i][1], E[j][1])
                  + o1(E[i][2], E[j][2]);
      } else {
        EET[i][j] = EET[j][i];
      }
    }
  }

  // Equation (21).
  Vec (&L)[3][3] = EET;
  Vec trace  = 0.5 * (EET[0][0] + EET[1][1] + EET[2][2]);
  for (int i = 0; i < 3; ++i) {
    L[i][i] -= trace;
  }

  // Equation (23).
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Vec LEij = o2(L[i][0], E[0][j])
               + o2(L[i][1], E[1][j])
               + o2(L[i][2], E[2][j]);
      M.row(mrow++) = LEij;
    }
  }

  return M;
}

} // namespace openMVG

