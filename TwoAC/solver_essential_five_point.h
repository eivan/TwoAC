#pragma once

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

// Copyright (c) 2018 Ivan Eichhardt.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//
//
// Five point relative pose computation using Groebner basis.
// We follow the algorithm of [1] and apply some optimization hints of [2].
//
// [1] H. Stewénius, C. Engels and D. Nistér,  "Recent Developments on Direct
//     Relative Orientation",  ISPRS 2006
//
// [2] D. Nistér,  "An Efficient Solution to the Five-Point Relative Pose",
//     PAMI 2004


#include "numeric.h"
#include <vector>

namespace TwoAC
{

	/**
	* @brief Compute the nullspace of the linear constraints given by the matches.
	* @param x1 Match position in first camera
	* @param x2 Match position in second camera
	* @return Nullspace (homography) that maps x1 points to x2 points
	*/
	Mat FivePointsNullspaceBasis(const AffineCorrespondences &x);

	/**
	* Builds the polynomial constraint matrix M.
	* @param E_basis Basis essential matrix
	* @return polynomial constraint associated to the essential matrix
	*/
	Mat FivePointsPolynomialConstraints(const Mat &E_basis);

	/**
	* Build a 9 x n matrix from point matches, where each row is equivalent to the
	* equation x'T*F*x = 0 for a single correspondence pair (x', x). The domain of
	* the matrix is a 9 element vector corresponding to F. In other words, set up
	* the linear system
	*
	*   Af = 0,
	*
	* where f is the F matrix as a 9-vector rather than a 3x3 matrix (row
	* major). If the points are well conditioned and there are 8 or more, then
	* the nullspace should be rank one. If the nullspace is two dimensional,
	* then the rank 2 constraint must be enforced to identify the appropriate F
	* matrix.
	*
	* Note that this does not resize the matrix A; it is expected to have the
	* appropriate size already.
	*/
	template<typename TMatA>
	inline void EncodeEpipolarEquation(const AffineCorrespondences &x, TMatA *A) {
		for (size_t i = 0; i < x.size(); ++i) {
			auto& xx1 = x[i].first.x;
			auto& xx2 = x[i].second.x;
			A->row(i) <<
				xx2(0) * xx1(0),  // 0 represents x coords,
				xx2(0) * xx1(1),  // 1 represents y coords.
				xx2(0) * xx1(2),
				xx2(1) * xx1(0),
				xx2(1) * xx1(1),
				xx2(1) * xx1(2),
				xx2(2) * xx1(0),
				xx2(2) * xx1(1),
				xx2(2) * xx1(2);
		}
	}

	template<typename TMatA>
	inline void EncodeEpipolarEquation(const AffineCorrespondences &x, TMatA *A, const std::vector<double>* weights) {
		for (size_t i = 0; i < x.size(); ++i) {
			auto& xx1 = x[i].first.x;
			Vec3 xx2 = weights->at(i) * x[i].second.x;
			A->row(i) <<
				xx2(0) * xx1(0),  // 0 represents x coords,
				xx2(0) * xx1(1),  // 1 represents y coords.
				xx2(0) * xx1(2),
				xx2(1) * xx1(0),
				xx2(1) * xx1(1),
				xx2(1) * xx1(2),
				xx2(2) * xx1(0),
				xx2(2) * xx1(1),
				xx2(2) * xx1(2);
		}
	}

	struct FivePTSolver
	{
		typedef typename Mat3 Model;
		typedef typename AffineCorrespondences Param;

		/// The minimal number of point required for the model estimation
		enum { MINIMUM_SAMPLES = 5 };

		/// The number of models that the minimal solver could return.
		enum { MAX_MODELS = 10 };

		static bool Solve(const Param& x, std::vector<Model>* models, const std::vector<double>* weights = nullptr)
		{
			// Step 1: Nullspace Extraction.
			Eigen::Matrix<double, Eigen::Dynamic, 9> action;
			action.setZero(std::max(int(x.size()), 9), 9);  // Make A square until Eigen supports rectangular SVD.

			if (weights) {
				EncodeEpipolarEquation(x, &action, weights);
			}
			else {
				EncodeEpipolarEquation(x, &action);
			}
			Eigen::Matrix<double, 9, 4> E_basis = action.jacobiSvd(Eigen::ComputeFullV).matrixV().topRightCorner<9, 4>();

			// Step 2: Constraint Expansion.
			const Eigen::Matrix<double, 10, 20> E_constraints = FivePointsPolynomialConstraints(E_basis);

			// Step 3: Gauss-Jordan Elimination (done thanks to a LU decomposition).
			typedef Eigen::Matrix<double, 10, 10> Mat10;
			Eigen::FullPivLU<Mat10> c_lu(E_constraints.block<10, 10>(0, 0));
			const Mat10 M = c_lu.solve(E_constraints.block<10, 10>(0, 10));

			// For next steps we follow the matlab code given in Stewenius et al [1].

			// Build action matrix.

			const Mat10 & B = M.topRightCorner<10, 10>();
			Mat10 At = Mat10::Zero(10, 10);
			At.block<3, 10>(0, 0) = B.block<3, 10>(0, 0);
			At.row(3) = B.row(4);
			At.row(4) = B.row(5);
			At.row(5) = B.row(7);
			At(6, 0) = At(7, 1) = At(8, 3) = At(9, 6) = -1;

			Eigen::EigenSolver<Mat10> eigensolver(At);
			const auto& eigenvectors = eigensolver.eigenvectors();
			const auto& eigenvalues = eigensolver.eigenvalues();

			// Build essential matrices for the real solutions.
			models->reserve(10);
			for (int s = 0; s < 10; ++s) {
				// Only consider real solutions.
				if (eigenvalues(s).imag() != 0) {
					continue;
				}
				Mat3 E;
				Eigen::Map<Vec9 >(E.data()) =
					E_basis * eigenvectors.col(s).tail<4>().real();
				models->emplace_back(E.transpose());
			}

			return models->size() > 0;
		}

	};

}