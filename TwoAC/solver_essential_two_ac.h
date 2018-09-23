#pragma once
// MIT License
// 
// Copyright(c) 2018 Ivan Eichhardt
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "numeric.h"
#include "solver_essential_five_point.h"
#include <vector>

namespace TwoAC
{

	template<typename TMatA>
	inline void EncodeDifferentialEpipolarEquationTwoPatch(const AffineCorrespondences &x, TMatA *A, const size_t offset = 0) {

		for (size_t i = 0; i < x.size(); ++i) {
			auto& xx1 = x[i].first.x;
			auto& xx2 = x[i].second.x;
			auto tmp = x[i].first.dx.transpose();
			auto tmp1 = x[i].second.dx.transpose();
			Mat23 r1 = xx2(0)*tmp + tmp1.col(0) * xx1.transpose();
			Mat23 r2 = xx2(1)*tmp + tmp1.col(1) * xx1.transpose();
			Mat23 r3 = xx2(2)*tmp + tmp1.col(2) * xx1.transpose();
			A->row(offset + 2 * i + 0) << r1.row(0), r2.row(0), r3.row(0);
			A->row(offset + 2 * i + 1) << r1.row(1), r2.row(1), r3.row(1);
		}

	}

	inline Mat TwoPatchesNullspaceBasis(const AffineCorrespondences &x)
	{
		Eigen::Matrix<double, 9, 9> action;
		action.setZero(); 
		EncodeEpipolarEquation(x, &action);
		EncodeDifferentialEpipolarEquationTwoPatch(x, &action, x.size());
		return action.jacobiSvd(Eigen::ComputeFullV).matrixV().topRightCorner<9, 4>();
	}

	struct TwoACSolver
	{
		typedef typename Mat3 Model;
		typedef typename AffineCorrespondences Param;

		/// The minimal number of point required for the model estimation
		enum { MINIMUM_SAMPLES = 2 };

		/// The number of models that the minimal solver could return.
		enum { MAX_MODELS = 10 };

		static bool Solve(const Param& x, std::vector<Model>* models)
		{
			if (x.size() > MINIMUM_SAMPLES) {
				return FivePTSolver::Solve(x, models);
			}

			// Step 1: Nullspace Extraction.
			Eigen::Matrix<double, 9, 4> E_basis = TwoPatchesNullspaceBasis(x);

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

		static bool Solve(const Param& x, std::vector<Model>* models, const std::vector<double>* weights)
		{
			return FivePTSolver::Solve(x, models, weights);
		}
	};
}

