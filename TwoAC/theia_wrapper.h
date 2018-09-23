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

#include "theia/solvers/estimator.h"
#include "types.h"

namespace TwoAC {

	template <typename Solver, typename ErrorModel>
	struct THEIA_Estimator_wrapper : theia::Estimator<IndexT, typename Solver::Model>
	{
		THEIA_Estimator_wrapper(const AffineCorrespondences* sample) : mSample(sample) {}

		double SampleSize() const override { return Solver::MINIMUM_SAMPLES; }

		bool EstimateModel(
			const std::vector<IndexT>& indices,
			std::vector<typename Solver::Model>* model) const {

			return Solver::Solve(SubSample(indices), model);
		}

		bool EstimateModelNonminimalWeighted(
			const std::vector<IndexT>& indices,
			std::vector<typename Solver::Model>* model,
			const std::vector<double>& weights) const {

			return Solver::Solve(SubSample(indices), model, &weights);
		}

		double Error(const IndexT& data, const typename Solver::Model& model) const {
			return ErrorModel::Error(model, mSample->operator[](data));
		}

		void GetInliers(std::vector<IndexT>& inliers, const typename Solver::Model& model, double error_threshold)
		{
			inliers.clear();
			inliers.reserve(NumSamples());
			for (int i = 0; i < errors.size(); ++i) {
				if (Error(i, model) <= error_threshold) {
					inliers.push_back(i);
				}
			}
		}

		// NOTE: virtual bool ValidModel(const Model& model) const { return true; }

		IndexT NumSamples() const { return mSample->size(); }

	private:
		const AffineCorrespondences* mSample;

		inline AffineCorrespondences SubSample(const std::vector<IndexT>& indices) const {
			AffineCorrespondences sub_sample;
			sub_sample.reserve(indices.size());
			for (const auto& i : indices) {
				sub_sample.emplace_back(mSample->operator[](i));
			}
			return std::move(sub_sample);
		}
	};
}