// Copyright (C) 2013 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

// TODOs for Ivan:
// - feltételes (#pragma omp parallel for) .. hogy mennyit kellene iterálni..
// - csekkolni hogy std::vector<double> residuals = ... mennyire gyors

#ifndef THEIA_SOLVERS_SAMPLE_CONSENSUS_ESTIMATOR_H_
#define THEIA_SOLVERS_SAMPLE_CONSENSUS_ESTIMATOR_H_

#include <glog/logging.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "theia/solvers/estimator.h"
#include "theia/solvers/inlier_support.h"
#include "theia/solvers/mle_quality_measurement.h"
#include "theia/solvers/quality_measurement.h"
#include "theia/solvers/sampler.h"
namespace theia {

	// Helper struct to hold parameters to various RANSAC schemes. error_thresh is
	// the threshold for consider data points to be inliers to a model. This is the
	// only variable that must be explitly set, and the rest can be used with the
	// default values unless other values are desired.
	struct RansacParameters {
		RansacParameters()
			: error_thresh(-1),
			failure_probability(0.01),
			min_inlier_ratio(0),
			min_iterations(100),
			max_iterations(std::numeric_limits<int>::max()),
			use_mle(false),
			use_Tdd_test(false),
			lo_enable(false), 
			lo_treat_as_resampling(false), // Added by Ivan, Experimental.
			lo_model_change_threshold(std::numeric_limits<double>::epsilon()),
			lo_min_RANSAC_iterations(20), lo_min_inliers(7),
			lo_min_sample_number(12), lo_inner_iterations(10),
			lo_threshold_multiplier(4*std::sqrt(2.0)), lo_max_LSq_iterations(4), lo_max_LSq_samples(50) {} // TODO

		// The random number generator used to compute random number during
		// RANSAC. This may be controlled by the caller for debugging purposes.
		std::shared_ptr<RandomNumberGenerator> rng;

		// Error threshold to determin inliers for RANSAC (e.g., squared reprojection
		// error). This is what will be used by the estimator to determine inliers.
		double error_thresh;

		// The failure probability of RANSAC. Set to 0.01 means that RANSAC has a 1%
		// chance of missing the correct pose.
		double failure_probability;

		// The minimal assumed inlier ratio, i.e., it is assumed that the given set
		// of correspondences has an inlier ratio of at least min_inlier_ratio.
		// This is required to limit the number of RANSAC iteratios.
		double min_inlier_ratio;

		// The minimum number of iterations required before exiting.
		int min_iterations;

		// Another way to specify the maximal number of RANSAC iterations. In effect,
		// the maximal number of iterations is set to min(max_ransac_iterations, T),
		// where T is the number of iterations corresponding to min_inlier_ratio.
		// This variable is useful if RANSAC is to be applied iteratively, i.e.,
		// first applying RANSAC with an min_inlier_ratio of x, then with one
		// of x-y and so on, and we want to avoid repeating RANSAC iterations.
		// However, the preferable way to limit the number of RANSAC iterations is
		// to set min_inlier_ratio and leave max_ransac_iterations to its default
		// value.
		// Per default, this variable is set to std::numeric_limits<int>::max().
		int max_iterations;

		// Instead of the standard inlier count, use the Maximum Likelihood Estimate
		// (MLE) to determine the best solution. Inliers are weighted by their error
		// and outliers count as a constant penalty.
		bool use_mle;

		// Whether to use the T_{d,d}, with d=1, test proposed in
		// Chum, O. and Matas, J.: Randomized RANSAC and T(d,d) test, BMVC 2002.
		// After computing the pose, RANSAC selects one match at random and evaluates
		// all poses. If the point is an outlier to one pose, the corresponding pose
		// is rejected. Notice that if the pose solver returns multiple poses, then
		// at most one pose is correct. If the selected match is correct, then only
		// the correct pose will pass the test. Per default, the test is disabled.
		//
		// NOTE: Not currently implemented!
		bool use_Tdd_test;

		// LO-RANSAC stuff
		bool lo_enable;
		bool lo_treat_as_resampling;
		double lo_model_change_threshold;
		int lo_min_RANSAC_iterations;
		int lo_min_inliers;
		size_t lo_min_sample_number;
		int lo_inner_iterations;
		double lo_threshold_multiplier;
		int lo_max_LSq_iterations;
		int lo_max_LSq_samples;
	};

	// A struct to hold useful outputs of Ransac-like methods.
	struct RansacSummary {
		// Contains the indices of all inliers.
		std::vector<int> inliers;

		// The number of iterations performed before stopping RANSAC.
		int num_iterations;

		int lo_count;

		// The confidence in the solution.
		double confidence;
	};

	template <class ModelEstimator> class SampleConsensusEstimator {
	public:
		typedef typename ModelEstimator::Datum Datum;
		typedef typename ModelEstimator::Model Model;

		SampleConsensusEstimator(const RansacParameters& ransac_params,
			const ModelEstimator& estimator);

		virtual bool Initialize() { return true; }

		virtual ~SampleConsensusEstimator() {}

		// Computes the best-fitting model using RANSAC. Returns false if RANSAC
		// calculation fails and true (with the best_model output) if successful.
		// Params:
		//   data: the set from which to sample
		//   estimator: The estimator used to estimate the model based on the Datum
		//     and Model type
		//   best_model: The output parameter that will be filled with the best model
		//     estimated from RANSAC
		virtual bool Estimate(const std::vector<Datum>& data,
			Model* best_model,
			RansacSummary* summary);

		struct LocalSummary
		{
			LocalSummary(const double& _best_cost = std::numeric_limits<double>::max(),
				const size_t& _best_inlier_number = std::numeric_limits<size_t>::min(),
				const int& _max_iterations = std::numeric_limits<int>::max())
				: found_better_model(false), best_cost(_best_cost), best_inlier_number(_best_inlier_number), max_iterations(_max_iterations) {}

			void Apply(double& _best_cost, size_t& _best_inlier_number, int& _max_iterations) {
				_best_cost = best_cost;
				_best_inlier_number = best_inlier_number;
				_max_iterations = max_iterations;
			}

			bool found_better_model;
			double best_cost;
			size_t best_inlier_number;
			int max_iterations;
		};

		virtual bool ChooseBestModel(const std::vector<Datum>& data,
			const std::vector<Model>& models,
			Model* best_model,
			RansacSummary* summary,
			LocalSummary& local_summary, int sample_size_used);

		virtual bool ChooseBestModel(const std::vector<Datum>& data,
			const std::vector<Model>& models,
			Model* best_model,
			RansacSummary* summary);

		virtual bool LocalOptimizationStep(const std::vector<Datum>& data,
			const Model* best_minimal_model,
			Model* best_LO_model,
			RansacSummary* summary,
			LocalSummary & local_summary);

		virtual bool IterativeLeastSquares(const std::vector<Datum>& data,
			Model* best_model,
			RansacSummary* summary,
			LocalSummary & local_summary);

	protected:
		// This method is called from derived classes to set up the sampling scheme
		// and the method for computing inliers. It must be called by derived classes
		// unless they override the Estimate(...) method. The method for computing
		// inliers (standar inlier support or MLE) is determined by the ransac params.
		//
		// sampler: The class that instantiates the sampling strategy for this
		//   particular type of sampling consensus.
		bool Initialize(Sampler<Datum>* sampler);

		// Computes the maximum number of iterations required to ensure the inlier
		// ratio is the best with a probability corresponding to log_failure_prob.
		int ComputeMaxIterations(const double min_sample_size,
			const double inlier_ratio,
			const double log_failure_prob) const;

		// The sampling strategy.
		std::unique_ptr<Sampler<Datum> > sampler_;

		// The quality metric for the estimated model and data.
		std::unique_ptr<QualityMeasurement> quality_measurement_;

		//std::unique_ptr<QualityMeasurement> weighting_;

		// Ransac parameters (see above struct).
		const RansacParameters& ransac_params_;

		// Estimator to use for generating models.
		const ModelEstimator& estimator_;
	};

	// --------------------------- Implementation --------------------------------//

	template <class ModelEstimator>
	SampleConsensusEstimator<ModelEstimator>::SampleConsensusEstimator(
		const RansacParameters& ransac_params, const ModelEstimator& estimator)
		: ransac_params_(ransac_params), estimator_(estimator) {
		CHECK_GT(ransac_params.error_thresh, 0)
			<< "Error threshold must be set to greater than zero";
		CHECK_LE(ransac_params.min_inlier_ratio, 1.0);
		CHECK_GE(ransac_params.min_inlier_ratio, 0.0);
		CHECK_LT(ransac_params.failure_probability, 1.0);
		CHECK_GT(ransac_params.failure_probability, 0.0);
		CHECK_GE(ransac_params.max_iterations, ransac_params.min_iterations);
	}

	template <class ModelEstimator>
	bool SampleConsensusEstimator<ModelEstimator>::Initialize(
		Sampler<Datum>* sampler) {
		CHECK_NOTNULL(sampler);
		sampler_.reset(sampler);

		if (!sampler_->Initialize()) {
			return false;
		}

		if (ransac_params_.use_mle) {
			quality_measurement_.reset(
				new MLEQualityMeasurement(ransac_params_.error_thresh));
		}
		else {
			quality_measurement_.reset(
				new InlierSupport(ransac_params_.error_thresh));
		}
		return quality_measurement_->Initialize();
	}

	template <class ModelEstimator>
	int SampleConsensusEstimator<ModelEstimator>::ComputeMaxIterations(
		const double min_sample_size,
		const double inlier_ratio,
		const double log_failure_prob) const {
		CHECK_GT(inlier_ratio, 0.0);
		if (inlier_ratio == 1.0) {
			return ransac_params_.min_iterations;
		}

		// If we use the T_{1,1} test, we have to adapt the number of samples
		// that needs to be generated accordingly since we use another
		// match for verification and a correct match is selected with probability
		// inlier_ratio.
		const double num_samples =
			ransac_params_.use_Tdd_test ? min_sample_size + 1 : min_sample_size;

		const double log_prob = std::log(1.0 - std::pow(inlier_ratio, num_samples))
			- std::numeric_limits<double>::epsilon();

		// NOTE: For very low inlier ratios the number of iterations can actually
		// exceed the maximum value for an int. We need to keep this variable as a
		// double until we do the check below against the minimum and maximum number
		// of iterations in the parameter settings.
		const double num_iterations = log_failure_prob / log_prob;

		return std::max(static_cast<double>(ransac_params_.min_iterations),
			std::min(num_iterations,
				static_cast<double>(ransac_params_.max_iterations)));
	}

	template <class ModelEstimator>
	bool SampleConsensusEstimator<ModelEstimator>::Estimate(
		const std::vector<Datum>& data,
		Model* best_model,
		RansacSummary* summary) {
		CHECK_GT(data.size(), 0)
			<< "Cannot perform estimation with 0 data measurements!";
		CHECK_NOTNULL(sampler_.get());
		CHECK_NOTNULL(quality_measurement_.get());
		CHECK_NOTNULL(summary);
		summary->inliers.clear();
		CHECK_NOTNULL(best_model);

		summary->lo_count = 0;

		const double log_failure_prob = std::log(ransac_params_.failure_probability);

		double best_cost = std::numeric_limits<double>::max();

		Model best_minimal_model = *best_model;
		LocalSummary best_minimal_summary(best_cost, 0, ransac_params_.max_iterations);
		int& max_iterations = best_minimal_summary.max_iterations;

		// Set the max iterations if the inlier ratio is set.
		if (ransac_params_.min_inlier_ratio > 0) {
			max_iterations = std::min(
				ComputeMaxIterations(estimator_.SampleSize(),
					ransac_params_.min_inlier_ratio,
					log_failure_prob),
				ransac_params_.max_iterations);
		}

		LocalSummary best_lo_summary(best_cost, 0, max_iterations);

		for (summary->num_iterations = 0;
			summary->num_iterations < max_iterations;
			summary->num_iterations++) {

			// Sample subset. Proceed if successfully sampled.
			std::vector<Datum> data_subset;
			if (!sampler_->Sample(data, &data_subset)) {
				continue;
			}

			// Estimate model from subset. Skip to next iteration if the model fails to
			// estimate.
			std::vector<Model> temp_models;
			if (!estimator_.EstimateModel(data_subset, &temp_models)) {
				continue;
			}

			if(!ChooseBestModel(data, temp_models, &best_minimal_model, summary, best_minimal_summary, estimator_.SampleSize()))
				continue;

			if (ransac_params_.lo_enable)
			{
				// Decide whether a local optimization is needed or not
				if (summary->num_iterations > ransac_params_.lo_min_RANSAC_iterations && 
					best_minimal_summary.best_inlier_number > ransac_params_.lo_min_inliers)
				{
					if (best_minimal_summary.best_cost < best_lo_summary.best_cost) {
						best_lo_summary = best_minimal_summary;
						*best_model = best_minimal_model;

						//std::cout << "LO: No better!" << std::endl;
					}

					//LocalOptimizationStep(data, &best_minimal_model, best_model, summary, best_lo_summary);
					if (LocalOptimizationStep(data, &best_minimal_model, best_model, summary, best_lo_summary))
						max_iterations = std::min(max_iterations, best_lo_summary.max_iterations);
					++(summary->lo_count);
				}
			}
		}

		if (ransac_params_.lo_enable) {
			// Apply a final local optimization if it hasn't been applied yet
			if (summary->lo_count == 0)
			{
				best_lo_summary = best_minimal_summary;
				*best_model = best_minimal_model;

				LocalOptimizationStep(data, &best_minimal_model, best_model, summary, best_lo_summary);
				++(summary->lo_count);
			}

			if (best_minimal_summary.best_cost < best_lo_summary.best_cost) {
				best_lo_summary = best_minimal_summary;
				*best_model = best_minimal_model;

				std::cout << "LO: No better!" << std::endl;
			}
			else {
				/*std::cout << "LO: Better! inl("
				<< best_lo_summary.best_inlier_number << " > " << best_minimal_summary.best_inlier_number << "); cost("
				<< best_lo_summary.best_cost << " < " << best_minimal_summary.best_cost << ")" << std::endl;*/
			}
		} 
		else {
			best_lo_summary = best_minimal_summary;
			*best_model = best_minimal_model;
		}

		// Compute the final inliers for the best model.
		const std::vector<double> best_residuals =
			estimator_.Residuals(data, *best_model);
		quality_measurement_->ComputeCost(best_residuals, &summary->inliers);

		const double inlier_ratio =
			static_cast<double>(summary->inliers.size()) / data.size();
		summary->confidence =
			1.0 - std::pow(1.0 - std::pow(inlier_ratio, estimator_.SampleSize()),
				summary->num_iterations);

		return true;
	}

	template<class ModelEstimator>
	inline bool theia::SampleConsensusEstimator<ModelEstimator>::ChooseBestModel(
		const std::vector<Datum>& data,
		const std::vector<Model>& models,
		Model* best_model,
		RansacSummary * summary,
		LocalSummary & local_summary, int sample_size_used)
	{
		const double log_failure_prob = std::log(ransac_params_.failure_probability);

		local_summary.found_better_model = false;

		// Test all models
		for (const Model& temp_model : models) {

			// Calculate residuals from estimated model.
			const std::vector<double> residuals =
				estimator_.Residuals(data, temp_model);

			// Determine cost of the generated model.
			std::vector<int> inlier_indices;
			const double sample_cost =
				quality_measurement_->ComputeCost(residuals, &inlier_indices);

			// Update best model if error is the best we have seen.
			if (sample_cost < local_summary.best_cost) {
				*best_model = temp_model;
				local_summary.best_cost = sample_cost;
				local_summary.best_inlier_number = inlier_indices.size();

				local_summary.found_better_model = true;

				const double inlier_ratio = static_cast<double>(inlier_indices.size()) / data.size();

				if (inlier_ratio <
					estimator_.SampleSize() / static_cast<double>(data.size())) {
					continue;
				}

				if (sample_size_used > 0) {
					// A better cost does not guarantee a higher inlier ratio (i.e, the MLE
					// case) so we only update the max iterations if the number decreases.
					local_summary.max_iterations = std::min(ComputeMaxIterations(
						ransac_params_.lo_treat_as_resampling ? sample_size_used : estimator_.SampleSize(),
						inlier_ratio,
						log_failure_prob),
						local_summary.max_iterations);
				}

				VLOG(3) << "Inlier ratio = " << inlier_ratio
				<< " and max number of iterations = " << local_summary.max_iterations;

				/*std::cout << "Inlier ratio = " << inlier_ratio
					<< " and max number of iterations = " << local_summary.max_iterations << std::endl;*/
			}
		}

		return local_summary.found_better_model;
	}

	template<class ModelEstimator>
	inline bool theia::SampleConsensusEstimator<ModelEstimator>::ChooseBestModel(
		const std::vector<Datum>& data,
		const std::vector<Model>& models,
		Model* best_model,
		RansacSummary * summary)
	{
		auto best_cost = std::numeric_limits<double>::max();

		// Test all models
		//for (const Model& temp_model : models) {
//#pragma omp parallel for
		for(int i = 0; i<models.size(); ++i){
			const auto& temp_model = models[i];

			// Calculate residuals from estimated model.
			const std::vector<double> residuals =
				estimator_.Residuals(data, temp_model);

			// Determine cost of the generated model.
			std::vector<int> inlier_indices;
			const double sample_cost =
				quality_measurement_->ComputeCost(residuals, &inlier_indices);
			
//#pragma omp critical
			{
				// Update best model if error is the best we have seen.
				if (sample_cost < best_cost) {
					*best_model = temp_model;
					best_cost = sample_cost;
				}
			}
		}

		return best_cost < std::numeric_limits<double>::max();
	}

	template <class ModelEstimator>
	bool SampleConsensusEstimator<ModelEstimator>::LocalOptimizationStep(
		const std::vector<Datum>& data,
		const Model * best_minimal_model,
		Model * best_LO_model,
		RansacSummary * summary, 
		LocalSummary & local_summary)
	{
		CHECK_GT(data.size(), 0) << "Cannot perform estimation with 0 data measurements!";
		CHECK_NOTNULL(sampler_.get());
		CHECK_NOTNULL(quality_measurement_.get());
		CHECK_NOTNULL(summary);
		CHECK_NOTNULL(best_minimal_model);
		CHECK_NOTNULL(best_LO_model);

		std::vector<Datum> I_base;
		{

			std::vector<Datum> data_subset;
			{
				/// get inlier set for increased theta
				// Calculate residuals from estimated model.
				const std::vector<double> residuals = estimator_.Residuals(data, *best_minimal_model);

				/// increase theta
				const double old_threshold = quality_measurement_->Threshold();
				quality_measurement_->Threshold() *= ransac_params_.lo_threshold_multiplier;

				// Determine cost of the generated model.
				std::vector<int> inlier_indices;
				quality_measurement_->ComputeCost(residuals, &inlier_indices);

				// reset theta
				quality_measurement_->Threshold() = old_threshold;

				// Sample.
				data_subset.resize(inlier_indices.size());
				for (int i = 0; i < inlier_indices.size(); ++i)
					data_subset[i] = data[inlier_indices[i]];
			}

			/// estimate model on new inlier set
			// TODO: [future] could that be replaced by random sampled input??
			std::vector<Model> temp_models;
			if (data_subset.size() == 0 || !estimator_.EstimateModelNonminimal(data_subset, &temp_models))
				return false;

			Model increased_theta_model;
			if (!ChooseBestModel(data, temp_models, &increased_theta_model, summary))
				return false; // whatever empy localsummary

			// get base inlier set using original theta on the increased_theta_model
			{
				// Calculate residuals from estimated model.
				const std::vector<double> residuals = estimator_.Residuals(data, increased_theta_model);

				// Determine cost of the generated model.
				std::vector<int> inlier_indices;
				quality_measurement_->ComputeCost(residuals, &inlier_indices);

				// Sample.
				I_base.resize(inlier_indices.size());
				for (int i = 0; i < inlier_indices.size(); ++i)
					I_base[i] = data[inlier_indices[i]];
			}
		}

		const size_t sample_number = std::min<size_t>(ransac_params_.lo_min_sample_number, I_base.size() / 2);
		
		if (sample_number <= estimator_.SampleSize())
			return false;

		RandomSampler<Datum> random_sampler(this->ransac_params_.rng, sample_number);

		//std::cout << "  I_base size: " << I_base.size() << std::endl;

		for (int local_num_iterations = 0; local_num_iterations < ransac_params_.lo_inner_iterations; ++local_num_iterations) {

			// sample from I_base
			std::vector<Datum> data_subset;
			if (!random_sampler.Sample(I_base, &data_subset)) {
				continue;
			}

			// estimate model from data_subset
			{
				std::vector<Model> temp_models;
				if (!estimator_.EstimateModelNonminimal(data_subset, &temp_models)) {
					continue;
				}

				ChooseBestModel(data, temp_models, best_LO_model, summary, local_summary, data_subset.size());

				//if (!local_summary.found_better_model) continue; // TODO: (nem szerepel az eredeti algoritmusban)
			}

			// estimate model using iterative LsQ
			if (!IterativeLeastSquares(data, best_LO_model, summary, local_summary))
				continue;

			//std::cout << "  LO step: " << local_num_iterations << std::endl;
		}

		return true;
	}


	template<class ModelEstimator>
	inline bool SampleConsensusEstimator<ModelEstimator>::IterativeLeastSquares(
		const std::vector<Datum>& data,
		Model * best_model,
		RansacSummary * summary,
		LocalSummary & local_summary)
	{
		const double old_threshold = quality_measurement_->Threshold();

		/// 1

		std::vector<Datum> data_subset;
		{
			const std::vector<double> residuals = estimator_.Residuals(data, *best_model);

			std::vector<int> inlier_indices;
			quality_measurement_->ComputeCost(residuals, &inlier_indices);

			data_subset.resize(inlier_indices.size());
			for (int i = 0; i < inlier_indices.size(); ++i)
				data_subset[i] = data[inlier_indices[i]];
		}

		std::vector<Model> temp_models;
		if (!estimator_.EstimateModelNonminimal(data_subset, &temp_models)) 
			return false;

		ChooseBestModel(data, temp_models, best_model, summary, local_summary, data_subset.size());

		quality_measurement_->Threshold() *= ransac_params_.lo_threshold_multiplier;
		const double delta_theta = (quality_measurement_->Threshold() - old_threshold) / ransac_params_.lo_max_LSq_iterations;

		// LO'
		RandomSampler<int> random_sampler(this->ransac_params_.rng, ransac_params_.lo_max_LSq_samples);

		Model best_LSq_model = *best_model;
		Model prev_LSq_model = best_LSq_model;

		for (int i = 0;
			i < ransac_params_.lo_max_LSq_iterations;
			++i, quality_measurement_->Threshold() -= delta_theta) /// 7
		{
			/// 4
			std::vector<Datum> data_subset;
			
			const std::vector<double> residuals = estimator_.Residuals(data, best_LSq_model);

			std::vector<int> inlier_indices;
			quality_measurement_->ComputeCost(residuals, &inlier_indices);

			if (inlier_indices.size() < estimator_.SampleSize())
				return false;

			// LO'
			if(ransac_params_.lo_max_LSq_samples > 0 && inlier_indices.size() > ransac_params_.lo_max_LSq_samples)
			{
				std::vector<int> tmp;
				random_sampler.Sample(inlier_indices, &tmp);
				inlier_indices = tmp;
			}

			data_subset.resize(inlier_indices.size());
			for (int i = 0; i < inlier_indices.size(); ++i)
				data_subset[i] = data[inlier_indices[i]];
			
			/// 5 compute weights
			std::vector<double> weights(inlier_indices.size());
			bool found_too_small_residual = false;
			for (int i = 0; i < inlier_indices.size(); ++i) {
				const auto& w = residuals[inlier_indices[i]];
				// TODO: smoothed weight???
				weights[i] = 1 / (w + 0.000001);
				/*if (w > std::numeric_limits<double>::epsilon())
					weights[i] = 1 / w; // TODO: more robust weighting funciton
				else {
					found_too_small_residual = true;
					break;
				}*/
			}
			if (found_too_small_residual)
				continue; // TODO: could be break

			/// 6
			std::vector<Model> temp_models;
			if (!estimator_.EstimateModelNonminimalWeighted(data_subset, &temp_models, weights)) {
				continue;
			}

			// Choose best model.
			// NOTE that with this construct, if no better model is found than the one described by local_summary,
			// best_model won't be replaced!
			{
				//const auto tmp = quality_measurement_->Threshold();
				//quality_measurement_->Threshold() = old_threshold;
				//ChooseBestModel(data, temp_models, best_model, summary, local_summary);
				ChooseBestModel(data, temp_models, &best_LSq_model, summary);
				//quality_measurement_->Threshold() = tmp;
			}

			{
				if (*best_model == best_LSq_model)
					continue;

				const std::vector<double> residuals = estimator_.Residuals(data, best_LSq_model);

				const auto tmp = quality_measurement_->Threshold();
				quality_measurement_->Threshold() = old_threshold;
				const double sample_cost =
					quality_measurement_->ComputeCost(residuals, &inlier_indices);
				quality_measurement_->Threshold() = tmp;

				if (sample_cost < local_summary.best_cost) {
					*best_model = best_LSq_model;
					local_summary.best_cost = sample_cost;
					local_summary.best_inlier_number = inlier_indices.size();

					local_summary.found_better_model = true;

					const double inlier_ratio = static_cast<double>(inlier_indices.size()) / data.size();

					if (inlier_ratio <
						estimator_.SampleSize() / static_cast<double>(data.size())) {
						continue;
					}

					const double log_failure_prob = std::log(ransac_params_.failure_probability);
					// A better cost does not guarantee a higher inlier ratio (i.e, the MLE
					// case) so we only update the max iterations if the number decreases.
					local_summary.max_iterations = std::min(ComputeMaxIterations(
						ransac_params_.lo_treat_as_resampling ? data_subset.size() : estimator_.SampleSize(), // TODO: Ivan replaced maxiter statistics
						inlier_ratio,
						log_failure_prob),
						local_summary.max_iterations);

					if (i && (prev_LSq_model - best_LSq_model).squaredNorm() < ransac_params_.lo_model_change_threshold)
						break;
				}
			}

			//std::cout << "    LSq step: " << i << std::endl;
			// TODO once:
			prev_LSq_model = best_LSq_model;
		}

		quality_measurement_->Threshold() = old_threshold;

		return true;
	}

}  // namespace theia

#endif  // THEIA_SOLVERS_SAMPLE_CONSENSUS_ESTIMATOR_H_
