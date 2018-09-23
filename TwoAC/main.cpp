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

#define _SILENCE_CXX17_ADAPTOR_TYPEDEFS_DEPRECATION_WARNING

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <memory>

#include "types.h"
#include "camera.h"
#include "regions.h"

#include <theia/solvers/ransac.h>
#include <theia/solvers/prosac.h>
#include "theia_wrapper.h"
#include "solver_essential_five_point.h"
#include "solver_essential_two_ac.h"

using namespace TwoAC;

int main(int argc, char* argv[])
{
	AffineCorrespondences ACs;

	// Try loading Affine Correspondences ...
	try {
		if (argc != 2) {
			throw new std::runtime_error("Invalid number of commandline arguments");
		}

		std::string base_path = argv[1];

		//std::string base_path = "data/sarok_1";
		//std::string base_path = "data/stre_f";

		std::string
			// File of LAFs extracted from the first image
			filename_image1 = base_path + "/camA.jpg",
			// File of LAFs extracted from the second image
			filename_image2 = base_path + "/camB.jpg",
			// List of matched LAFs
			filename_matches = base_path + "/pairwise_0000_0001.matches";

		std::cout << "Loading dataset '" << base_path << "'" << std::endl << std::endl;

		// Load features and camera parameters for image 1
		auto regions1 = Regions::Load(filename_image1 + ".lafs");
		auto camera1 = std::make_unique<Camera_Radial>();
		camera1->LoadParams(filename_image1 + ".intrinsics");

		// Load features and camera parameters for image 2
		auto regions2 = Regions::Load(filename_image2 + ".lafs");
		auto camera2 = std::make_unique<Camera_Radial>();
		camera2->LoadParams(filename_image2 + ".intrinsics");

		// Load matches between images
		auto matches = Matches::Load(filename_matches, true);

		// Get normalized Affine Correspondences based on matches
		ACs = matches->GetAffineCorrespondences(regions1.get(), regions2.get(), camera1.get(), camera2.get());
	}
	catch (std::runtime_error e) {
		std::cerr << e.what() << std::endl;
		std::cin.get();
		exit(EXIT_FAILURE);
	}

	// Output model (essential matrix) of robust estimation
	Mat3 E;

	// Perform robust estimation
	{
		using namespace theia;

		RansacParameters params;
		srand(time(0));
		// Theia solvers parameters
		params.rng = std::make_shared<theia::RandomNumberGenerator>(rand());
		params.error_thresh = 0.5; // angular: degrees
		params.max_iterations = 2000;
		params.min_iterations = 10;
		params.failure_probability = 0.05;

		// LO-RANSAC parameters
		params.lo_min_RANSAC_iterations = 20;
		//params.lo_treat_as_resampling = false;
		params.lo_enable = true;
		params.use_mle = true;

		// Indices of Affine Correspondences
		std::vector<IndexT> input_points(ACs.size());
		std::iota(std::begin(input_points), std::end(input_points), 0);

		// Performing tests 1, 2, ...

		// 2AC (1)
		{
			std::cout << "Running LO+ with 2AC and Prosac sampling..." << std::endl;

			using WRAPPER = TwoAC::THEIA_Estimator_wrapper<TwoACSolver, AngularError>;
			WRAPPER theia_opendmvg_wrapper(&ACs);

			Prosac<WRAPPER> method_2AC(params, theia_opendmvg_wrapper);
			method_2AC.Initialize();

			RansacSummary summary;

			// Run LO+
			auto begin = std::chrono::high_resolution_clock::now();
			method_2AC.Estimate(input_points, &E, &summary);
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - begin);

			std::cout
				<< "\tconfidence:\t" << summary.confidence << std::endl
				<< "\tinliers:\t" << summary.inliers.size() << " of " << ACs.size()
				<< " (" << (summary.inliers.size() / double(ACs.size())) * 100 << "%)" << std::endl
				<< "\tnum_iterations:\t" << summary.num_iterations << std::endl
				<< "\tlo_count:\t" << summary.lo_count << std::endl
				<< "\truntime:\t" << duration.count() << "s"
				<< std::endl;
		}

		std::cout << std::endl;

		// 2AC (2)
		{
			std::cout << "Running LO+ with 2AC..." << std::endl;

			using WRAPPER = TwoAC::THEIA_Estimator_wrapper<TwoACSolver, AngularError>;
			WRAPPER theia_opendmvg_wrapper(&ACs);

			Ransac<WRAPPER> method_2AC(params, theia_opendmvg_wrapper);
			method_2AC.Initialize();

			RansacSummary summary;

			// Run LO+
			auto begin = std::chrono::high_resolution_clock::now();
			method_2AC.Estimate(input_points, &E, &summary);
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - begin);

			std::cout
				<< "\tconfidence:\t" << summary.confidence << std::endl
				<< "\tinliers:\t" << summary.inliers.size() << " of " << ACs.size()
				<< " (" << (summary.inliers.size() / double(ACs.size())) * 100 << "%)" << std::endl
				<< "\tnum_iterations:\t" << summary.num_iterations << std::endl
				<< "\tlo_count:\t" << summary.lo_count << std::endl
				<< "\truntime:\t" << duration.count() << "s"
				<< std::endl;
		}

		// 5PT (3)
		{
			using WRAPPER = TwoAC::THEIA_Estimator_wrapper<FivePTSolver, AngularError>;
			WRAPPER theia_opendmvg_wrapper(&ACs);

			std::cout << "Running LO+ with 5PT..." << std::endl;
			Ransac<WRAPPER> method_5PT(params, theia_opendmvg_wrapper);
			method_5PT.Initialize();

			RansacSummary summary;

			// Run LO+
			auto begin = std::chrono::high_resolution_clock::now();
			method_5PT.Estimate(input_points, &E, &summary);
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - begin);

			std::cout
				<< "\tconfidence:\t" << summary.confidence << std::endl
				<< "\tinliers:\t" << summary.inliers.size() << " of " << ACs.size()
				<< " (" << (summary.inliers.size() / double(ACs.size())) * 100 << "%)" << std::endl
				<< "\tnum_iterations:\t" << summary.num_iterations << std::endl
				<< "\tlo_count:\t" << summary.lo_count << std::endl
				<< "\truntime:\t" << duration.count() << "s" << std::endl;
		}
	}

	std::cin.get();
}