// Copyright (c) 2018 Ivan Eichhardt.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "regions.h"
#include <string_view>
#include <fstream>
#include <iterator>

namespace TwoAC {

	const LocalAffineFrame & TwoAC::Regions::GetLAF(IndexT i) const {
		assert(mLAFs.size() > i);
		return mLAFs[i];
	}

	size_t TwoAC::Regions::Size() const {
		return mLAFs.size();
	}

	std::unique_ptr<Regions> TwoAC::Regions::Load(std::string_view filename) {
		auto regions = std::make_unique<Regions>();

		std::ifstream fileIn(filename.data());

		if (!fileIn.is_open()) {
			throw std::runtime_error("Failed to open Regions file: " + std::string(filename));
		}

		while (!fileIn.eof() && !fileIn.bad()) {
			LocalAffineFrame LAF;
			fileIn >> LAF.x(0) >> LAF.x(1) >> LAF.M(0, 0) >> LAF.M(0, 1) >> LAF.M(1, 0) >> LAF.M(1, 1);
			regions->mLAFs.emplace_back(LAF);
		}

		if (fileIn.bad()) {
			throw std::runtime_error("Failed to load Regions from file: " + std::string(filename));
		}

		return regions;
	}

	std::unique_ptr<Matches> TwoAC::Matches::Load(std::string_view filename, bool sortMatches) {
		auto matches = std::make_unique<Matches>();

		std::ifstream fileIn(filename.data());

		if (!fileIn.is_open()) {
			throw std::runtime_error("Failed to open Matches file: " + std::string(filename));
		}

		std::copy(
			std::istream_iterator<Match>(fileIn),
			std::istream_iterator<Match>(),
			std::back_inserter(matches->mMatches));

		if (fileIn.bad()) {
			throw std::runtime_error("Failed to load Matches from file: " + std::string(filename));
		}

		// sort matches by confidence
		if (sortMatches) {
			std::sort(begin(matches->mMatches), end(matches->mMatches),
				[](auto& left, auto& right) { return std::get<2>(left) < std::get<2>(right); });
		}

		return matches;
	}

	AffineCorrespondences TwoAC::Matches::GetAffineCorrespondences(const Regions * regions_1, const Regions * regions_2, const Camera * camera_1, const Camera * camera_2) const {

		AffineCorrespondences result;
		result.reserve(mMatches.size());

		for (const auto& match : mMatches) {
			auto& laf1 = regions_1->GetLAF(std::get<0>(match));
			auto& laf2 = regions_2->GetLAF(std::get<1>(match));

			auto[x1, dx1] = camera_1->q_gradient(laf1.x);
			auto[x2, dx2] = camera_2->q_gradient(laf2.x);

			result.emplace_back(std::make_pair(
				NormalizedLocalAffineFrame({ x1, dx1 * laf1.M }),
				NormalizedLocalAffineFrame({ x2, dx2 * laf2.M })));
		}

		return std::move(result);
	}

}