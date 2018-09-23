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

#include <memory>
#include "types.h"
#include "camera.h"

namespace TwoAC {

	// Output of feature extractors
	struct LocalAffineFrame {
		Vec2 x;
		Mat2 M;
	};

	// A container of LAFs, returned by a feature extractor.
	class Regions {
	public:

		static std::unique_ptr<Regions> Load(std::string_view filename);

		const LocalAffineFrame& GetLAF(IndexT i) const;

		size_t Size() const;

	private:
		std::vector<LocalAffineFrame> mLAFs;
	};

	// A container of matched pairs of LAFs.
	class Matches {
	public:

		static std::unique_ptr<Matches> Load(std::string_view filename, bool sortMatches = false);

		// Returns Affine Correspondences, transformed to camera space.
		AffineCorrespondences GetAffineCorrespondences(
			const Regions* regions_1, const Regions* regions_2,
			const Camera* camera_1, const Camera* camera_2) const;

	private:
		std::vector<Match> mMatches;
	};

}