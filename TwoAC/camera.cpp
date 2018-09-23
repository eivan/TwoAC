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

#include "camera.h"
#include <string_view>
#include <fstream>
#include <iterator>

TwoAC::Camera_Radial::Camera_Radial() : Camera(), mK(Mat3::Identity()) {}

TwoAC::Vec2 TwoAC::Camera_Radial::p(const Vec3 & X) const {
	return p_radial(X);
}

std::pair<TwoAC::Vec2, TwoAC::Mat23> TwoAC::Camera_Radial::p_gradient(const Vec3 & X) const {
	return p_radial_gradient(X);
}

TwoAC::Vec3 TwoAC::Camera_Radial::q(const Vec2 & x) const {
	return q_radial(x);
}

std::pair<TwoAC::Vec3, TwoAC::Mat32> TwoAC::Camera_Radial::q_gradient(const Vec2 & x) const {
	return q_radial_gradient(x);
}

void TwoAC::Camera_Radial::SetParams(const std::vector<double>& params) {
	assert(params.size() == 7);
	fx() = params[0]; fy() = params[1];
	cx() = params[2]; cy() = params[3];
	for (int i = 0; i < 3; ++i) {
		mDistortionParameters(i) = params[4 + i];
	}
}

std::vector<double> TwoAC::Camera_Radial::GetParams() const {
	return {
		fx(), fy(), cx(), cy(),
		mDistortionParameters(0), mDistortionParameters(1), mDistortionParameters(2)
	};
}

inline TwoAC::Camera_Radial::DistortionFunctor::DistortionFunctor(const Camera_Radial & cam) : mCamera(cam) {}

void TwoAC::Camera::LoadParams(const std::string_view filename) {

	std::ifstream fileIn(filename.data());

	if (!fileIn.is_open()) {
		throw std::runtime_error("Failed to open Matches file: " + std::string(filename));
	}

	std::vector<double> params;
	std::copy(
		std::istream_iterator<double>(fileIn),
		std::istream_iterator<double>(),
		std::back_inserter(params));

	if (fileIn.bad()) {
		throw std::runtime_error("Failed to load Matches from file: " + std::string(filename));
	}

	SetParams(params);
}
