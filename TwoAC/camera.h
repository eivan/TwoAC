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

#include "types.h"
#include "Dual_Number.h"

namespace TwoAC {

	// Camera base class
	class Camera {
	public:
		// The camera to image projection funciton, returns the value.
		virtual Vec2 p(const Vec3& X) const = 0;

		// The gradient camera to image projection funciton, returns the value and the gradient as a pair.
		virtual std::pair<Vec2, Mat23> p_gradient(const Vec3& X) const = 0;

		// The image to camera projection function, returns the value.
		virtual Vec3 q(const Vec2& x) const = 0;

		// The gradient image to camera projection function, returns the value and the gradient as a pair.
		virtual std::pair<Vec3, Mat32> q_gradient(const Vec2& x) const = 0;

		// Sets camera parameters.
		virtual void SetParams(const std::vector<double>& params) = 0;

		// Returns camera parameters.
		virtual std::vector<double> GetParams() const = 0;

		void LoadParams(const std::string_view filename);
	};

	// Pinhole camera with 3 radial distortion parameters
	class Camera_Radial : public Camera {

	public:
		Camera_Radial();

		Vec2 p(const Vec3& X) const final override;
		std::pair<Vec2, Mat23> p_gradient(const Vec3& X) const final override;

		Vec3 q(const Vec2& x) const final override;
		std::pair<Vec3, Mat32> q_gradient(const Vec2& x) const final override;

		void SetParams(const std::vector<double>& params) final override;

		std::vector<double> GetParams() const final override;

	private:
		inline double& fx();
		inline double& fy();
		inline double& cx();
		inline double& cy();

		inline const double& fx() const;
		inline const double& fy() const;
		inline const double& cx() const;
		inline const double& cy() const;

		inline double& k1();
		inline double& k2();
		inline double& k3();

		inline const double& k1() const;
		inline const double& k2() const;
		inline const double& k3() const;

		template <typename V>
		inline V distortion_formula(const V& r2) const;

		template <typename V>
		inline Eigen::Matrix<V, 2, 1> add_disto_cam(const Eigen::Matrix<V, 2, 1>& p) const;

		struct DistortionFunctor {
			const Camera_Radial& mCamera;

			explicit DistortionFunctor(const Camera_Radial& cam);

			template <typename T>
			T operator()(const T& r2) const {
				return r2 * Square(mCamera.distortion_formula(r2));
			}

			DEFINE_REAL_DERIVATIVE;
		};

		template <typename V>
		inline Eigen::Matrix<V, 2, 1> remove_disto_cam(const Eigen::Matrix<V, 2, 1>& p) const;

		template <typename V>
		inline Eigen::Matrix<V, 2, 1> p_pinhole(const Eigen::Matrix<V, 2, 1>& p) const;

		template <typename V>
		inline Eigen::Matrix<V, 2, 1> q_pinhole(const Eigen::Matrix<V, 2, 1>& p) const;

		template <typename T>
		inline Eigen::Matrix<T, 2, 1> p_radial(const Eigen::Matrix<T, 3, 1>& X) const;
		DIFFERENTIATE(p_radial, 3, 2);

		template <typename T>
		inline Eigen::Matrix<T, 3, 1> q_radial(const Eigen::Matrix<T, 2, 1>& x) const;
		DIFFERENTIATE(q_radial, 2, 3);

	private:
		Mat3 mK;
		Vec3 mDistortionParameters;

	};

	inline double & TwoAC::Camera_Radial::fx() { return mK(0, 0); }

	inline double & TwoAC::Camera_Radial::fy() { return mK(1, 1); }

	inline double & TwoAC::Camera_Radial::cx() { return mK(0, 2); }

	inline double & TwoAC::Camera_Radial::cy() { return mK(1, 2); }

	inline const double & TwoAC::Camera_Radial::fx() const { return mK(0, 0); }

	inline const double & TwoAC::Camera_Radial::fy() const { return mK(1, 1); }

	inline const double & TwoAC::Camera_Radial::cx() const { return mK(0, 2); }

	inline const double & TwoAC::Camera_Radial::cy() const { return mK(1, 2); }

	inline double & TwoAC::Camera_Radial::k1() { return mDistortionParameters[0]; }

	inline double & TwoAC::Camera_Radial::k2() { return mDistortionParameters[1]; }

	inline double & TwoAC::Camera_Radial::k3() { return mDistortionParameters[2]; }

	inline const double & TwoAC::Camera_Radial::k1() const { return mDistortionParameters[0]; }

	inline const double & TwoAC::Camera_Radial::k2() const { return mDistortionParameters[1]; }

	inline const double & TwoAC::Camera_Radial::k3() const { return mDistortionParameters[2]; }

	template<typename V>
	inline V Camera_Radial::distortion_formula(const V & r2) const {
		return 1. + r2 * (k1() + r2 * (k2() + r2 * k3()));
	}

	template<typename V>
	inline Eigen::Matrix<V, 2, 1> Camera_Radial::add_disto_cam(const Eigen::Matrix<V, 2, 1>& p) const {
		const auto r2 = p.squaredNorm();
		return p * distortion_formula(r2);
	}

	template<typename V>
	inline Eigen::Matrix<V, 2, 1> Camera_Radial::remove_disto_cam(const Eigen::Matrix<V, 2, 1>& p) const {
		const V r2 = p.squaredNorm();

		if (r2 == 0.0) {
			return p;
		}
		else {
			DistortionFunctor distoFunctor(*this);

			V radius = TwoAC::sqrt(bisection(distoFunctor, r2) / r2);
			return radius * p;
		}
	}

	template<typename V>
	inline Eigen::Matrix<V, 2, 1> Camera_Radial::p_pinhole(const Eigen::Matrix<V, 2, 1>& p) const {
		return { fx() * p(0) + cx(), fy() * p(1) + cy() };
	}

	template<typename V>
	inline Eigen::Matrix<V, 2, 1> Camera_Radial::q_pinhole(const Eigen::Matrix<V, 2, 1>& p) const {
		return { (p(0) - cx()) / fx(), (p(1) - cy()) / fy() };
	}

	template<typename T>
	inline Eigen::Matrix<T, 2, 1> Camera_Radial::p_radial(const Eigen::Matrix<T, 3, 1>& X) const {
		Eigen::Matrix<T, 2, 1> p = X.template head<2>() / X(2);
		return p_pinhole(add_disto_cam(p));
	}

	template<typename T>
	inline Eigen::Matrix<T, 3, 1> Camera_Radial::q_radial(const Eigen::Matrix<T, 2, 1>& x) const {
		auto p2 = remove_disto_cam(q_pinhole(x));
		return Eigen::Matrix<T, 3, 1>(p2(0), p2(1), T(1.0)).normalized();
	}

}