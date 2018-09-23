#pragma once
#include "types.h"

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

namespace TwoAC
{
	inline double R2D(double radian)
	{
		return radian / M_PI * 180;
	}

	template <typename T>
	inline T Square(const T& x)
	{
		return x * x;
	}

	template <typename T>
	inline T Cube(const T& x)
	{
		return x * x * x;
	}

	template <typename T>
	inline T TripleProduct(const Eigen::Matrix<T, 3, 1>& a, const Eigen::Matrix<T, 3, 1>& b, const Eigen::Matrix<T, 3, 1>& c)
	{
		return a.dot(b.cross(c));
	}

	Mat3 LookAt(const Vec3 &center, const Vec3 & up = Vec3::UnitY());

	double getRotationMagnitude(const Mat3 & R2);

	inline Mat3 crossProduct(const Vec3& v)
	{
		Mat3 result; result <<
			0, -v(2), v(1),
			v(2), 0, -v(0),
			-v(1), v(0), 0;
		return result;
	}

	Mat3 RotationAroundX(double angle);

	Mat3 RotationAroundY(double angle);

	Mat3 RotationAroundZ(double angle);

	template<typename TMat>
	inline double FrobeniusNormSq(const TMat &A)
	{
		return A.array().abs2().sum();
	}

	template<typename TMat>
	inline double FrobeniusNorm(const TMat &A)
	{
		return std::sqrt(FrobeniusNormSq(A));
	}

	template<typename TMat>
	inline double MatrixError(const TMat& m1, const TMat& m2)
	{
		auto m1n = m1 / FrobeniusNorm(m1);
		auto m2n = m2 / FrobeniusNorm(m2);
		return std::min(FrobeniusNorm(m1n - m2n), FrobeniusNorm(m1n + m2n));
	}

	// Angular error metric, measured in degrees.
	struct AngularError {
		static double Error(const Mat3 &model, const AffineCorrespondence &x) {
			const Vec3 Em1 = model * x.first.x;
			double angleVal = x.second.x.dot(Em1) / (Em1.norm() * x.second.x.norm());
			return std::abs(R2D(asin(std::clamp(angleVal, -1.0 + 1.e-8, 1.0 - 1.e-8))));
		}
	};
}