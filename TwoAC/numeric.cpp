#include "numeric.h"

TwoAC::Mat3 TwoAC::LookAt(const Vec3 & center, const Vec3 & up)
{
	const Vec3 zc = center.normalized();
	const Vec3 xc = up.cross(zc).normalized();
	const Vec3 yc = zc.cross(xc).normalized();
	Mat3 R;
	R.row(0) = xc;
	R.row(1) = yc;
	R.row(2) = zc;
	return R;
}

TwoAC::Mat3 TwoAC::RotationAroundX(double angle)
{
	return Eigen::AngleAxisd(angle, Vec3::UnitX()).toRotationMatrix();
}

TwoAC::Mat3 TwoAC::RotationAroundY(double angle)
{
	return Eigen::AngleAxisd(angle, Vec3::UnitY()).toRotationMatrix();
}

TwoAC::Mat3 TwoAC::RotationAroundZ(double angle)
{
	return Eigen::AngleAxisd(angle, Vec3::UnitZ()).toRotationMatrix();
}

double TwoAC::getRotationMagnitude(const TwoAC::Mat3 & R2) {
	const Mat3 R1 = Mat3::Identity();
	double cos_theta = (R1.array() * R2.array()).sum() / 3.0;
	cos_theta = std::clamp(cos_theta, -1.0, 1.0);
	//return (std::acos(R2(0, 0)) + std::acos(R2(1, 1)) + std::acos(R2(2, 2))) / 3.0;
	return std::acos(cos_theta);
}