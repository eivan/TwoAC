#pragma once
#include <cstdint>
#include <limits>
#include <utility>
#include <set>
#include <vector>
#include <map>
#include <Eigen/Eigen>

namespace TwoAC
{
	// Check GCC
#if __GNUC__
#if __x86_64__ || __ppc64__ || _LP64
#define ENV64BIT
#else
#define ENV32BIT
#endif
#endif

	using Eigen::Map;

	typedef Eigen::NumTraits<double> EigenDoubleTraits;

	typedef Eigen::Vector3d Vec3;
	typedef Eigen::Vector2i Vec2i;
	typedef Eigen::Vector2f Vec2f;
	typedef Eigen::Vector3f Vec3f;

#if defined(ENV32BIT)
	typedef Eigen::Matrix<double, 2, 1, Eigen::DontAlign> Vec2;
	typedef Eigen::Matrix<double, 4, 1, Eigen::DontAlign> Vec4;
	typedef Eigen::Matrix<double, 6, 1, Eigen::DontAlign> Vec6;
	typedef Eigen::Matrix<double, 9, 1, Eigen::DontAlign> Vec9;
	typedef Eigen::Matrix<double, 2, 2, Eigen::DontAlign> Mat2;
	typedef Eigen::Matrix<double, 2, 3, Eigen::DontAlign> Mat23;
	typedef Eigen::Matrix<double, 3, 2, Eigen::DontAlign> Mat32;
	typedef Eigen::Matrix<double, 3, 3, Eigen::DontAlign> Mat3;
	typedef Eigen::Matrix<double, 3, 4, Eigen::DontAlign> Mat34;
	typedef Eigen::Matrix<double, 4, 4, Eigen::DontAlign> Mat4;
#else // 64 bits compiler
	typedef Eigen::Vector2d Vec2;
	typedef Eigen::Vector4d Vec4;
	typedef Eigen::Matrix<double, 6, 1> Vec6;
	typedef Eigen::Matrix<double, 9, 1> Vec9;
	typedef Eigen::Matrix<double, 2, 2> Mat2;
	typedef Eigen::Matrix<double, 2, 3> Mat23;
	typedef Eigen::Matrix<double, 3, 2> Mat32;
	typedef Eigen::Matrix<double, 3, 3> Mat3;
	typedef Eigen::Matrix<double, 3, 4> Mat34;
	typedef Eigen::Matrix<double, 4, 4> Mat4;
#endif

	//-- General purpose Matrix and Vector
	typedef Eigen::MatrixXd Mat;
	typedef Eigen::VectorXd Vec;

	typedef Eigen::Matrix<double, 2, Eigen::Dynamic> Mat2X;
	typedef Eigen::Matrix<double, 3, Eigen::Dynamic> Mat3X;
	typedef Eigen::Matrix<double, 4, Eigen::Dynamic> Mat4X;
	typedef Eigen::Matrix<double, Eigen::Dynamic, 9> MatX9;

	//-- Sparse Matrix (Column major, and row major)
	typedef Eigen::SparseMatrix<double> sMat;
	typedef Eigen::SparseMatrix<double, Eigen::RowMajor> sRMat;

	/// Portable type used to store an index
	typedef uint32_t IndexT;

	// Normalized using first order approximation of an image-to-camera 
	// projeciton function.
	struct NormalizedLocalAffineFrame {
		template<typename Derived>
		EIGEN_STRONG_INLINE NormalizedLocalAffineFrame(const Vec3& x_, const Eigen::DenseBase<Derived> &dx_)
			: x(x_), dx(dx_) {
		}

		Vec3 x;
		Mat32 dx;
	};

	// A pair of Normalized LAFs in this application is an Affine Correspondence 
	// in normalized camera space.
	using AffineCorrespondence = std::pair<NormalizedLocalAffineFrame, NormalizedLocalAffineFrame>;

	// A vector of Affine Correspondences
	using AffineCorrespondences = std::vector<AffineCorrespondence>;

	// Indices and reliability score of a matching pair of LAFs.
	struct Match : std::tuple<IndexT, IndexT, double> {};

	// Output stream definition of a Match
	inline std::ostream& operator<<(std::ostream& out, const Match& obj)
	{
		out << std::get<0>(obj) << " " << std::get<1>(obj) << " " << std::get<2>(obj);
		return out;
	}

	// Input stream definition of a Match
	inline std::istream& operator >> (std::istream& in, Match& obj)
	{
		in >> std::get<0>(obj) >> std::get<1>(obj) >> std::get<2>(obj);
		return in;
	}
}
