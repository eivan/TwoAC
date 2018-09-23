#pragma once
#include "numeric.h"
#include <Eigen/Core>
//#include <boost/math/special_functions/erf.hpp>

// inspiration: Google Ceres solver > Jet.h

namespace TwoAC
{
	// Dual Number class for automatic differentiation
	template<typename T, int N>
	class Dual_Number
	{
	protected:
		// Value part.
		T f_;

		// The infinitesimal part.
		//
		// We allocate Jets on the stack and other places they might not be aligned
		// to X(=16 [SSE], 32 [AVX] etc)-byte boundaries, which would prevent the safe
		// use of vectorisation.  If we have C++11, we can specify the alignment.
		// However, the standard gives wide latitude as to what alignments are valid,
		// and it might be that the maximum supported alignment *guaranteed* to be
		// supported is < 16, in which case we do not specify an alignment, as this
		// implies the host is not a modern x86 machine.  If using < C++11, we cannot
		// specify alignment.

#if defined(EIGEN_DONT_VECTORIZE)
		typedef Eigen::Matrix<T, N, 1, Eigen::DontAlign> VecX;
		Eigen::Matrix<T, N, 1, Eigen::DontAlign> grad_;
#else
		// Enable vectorisation iff the maximum supported scalar alignment is >=
		// 16 bytes, as this is the minimum required by Eigen for any vectorisation.
		//
		// NOTE: It might be the case that we could get >= 16-byte alignment even if
		//       max_align_t < 16.  However we can't guarantee that this
		//       would happen (and it should not for any modern x86 machine) and if it
		//       didn't, we could get misaligned Jets.
		static constexpr int kAlignOrNot =
			// Work around a GCC 4.8 bug
			// (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=56019) where
			// std::max_align_t is misplaced.
#if defined (__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ == 8
			alignof(::max_align_t) >= 16
#else
			alignof(std::max_align_t) >= 16
#endif
			? Eigen::AutoAlign : Eigen::DontAlign;

#if defined(EIGEN_MAX_ALIGN_BYTES)
		// Eigen >= 3.3 supports AVX & FMA instructions that require 32-byte alignment
		// (greater for AVX512).  Rather than duplicating the detection logic, use
		// Eigen's macro for the alignment size.
		//
		// NOTE: EIGEN_MAX_ALIGN_BYTES can be > 16 (e.g. 32 for AVX), even though
		//       kMaxAlignBytes will max out at 16.  We are therefore relying on
		//       Eigen's detection logic to ensure that this does not result in
		//       misaligned Jets.
#define CERES_JET_ALIGN_BYTES EIGEN_MAX_ALIGN_BYTES
#else
		// Eigen < 3.3 only supported 16-byte alignment.
#define CERES_JET_ALIGN_BYTES 16
#endif

		// Default to the native alignment if 16-byte alignment is not guaranteed to
		// be supported.  We cannot use alignof(T) as if we do, GCC 4.8 complains that
		// the alignment 'is not an integer constant', although Clang accepts it.
		static constexpr size_t kAlignment = kAlignOrNot == Eigen::AutoAlign
			? CERES_JET_ALIGN_BYTES : alignof(double);

#undef CERES_JET_ALIGN_BYTES
		typedef Eigen::Matrix<T, N, 1, kAlignOrNot> VecX;
		alignas(kAlignment)Eigen::Matrix<T, N, 1, kAlignOrNot> grad_;
#endif

	public:
		enum { DIMENSION = N };

		Dual_Number() : f_() { grad_.setZero(); }
		template<typename V>
		explicit Dual_Number(const V& f) { f_ = T(f); grad_.setZero(); }

		// FALLBACK OPERATORS
		explicit Dual_Number(const Dual_Number<float, N>& f) { f_ = T(f.f()); grad_ = f.grad().cast<T>(); }
		explicit operator double() const { return double(f_); }
		explicit operator int() const { return int(f_); }
		template <typename V>
		Dual_Number<V, N> cast() const { return { V(f_), grad_.template cast<V>() }; }

		Dual_Number(const T& f, int k) {
			f_ = f;
			grad_.setZero();
			grad_[k] = T(1.0);
		}
		void setPartial(const T& f, int k) {
			f_ = f;
			grad_.setZero();
			grad_[k] = T(1.0);
		}

		// Constructor from scalar and vector part
		// The use of Eigen::DenseBase allows Eigen expressions
		// to be passed in without being fully evaluated until
		// they are assigned to v
		template<typename Derived>
		EIGEN_STRONG_INLINE Dual_Number(const T& a, const Eigen::DenseBase<Derived> &v)
			: f_(a), grad_(v) {
		}

		const T& f() const { return f_; }
		T& f() { return f_; }
		const VecX& grad() const { return grad_; }
		VecX& grad() { return grad_; }

		/// basic operators

		Dual_Number& operator+= (const Dual_Number& rhs)
		{
			f_ += rhs.f_;
			grad_ += rhs.grad_;
			return *this;
		}

		Dual_Number& operator-= (const Dual_Number& rhs)
		{
			f_ -= rhs.f_;
			grad_ -= rhs.grad_;
			return *this;
		}

		Dual_Number& operator*= (const Dual_Number& rhs)
		{
			f_ *= rhs.f_;
			grad_.noalias() = grad_ * rhs.f_ + rhs.grad_*f_;
			return *this;
		}

		Dual_Number& operator/= (const Dual_Number& rhs)
		{
			const T rhs_inverse = T(1.0) / rhs.f_;
			f_ *= rhs_inverse;
			grad_ = (grad_ - f_ * rhs.grad_) * rhs_inverse;
			return *this;
		}

		Dual_Number operator* (const double& rhs)
		{
			return{ f_ * rhs, grad_ * rhs };
		}

		Dual_Number operator/ (const double& rhs)
		{
			return{ f_ / rhs, grad_ / rhs };
		}

		Dual_Number& operator*= (const double & rhs)
		{
			f_ *= rhs;
			grad_ *= rhs;
			return *this;
		}

		Dual_Number& operator/= (const double& rhs)
		{
			f_ /= rhs;
			grad_ /= rhs;
			return *this;
		}

		Dual_Number operator- (const double& rhs)
		{
			return{ f_ - rhs, grad_ };
		}

		Dual_Number operator+ (const double& rhs)
		{
			return{ f_ + rhs, grad_ };
		}
	};

	// Unary +
	template<typename T, int N> inline
		Dual_Number<T, N> const& operator+(const Dual_Number<T, N>& f) {
		return f;
	}

	// Unary -
	template<typename T, int N> inline
		Dual_Number<T, N> operator-(const Dual_Number<T, N>&f) {
		return{ -f.f(), -f.grad() };
	}

	// Binary +
	template<typename T, int N> inline
		Dual_Number<T, N> operator+(const Dual_Number<T, N>& f,
			const Dual_Number<T, N>& g) {
		return{ f.f() + g.f(), f.grad() + g.grad() };
	}

	// Binary + with a scalar: x + s
	template<typename T, int N> inline
		Dual_Number<T, N> operator+(const Dual_Number<T, N>& f, T s) {
		return{ f.f() + s, f.grad() };
	}

	// Binary + with a scalar: s + x
	template<typename T, int N> inline
		Dual_Number<T, N> operator+(T s, const Dual_Number<T, N>& f) {
		return{ f.f() + s, f.grad() };
	}

	// Binary -
	template<typename T, int N> inline
		Dual_Number<T, N> operator-(const Dual_Number<T, N>& f,
			const Dual_Number<T, N>& g) {
		return{ f.f() - g.f(), f.grad() - g.grad() };
	}

	// Binary - with a scalar: x - s
	template<typename T, int N> inline
		Dual_Number<T, N> operator-(const Dual_Number<T, N>& f, T s) {
		return{ f.f() - s, f.grad() };
	}

	// Binary - with a scalar: s - x
	template<typename T, int N> inline
		Dual_Number<T, N> operator-(T s, const Dual_Number<T, N>& f) {
		return{ s - f.f(), -f.grad() };
	}

	// Binary *
	template<typename T, int N> inline
		Dual_Number<T, N> operator*(const Dual_Number<T, N>& f,
			const Dual_Number<T, N>& g) {
		return{ f.f() * g.f(), f.f() * g.grad() + f.grad() * g.f() };
	}

	// Binary * with a scalar: x * s
	template<typename T, int N> inline
		Dual_Number<T, N> operator*(const Dual_Number<T, N>& f, T s) {
		return{ f.f() * s, f.grad() * s };
	}

	// Binary * with a scalar: s * x
	template<typename T, int N> inline
		Dual_Number<T, N> operator*(T s, const Dual_Number<T, N>& f) {
		return{ f.f() * s, f.grad() * s };
	}

	// Binary /
	template<typename T, int N> inline
		Dual_Number<T, N> operator/(const Dual_Number<T, N>& f,
			const Dual_Number<T, N>& g) {
		// This uses:
		//
		//   a + u   (a + u)(b - v)   (a + u)(b - v)
		//   ----- = -------------- = --------------
		//   b + v   (b + v)(b - v)        b^2
		//
		// which holds because v*v = 0.
		const T g_a_inverse = T(1.0) / g.f();
		const T f_a_by_g_a = f.f() * g_a_inverse;
		return{ f_a_by_g_a, (f.grad() - f_a_by_g_a * g.grad()) * g_a_inverse };
	}

	// Binary / with a scalar: s / x
	template<typename T, int N> inline
		Dual_Number<T, N> operator/(T s, const Dual_Number<T, N>& g) {
		const T minus_s_g_a_inverse2 = -s / (g.f() * g.f());
		return{ s / g.f(), g.grad() * minus_s_g_a_inverse2 };
	}

	// Binary / with a scalar: x / s
	template<typename T, int N> inline
		Dual_Number<T, N> operator/(const Dual_Number<T, N>& f, T s) {
		const T s_inverse = 1.0 / s;
		return{ f.f() * s_inverse, f.grad() * s_inverse };
	}

	// Binary comparison operators for both scalars and jets.
#define DEFINE_DN_COMPARISON_OPERATOR(op) \
	template<typename T, int N> inline \
	bool operator op(const T& s, const Dual_Number<T, N>& g) { \
	  return s op g.f(); \
	} \
	template<typename T, int N> inline \
	bool operator op(const Dual_Number<T, N>& f, const T& s) { \
	  return f.f() op s; \
	} \
	template<typename T, int N> inline \
	bool operator op(int s, const Dual_Number<T, N>& g) { \
	  return s op g.f(); \
	} \
	template<typename T, int N> inline \
	bool operator op(const Dual_Number<T, N>& f, int s) { \
	  return f.f() op s; \
	}
	DEFINE_DN_COMPARISON_OPERATOR(< );
	DEFINE_DN_COMPARISON_OPERATOR(<= );
	DEFINE_DN_COMPARISON_OPERATOR(> );
	DEFINE_DN_COMPARISON_OPERATOR(>= );
	DEFINE_DN_COMPARISON_OPERATOR(== );
	DEFINE_DN_COMPARISON_OPERATOR(!= );

	/*template<typename T, int N> inline \
	bool operator op(const Dual_Number<T, N>& f, const Dual_Number<T, N>& g) { \
	return f.f() op g.f(); \
	} \*/

	template<typename T, int N> inline
		bool operator <(const Dual_Number<T, N>& f, const Dual_Number<T, N>& g) {
		if (f.f() < g.f())
			return true;
		if (f.f() == g.f() && std::lexicographical_compare(&f.grad()[0], &f.grad()[0] + N, &g.grad()[0], &g.grad()[0] + N))
			return true;
		return false;
	}

	template<typename T, int N> inline
		bool operator >(const Dual_Number<T, N>& f, const Dual_Number<T, N>& g) {
		return g < f;
	}

	template<typename T, int N> inline
		bool operator ==(const Dual_Number<T, N>& f, const Dual_Number<T, N>& g) {
		return f.f() == g.f() && std::equal(&f.grad()[0], &f.grad()[0] + N, &g.grad()[0], &g.grad()[0] + N);
	}

#undef DEFINE_DN_COMPARISON_OPERATOR

	using std::abs;
	using std::acos;
	using std::asin;
	using std::atan;
	using std::atan2;
	using std::cbrt;
	using std::ceil;
	using std::cos;
	using std::cosh;
	using std::erf;
	using std::erfc;
	//using boost::math::erf_inv;
	//using boost::math::erfc_inv;
	using std::exp;
	using std::exp2;
	using std::floor;
	using std::fmax;
	using std::fmin;
	using std::hypot;
	using std::isfinite;
	using std::isinf;
	using std::isnan;
	using std::isnormal;
	using std::log;
	using std::log2;
	using std::pow;
	using std::sin;
	using std::sinh;
	using std::sqrt;
	using std::tan;
	using std::tanh;

	inline double normSq(double x) { return Square(x); }
	template<typename T, int N> inline
		T normSq(const Dual_Number<T, N>& f)
	{
		return normSq(f.f()) + f.grad().squaredNorm();
	}

#pragma region Solvers

	template <typename FUN, typename V>
	inline V bisection(const FUN& functor, const V& targetValue, const double epsilon = 1e-12, const int max_iter = 1000)
	{
		// Guess plausible upper and lower bound
		V lowerbound = targetValue, upbound = targetValue;
		while (functor(lowerbound) > targetValue) lowerbound /= V(1.05);
		while (functor(upbound) < targetValue) upbound *= V(1.05);

		// Perform a bisection until epsilon accuracy is not reached
		for (int i = 0; i < max_iter && (epsilon < upbound - lowerbound); ++i)
			//while (epsilon < upbound - lowerbound)
		{
			const V mid = V(.5) * (lowerbound + upbound);
			if (functor(mid) > targetValue)
				upbound = mid;
			else
				lowerbound = mid;
		}
		return .5 * (lowerbound + upbound);
	}

	template <typename FUN, typename T, int N>
	inline Dual_Number<T, N> bisection(const FUN& functor, const Dual_Number<T, N>& targetValue, const double epsilon = 1e-12, const int max_iter = 1000)
	{
		T sourceValue = bisection(functor, targetValue.f(), epsilon, max_iter);

		auto functor_derivative = functor.derivative(sourceValue);

		return{
			/*f */ sourceValue,
			/*f'*/ targetValue.grad() / functor_derivative // inverse function theorem
		};
	}

#pragma endregion

	template<typename T>
	struct Ops {
		static bool IsScalar() {
			return true;
		}
		static const T& GetScalar(const T& t) {
			return t;
		}
		static void SetScalar(const T& scalar, T& t) {
			t = scalar;
		}
		//static void ScaleDerivative(double scale_by, const T& value) {
		//	// For double, there is no derivative to scale.
		//	(void)scale_by;  // Ignored.
		//	(void)value;  // Ignored.
		//}
		/*template<int N, int M>
		static void ChainRule(const Eigen::Matrix<T, M, N>& fx, T& dx)
		{
			(void)fx;
			(void)dx;
		}*/
		static T ChainRule(const T& fx, const T& x)
		{
			(void)x;
			return fx;
		}
	};

	template<typename T, int N>
	struct Ops<Dual_Number<T, N> > {
		static bool IsScalar() {
			return false;
		}
		static const T& GetScalar(const Dual_Number<T, N>& t) {
			return t.f();
		}
		static void SetScalar(const T& scalar, Dual_Number<T, N>& t) {
			t.f() = scalar;
		}
		//static void ScaleDerivative(double scale_by, const Dual_Number<T, N>& value) {
		//	value->v *= scale_by;
		//}
		/*template<int M>
		static void ChainRule(const Eigen::Matrix<T, M, N>& fx, T& dx)
		{
			(void)fx;
			(void)dx;
		}*/
		static Dual_Number<T, N> ChainRule(const Dual_Number<T, N>& fx, const Dual_Number<T, N>& x)
		{
			return{
				fx.f(),
				fx.grad().array() * dx.grad().array() // dfxdx
			};
		}
	};

#pragma region Regular math functions

	// In general, f(a + h) ~= f(a) + f'(a) h, via the chain rule.

	// abs(x + h) ~= x + h or -(x + h)
	template <typename T, int N> inline
		Dual_Number<T, N> abs(const Dual_Number<T, N>& f) {
		return f.f() < T(0.0) ? -f : f;
	}

	// log(a + h) ~= log(a) + h / a
	template <typename T, int N> inline
		Dual_Number<T, N> log(const Dual_Number<T, N>& f) {
		const T a_inverse = T(1.0) / f.f();
		return{ log(f.f()), f.grad() * a_inverse };
	}

	// exp(a + h) ~= exp(a) + exp(a) h
	template <typename T, int N> inline
		Dual_Number<T, N> exp(const Dual_Number<T, N>& f) {
		const T tmp = exp(f.f());
		return{ tmp, tmp * f.grad() };
	}

	// http://mathworld.wolfram.com/Erf.html
	template <typename T, int N> inline
		Dual_Number<T, N> erf(const Dual_Number<T, N>& f) {
		return{ erf(f.f()), ((2 / sqrt(M_PI)) * exp(-Square(f.f()))) * f.grad() };
	}

	// http://mathworld.wolfram.com/Erfc.html
	template <typename T, int N> inline
		Dual_Number<T, N> erfc(const Dual_Number<T, N>& f) {
		return{ erfc(f.f()), ((-2 / sqrt(M_PI)) * exp(-Square(f.f()))) * f.grad() };
	}

	/*// http://mathworld.wolfram.com/InverseErf.html
	template <typename T, int N> inline
		Dual_Number<T, N> erf_inv(const Dual_Number<T, N>& f) {
		const T tmp = erf_inv(f.f());
		return{ tmp, ((sqrt(M_PI)/2.0)*exp(Square(tmp))) * f.grad() };
	}

	// http://mathworld.wolfram.com/InverseErfc.html
	template <typename T, int N> inline
		Dual_Number<T, N> erfc_inv(const Dual_Number<T, N>& f) {
		const T tmp = erfc_inv(f.f());
		return{ tmp, ((sqrt(M_PI) / 2.0)*exp(Square(tmp))) * f.grad() };
	}*/

	// sqrt(a + h) ~= sqrt(a) + h / (2 sqrt(a))
	template <typename T, int N> inline
		Dual_Number<T, N> sqrt(const Dual_Number<T, N>& f) {
		const T tmp = sqrt(f.f());
		const T two_a_inverse = T(1.0) / (T(2.0) * tmp);
		return{ tmp, f.grad() * two_a_inverse };
	}

	// cos(a + h) ~= cos(a) - sin(a) h
	template <typename T, int N> inline
		Dual_Number<T, N> cos(const Dual_Number<T, N>& f) {
		return{ cos(f.f()), -sin(f.f()) * f.grad() };
	}

	// acos(a + h) ~= acos(a) - 1 / sqrt(1 - a^2) h
	template <typename T, int N> inline
		Dual_Number<T, N> acos(const Dual_Number<T, N>& f) {
		const T tmp = -T(1.0) / sqrt(T(1.0) - f.f() * f.f());
		return{ acos(f.f()), tmp * f.grad() };
	}

	// sin(a + h) ~= sin(a) + cos(a) h
	template <typename T, int N> inline
		Dual_Number<T, N> sin(const Dual_Number<T, N>& f) {
		return{ sin(f.f()), cos(f.f()) * f.grad() };
	}

	// asin(a + h) ~= asin(a) + 1 / sqrt(1 - a^2) h
	template <typename T, int N> inline
		Dual_Number<T, N> asin(const Dual_Number<T, N>& f) {
		const T tmp = T(1.0) / sqrt(T(1.0) - f.f() * f.f());
		return{ asin(f.f()), tmp * f.grad() };
	}

	// tan(a + h) ~= tan(a) + (1 + tan(a)^2) h
	template <typename T, int N> inline
		Dual_Number<T, N> tan(const Dual_Number<T, N>& f) {
		const T tan_a = tan(f.f());
		const T tmp = T(1.0) + tan_a * tan_a;
		return{ tan_a, tmp * f.grad() };
	}

	// atan(a + h) ~= atan(a) + 1 / (1 + a^2) h
	template <typename T, int N> inline
		Dual_Number<T, N> atan(const Dual_Number<T, N>& f) {
		const T tmp = T(1.0) / (T(1.0) + f.f() * f.f());
		return{ atan(f.f()), tmp * f.grad() };
	}

	// sinh(a + h) ~= sinh(a) + cosh(a) h
	template <typename T, int N> inline
		Dual_Number<T, N> sinh(const Dual_Number<T, N>& f) {
		return{ sinh(f.f()), cosh(f.f()) * f.grad() };
	}

	// cosh(a + h) ~= cosh(a) + sinh(a) h
	template <typename T, int N> inline
		Dual_Number<T, N> cosh(const Dual_Number<T, N>& f) {
		return{ cosh(f.f()), sinh(f.f()) * f.grad() };
	}

	// tanh(a + h) ~= tanh(a) + (1 - tanh(a)^2) h
	template <typename T, int N> inline
		Dual_Number<T, N> tanh(const Dual_Number<T, N>& f) {
		const T tanh_a = tanh(f.f());
		const T tmp = T(1.0) - tanh_a * tanh_a;
		return{ tanh_a, tmp * f.grad() };
	}

	// The floor function should be used with extreme care as this operation will
	// result in a zero derivative which provides no information to the solver.
	//
	// floor(a + h) ~= floor(a) + 0
	template <typename T, int N> inline
		Dual_Number<T, N> floor(const Dual_Number<T, N>& f) {
		return Dual_Number<T, N>(floor(f.f()));
	}

	// The ceil function should be used with extreme care as this operation will
	// result in a zero derivative which provides no information to the solver.
	//
	// ceil(a + h) ~= ceil(a) + 0
	template <typename T, int N> inline
		Dual_Number<T, N> ceil(const Dual_Number<T, N>& f) {
		return Dual_Number<T, N>(ceil(f.f()));
	}

	// atan2(b + db, a + da) ~= atan2(b, a) + (- b da + a db) / (a^2 + b^2)
	//
	// In words: the rate of change of theta is 1/r times the rate of
	// change of (x, y) in the positive angular direction.
	template <typename T, int N> inline
		Dual_Number<T, N> atan2(const Dual_Number<T, N>& g, const Dual_Number<T, N>& f) {
		// Note order of arguments:
		//
		//   f = a + da
		//   g = b + db

		T const tmp = T(1.0) / (f.f() * f.f() + g.f() * g.f());
		return{ atan2(g.f(), f.f()), tmp * (-g.f() * f.grad() + f.f() * g.grad()) };
	}


	// pow -- base is a differentiable function, exponent is a constant.
	// (a+da)^p ~= a^p + p*a^(p-1) da
	template <typename T, int N> inline
		Dual_Number<T, N> pow(const Dual_Number<T, N>& f, double g) {
		T const tmp = g * pow(f.f(), g - T(1.0));
		return{ pow(f.f(), g), tmp * f.grad() };
	}

	/*
	// pow -- base is a constant, exponent is a differentiable function.
	// (a)^(p+dp) ~= a^p + a^p log(a) dp
	template <typename T, int N> inline
	Dual_Number<T, N> pow(double f, const Dual_Number<T, N>& g) {
	T const tmp = pow(f, g.f());
	return{ tmp, log(f) * tmp * g.grad() };
	}

	// pow -- both base and exponent are differentiable functions.
	// (a+da)^(b+db) ~= a^b + b * a^(b-1) da + a^b log(a) * db
	template <typename T, int N> inline
	Dual_Number<T, N> pow(const Dual_Number<T, N>& f, const Dual_Number<T, N>& g) {
	T const tmp1 = pow(f.f(), g.f());
	T const tmp2 = g.f() * pow(f.f(), g.f() - T(1.0));
	T const tmp3 = tmp1 * log(f.f());

	return{ tmp1, tmp2 * f.grad() + tmp3 * g.grad() };
	}*/

	// pow -- base is a constant, exponent is a differentiable function.
	// We have various special cases, see the comment for pow(Jet, Jet) for
	// analysis:
	//
	// 1. For f > 0 we have: (f)^(g + dg) ~= f^g + f^g log(f) dg
	//
	// 2. For f == 0 and g > 0 we have: (f)^(g + dg) ~= f^g
	//
	// 3. For f < 0 and integer g we have: (f)^(g + dg) ~= f^g but if dg
	// != 0, the derivatives are not defined and we return NaN.

	template <typename T, int N> inline
		Dual_Number<T, N> pow(double f, const Dual_Number<T, N>& g) {
		if (f == 0 && g.f() > 0) {
			// Handle case 2.
			return Dual_Number<T, N>(T(0.0));
		}
		if (f < 0 && g.f() == floor(g.f())) {
			// Handle case 3.
			Dual_Number<T, N> ret(pow(f, g.f()));
			for (int i = 0; i < N; i++) {
				if (g.grad()[i] != T(0.0)) {
					// Return a NaN when g.grad() != 0.
					ret.grad()[i] = std::numeric_limits<T>::quiet_NaN();
				}
			}
			return ret;
		}
		// Handle case 1.
		T const tmp = pow(f, g.f());
		return{ tmp, log(f) * tmp * g.grad() };
	}

	// pow -- both base and exponent are differentiable functions. This has a
	// variety of special cases that require careful handling.
	//
	// 1. For f > 0:
	//    (f + df)^(g + dg) ~= f^g + f^(g - 1) * (g * df + f * log(f) * dg)
	//    The numerical evaluation of f * log(f) for f > 0 is well behaved, even for
	//    extremely small values (e.g. 1e-99).
	//
	// 2. For f == 0 and g > 1: (f + df)^(g + dg) ~= 0
	//    This cases is needed because log(0) can not be evaluated in the f > 0
	//    expression. However the function f*log(f) is well behaved around f == 0
	//    and its limit as f-->0 is zero.
	//
	// 3. For f == 0 and g == 1: (f + df)^(g + dg) ~= 0 + df
	//
	// 4. For f == 0 and 0 < g < 1: The value is finite but the derivatives are not.
	//
	// 5. For f == 0 and g < 0: The value and derivatives of f^g are not finite.
	//
	// 6. For f == 0 and g == 0: The C standard incorrectly defines 0^0 to be 1
	//    "because there are applications that can exploit this definition". We
	//    (arbitrarily) decree that derivatives here will be nonfinite, since that
	//    is consistent with the behavior for f == 0, g < 0 and 0 < g < 1.
	//    Practically any definition could have been justified because mathematical
	//    consistency has been lost at this point.
	//
	// 7. For f < 0, g integer, dg == 0: (f + df)^(g + dg) ~= f^g + g * f^(g - 1) df
	//    This is equivalent to the case where f is a differentiable function and g
	//    is a constant (to first order).
	//
	// 8. For f < 0, g integer, dg != 0: The value is finite but the derivatives are
	//    not, because any change in the value of g moves us away from the point
	//    with a real-valued answer into the region with complex-valued answers.
	//
	// 9. For f < 0, g noninteger: The value and derivatives of f^g are not finite.

	template <typename T, int N> inline
		Dual_Number<T, N> pow(const Dual_Number<T, N>& f, const Dual_Number<T, N>& g) {
		if (f.f() == 0.0 && g.f() >= 1.0) {
			// Handle cases 2 and 3.
			if (g.f() > 1.0) {
				return Dual_Number<T, N>(T(0.0));
			}
			return f;
		}
		if (f.f() < 0.0 && g.f() == floor(g.f())) {
			// Handle cases 7 and 8.
			T const tmp = g.f() * pow(f.f(), g.f() - T(1.0));
			Dual_Number<T, N> ret(pow(f.f(), g.f()), tmp * f.grad());
			for (int i = 0; i < N; i++) {
				if (g.grad()[i] != T(0.0)) {
					// Return a NaN when g.grad() != 0.
					ret.grad()[i] = std::numeric_limits<T>::quiet_NaN();
				}
			}
			return ret;
		}
		// Handle the remaining cases. For cases 4,5,6,9 we allow the log() function
		// to generate -HUGE_VAL or NaN, since those cases result in a nonfinite
		// derivative.
		T const tmp1 = pow(f.f(), g.f());
		T const tmp2 = g.f() * pow(f.f(), g.f() - T(1.0));
		T const tmp3 = tmp1 * log(f.f());
		return Dual_Number<T, N>(tmp1, tmp2 * f.grad() + tmp3 * g.grad());
	}

#pragma endregion

#pragma region Bessel

	// Bessel functions of the first kind with integer order equal to 0, 1, n.
	//
	// Microsoft has deprecated the j[0,1,n]() POSIX Bessel functions in favour of
	// _j[0,1,n]().  Where available on MSVC, use _j[0,1,n]() to avoid deprecated
	// function errors in client code (the specific warning is suppressed when
	// Ceres itself is built).
	inline double BesselJ0(double x) {
#if defined(_MSC_VER) && _MSC_VER >= 1800
		return _j0(x);
#else
		return j0(x);
#endif
	}
	inline double BesselJ1(double x) {
#if defined(_MSC_VER) && _MSC_VER >= 1800
		return _j1(x);
#else
		return j1(x);
#endif
	}
	inline double BesselJn(int n, double x) {
#if defined(_MSC_VER) && _MSC_VER >= 1800
		return _jn(n, x);
#else
		return jn(n, x);
#endif
	}

	// For the formulae of the derivatives of the Bessel functions see the book:
	// Olver, Lozier, Boisvert, Clark, NIST Handbook of Mathematical Functions,
	// Cambridge University Press 2010.
	//
	// Formulae are also available at http://dlmf.nist.gov

	// See formula http://dlmf.nist.gov/10.6#E3
	// j0(a + h) ~= j0(a) - j1(a) h
	template <typename T, int N> inline
		Dual_Number<T, N> BesselJ0(const Dual_Number<T, N>& f) {
		return{ BesselJ0(f.f()), -BesselJ1(f.f()) * f.grad() };
	}

	// See formula http://dlmf.nist.gov/10.6#E1
	// j1(a + h) ~= j1(a) + 0.5 ( j0(a) - j2(a) ) h
	template <typename T, int N> inline
		Dual_Number<T, N> BesselJ1(const Dual_Number<T, N>& f) {
		return{ BesselJ1(f.f()),
			T(0.5) * (BesselJ0(f.f()) - BesselJn(2, f.f())) * f.grad() };
	}

	// See formula http://dlmf.nist.gov/10.6#E1
	// j_n(a + h) ~= j_n(a) + 0.5 ( j_{n-1}(a) - j_{n+1}(a) ) h
	template <typename T, int N> inline
		Dual_Number<T, N> BesselJn(int n, const Dual_Number<T, N>& f) {
		return{ BesselJn(n, f.f()),
			T(0.5) * (BesselJn(n - 1, f.f()) - BesselJn(n + 1, f.f())) * f.grad() };
	}

#pragma endregion

#pragma region Dual Number classification
	// Classification. It is not clear what the appropriate semantics are for
	// these classifications. This picks that IsFinite and isnormal are "all"
	// operations, i.e. all elements of the jet must be finite for the jet itself
	// to be finite (or normal). For IsNaN and IsInfinite, the answer is less
	// clear. This takes a "any" approach for IsNaN and IsInfinite such that if any
	// part of a jet is nan or inf, then the entire jet is nan or inf. This leads
	// to strange situations like a jet can be both IsInfinite and IsNaN, but in
	// practice the "any" semantics are the most useful for e.g. checking that
	// derivatives are sane.

	// The jet is finite if all parts of the jet are finite.
	template <typename T, int N> inline
		bool isfinite(const Dual_Number<T, N>& f) {
		if (!isfinite(f.f())) {
			return false;
		}
		for (int i = 0; i < N; ++i) {
			if (!isfinite(f.grad()[i])) {
				return false;
			}
		}
		return true;
	}


	// The jet is infinite if any part of the jet is infinite.
	template <typename T, int N> inline
		bool isinf(const Dual_Number<T, N>& f) {
		if (isinf(f.f())) {
			return true;
		}
		for (int i = 0; i < N; i++) {
			if (isinf(f.grad()[i])) {
				return true;
			}
		}
		return false;
	}

	// The jet is NaN if any part of the jet is NaN.
	template <typename T, int N> inline
		bool isnan(const Dual_Number<T, N>& f) {
		if (isnan(f.f())) {
			return true;
		}
		for (int i = 0; i < N; ++i) {
			if (isnan(f.grad()[i])) {
				return true;
			}
		}
		return false;
	}

	// The jet is normal if all parts of the jet are normal.
	template <typename T, int N> inline
		bool isnormal(const Dual_Number<T, N>& f) {
		if (!isnormal(f.f())) {
			return false;
		}
		for (int i = 0; i < N; ++i) {
			if (!isnormal(f.grad()[i])) {
				return false;
			}
		}
		return true;
	}
#pragma endregion

#pragma region Differentiation

	template <typename T, int N, int M, typename FUN>
	void gradient(
		const FUN& OP,
		const Eigen::Matrix<T, N, 1>& param,
		Eigen::Matrix<T, M, 1>& value,
		Eigen::Matrix<T, M, N>& grad)
	{
		Eigen::Matrix<Dual_Number<T, N>, N, 1> dn_param;
		for (auto i = 0; i < N; ++i)
		{
			dn_param[i].setPartial(param[i], i);
		}

		Eigen::Matrix<Dual_Number<T, N>, M, 1> f = OP(dn_param);

		for (auto i = 0; i < M; ++i)
		{
			value[i] = f[i].f();
			grad.row(i) = f[i].grad();
		}
	}

	template <typename V, typename FUN>
	void derivative(const FUN& OP, const V& param, V& value, V& grad)
	{
		Dual_Number<V, 1> dn_param;
		dn_param.setPartial(param, 0);
		Dual_Number<V, 1> f = OP(dn_param);
		value = f.f();
		grad = f.grad()[0];
	}

	template <typename V, typename FUN>
	V derivative(const FUN& OP, const V& param)
	{
		Dual_Number<V, 1> dn_param(param, 0);
		Dual_Number<V, 1> f = OP(dn_param);
		return f.grad()[0];
	}

#define DEFINE_REAL_DERIVATIVE \
	template <typename T> \
	inline T derivative(const T& x) const \
	{ \
		return TwoAC::derivative(*this, x); \
	}

#define DEFINE_VECTOR_GRADIENT(N,M) \
	template <typename T> \
	inline Eigen::Matrix<T, M, N> gradient(const Eigen::Matrix<T, N, 1>& param) const \
	{ \
		Eigen::Matrix<TwoAC::Dual_Number<T, N>, N, 1> dn_param; \
		for (int i = 0; i < N; ++i) \
			dn_param[i].setPartial(param[i], i); \
		Eigen::Matrix<TwoAC::Dual_Number<T, N>, M, 1> f_ = this->operator()(dn_param);  \
		Eigen::Matrix<T, M, N> grad; \
		for (int i = 0; i < M; ++i) \
			grad.row(i) = f_[i].grad(); \
		return grad; \
	}

#define DIFFERENTIATE(FUN,N,M) \
	template <typename T> \
	inline std::pair<Eigen::Matrix<T, M, 1>, Eigen::Matrix<T, M, N>> FUN ## _gradient(const Eigen::Matrix<T, N, 1>& param) const \
	{ \
		Eigen::Matrix<TwoAC::Dual_Number<T, N>, N, 1> dn_param; \
		for (int i = 0; i < N; ++i) \
			dn_param[i].setPartial(param[i], i); \
		Eigen::Matrix<TwoAC::Dual_Number<T, N>, M, 1> f_ = ##FUN(dn_param);  \
		Eigen::Matrix<T, M, N> grad; \
		Eigen::Matrix<T, M, 1> val; \
		for (int i = 0; i < M; ++i) { \
			grad.row(i) = f_[i].grad(); \
			val(i) = f_[i].f();\
		} \
		return { val, grad }; \
	}

#define DEFINE_SCALAR_VECTOR_GRADIENT(N) \
	template <typename T> \
	inline Eigen::Matrix<T, 1, N> gradient(const Eigen::Matrix<T, N, 1>& param) const \
	{ \
		Eigen::Matrix<TwoAC::Dual_Number<T, N>, N, 1> dn_param; \
		for (int i = 0; i < N; ++i) \
			dn_param[i].setPartial(param[i], i); \
		TwoAC::Dual_Number<T, N> f_ = this->operator()(dn_param);  \
		return f_.grad(); \
	}\
	\
	template <typename T> \
	inline TwoAC::Dual_Number<T, N> value_and_gradient(const Eigen::Matrix<T, N, 1>& param) const \
	{ \
		Eigen::Matrix<TwoAC::Dual_Number<T, N>, N, 1> dn_param; \
		for (int i = 0; i < N; ++i) \
			dn_param[i].setPartial(param[i], i); \
		return this->operator()(dn_param);  \
	}

#define OPERATOR_SCALAR_SCALAR(EXPRESSION) \
	template <typename T> T operator()(const T& x) const { \
		return EXPRESSION; \
	}

#define OPERATOR_VECTOR_VECTOR(N, M, EXPRESSION) \
	template <typename T> \
	Eigen::Matrix<T,M,1> operator()(const Eigen::Matrix<T,N,1>& x) const { \
		return EXPRESSION; \
	}

#define OPERATOR_VECTOR_VECTOR_BLOCK(N, M, EXPRESSION) \
	template <typename T> \
	Eigen::Matrix<T,M,1> operator()(const Eigen::Matrix<T,N,1>& x) const { \
		EXPRESSION\
	}

#define DIFFERENTIABLE(TYPENAME, EXPRESSION) \
struct TYPENAME { \
	OPERATOR_SCALAR_SCALAR(EXPRESSION) \
	DEFINE_REAL_DERIVATIVE\
}

#define DIFFERENTIABLE_VECTOR_VECTOR(TYPENAME, N, M, EXPRESSION) \
struct TYPENAME { \
	OPERATOR_VECTOR_VECTOR(N, M, EXPRESSION) \
	DEFINE_VECTOR_GRADIENT(N,M)\
}

#pragma endregion

	template <typename T, int N>
	inline std::ostream &operator<<(std::ostream &s, const Dual_Number<T, N>& z) {
		return s << "[" << z.f() << " ; " << z.grad().transpose() << "]";
	}
}

namespace Eigen {

	// Creating a specialization of NumTraits enables placing Dual_Number objects inside
	// Eigen arrays, getting all the goodness of Eigen combined with autodiff.
	template<typename T, int N>
	struct NumTraits<TwoAC::Dual_Number<T, N> > {
		typedef TwoAC::Dual_Number<T, N> Real;
		typedef TwoAC::Dual_Number<T, N> NonInteger;
		typedef TwoAC::Dual_Number<T, N> Nested;
		typedef TwoAC::Dual_Number<T, N> Literal;

		static inline TwoAC::Dual_Number<T, N> dummy_precision() {
			//return TwoAC::Dual_Number<T, N>(1e-12);
			return Real(NumTraits<T>::dummy_precision());
		}

		static inline Real epsilon() {
			//return Real(std::numeric_limits<T>::epsilon());
			return Real(NumTraits<T>::epsilon());
		}

		enum {
			IsComplex = NumTraits<T>::IsComplex,
			IsInteger = NumTraits<T>::IsInteger,
			IsSigned = NumTraits<T>::IsSigned,
			ReadCost = (N + 1) * NumTraits<T>::ReadCost,
			AddCost = (N + 1) * NumTraits<T>::AddCost,
			MulCost = N * NumTraits<T>::AddCost + (2 * N + 1) * NumTraits<T>::MulCost,
			HasFloatingPoint = 1,
			RequireInitialization = 1
		};

		template<bool Vectorized>
		struct Div {
			enum {
#define EIGEN_VECTORIZE_AVX // TODO
#if defined(EIGEN_VECTORIZE_AVX) 
				AVX = true,
#else
				AVX = false,
#endif

				// Assuming that for Jets, division is as expensive as
				// multiplication.
				Cost = 3
			};
		};
	};

}  // namespace Eigen