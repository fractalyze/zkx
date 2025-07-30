#include "zkx/math/elliptic_curves/short_weierstrass/jacobian_point.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/status.h"
#include "zkx/base/buffer/vector_buffer.h"
#include "zkx/math/elliptic_curves/short_weierstrass/test/sw_curve_config.h"

namespace zkx::math::test {

TEST(JacobianPointTest, Zero) {
  EXPECT_TRUE(JacobianPoint::Zero().IsZero());
  EXPECT_FALSE(JacobianPoint(1, 2, 1).IsZero());
}

TEST(JacobianPointTest, One) {
  auto generator = JacobianPoint::Generator();
  EXPECT_EQ(generator, JacobianPoint(JacobianPoint::Curve::Config::kX,
                                     JacobianPoint::Curve::Config::kY, 1));
  EXPECT_EQ(JacobianPoint::Generator(), JacobianPoint::One());
  EXPECT_TRUE(generator.IsOne());
}

TEST(JacobianPointTest, EqualityOperations) {
  {
    SCOPED_TRACE("p.IsZero() && p2.IsZero()");
    JacobianPoint p(1, 2, 0);
    JacobianPoint p2(3, 4, 0);
    EXPECT_EQ(p, p2);
    EXPECT_EQ(p2, p);
  }

  {
    SCOPED_TRACE("!p.IsZero() && p2.IsZero()");
    JacobianPoint p(1, 2, 1);
    JacobianPoint p2(3, 4, 0);
    EXPECT_NE(p, p2);
    EXPECT_NE(p2, p);
  }

  {
    SCOPED_TRACE("other");
    JacobianPoint p(1, 2, 3);
    JacobianPoint p2(1, 2, 3);
    EXPECT_EQ(p, p2);
    EXPECT_EQ(p2, p);
  }
}

TEST(JacobianPointTest, GroupOperations) {
  JacobianPoint p(5, 5, 1);
  JacobianPoint p2(3, 2, 1);
  JacobianPoint p3(3, 5, 1);
  JacobianPoint p4(6, 5, 1);
  TF_ASSERT_OK_AND_ASSIGN(AffinePoint ap, p.ToAffine());
  TF_ASSERT_OK_AND_ASSIGN(AffinePoint ap2, p2.ToAffine());

  EXPECT_EQ(p + p2, p3);
  EXPECT_EQ(p - p3, -p2);
  EXPECT_EQ(p + p, p4);
  EXPECT_EQ(p - p4, -p);

  {
    JacobianPoint p_tmp = p;
    p_tmp += p2;
    EXPECT_EQ(p_tmp, p3);
    p_tmp -= p2;
    EXPECT_EQ(p_tmp, p);
  }

  EXPECT_EQ(p + ap2, p3);
  EXPECT_EQ(p + ap, p4);
  EXPECT_EQ(p - p3, -p2);
  EXPECT_EQ(p - p4, -p);

  EXPECT_EQ(p.Double(), p4);

  EXPECT_EQ(-p, JacobianPoint(5, 2, 1));

  EXPECT_EQ(p * 2, p4);
  EXPECT_EQ(Fr(2) * p, p4);
  EXPECT_EQ(p *= 2, p4);
}

TEST(JacobianPointTest, CyclicScalarMul) {
  std::vector<AffinePoint> points;
  for (size_t i = 0; i < 7; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(AffinePoint ap,
                            (Fr(i) * JacobianPoint::Generator()).ToAffine());
    points.push_back(ap);
  }

  EXPECT_THAT(points,
              testing::UnorderedElementsAreArray(std::vector<AffinePoint>{
                  AffinePoint(0, 0),
                  AffinePoint(3, 2),
                  AffinePoint(5, 2),
                  AffinePoint(6, 2),
                  AffinePoint(3, 5),
                  AffinePoint(5, 5),
                  AffinePoint(6, 5),
              }));
}

TEST(JacobianPointTest, ToAffine) {
  JacobianPoint p(1, 2, 0);
  JacobianPoint p2(1, 2, 1);
  JacobianPoint p3(1, 2, 3);
  TF_ASSERT_OK_AND_ASSIGN(AffinePoint ap, p.ToAffine());
  TF_ASSERT_OK_AND_ASSIGN(AffinePoint ap2, p2.ToAffine());
  TF_ASSERT_OK_AND_ASSIGN(AffinePoint ap3, p3.ToAffine());
  EXPECT_EQ(ap, AffinePoint::Zero());
  EXPECT_EQ(ap2, AffinePoint(1, 2));
  EXPECT_EQ(ap3, AffinePoint(4, 5));
}

TEST(JacobianPointTest, ToXyzz) {
  auto p = JacobianPoint::Random();
  EXPECT_EQ(p.ToXyzz(), p.ToAffine()->ToXyzz());
}

TEST(JacobianPointTest, BatchToAffine) {
  std::vector<JacobianPoint> jacobian_points = {
      JacobianPoint(1, 2, 0),
      JacobianPoint(1, 2, 1),
      JacobianPoint(1, 2, 3),
  };

  absl::Span<AffinePoint> affine_points_span;
  ASSERT_FALSE(
      JacobianPoint::BatchToAffine(jacobian_points, &affine_points_span).ok());

  std::vector<AffinePoint> affine_points;
  ASSERT_TRUE(
      JacobianPoint::BatchToAffine(jacobian_points, &affine_points).ok());

  std::vector<AffinePoint> expected_affine_points = {
      AffinePoint::Zero(), AffinePoint(1, 2), AffinePoint(4, 5)};
  EXPECT_EQ(affine_points, expected_affine_points);
}

TEST(JacobianPointTest, MontReduce) {
  JacobianPoint p(3, 2, 1);
  JacobianPoint reduced = p.MontReduce();

  EXPECT_EQ(reduced.x(), Fq(3).MontReduce());
  EXPECT_EQ(reduced.y(), Fq(2).MontReduce());
  EXPECT_EQ(reduced.z(), Fq(1).MontReduce());
}

TEST(JacobianPointTest, Serde) {
  JacobianPoint expected = JacobianPoint::Random();

  base::Uint8VectorBuffer write_buf;
  TF_ASSERT_OK(write_buf.Grow(base::EstimateSize(expected)));
  TF_ASSERT_OK(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  JacobianPoint value;
  TF_ASSERT_OK(write_buf.Read(&value));
  EXPECT_EQ(expected, value);
}

TEST(JacobianPointTest, JsonSerde) {
  rapidjson::Document doc;

  JacobianPoint expected = JacobianPoint::Random();
  rapidjson::Value json_value =
      base::JsonSerde<JacobianPoint>::From(expected, doc.GetAllocator());
  EXPECT_TRUE(json_value.IsObject());
  EXPECT_EQ(Fq::FromUnchecked(json_value["x"].GetInt()), expected.x());
  EXPECT_EQ(Fq::FromUnchecked(json_value["y"].GetInt()), expected.y());
  EXPECT_EQ(Fq::FromUnchecked(json_value["z"].GetInt()), expected.z());

  TF_ASSERT_OK_AND_ASSIGN(JacobianPoint value,
                          base::JsonSerde<JacobianPoint>::To(json_value, ""));
  EXPECT_EQ(expected, value);
}

}  // namespace zkx::math::test
