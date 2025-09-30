#include "zkx/math/elliptic_curves/short_weierstrass/affine_point.h"

#include <vector>

#include "absl/strings/substitute.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/status.h"
#include "zkx/base/auto_reset.h"
#include "zkx/base/buffer/vector_buffer.h"
#include "zkx/math/elliptic_curves/short_weierstrass/test/sw_curve_config.h"

namespace zkx::math::test {

TEST(AffinePointTest, Zero) {
  EXPECT_TRUE(AffinePoint::Zero().IsZero());
  EXPECT_FALSE(AffinePoint(1, 2).IsZero());
}

TEST(AffinePointTest, One) {
  auto generator = AffinePoint::Generator();
  EXPECT_EQ(generator, AffinePoint(AffinePoint::Curve::Config::kX,
                                   AffinePoint::Curve::Config::kY));
  EXPECT_EQ(AffinePoint::Generator(), AffinePoint::One());
  EXPECT_TRUE(generator.IsOne());
}

TEST(AffinePointTest, EqualityOperations) {
  AffinePoint p(1, 2);
  AffinePoint p2(3, 4);
  EXPECT_EQ(p, p);
  EXPECT_NE(p, p2);
}

TEST(AffinePointTest, GroupOperations) {
  AffinePoint ap(5, 5);
  AffinePoint ap2(3, 2);
  AffinePoint ap3(3, 5);
  AffinePoint ap4(6, 5);
  JacobianPoint jp = ap.ToJacobian();
  JacobianPoint jp2 = ap2.ToJacobian();
  JacobianPoint jp3 = ap3.ToJacobian();
  JacobianPoint jp4 = ap4.ToJacobian();

  EXPECT_EQ(ap + ap2, jp3);
  EXPECT_EQ(ap + ap, jp4);
  EXPECT_EQ(ap3 - ap2, jp);
  EXPECT_EQ(ap4 - ap, jp);

  EXPECT_EQ(ap + jp2, jp3);
  EXPECT_EQ(ap + jp, jp4);
  EXPECT_EQ(ap - jp3, -jp2);
  EXPECT_EQ(ap - jp4, -jp);

  EXPECT_EQ(-ap, AffinePoint(5, 2));

  EXPECT_EQ(ap * 2, jp4);
  EXPECT_EQ(Fr(2) * ap, jp4);
}

TEST(AffinePointTest, CyclicScalarMul) {
  std::vector<AffinePoint> points;
  for (size_t i = 0; i < 7; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(AffinePoint ap,
                            (Fr(i) * AffinePoint::Generator()).ToAffine());
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

TEST(AffinePointTest, ToJacobian) {
  EXPECT_EQ(AffinePoint::Zero().ToJacobian(), JacobianPoint::Zero());
  AffinePoint p(3, 2);
  EXPECT_EQ(p.ToJacobian(), JacobianPoint(3, 2, 1));
}

TEST(AffinePointTest, ToPointXyzz) {
  EXPECT_EQ(AffinePoint::Zero().ToXyzz(), PointXyzz::Zero());
  AffinePoint p(3, 2);
  EXPECT_EQ(p.ToXyzz(), PointXyzz(3, 2, 1, 1));
}

TEST(AffinePointTest, MontReduce) {
  AffinePoint p(3, 2);
  AffinePoint::StdType reduced = p.MontReduce();

  EXPECT_EQ(reduced.x(), Fq(3).MontReduce());
  EXPECT_EQ(reduced.y(), Fq(2).MontReduce());
}

TEST(AffinePointTest, Serde) {
  AffinePoint expected = AffinePoint::Random();

  base::Uint8VectorBuffer write_buf;
  TF_ASSERT_OK(write_buf.Grow(base::EstimateSize(expected)));
  TF_ASSERT_OK(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  AffinePoint value;
  TF_ASSERT_OK(write_buf.Read(&value));
  EXPECT_EQ(expected, value);
}

TEST(AffinePointTest, SerdeWithGnark) {
  for (size_t i = 0; i < 2; ++i) {
    AffinePointSerdeMode mode = static_cast<AffinePointSerdeMode>(i);
    SCOPED_TRACE(absl::Substitute("mode: $0", i));
    base::AutoReset<AffinePointSerdeMode> auto_reset(
        &base::Serde<AffinePoint>::s_mode, mode);
    base::AutoReset<bool> auto_reset2(&base::Serde<Fq>::s_is_in_montgomery,
                                      false);

    AffinePoint expected = AffinePoint::Random();

    base::Uint8VectorBuffer write_buf;
    write_buf.set_endian(base::Endian::kBig);
    TF_ASSERT_OK(write_buf.Grow(base::EstimateSize(expected)));
    TF_ASSERT_OK(write_buf.Write(expected));
    ASSERT_TRUE(write_buf.Done());

    write_buf.set_buffer_offset(0);

    AffinePoint value;
    TF_ASSERT_OK(write_buf.Read(&value));
    EXPECT_EQ(expected, value);
  }
}

TEST(AffinePointTest, JsonSerde) {
  rapidjson::Document doc;

  AffinePoint expected = AffinePoint::Random();
  rapidjson::Value json_value =
      base::JsonSerde<AffinePoint>::From(expected, doc.GetAllocator());
  EXPECT_TRUE(json_value.IsObject());
  EXPECT_EQ(Fq::FromUnchecked(json_value["x"].GetInt()), expected.x());
  EXPECT_EQ(Fq::FromUnchecked(json_value["y"].GetInt()), expected.y());

  TF_ASSERT_OK_AND_ASSIGN(AffinePoint value,
                          base::JsonSerde<AffinePoint>::To(json_value, ""));
  EXPECT_EQ(expected, value);
}

}  // namespace zkx::math::test
