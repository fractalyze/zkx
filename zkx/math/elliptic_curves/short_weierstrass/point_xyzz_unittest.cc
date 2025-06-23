#include "zkx/math/elliptic_curves/short_weierstrass/point_xyzz.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/status.h"
#include "zkx/base/buffer/vector_buffer.h"
#include "zkx/math/elliptic_curves/short_weierstrass/test/sw_curve_config.h"

namespace zkx::math::test {

TEST(PointXyzzTest, Zero) {
  EXPECT_TRUE(PointXyzz(1, 2, 0, 0).IsZero());
  EXPECT_FALSE(PointXyzz(1, 2, 1, 0).IsZero());
  EXPECT_TRUE(PointXyzz(1, 2, 0, 1).IsZero());
}

TEST(PointXyzzTest, One) {
  auto generator = PointXyzz::Generator();
  EXPECT_EQ(generator, PointXyzz(PointXyzz::Curve::Config::kX,
                                 PointXyzz::Curve::Config::kY, 1, 1));
  EXPECT_EQ(PointXyzz::Generator(), PointXyzz::One());
  EXPECT_TRUE(generator.IsOne());
}

TEST(PointXyzzTest, EqualityOperations) {
  {
    SCOPED_TRACE("p.IsZero() && p2.IsZero()");
    PointXyzz p(1, 2, 0, 0);
    PointXyzz p2(3, 4, 0, 0);
    EXPECT_EQ(p, p2);
    EXPECT_EQ(p2, p);
  }

  {
    SCOPED_TRACE("!p.IsZero() && p2.IsZero()");
    PointXyzz p(1, 2, 1, 0);
    PointXyzz p2(3, 4, 0, 0);
    EXPECT_NE(p, p2);
    EXPECT_NE(p2, p);
  }

  {
    SCOPED_TRACE("other");
    PointXyzz p(1, 2, 2, 6);
    PointXyzz p2(1, 2, 2, 6);
    EXPECT_EQ(p, p2);
    EXPECT_EQ(p2, p);
  }
}

TEST(PointXyzzTest, GroupOperations) {
  PointXyzz p(5, 5, 1, 1);
  PointXyzz p2(3, 2, 1, 1);
  PointXyzz p3(3, 5, 1, 1);
  PointXyzz p4(6, 5, 1, 1);
  TF_ASSERT_OK_AND_ASSIGN(AffinePoint ap, p.ToAffine());
  TF_ASSERT_OK_AND_ASSIGN(AffinePoint ap2, p2.ToAffine());

  EXPECT_EQ(p + p2, p3);
  EXPECT_EQ(p - p3, -p2);
  EXPECT_EQ(p + p, p4);
  EXPECT_EQ(p - p4, -p);

  {
    PointXyzz p_tmp = p;
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

  EXPECT_EQ(-p, PointXyzz(5, 2, 1, 1));

  EXPECT_EQ(p * 2, p4);
  EXPECT_EQ(Fr(2) * p, p4);
  EXPECT_EQ(p *= 2, p4);
}

TEST(PointXyzzTest, CyclicScalarMul) {
  std::vector<AffinePoint> points;
  for (size_t i = 0; i < 7; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(AffinePoint ap,
                            (Fr(i) * PointXyzz::Generator()).ToAffine());
    points.push_back(ap);
  }

  EXPECT_THAT(points,
              testing::UnorderedElementsAreArray(std::vector<AffinePoint>{
                  AffinePoint(0, 0), AffinePoint(3, 2), AffinePoint(5, 2),
                  AffinePoint(6, 2), AffinePoint(3, 5), AffinePoint(5, 5),
                  AffinePoint(6, 5)}));
}

TEST(PointXyzzTest, ToAffine) {
  PointXyzz p(1, 2, 0, 0);
  PointXyzz p2(1, 2, 1, 1);
  PointXyzz p3(1, 2, 2, 6);
  TF_ASSERT_OK_AND_ASSIGN(AffinePoint ap, p.ToAffine());
  TF_ASSERT_OK_AND_ASSIGN(AffinePoint ap2, p2.ToAffine());
  TF_ASSERT_OK_AND_ASSIGN(AffinePoint ap3, p3.ToAffine());
  EXPECT_EQ(ap, AffinePoint::Zero());
  EXPECT_EQ(ap2, AffinePoint(1, 2));
  EXPECT_EQ(ap3, AffinePoint(4, 5));
}

TEST(PointXyzzTest, ToJacobian) {
  auto p = PointXyzz::Random();
  EXPECT_EQ(p.ToJacobian(), p.ToAffine()->ToJacobian());
}

TEST(PointXyzzTest, BatchToAffine) {
  std::vector<PointXyzz> point_xyzzs = {
      PointXyzz(1, 2, 0, 0),
      PointXyzz(1, 2, 1, 1),
      PointXyzz(1, 2, 2, 6),
  };

  absl::Span<AffinePoint> affine_points_span;
  ASSERT_FALSE(PointXyzz::BatchToAffine(point_xyzzs, &affine_points_span).ok());

  std::vector<AffinePoint> affine_points;
  ASSERT_TRUE(PointXyzz::BatchToAffine(point_xyzzs, &affine_points).ok());

  std::vector<AffinePoint> expected_affine_points = {
      AffinePoint::Zero(), AffinePoint(1, 2), AffinePoint(4, 5)};
  EXPECT_EQ(affine_points, expected_affine_points);
}

TEST(PointXyzzTest, Serde) {
  PointXyzz expected = PointXyzz::Random();

  base::Uint8VectorBuffer write_buf;
  TF_ASSERT_OK(write_buf.Grow(base::EstimateSize(expected)));
  TF_ASSERT_OK(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  PointXyzz value;
  TF_ASSERT_OK(write_buf.Read(&value));
  EXPECT_EQ(expected, value);
}

TEST(PointXyzzTest, JsonSerde) {
  rapidjson::Document doc;

  PointXyzz expected = PointXyzz::Random();
  rapidjson::Value json_value =
      base::JsonSerde<PointXyzz>::From(expected, doc.GetAllocator());
  EXPECT_TRUE(json_value.IsObject());
  EXPECT_EQ(Fq::FromUnchecked(json_value["x"].GetInt()), expected.x());
  EXPECT_EQ(Fq::FromUnchecked(json_value["y"].GetInt()), expected.y());
  EXPECT_EQ(Fq::FromUnchecked(json_value["zz"].GetInt()), expected.zz());
  EXPECT_EQ(Fq::FromUnchecked(json_value["zzz"].GetInt()), expected.zzz());

  TF_ASSERT_OK_AND_ASSIGN(PointXyzz value,
                          base::JsonSerde<PointXyzz>::To(json_value, ""));
  EXPECT_EQ(expected, value);
}

}  // namespace zkx::math::test
