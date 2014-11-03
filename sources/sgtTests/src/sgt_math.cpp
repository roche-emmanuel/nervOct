// each test module could contain no more then one 'main' file with init function defined
// alternatively you could define init function yourself
// #define BOOST_TEST_MODULE "sgtMX Basic Unit tests"

#include <boost/test/unit_test.hpp>

#include <sgtmx.h>

#include <sgt/math.h>

using namespace sgt;

BOOST_AUTO_TEST_SUITE( sgt_math )

BOOST_AUTO_TEST_CASE( test_Vec2 )
{
  logDEBUG("Testing Vec2 types");

  Vec2i veci(1,2);
  Vec2f vecf(1.1f,2.2f);
  Vec2d vecd(1.1,2.2);

  BOOST_CHECK_EQUAL(veci.x(),1);
  BOOST_CHECK_EQUAL(veci.y(),2);
  BOOST_CHECK_EQUAL(vecf.x(),1.1f);
  BOOST_CHECK_EQUAL(vecd.x(),1.1);

  std::ostringstream os;
  os << vecf;
  BOOST_CHECK_EQUAL(os.str(),"Vec2f(1.1, 2.2)");
}

BOOST_AUTO_TEST_CASE( test_Vec3 )
{
  logDEBUG("Testing Vec3 types");

  Vec3i veci(1,2,3);
  Vec3f vecf(1.1f,2.2f,3.3f);
  Vec3d vecd(1.1,2.2,3.3);

  BOOST_CHECK_EQUAL(veci.x(),1);
  BOOST_CHECK_EQUAL(veci.y(),2);
  BOOST_CHECK_EQUAL(vecf.x(),1.1f);
  BOOST_CHECK_EQUAL(vecd.x(),1.1);
}

BOOST_AUTO_TEST_CASE( test_Vec4 )
{
  logDEBUG("Testing Vec4 types");

  Vec4i veci(1,2,3,4);
  Vec4f vecf(1.1f,2.2f,3.3f,4.4f);
  Vec4d vecd(1.1,2.2,3.3,4.4);

  BOOST_CHECK_EQUAL(veci.x(),1);
  BOOST_CHECK_EQUAL(veci.y(),2);
  BOOST_CHECK_EQUAL(vecf.x(),1.1f);
  BOOST_CHECK_EQUAL(vecd.x(),1.1);
}

BOOST_AUTO_TEST_CASE( test_quat )
{
  logDEBUG("Testing Quat");

  Quat q;

  BOOST_CHECK_EQUAL(q.w(),1.0);
}

BOOST_AUTO_TEST_CASE( test_quat_to_string )
{
  logDEBUG("Testing Quat to string");

  Quat q;

  BOOST_CHECK_EQUAL(q.toString(),"Quat(0, 0, 0, 1)");

  q.set(1.0,2.0,3.0,4.1);
  BOOST_CHECK_EQUAL(q.toString(),"Quat(1, 2, 3, 4.1)");
}

BOOST_AUTO_TEST_CASE( test_matrixf_to_string )
{
  logDEBUG("Testing Matrixf to string");

  Matrixf mat = Matrixf::identity();

  BOOST_CHECK_EQUAL(mat.toString(),"Matrixf(\n  1, 0, 0, 0,\n  0, 1, 0, 0,\n  0, 0, 1, 0,\n  0, 0, 0, 1)");
}

BOOST_AUTO_TEST_CASE( test_matrixd_to_string )
{
  logDEBUG("Testing Matrixd to string");

  Matrixd mat = Matrixd::identity();

  BOOST_CHECK_EQUAL(mat.toString(),"Matrixd(\n  1, 0, 0, 0,\n  0, 1, 0, 0,\n  0, 0, 1, 0,\n  0, 0, 0, 1)");
}

BOOST_AUTO_TEST_CASE( test_matrixf )
{
  logDEBUG("Testing Matrixf");

  Matrixf mat = Matrixf::identity();

  BOOST_CHECK_EQUAL(mat(2,2),1.0f);
}

BOOST_AUTO_TEST_CASE( test_matrixd )
{
  logDEBUG("Testing Matrixd");

  Matrixd mat = Matrixd::identity();

  BOOST_CHECK_EQUAL(mat(2,2),1.0);
}




BOOST_AUTO_TEST_SUITE_END()
