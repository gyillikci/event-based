/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV3D_LINE_UTILS_H
#define METAVISION_SDK_CV3D_LINE_UTILS_H

#include <algorithm>
#include <Eigen/Core>

namespace Metavision {

/// @brief Implements Liang-Barsky algorithm for line clipping inside a rectangle window defined diagonally by
/// pmin(xmin, ymin) to pmax(xmax, ymax). p0 and p1 are modified in place to their clipped values if the segment is
/// partially visible
/// @param pmin Lower corner of the viewport
/// @param pmax Higher corner of the viewport
/// @param p0 Head of the segment to be clipped
/// @param p1 Tail of the segment to be clipped
/// @return True if the segment is partially or totally visible in the viewport, false otherwise
bool liang_barsky_line_clip(const Eigen::Vector2f &pmin, const Eigen::Vector2f &pmax, Eigen::Vector2f &p0,
                            Eigen::Vector2f &p1);

/// @brief Cohen–Sutherland line clipping algorithm (from wikipedia)
/// @param pmin Lower corner of the viewport
/// @param pmax Higher corner of the viewport
/// @param p0 Head of the segment to be clipped
/// @param p1 Tail of the segment to be clipped
/// @return True if the segment is partially or totally visible in the viewport, false otherwise
bool cohen_sutherland_line_clip(const Eigen::Vector2f &pmin, const Eigen::Vector2f &pmax, Eigen::Vector2f &p0,
                                Eigen::Vector2f &p1);

template<int N>
using VecN = Eigen::Matrix<float, N, 1, 0, N, 1>; // we use an alias because MSVC fails at recognizing default template
                                                  // arguments during template instantiation

/// @brief Computes the orthogonal projection of a point onto a line represented by two points
/// @tparam N Space dimensions (i.e. 2 or 3)
/// @param point The point to project
/// @param a First endpoint of the segment
/// @param b Second endpoint of the segment
/// @return The scalar \f$t\f$ so that the projected point \f$pp\f$ is \f$pp = a + (b - a) \times t\f$
template<int N>
float project_point_on_line(const VecN<N> &point, const VecN<N> &a, const VecN<N> &b);

/// @brief Computes the distance from a point to a segment (ab). It is the orthogonal distance if the projection of
/// the point falls within the segment.
/// @tparam N Space dimensions (i.e. 2 or 3)
/// @param point The point to consider
/// @param a First endpoint of the segment
/// @param b Second endpoint of the segment
template<int N>
float distance_point_to_segment(const VecN<N> &point, const VecN<N> &a, const VecN<N> &b);

/// @brief Computes the distance from a point to a line. It is the orthogonal distance between the point and the line
/// @tparam N Space dimensions (i.e. 2 or 3)
/// @param point The point to consider
/// @param a First point to define the line
/// @param b Second point to define the line
template<int N>
float distance_point_to_line(const VecN<N> &point, const VecN<N> &a, const VecN<N> &b);

/// @brief Returns true if the 2D line segments (ab) and (cd) intersect. Points a and b are assumed to be different,
/// same for c and d.
/// @throw std::runtime error if a and b (resp. c and d) are not different
bool do_segments_intersect(const Eigen::Vector2f &a, const Eigen::Vector2f &b, const Eigen::Vector2f &c,
                           const Eigen::Vector2f &d);

} // namespace Metavision

#endif // METAVISION_SDK_CV3D_LINE_UTILS_H