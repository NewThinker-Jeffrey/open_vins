/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2022 Patrick Geneva
 * Copyright (C) 2018-2022 Guoquan Huang
 * Copyright (C) 2018-2022 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef OV_CORE_RANSAC_HELPER_H
#define OV_CORE_RANSAC_HELPER_H

#include <random>
#include <algorithm>

namespace ov_core {

inline std::vector<size_t> select_samples(size_t n_total, size_t n_select) {
  std::vector<size_t> selected_indices;
  selected_indices.reserve(n_select);
  std::random_device rd;
  std::mt19937 generator(rd()); 

  std::map<size_t, size_t> idx_to_i;
  for (size_t i=0; i<n_select; i++) {
    size_t idx = i + generator() % (n_total - i);
    if (idx_to_i.count(idx)) {
      idx = idx_to_i[idx];
    }
    idx_to_i[idx] = i;
  }
  for (auto item : idx_to_i) {
    selected_indices.push_back(item.first);
  }
  return selected_indices;
}


} // namespace ov_core

#endif /* OV_CORE_RANSAC_HELPER_H */