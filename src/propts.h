/*
 * Copyright Peter G. Jensen
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * File:   propts.h
 * Author: Peter G. Jensen
 *
 * Created on February 14, 2019, 4:11 PM
 */

#ifndef PROPTS_H
#define PROPTS_H

#include <cstddef>
#include <limits>

namespace prlearn {
    // primed for Q-learning

    struct propts_t {
        double _upper_t = 1.75;
        double _lower_t = 0.15;
        double _ks_limit = 0.25;
        // the filter can be disabled (_filter_rate = 1) for MLearning
        // We recommend, however, a high filter-value (_filter_rate = 0.95).
        double _filter_rate = 0.02;
        // Can probably be omitted
        double _filter_val = 0.99;
        double _discount = 0.99;
        double _indefference = 0.005;
    };
}

#endif /* PROPTS_H */

