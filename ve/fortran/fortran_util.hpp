/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the
GNU Lesser General Public License along with Bohrium.

If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __FORTRAN_FORTRAN_UTIL_HPP
#define __FORTRAN_FORTRAN_UTIL_HPP

#include <bh_opcode.h>
#include <jitk/base_db.hpp>

// Return the Fortran reduction symbol
const char* fortran_reduce_symbol(bh_opcode opcode) {
    switch (opcode) {
        case BH_ADD_REDUCE:
            return "+";
        case BH_MULTIPLY_REDUCE:
            return "*";
        case BH_BITWISE_AND_REDUCE:
            return "&";
        case BH_BITWISE_OR_REDUCE:
            return "|";
        case BH_BITWISE_XOR_REDUCE:
            return "^";
        case BH_MAXIMUM_REDUCE:
            return "max";
        case BH_MINIMUM_REDUCE:
            return "min";
        default:
            return NULL;
    }
}

// Is 'opcode' compatible with Fortran reductions such as reduction(+:var)
bool fortran_reduce_compatible(bh_opcode opcode) {
    return fortran_reduce_symbol(opcode) != NULL;
}

// Is the 'block' compatible with Fortran
bool fortran_compatible(const bohrium::jitk::LoopB &block) {
    // For now, all sweeps must be reductions
    for (const bohrium::jitk::InstrPtr instr: block._sweeps) {
        if (not bh_opcode_is_reduction(instr->opcode)) {
            return false;
        }
    }
    return true;
}

// Is the 'block' compatible with Fortran SIMD
bool simd_compatible(const bohrium::jitk::LoopB &block,
                     const bohrium::jitk::Scope &scope) {

    // Check for non-compatible reductions
    for (const bohrium::jitk::InstrPtr instr: block._sweeps) {
        if (not fortran_reduce_compatible(instr->opcode))
            return false;
    }

    // An Fortran SIMD loop does not support ANY Fortran pragmas
    for (bohrium::jitk::InstrPtr instr: block.getAllInstr()) {
        for(const bh_view *view: instr->get_views()) {
            if (scope.isFortranAtomic(*view) or scope.isFortranCritical(*view))
                return false;
        }
    }
    return true;
}

// Does 'opcode' support the Fortran Atomic guard?
bool fortran_atomic_compatible(bh_opcode opcode) {
    switch (opcode) {
        case BH_ADD_REDUCE:
        case BH_MULTIPLY_REDUCE:
        case BH_BITWISE_AND_REDUCE:
        case BH_BITWISE_OR_REDUCE:
        case BH_BITWISE_XOR_REDUCE:
            return true;
        default:
            return false;
    }
}

#endif
