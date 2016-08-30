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

#ifndef __BH_VE_UNI_INSTRUCTION_HPP
#define __BH_VE_UNI_INSTRUCTION_HPP

#include <iostream>

#include <bh_idmap.hpp>
#include <bh_instruction.hpp>

namespace bohrium {

// Write the source code of an instruction
void write_instr(const IdMap<bh_base*> &base_ids, const std::set<bh_base*> &temps, const bh_instruction &instr, std::stringstream &out);

// Return the axis that 'instr' reduces over or 'BH_MAXDIM' if 'instr' isn't a reduction
int sweep_axis(const bh_instruction &instr);

} // bohrium

#endif
