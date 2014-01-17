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
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <bh.h>

using namespace std;

void pow_to_mul(bh_ir* bhir, bh_node_index idx, vector<bool> &visited, set<bh_node_index> &hits)
{
    visited[idx] = true;    // Update to avoid revisiting this node.

    if ((NODE_LOOKUP(idx).type == BH_INSTRUCTION)) {    // Found one
        bh_instruction *instr = &INSTRUCTION_LOOKUP(NODE_LOOKUP(idx).instruction);
        if ((instr->opcode == BH_POWER) && bh_is_constant(&instr->operand[2])) {
            bool roll = false;
            switch(instr->constant.type) {
                case BH_INT8:
                    roll = instr->constant.value.int8 == 2;
                    break;
                case BH_INT16:
                    roll = instr->constant.value.int16 == 2;
                    break;
                case BH_INT32:
                    roll = instr->constant.value.int32 == 2;
                    break;
                case BH_INT64:
                    roll = instr->constant.value.int64 == 2;
                    break;
                case BH_UINT8:
                    roll = instr->constant.value.uint8 == 2;
                    break;
                case BH_UINT16:
                    roll = instr->constant.value.uint16 == 2;
                    break;
                case BH_UINT32:
                    roll = instr->constant.value.uint32 == 2;
                    break;
                case BH_UINT64:
                    roll = instr->constant.value.uint64 == 2;
                    break;
                case BH_FLOAT16:
                    roll = instr->constant.value.float16 == 2;
                    break;
                case BH_FLOAT32:
                    roll = instr->constant.value.float32 == 2;
                    break;
                case BH_FLOAT64:
                    roll = instr->constant.value.float64 == 2;
                    break;
            }
            if (roll) {
                instr->opcode = BH_MULTIPLY;
                instr->operand[2] = instr->operand[1];
                hits.insert(idx);
            }
        }
    }

    bh_node_index left  = NODE_LOOKUP(idx).left_child;  // Continue
    bh_node_index right = NODE_LOOKUP(idx).right_child;
    if ((left!=INVALID_NODE) && (!visited[left])) {
        pow_to_mul(bhir, left, visited, hits);
    }
    if ((right!=INVALID_NODE) && (!visited[right])) {
        pow_to_mul(bhir, right, visited, hits);
    }
}

/**
 *  Replace all POW(x,y,2) instructions with MUL(x,y,y).
 */
void power_filter(bh_ir* bhir)
{
    set<bh_node_index> hits;
    vector<bool> visited(bhir->nodes->count, false);

    cout << "### Power-filter searching through " << bhir->nodes->count << " nodes." << endl;
    pow_to_mul(bhir, 0, visited, hits);
    cout << "## transformed " << hits.size() << " instructions." << endl;

    if (hits.size()>0) {
        cout << "## [";
        bool first = true;
        for(set<bh_node_index>::iterator it=hits.begin(); it != hits.end(); ++it) {
            if (!first) {
                cout << ", ";
            }
            first = false;
            std::cout << *it;
        }
        std::cout << "]" << std::endl;
    }

    cout << "###" << endl;
}
