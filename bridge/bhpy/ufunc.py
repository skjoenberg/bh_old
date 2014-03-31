#!/usr/bin/env python
"""
/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
http://bohrium.bitbucket.org

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
"""
import _util
import _bh
import array_create
import bhc
import numpy as np
import _info
from _util import dtype_name
from ndarray import get_bhc
import ndarray
from numbers import Number

def assign(a, out):
    out_dtype = dtype_name(out)
    out_bhc = get_bhc(out)
    if np.isscalar(a):
        exec "bhc.bh_multi_array_%s_assign_scalar(out_bhc,a)"%(out_dtype)
    else:
        a_bhc = get_bhc(a)
        a_dtype = dtype_name(a)
        np.broadcast(a,out)#We only do this for the dimension mismatch check
        if out_dtype != a_dtype:
            exec "a_bhc = bhc.bh_multi_array_%s_convert_%s(a_bhc)"%(out_dtype, a_dtype)
        exec "bhc.bh_multi_array_%s_assign_array(out_bhc,a_bhc)"%(out_dtype)

class ufunc:
    def __init__(self, info):
        """A Bohrium Universal Function"""
        self.info = info
    def __str__(self):
        return "<bohrium ufunc '%s'>"%self.info['bhc_name']
    def __call__(self, *args):

        #Check number of arguments
        if len(args) != self.info['nop'] and len(args) != self.info['nop']-1:
            raise ValueError("invalid number of arguments")

        #Check for shape mismatch and get the final output shape
        out_shape = np.broadcast(*args).shape if len(args) > 1 else args[0].shape

        #Pop the output from the 'args' list
        out = None
        args = list(args)
        if len(args) == self.info['nop']:#output given
            out = args.pop()
            if out.shape != out_shape:
                raise ValueError("Could not broadcast to the shape of the output array")

        if len(args) > 2:
            raise ValueError("Bohrium do not support ufunc with more than two inputs")

        #Find the type signature
        (out_dtype,in_dtype) = _util.type_sig(self.info['np_name'], args)

        #Check for Python scalars
        py_scalar = None
        for i, a in enumerate(args):
            if isinstance(a, Number):
                if py_scalar is not None:
                    raise ValueError("Bohrium ufuncs do not support multiple scalar inputs")
                py_scalar = i#The i'th input is a Python scalar

        #Convert 'args' to Bohrium-C arrays
        bhcs = []
        for a in args:
            if isinstance(a, Number):
                bhcs.append(a)
            elif ndarray.check(a):
                bhcs.append(get_bhc(a))
            else:
                bhcs.append(get_bhc(array_create.array(a)))

        #Convert dtype of all inputs
        inputs = []
        for i, a in enumerate(bhcs):
            a_dtype = _util.dtype_from_bhc(a)
            t_dtype = dtype_name(in_dtype)
            if i != py_scalar and a_dtype != t_dtype:
                exec "t = bhc.bh_multi_array_%s_convert_%s(a)"%(t_dtype, a_dtype)
                inputs.append(t)
            else:
                inputs.append(a)

        cmd = "bhc.bh_multi_array_%s_%s"%(dtype_name(in_dtype), self.info['bhc_name'])
        if py_scalar is not None:
            if py_scalar == 0:
                cmd += "_scalar_lhs"
            else:
                cmd += "_scalar_rhs"

        f = eval(cmd)
        ret = f(*inputs)

        if out is None: #Create a new output with the returned Bohrium-C array
            out = _bh.ndarray(out_shape, dtype=out_dtype)
            out.bhc_ary = ret
        else: #We have to use the output given
            if not ndarray.check(out):
                raise NotImplementedError("For now, the output must be a Bohrium array")
            if out.dtype == out_dtype:
                t = ret
            else:
                exec "t = bhc.bh_multi_array_%s_convert_%s(ret)"%(dtype_name(out.dtype), dtype_name(out_dtype))
            #Copy result into the output array
            exec "bhc.bh_multi_array_%s_assign_array(get_bhc(out),t)"%(dtype_name(out_dtype))
        return out

ufuncs = []
for op in _info.op.itervalues():
    ufuncs.append(ufunc(op))


###############################################################################
################################ UNIT TEST ####################################
###############################################################################

import unittest

class Tests(unittest.TestCase):

    def test_assign_copy(self):
        A = array_create.empty((4,4), dtype=int)
        B = array_create.empty((4,4), dtype=int)
        assign(42, A)
        assign(A, B)
        A._data_bhc2np()
        B._data_bhc2np()
        #Compare result to NumPy
        N = np.empty((4,4), dtype=int)
        N[:] = 42
        self.assertTrue(np.array_equal(B,N))
        self.assertTrue(np.array_equal(A,N))

    def test_ufunc(self):
        for f in ufuncs:
            for type_sig in f.info['type_sig']:
                if f.info['bhc_name'] == "assign":
                    continue
                print f, type_sig
                A = array_create.empty((4,4), dtype=type_sig[1])
                if type_sig[1] == "bool":
                    assign(False, A)
                else:
                    assign(2, A)
                if f.info['nop'] == 2:
                    res = f(A)
                elif f.info['nop'] == 3:
                    B = array_create.empty((4,4), dtype=type_sig[2])
                    if type_sig[1] == "bool":
                        assign(True, B)
                    else:
                        assign(3, B)
                    res = f(A,B)
                res._data_bhc2np()
                #Compare result to NumPy
                A = np.empty((4,4), dtype=type_sig[1])
                A[:] = 2
                B = np.empty((4,4), dtype=type_sig[1])
                B[:] = 3
                if f.info['nop'] == 2:
                    exec "np_res = np.%s(A)"%f.info['np_name']
                elif f.info['nop'] == 3:
                    exec "np_res = np.%s(A,B)"%f.info['np_name']
                self.assertTrue(np.allclose(res,np_res))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Tests)
    unittest.TextTestRunner(verbosity=2).run(suite)