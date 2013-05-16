import bohrium as np
import bohrium.examples.sor as sor
import util

B = util.Benchmark()
H = B.size[0]
W = B.size[1]
I = B.size[2]

ft = sor.freezetrap(H,W,dtype=B.dtype,bohrium=B.bohrium)

B.start()
ft = sor.solve(ft,max_iterations=I)
r = np.add.reduce(np.add.reduce(ft[0] + ft[1] + ft[2] + ft[3]))
B.stop()
B.pprint()
