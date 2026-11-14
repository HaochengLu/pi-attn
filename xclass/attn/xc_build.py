import sys
import xc_lib
xc_lib.build(int(sys.argv[1]), int(sys.argv[2]))
print("built", sys.argv[1:], flush=True)
