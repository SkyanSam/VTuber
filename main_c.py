#import pyximport
#import numpy as np
#pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)
import vtuber_c
vtuber_c.main()
