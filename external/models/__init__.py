def import_model_by_backend(tensorflow_cmd, pytorch_cmd):
    import sys
    for _backend in sys.modules["external"].backend:
        if _backend == "tensorflow":
            exec(tensorflow_cmd)
        elif _backend == "pytorch":
            exec(pytorch_cmd)
            break


import sys
for _backend in sys.modules["external"].backend:
    if _backend == "tensorflow":
        from .rmg.RMG import RMG
        from .svdpp.SVDpp import SVDpp
    elif _backend == "pytorch":
        from .gat.GAT import GAT
        from .gcmc.GCMC import GCMC
        from .lightgcn.LightGCN import LightGCN
        from .egcf.EGCF import EGCF
        from .roger.RoGER import RoGER
        from .ncf.NCF import NCF
        from .mf.MF import MF
