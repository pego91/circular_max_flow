import numpy as np
from numba import jit, prange, set_num_threads,config
from diffusion_utils import dist_from_pts_periodic_boundaries
from itertools import chain, combinations, product
import multiprocessing as mp
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from reeb_aux import make_neigh

def reeb_wrap(LIST):
    
    from reeb_graph import Reeb_Graph
    inputfile,box,r,n_grid,V,fat,covering,relax=LIST 
    
    reeb = Reeb_Graph(inputfile, box = box, sign_var = r,
                            grid_size = n_grid, axes = [0,1], dim = 3,  
                            simply_connected_top_bottom = True,
                            fat_radius = fat,
                            covering = covering,
                            transform_points = V,
                            relax_z_axis = relax,
                            verbose = False)

    reeb.make_reeb_graph(plot=False)
    reeb.compute_max_flow()
    
    return reeb


def make_conn_comp_pool(LVLs,graph):
    
#    pool = mp.Pool(processes=np.min([mp.cpu_count(),20]))
    pool = mp.Pool(processes=mp.cpu_count())

    RES = pool.map(connected_components_pool,
                     ([graph,lvl] 
                             for lvl in LVLs))
    pool.close()
    
    CONN = np.zeros_like(LVLs)-1
    N_CONN = np.zeros_like(LVLs[:,0])
    
    for i,fat_lvl in enumerate(LVLs):
        fat_lvl = fat_lvl>0
        n_C,C = RES[i]
        CONN[i,fat_lvl] = C
        N_CONN[i]=n_C
    
    return CONN, N_CONN


def connected_components_pool(LIST):
    graph,lvl = LIST
    lvl = lvl>0
    return connected_components(graph[:,lvl][lvl,:])


@jit(nopython=True, parallel=True, fastmath=True)
def make_fat_lvls(f,values,u,l,delta,reeb_size):

    FAT = np.zeros((reeb_size,len(f)),dtype=np.int32)
    
    for i in prange(len(values)):    
        v = values[i]
        sub_lvl = (f<=v+u*delta)
        sup_lvl = (f>=v+l*delta)
        fat_lvl = np.multiply(sub_lvl,sup_lvl)
        
        FAT[i,:] = fat_lvl 

    return FAT

def make_conn_comp(LVLs,graph):
    
    CONN = np.zeros_like(LVLs)-1
    N_CONN = np.zeros_like(LVLs[:,0])
    
    for i,fat_lvl in enumerate(LVLs):
        fat_lvl = fat_lvl>0
        n_C,C = connected_components(graph[:,fat_lvl][fat_lvl,:])
        CONN[i,fat_lvl] = C
        N_CONN[i]=n_C
    
    return CONN, N_CONN


def reeb_graph_aux(grid, graph, axis=-1, stride = 2, covering = np.array([-1,1]), MP=True):

    g = lambda x: len(x)

    f = grid[:,axis]
    values = np.unique(f)
    l,u = covering
    d_z = values[1]-values[0]
    
    print('Faccio delta')
    delta = make_neigh(grid, graph.indptr, graph.indices,f)

    """
    delta -> di quanto cambia la funzione in un intorno di un punto: devo essere sicuro che tutti i punti di un intorno 
             di un punto di un aperto del ricoprimento, stiamo o nell'aperto precedente, o nel successivo. 
             Quindi delta e' il raggio degli aperti del ricoprimento.
    norm ->  il diametro di un aperto del ricoprimento é DELTA*sum(covering). Calcolando quante volte ci sta un passo della
             griglia lungo z in un aperto del ricoprimento ottengo da quanti "PIANI" é composta la "TORTA" dell intersez di 
             f^-1. 
             Devo aggiungere 1 perché il piano 0 conta. Quindi prendo il numero medio di punti per piano. Con minimo 1.
    """

    delta = 1.001*delta        
    norm = ((delta*np.max(covering))//(d_z*stride)) + 1

    """
    Tengo conto dello stride
    """
    reeb_size = ((values.shape[0]-1)//stride)*stride
    reeb_idxs = np.arange(0,reeb_size+1,stride)        
    reeb_idxs = np.concatenate([reeb_idxs, 
                                np.arange(reeb_idxs[-1]+1,values.shape[0],1)])
    reeb_size = len(reeb_idxs)

    """
    Faccio il reeb
    """
    print('Faccio LVLs Pool')
    reeb_values = values[reeb_idxs]   
    LVLs = make_fat_lvls(f,reeb_values,u,l,delta,reeb_size)

    if MP:     
        CONN, N_CONN = make_conn_comp_pool(LVLs,graph)
    else:
        CONN, N_CONN = make_conn_comp(LVLs,graph)

    print('Faccio Reeb')
    REEB_GRAPH, weights, fun, emb = make_reeb_numba(CONN, N_CONN, grid, reeb_size, norm)

    return REEB_GRAPH, weights, fun, emb, delta, norm
    

@jit(nopython=True, parallel=True, fastmath=True)    
def make_reeb_numba(CONN, N_CONN, grid, reeb_size, norm):
    
    N = np.sum(N_CONN)
    L = len(N_CONN)
    REEB_GRAPH = -1*np.ones((grid.shape[0],reeb_size),dtype=np.int32)

    fun = np.zeros((N,))
    emb = np.zeros((N,3))
    weights = np.zeros((N,N))    
    count = 0
    
    n_c = N_CONN[0]

    for c in range(n_c):
        tmp = CONN[0,:]==c       
        REEB_GRAPH[tmp,0] = count 
        fun[count] = np.max(np.array([np.rint(np.sum(tmp)/norm),1]))
        emb[count] = np.array([np.mean(grid[tmp][:,ax]) for ax in range(3)])
        count += 1

    for i_ in range(L-1):

        i = i_+1
        n_c = N_CONN[i]
        REEB_GRAPH, weights, fun, emb, count = connect_lvl_sets(i, n_c, grid, norm, CONN, REEB_GRAPH, weights, fun, emb, count)

    return REEB_GRAPH, weights, fun, emb


@jit(nopython=True, parallel=True, fastmath=True)    
def connect_lvl_sets(i, n_c, grid, norm, CONN, REEB_GRAPH, weights, fun, emb, count):
    
    for c in range(n_c):
        tmp = CONN[i,:]==c       
        
        REEB_GRAPH[tmp,i] = count 
        idxs_aux = np.unique(REEB_GRAPH[tmp,i-1])        
        idxs_aux = idxs_aux[idxs_aux>-1]
        fun[count] = np.max(np.array([np.rint(np.sum(tmp)/norm),1]))
        emb[count] =  np.array([np.mean(grid[tmp][:,ax]) for ax in range(3)])

        for j in prange(len(idxs_aux)):
            old_c = idxs_aux[j]
            now = (REEB_GRAPH[:,i] == count)
            old = (REEB_GRAPH[:,i-1] == old_c)
            inters = np.sum(np.multiply(now,old))

            if inters>0:
                weights[old_c,count] = np.max(np.array([inters//norm,1]))
                
        count += 1

    return  REEB_GRAPH, weights, fun, emb, count 

@jit(nopython=True, parallel=True, fastmath=True)    
def connect_lvl_sets_aux(i, n_c, grid, norm, C, REEB_GRAPH,count):
    
    AUX_w = np.zeros((n_c*n_c,3))
    AUX_f = np.zeros((n_c*n_c,))
    AUX_e = np.zeros((n_c*n_c,3)) 
    aux = 0

    for c in range(n_c):
        tmp = np.where(C==c)[0]     
        REEB_GRAPH[tmp,i] = count 
        idxs_aux = np.unique(REEB_GRAPH[tmp,i-1])        
        idxs_aux = idxs_aux[idxs_aux>-1]
        AUX_f[aux] = np.max(np.array([np.rint(np.sum(tmp)/norm),1]))
        AUX_e[aux] =  np.array([np.mean(grid[tmp][:,ax]) for ax in range(3)])

        for j in prange(len(idxs_aux)):
            old_c = idxs_aux[j]
            now = (REEB_GRAPH[:,i] == count)
            old = (REEB_GRAPH[:,i-1] == old_c)
            inters = np.sum(np.multiply(now,old))

            if inters>0:
                AUX_w[aux,:] = [old_c,count,np.max(np.array([inters//norm,1]))]
                aux+=1
        count += 1

    return  REEB_GRAPH, AUX_w, AUX_f, AUX_e, count 



