#!/usr/bin/env python
# coding: utf-8

import argparse
import math
import os.path
import timeit
import json
import random

import maxflow
import numpy as np
import rasterio
from skimage.filters import sobel
from skimage.morphology import dilation
from skimage.segmentation import watershed
from skimage.segmentation import join_segmentations

#  example:
# python PyMoSP.py -i ParayLeMonial\L1_opi_4i.vrt -s ParayLeMonial\L1_quality_4i.vrt -c 128 -a 256 -o ParayLeMonial\debug.tif

# todo:
#   lecture/ecriture sous forme d'une image de label en couleur (avec table de correspondance OPI -- couleur)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--opis", required=True, help="input image with N channels", type=str)
parser.add_argument("-s", "--scores", required=True, help="input images with socres", type=str)
parser.add_argument("-k", "--constraint", required=False, help="input constraint", type=str)
parser.add_argument("-o", "--outputfile", required=True, help="outputfile", type=str)

parser.add_argument("-a", "--alpha", required=True, help="alpha value", action="extend", nargs="+", type=float)
parser.add_argument("-c", "--cellsize", help="average nb of pixels by SP (default: 512)", type=int, default=512)

parser.add_argument("-w", "--export_ortho", help="debug (default: False)", type=bool, default=False)
parser.add_argument("-d", "--debug", help="debug (default: False)", type=bool, default=False)
parser.add_argument("-v", "--verbose", help="verbose (default: 0)", type=int, default=0)
args = parser.parse_args()
verbose = args.verbose
if verbose > 0:
    print("Arguments: ", args)

def saveImage(S, name, transform=None, crs=None):
    # creation d'une image
    with rasterio.open(name,'w',driver='GTiff',
        height=S.shape[0],
        width=S.shape[1],
        count=1,
        dtype=S.dtype,
        transform=transform,
        crs=crs
    ) as dst:
        dst.write(S, 1)

def superPixels(O, cell_size):
    gradient = np.amax([np.abs(sobel(O[i])) for i in range(O.shape[0])], axis=0)
    regions = watershed(gradient, markers = O.shape[1] * O.shape[2] / cell_size, compactness = 0.001)
    uniques, inverses, counts = np.unique(regions, return_inverse=True, return_counts=True)
    # on filtre les labels avec peu de pixels
    uniques[counts<math.sqrt(cell_size)]=0
    # on applique le filtrage
    S = regions.shape
    regions = uniques[inverses].reshape(S)
    # on comble le noir avec des labels neighbors
    while np.sum(regions < 1) > 0:
        regions=regions + dilation(regions) * (regions < 1)
    # on re-index les labels
    regions = np.unique(regions, return_inverse=True)[1].reshape(S)
    gradient=None
    return regions

def buildGraph(cells):
    # dictionnaire des voisinages (contient la liste des neighbors pour region)
    neighbors = {}
    # contient les index des pixels de chaque region
    inside_indices = {}
    # pour rechercher les neighbors, on compare cells avec une version décalé d'un pixel en colonne et en ligne
    cells_ref = np.delete(np.delete(cells, -1, 0), -1, 1).flatten().tolist()
    cells_col = np.delete(np.delete(cells, 0, 0), -1, 1).flatten().tolist()
    cells_lig = np.delete(np.delete(cells, 0, 1), -1, 0).flatten().tolist()
    # pour les iterations, il est plus rapide de repasser a des listes natives python
    index = 0
    index_border = 0
    borders = []
    for i_ref, i_col, i_lig  in zip(cells_ref, cells_col, cells_lig):
        if not(i_ref in neighbors):
            neighbors[i_ref] = {}
        if not(i_col in neighbors):
            neighbors[i_col] = {}
        if not(i_lig in neighbors):
            neighbors[i_lig] = {}
        if not(i_ref in inside_indices):
            inside_indices[i_ref] = []
        inside_indices[i_ref].append(index)
        if (i_ref != i_col):
            # voisin en colonne
            i1 = min(i_ref, i_col)
            i2 = max(i_ref, i_col)
            # on cherche si c'est une frontiere deja connue
            if not(i2 in neighbors[i1]):
                # c'est une nouvelle frontiere
                neighbors[i1][i2] = index_border
                index_border += 1
                borders.append({'ids':[i1, i2], i1:{'col': [], 'lig':[]}, i2:{'col': [], 'lig':[]}})
            borders[neighbors[i1][i2]][i_ref]['col'].append(index)

        if (i_ref != i_lig):
            # voisin en ligne
            i1 = min(i_ref, i_lig)
            i2 = max(i_ref, i_lig)
            # on cherche si c'est une frontiere deja connue
            if not(i2 in neighbors[i1]):
                # c'est une nouvelle frontiere
                neighbors[i1][i2] = index_border
                index_border += 1
                borders.append({'ids':[i1, i2], i1:{'col': [], 'lig': []}, i2:{'col': [], 'lig': []}})
            borders[neighbors[i1][i2]][i_ref]['lig'].append(index)
        index += 1
    return inside_indices, borders 

def coutInt(scores, constraint, inside_indices):
    with_constraint = not constraint is None
    nb_opis = scores.shape[0]
    nb_regions = len(inside_indices)
    score_int = np.zeros((nb_opis, nb_regions))
    for o in range(nb_opis):
        score_o = score_int[o]
        s = np.delete(np.delete(scores[o], -1, 0), -1, 1).flatten()
        for region in inside_indices:
            R = s[inside_indices[region]]
            mi = np.min(R)
            su = np.sum(R)
            if mi == 0:
                # il y a au moins un pixel interdit, donc le super pixel est interdit
                score_o[region] = 0
            else:
                score_o[region] = su
    if with_constraint:
        k = np.delete(np.delete(constraint, -1, 0), -1, 1).flatten()
        for region in inside_indices:
            val = np.unique(k[inside_indices[region]])
            if len(val) > 1:
                print("Erreur : ", val)
                exit(1)
            if val[0] < nb_opis:
                # valeur contrainte
                for o in range(nb_opis):
                    if o != val[0]:
                        score_int[o][region] = 0
    return score_int

def exportResult_with_ortho(O, regions, result, filename, selected_opi, transform, crs):
    R = regions.flatten().tolist()
    OPI = [O[i].flatten().tolist() for i in range(O.shape[0])]
    out_G = [None] * (O.shape[1] * O.shape[2])
    out_O = [None] * (O.shape[1] * O.shape[2])
    index = 0
    if selected_opi is None:
        for i_R in R:
            o = result[i_R]
            out_G[index] = o
            out_O[index] = OPI[o][index]
            index += 1
    else:
        for i_R in R:
            o = result[i_R]
            out_G[index] = selected_opi[o]
            out_O[index] = OPI[o][index]
            index += 1
    saveImage(np.array(out_G).astype(np.uint8).reshape((O.shape[1], O.shape[2])), filename+"_graph.tif", transform, crs)
    saveImage(np.array(out_O).astype(np.uint8).reshape((O.shape[1], O.shape[2])), filename+"_ortho.tif", transform, crs)

def exportResult_without_ortho(shape, regions, result, filename, selected_opi, transform, crs):
    R = regions.flatten().tolist()
    out_G = [None] * (shape[0] * shape[1])
    index = 0
    if selected_opi is None:
        for i_R in R:
            o = result[i_R]
            out_G[index] = o
            index += 1
    else:
        for i_R in R:
            o = result[i_R]
            out_G[index] = selected_opi[o]
            index += 1
    saveImage(np.array(out_G).astype(np.uint8).reshape((shape[0], shape[1])), filename+".tif", transform, crs)

def coutTrans_diffsimple(O, borders, indices):
    nb_opis = O.shape[0]
    img_ref = [np.delete(np.delete(O[o], -1, 0), -1, 1).flatten() for o in range(nb_opis)]
    # img_col = [np.delete(np.delete(O[o], 0, 0), -1, 1).flatten() for o in range(nb_opis)] 
    # img_lig = [np.delete(np.delete(O[o], 0, 1), -1, 0).flatten() for o in range(nb_opis)] 
    # nb_edges = len(borders)
    for b in range(len(borders)):
        border = borders[b]
        c1 = border['ids'][0]
        c2 = border['ids'][1]
        score_trans = np.zeros(indices[-1][-2] + 1)
        # liste des pixels du cote c1 en colonne et en ligne
        l1 = np.concatenate((border[c1]['col'], border[c1]['lig'])).astype('int64')
        # liste des pixels du cote c2 en colonne et en ligne
        l2 = np.concatenate((border[c2]['col'], border[c2]['lig'])).astype('int64')
        # print(l1, l2)
        # on fait la différence des deux cotes
        l = np.concatenate((l1, l2)).astype('int64')
        # print(l)
        for i in range(nb_opis):
            for j in range(i+1, nb_opis):
                diff = np.sum(np.absolute(img_ref[i][l].astype('int64') - img_ref[j][l].astype('int64')))
                # print(l, img_ref[i][l].astype('int64'), img_ref[j][l].astype('int64'), np.absolute(img_ref[i][l].astype('int64') - img_ref[j][l].astype('int64')), diff)
                score_trans[indices[i][j]] = diff
        border['cout'] = score_trans
    return borders

def coutTrans_diffnorm(O, borders, indices):
    nb_opis = O.shape[0]
    img_ref = [np.delete(np.delete(O[o], -1, 0), -1, 1).flatten() for o in range(nb_opis)]
    img_col = [np.delete(np.delete(O[o], 0, 0), -1, 1).flatten() for o in range(nb_opis)] 
    img_lig = [np.delete(np.delete(O[o], 0, 1), -1, 0).flatten() for o in range(nb_opis)] 
    # nb_edges = len(borders)
    for b in range(len(borders)):
        border = borders[b]
        c1 = border['ids'][0]
        c2 = border['ids'][1]
        score_trans = np.zeros(indices[-1][-2] + 1)
        list_c1_col = border[c1]['col']
        list_c2_col = border[c2]['col']
        list_c1_lig = border[c1]['lig']
        list_c2_lig = border[c2]['lig']
        # on fait la différence des deux cotes
        l = np.concatenate((np.concatenate((list_c1_col, list_c1_lig)), np.concatenate((list_c2_col, list_c2_lig)))).astype('int64')
        for i in range(nb_opis):
            for j in range(i+1, nb_opis):
                diff = np.sum(np.absolute(img_ref[i][l].astype('int64') - img_ref[j][l].astype('int64')))
                ct_i = np.sum(np.concatenate((    np.concatenate((np.absolute(img_ref[i][list_c1_col].astype('int64') - img_col[i][list_c1_col].astype('int64')),
                                                    np.absolute(img_ref[i][list_c1_lig].astype('int64') - img_lig[i][list_c1_lig].astype('int64')))),
                                    np.concatenate((np.absolute(img_ref[i][list_c2_col].astype('int64') - img_col[i][list_c2_col].astype('int64')),
                                                    np.absolute(img_ref[i][list_c2_lig].astype('int64') - img_lig[i][list_c2_lig].astype('int64'))))
                                )))
                ct_j = np.sum(np.concatenate((    np.concatenate((np.absolute(img_ref[j][list_c1_col].astype('int64') - img_col[j][list_c1_col].astype('int64')),
                                                    np.absolute(img_ref[j][list_c1_lig].astype('int64') - img_lig[j][list_c1_lig].astype('int64')))),
                                    np.concatenate((np.absolute(img_ref[j][list_c2_col].astype('int64') - img_col[j][list_c2_col].astype('int64')),
                                                    np.absolute(img_ref[j][list_c2_lig].astype('int64') - img_lig[j][list_c2_lig].astype('int64'))))
                                )))
                # print(l, img_ref[i][l].astype('int64'), img_ref[j][l].astype('int64'), np.absolute(img_ref[i][l].astype('int64') - img_ref[j][l].astype('int64')), diff)
                score_trans[indices[i][j]] = diff / max(1, min(ct_i, ct_j))
                # score_trans[indices[i][j]] = diff / max(1, ct_i)
        border['cout'] = score_trans
    return borders

def coutTrans(O, borders, indices):
    nb_opis = O.shape[0]
    img_ref = [np.delete(np.delete(O[o], -1, 0), -1, 1).flatten() for o in range(nb_opis)]
    img_col = [np.delete(np.delete(O[o], 0, 0), -1, 1).flatten() for o in range(nb_opis)] 
    img_lig = [np.delete(np.delete(O[o], 0, 1), -1, 0).flatten() for o in range(nb_opis)] 
    nb_edges = len(borders)
    for b in range(len(borders)):
        border = borders[b]
        c1 = border['ids'][0]
        c2 = border['ids'][1]
        score_trans = np.zeros(indices[-1][-2] + 1)
        list_c1_col = border[c1]['col']
        list_c2_col = border[c2]['col']
        list_c1_lig = border[c1]['lig']
        list_c2_lig = border[c2]['lig']
        for i in range(nb_opis):
            for j in range(i+1, nb_opis):

	# float contraste = std::max(std::abs(_radioms(i1, j1, b1) - _radioms(i2, j2, b1)),
	# 						   std::abs(_radioms(i1, j1, b2) - _radioms(i2, j2, b2)));
	# float difference = std::min(std::abs(_radioms(i1, j1, b1) - _radioms(i1, j1, b2)),
	# 							std::abs(_radioms(i2, j2, b1) - _radioms(i2, j2, b2)));
	# float cout = 1.f;
	# if (contraste > 0.f)
	# {
	# 	cout = std::min(1.f, (difference / contraste) / 5.f); // todo Q:
	# }
	# else
	# {
	# 	cout = std::min(1.f, difference / 5.f); // todo Q:
	# }
	# return cout;


                i_ref_c1_col = img_ref[i][list_c1_col].astype('int64')
                i_ref_c2_col = img_ref[i][list_c2_col].astype('int64')
                i_ref_c1_lig = img_ref[i][list_c1_lig].astype('int64')
                i_ref_c2_lig = img_ref[i][list_c2_lig].astype('int64')

                j_ref_c1_col = img_ref[j][list_c1_col].astype('int64')
                j_ref_c2_col = img_ref[j][list_c2_col].astype('int64')
                j_ref_c1_lig = img_ref[j][list_c1_lig].astype('int64')
                j_ref_c2_lig = img_ref[j][list_c2_lig].astype('int64')

                i_col_c1_col = img_col[i][list_c1_col].astype('int64')
                i_col_c2_col = img_col[i][list_c2_col].astype('int64')
                i_lig_c1_lig = img_lig[i][list_c1_lig].astype('int64')
                i_lig_c2_lig = img_lig[i][list_c2_lig].astype('int64')

                j_col_c1_col = img_col[j][list_c1_col].astype('int64')
                j_col_c2_col = img_col[j][list_c2_col].astype('int64')
                j_lig_c1_lig = img_lig[j][list_c1_lig].astype('int64')
                j_lig_c2_lig = img_lig[j][list_c2_lig].astype('int64')

                # print(i_ref_c1_col, i_col_c1_col, j_ref_c1_col, j_col_c1_col)
                # print(np.absolute(i_ref_c1_col - i_col_c1_col), np.absolute(j_ref_c1_col - j_col_c1_col))
                # print(np.maximum(np.absolute(i_ref_c1_col - i_col_c1_col), np.absolute(j_ref_c1_col - j_col_c1_col)))
                # print(np.sum(np.maximum(np.absolute(i_ref_c1_col - i_col_c1_col), np.absolute(j_ref_c1_col - j_col_c1_col))))

                contraste = np.sum(np.maximum(np.absolute(i_ref_c1_col - i_col_c1_col), np.absolute(j_ref_c1_col - j_col_c1_col))) +\
                            np.sum(np.maximum(np.absolute(i_ref_c1_lig - i_lig_c1_lig), np.absolute(j_ref_c1_lig - j_lig_c1_lig))) +\
                            np.sum(np.maximum(np.absolute(i_ref_c2_col - i_col_c2_col), np.absolute(j_ref_c2_col - j_col_c2_col))) +\
                            np.sum(np.maximum(np.absolute(i_ref_c2_lig - i_lig_c2_lig), np.absolute(j_ref_c2_lig - j_lig_c2_lig)))
                difference = np.sum(np.minimum(np.absolute(i_ref_c1_col - j_ref_c1_col), np.absolute(i_col_c1_col - j_col_c1_col))) +\
                            np.sum(np.minimum(np.absolute(i_ref_c1_lig - j_ref_c1_lig), np.absolute(i_lig_c1_lig - j_lig_c1_lig))) +\
                            np.sum(np.minimum(np.absolute(i_ref_c2_col - j_ref_c2_col), np.absolute(i_col_c2_col - j_col_c2_col))) +\
                            np.sum(np.minimum(np.absolute(i_ref_c2_lig - j_ref_c2_lig), np.absolute(i_lig_c2_lig - j_lig_c2_lig)))
                
                if contraste > 0.:
                    score_trans[indices[i][j]] = min(1., (difference/contraste))
                else:
                    score_trans[indices[i][j]] = min(1., difference)
        border['cout'] = score_trans
    return borders

def optAlpha(alpha, score_int, borders, result, nb_nodes, indices, coeff_alpha):
    Emax = 1000000
    nb_edges = len(borders)
    # cout_min = 100 * coeff_alpha
    # alpha expension pour cette OPI
    # etat 0 : on garde la solution actuelle
    # etat 1 : on bascule sur l'OPI alpha
    print('creation du graphe...')
    G = maxflow.Graph[float](nb_nodes, nb_edges)
    N = G.add_nodes(nb_nodes)
    # capacité vers S
    eS = np.zeros(nb_nodes, dtype=float)
    # capacité vers T
    eT = np.zeros(nb_nodes, dtype=float)
    # pour chaque voisinage
    for e in range(nb_edges):
        border = borders[e]
        c1 = border['ids'][0]
        c2 = border['ids'][1]
        # on recupere les labels dans la solution initiale
        opi_c1 = result[c1]
        opi_c2 = result[c2]
        E00 = 0
        if not(opi_c1 == opi_c2):
            E00 += border['cout'][indices[opi_c1][opi_c2]]
        E01 = 0
        if not(alpha == opi_c1):
            E01 += border['cout'][indices[alpha][opi_c1]]
        E10 = 0
        if not(alpha == opi_c2):
                E10 += border['cout'][indices[alpha][opi_c2]]
        E11 = 0
        # if test_c1 == False or test_c2 == False:
        #     print(test_c1, test_c2, score_int[alpha][c1], score_int[alpha][c2], E00, E10, E01, E11, c1, c2, opi_c1, opi_c2, alpha)
        # le terme C-A de l'article (à savoir E(1,0)-E(0,0)) qui est a mettre entre le noeud c1 et T/S
        # c'est le cout lie au passage de c1 à alpha en laissant c2 inchangé
        CA = E10-E00
        # le terme D-C de l'article (à savoir E(1,1)-E(1,0)) qui est a mettre entre le noeud c2 et T/S
        # c'est le cout lie au passage de c2 à alpha avec c1 également à alpha
        DC = E11-E10
        BCAD = E01+E10-E00-E11
        if (BCAD<0):
            # print("Attention: ", CA, DC, BCAD)
            CA -= BCAD
            DC -= BCAD
            BCAD = 0
        # print(CA>0, DC>0, CA, DC, BCAD)
        if CA>=0 and DC<=0:
            G.add_edge(N[c1], N[c2], coeff_alpha*BCAD, 0)
            eS[c1] += coeff_alpha*CA
            eT[c2] -= coeff_alpha*DC
        elif CA<=0 and DC>=0:
            G.add_edge(N[c2], N[c1], coeff_alpha*BCAD, 0)
            eT[c1] -= coeff_alpha*CA
            eS[c2] += coeff_alpha*DC
    # pour chaque noeud
    for n in range(nb_nodes):
        # on verifie si le label alpha est autorise pour ces deux noeuds
        # dans le cas OptAE, si aucune image n'est valide, on laisse le label initial (celui du maxOfScore)
        valid = score_int[alpha][n] > 0
        if valid:
            eS[n] += score_int[result[n]][n]
            eT[n] += score_int[alpha][n]
            G.add_tedge(N[n], eS[n] , eT[n])
        else:
            G.add_tedge(N[n], Emax , 0)
    print('optimisation...')
    flow = G.maxflow()
    print(flow)
    print('mise a jour...')
    nb_update=0
    for n in range(nb_nodes):
        if G.get_segment(N[n])>0:
            result[n]=alpha
            nb_update+=1
    print('nombre de changements: ', nb_update)
    return result, nb_update

def optBinaire(score_int, borders, nb_nodes, coeff_alpha):
    Emax = 1000000
    nb_edges = len(borders)
    # alpha expension pour cette OPI
    # etat 0 : opi 0 
    # etat 1 : opi 1
    print('creation du graphe...')
    G = maxflow.Graph[float](nb_nodes, nb_edges)
    N = G.add_nodes(nb_nodes)
    # capacité vers S
    eS = np.zeros(nb_nodes, dtype=float)
    # capacité vers T
    eT = np.zeros(nb_nodes, dtype=float)
    # pour chaque voisinage
    for e in range(nb_edges):
        border = borders[e]
        c1 = border['ids'][0]
        c2 = border['ids'][1]
        E00 = 0
        E01 = border['cout'][0]
        E10 = E01
        E11 = 0
        # le terme C-A de l'article (à savoir E(1,0)-E(0,0)) qui est a mettre entre le noeud c1 et T/S
        # c'est le cout lie au passage de c1 à alpha en laissant c2 inchangé
        CA = E10-E00
        # le terme D-C de l'article (à savoir E(1,1)-E(1,0)) qui est a mettre entre le noeud c2 et T/S
        # c'est le cout lie au passage de c2 à alpha avec c1 également à alpha
        DC = E11-E10
        BCAD = E01+E10-E00-E11
        if (BCAD<0):
            CA -= BCAD
            DC -= BCAD
            BCAD = 0
        if CA>=0 and DC<=0:
            G.add_edge(N[c1], N[c2], coeff_alpha*BCAD, 0)
            eS[c1] += coeff_alpha*CA
            eT[c2] -= coeff_alpha*DC
        elif CA<=0 and DC>=0:
            G.add_edge(N[c2], N[c1], coeff_alpha*BCAD, 0)
            eT[c1] -= coeff_alpha*CA
            eS[c2] += coeff_alpha*DC
    # pour chaque noeud
    for n in range(nb_nodes):
        # dans le cas binaire, si aucune image n'est valide, on prend la premiere
        valid = score_int[1][n] > 0
        if valid:
            eS[n] += score_int[0][n]
            eT[n] += score_int[1][n]
            G.add_tedge(N[n], eS[n] , eT[n])
        else:
            G.add_tedge(N[n], Emax , 0)
    print('optimisation...')
    flow = G.maxflow()
    print(flow)
    print('mise a jour...')
    nb_update=0
    result = np.zeros(nb_nodes, dtype=int)
    for n in range(nb_nodes):
        if G.get_segment(N[n])>0:
            result[n]=1
            nb_update+=1
    print('nombre de changements: ', nb_update)
    return result

def main():
    # filtrage en amont avec un maxOfScore pour gagner du temps
    print("choix des OPI avec un MaxOfScore...")
    t1 = timeit.default_timer()
    scores = rasterio.open(args.scores)
    S = scores.read()
    selected_opi  = np.unique(S.argmax(axis=0))
    t2 = timeit.default_timer()
    print('temps de traitement -- : ', t2-t1, 's')

    print('opi selectionnees: ', selected_opi)
    nb_opis = len(selected_opi)
    print('nb opis: ', nb_opis)
    S = S[selected_opi]
    
    ortho = rasterio.open(args.opis)
    O = ortho.read(selected_opi.tolist())

    shape=O[0].shape

    # creation des super pixels
    print('creation des super-pixels...')
    t1 = timeit.default_timer()
    regions = superPixels(O, args.cellsize)
    K=None
    if args.constraint:
        print('prise en compte des contraintes pour les super pixels...')
        constraint = rasterio.open(args.constraint)
        K = constraint.read()[0]
        # pour un test
        # K[512:1536, 512:1536] = 10
        # saveImage(K, os.path.splitext(args.outputfile)[0]+"_cst.tif", ortho.transform, ortho.crs)
        regions = join_segmentations(regions, K)
        regions = np.unique(regions, return_inverse=True)[1].reshape(regions.shape)
    nb_regions = np.max(regions) + 1
    print('nb regions : ', nb_regions)
    t2 = timeit.default_timer()
    print('temps de traitement -- : ', t2-t1, 's')
    if verbose > 0:
        saveImage(np.mod(np.absolute(regions), np.iinfo(np.uint16).max).astype(np.uint16),
            os.path.splitext(args.outputfile)[0]+"_sp.tif", ortho.transform, ortho.crs)

    # pour un test
    # pour un test on interdit une zone sur l'une des OPI
    # S[1, 100:500, 100:500] = 0

    print("recherche des voisinages...")
    t1 = timeit.default_timer()
    inside_indices, borders = buildGraph(regions)
    t2 = timeit.default_timer()
    print('temps de traitement -- : ', t2-t1, 's')

    nb_nodes = nb_regions
    nb_edges = len(borders)
    print('taille du graphe : ', nb_nodes, nb_edges)

    print("calcul des couts int...")
    t1 = timeit.default_timer()
    score_int = coutInt(S, K, inside_indices)
    t2 = timeit.default_timer()
    print('temps de traitement -- : ', t2-t1, 's')
    # on n'a plus besoin des images de qualité
    S = None

    print("max of score...")
    t1 = timeit.default_timer()
    result = score_int.argmax(axis=0)
    t2 = timeit.default_timer()
    print('temps de traitement -- : ', t2-t1, 's')
    
    if verbose > 0:
        print("export...")
        t1 = timeit.default_timer()
        exportResult_with_ortho(O, regions, result, os.path.splitext(args.outputfile)[0]+"_mos", None, ortho.transform, ortho.crs)
        t2 = timeit.default_timer()
        print('temps de traitement -- : ', t2-t1, 's')
    
    # on supprime les OPI qui ne sont pas presentes dans le MaxOfScore
    # selected_opi, result = np.unique(result, return_inverse=True)
    # print('liste des OPI selectionnees : ', selected_opi)
    # nb_opis = len(selected_opi)
    # O = O[selected_opi]
    # S = S[selected_opi]
    # score_int = score_int[selected_opi]

    print("calcul des couts trans...")
    t1 = timeit.default_timer()
    # construction d'une table pour indicer les couts de trans 
    indices = []
    index = 0
    for i in range(nb_opis):
        I = []
        for j in range(nb_opis):
            if j <= i:
                I.append(None)
            else:
                I.append(index)
                index += 1
        indices.append(I)
    for i in range(nb_opis):
        for j in range(nb_opis):
            if j < i:
                indices[i][j] = indices[j][i]
    # borders = coutTrans_diffnorm(O, borders, indices)
    borders = coutTrans_diffsimple(O, borders, indices)
    t2 = timeit.default_timer()
    print('temps de traitement -- : ', t2-t1, 's')

    if verbose > 1:
        export_borders = []
        for b in range(len(borders)):
            border = borders[b]
            c1 = border['ids'][0]
            c2 = border['ids'][1]
            export_borders.append({'id': b, 'c1': c1, 'c2': c2, 'cout': border['cout'].tolist()})
        with open(os.path.splitext(args.outputfile)[0]+"_borders.json", "w") as out:
            json.dump(export_borders, out)
    else:
        # on n'a plus besoin des OPI
        if not args.export_ortho:
            O=None

    # optimisation
    for coeff_alpha in args.alpha:
        print("optimisation...")
        t1 = timeit.default_timer()
        if nb_opis > 2:
            # for n in range(nb_nodes):
            #     result[n] = random.randint(0, nb_opis-1)
            step = 0
            nb_chg = nb_nodes
            while step < 5 and nb_chg > (nb_nodes / 100):
                nb_chg = 0
                step += 1
                for alpha in range(nb_opis):
                    result, nb = optAlpha(alpha, score_int, borders, result, nb_nodes, indices, coeff_alpha)
                    nb_chg += nb
        else:
            result = optBinaire(score_int, borders, nb_nodes, coeff_alpha)
        t2 = timeit.default_timer()
        print('temps de traitement -- : ', t2-t1, 's')

        print("export...")
        t1 = timeit.default_timer()
        output_name = os.path.splitext(args.outputfile)[0]
        if len(args.alpha)>1:
            output_name += '_'+str(coeff_alpha)
        else:
            # si on a une seule valeur de alpha a utiliser
            # on peut liberer la memoire en purgeant borders
            borders = None
        if args.export_ortho:
            exportResult_with_ortho(O, regions, result, output_name, selected_opi, ortho.transform, ortho.crs)
        else:
            exportResult_without_ortho(shape, regions, result, output_name, selected_opi, ortho.transform, ortho.crs)
        t2 = timeit.default_timer()
        print('temps de traitement -- : ', t2-t1, 's')

if __name__ == "__main__":
    main()