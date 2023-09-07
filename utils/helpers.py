
def cal_savi(nir, r, coef=0.5):
    # ((NIR - R) / (NIR + R + L)) * (1 + L)
    savi = ((nir - r)/(nir+r+coef)) * (1+coef)
    return savi

def cal_ndvi(nir, r):
    ndvi = (nir - r)/(nir+r)
    return ndvi

def cal_ndwi(nir, g):
    # NDWI = (G-NIR)/(G+NIR)
    ndwi = (g - nir)/(nir+g)
    return ndwi

def ratio_indices(band1, band2):
    # ratio indices
    ratio = band1/band2
    return ratio