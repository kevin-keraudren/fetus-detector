import irtk
import numpy as np
import scipy.ndimage as nd

from sklearn.cluster import MeanShift
from skimage import morphology

def random_pick(choices, probs, n):
    '''
    http://stackoverflow.com/questions/8973350/draw-random-element-in-numpy
    >>> a = ['Hit', 'Out']
    >>> b = [.3, .7]
    >>> random_pick(a,b)
    '''
    cutoffs = np.cumsum(probs)
    idx = cutoffs.searchsorted(np.random.uniform(0, cutoffs[-1], size=n))
    return choices[idx], probs[idx]

def mean_shift_selection( votes,
                          n_points=5000,
                          bandwidth=10,
                          n=10,
                          cutoff=None,
                          debug=False,
                          debug_output="./" ):
    points = np.argwhere(votes)
    probas = votes[points[:,0],
                   points[:,1],
                   points[:,2]]
    points, probas = random_pick(points,probas,n_points)

    ms = MeanShift(bandwidth=bandwidth)
    ms.fit(points)

    weights = np.zeros( ms.cluster_centers_.shape[0], dtype='float' )
    for i,c in enumerate(ms.cluster_centers_):
        weights[i] = np.sum(probas[ms.labels_==i])

    detection = ms.cluster_centers_[np.argsort(weights)[::-1]]
    #points = points[np.argsort(weights)[::-1]]
    #ms.labels_ = ms.labels_[np.argsort(weights)[::-1]]
    weights = np.sort(weights)[::-1]

    weights /= weights.max()

    if debug:
        print weights.tolist()
        res = irtk.zeros(votes.get_header(),dtype='int')
        points = np.array(detection,dtype='int')
        if points.shape[0] > 30:
            points = points[:30]
        print np.arange(1,int(points.shape[0]+1))[::-1]
        res[points[:,0],
            points[:,1],
            points[:,2]] = np.arange(1,int(points.shape[0]+1))#[::-1]
        irtk.imwrite(debug_output+"/clusters.nii.gz",res)
        irtk.imwrite(debug_output+"/clusters_centers.nii.gz",irtk.landmarks_to_spheres(res,r=5))
        
    if cutoff is not None:
        selected = np.sum( weights >= cutoff )
        detection = detection[:selected]
        weights = weights[:selected]
    else:
        if len(detection) > n:
            detection = detection[:n]
            weights = weights[:n]

    return detection, weights

def gaussian( x, mu, sigma, normalised=False ):
    """ mu and sigma are respectively the mean and standard deciation of the distribution """
    if normalised:
        norm = np.sqrt(2.0*np.pi)*sigma
    else:
        norm = 1.0
    return 1.0/norm*np.exp(-(x-mu)**2/(2.0*sigma**2))

def mgaussian( x, mu, cov, normalised=False ):
    """ http://en.wikipedia.org/wiki/Multivariate_normal_distribution """
    inv_cov = np.linalg.pinv(cov)
    if normalised:
        norm = (2.0*np.pi)**(3.0/2.0)*np.linalg.det(cov)**(1.0/2.0)
    else:
        norm = 1.0
    if len(x.shape) == 2:
        d = np.sum( np.dot(x-mu, inv_cov) * (x-mu), 1)
        return np.exp( -0.5 * d )
    else:
        return 1.0/norm * np.exp( -0.5 * np.dot( np.transpose(x - mu), np.dot(inv_cov,x-mu) ) )

# def landmarks_to_spheres(seg, r=10):
#     nb_landmarks = np.unique(seg)[1:]
#     centers = []
#     for l in nb_landmarks:
#         centers.append( nd.center_of_mass( (seg == l).view(np.ndarray) ) )
#     centers= np.array(centers)[:,::-1]
#     centers = seg.ImageToWorld(centers)

#     res = irtk.zeros( seg.get_header(), dtype='uint8' )
#     ball = irtk.Image( morphology.ball(r) )
#     #ball.header['pixelSize'] = res.header['pixelSize']

#     for l in range(len(centers)):
#         ball.header['origin'] = centers[l]
#         ball2 = ball.transform(target=res)
#         res[ball2>0] = l+1
        
#     return res

def get_centers( seg ):
    brain_center = np.array(nd.center_of_mass( (seg == 2).view(np.ndarray) ),
                            dtype='float32')

    heart_center = np.array(nd.center_of_mass( (seg == 5).view(np.ndarray) ),
                                                  dtype='float32')

    left_lung = np.array(nd.center_of_mass( (seg == 3).view(np.ndarray) ),
                                              dtype='float')
    right_lung = np.array(nd.center_of_mass( (seg == 4).view(np.ndarray) ),
                                               dtype='float')

    return brain_center, heart_center, left_lung, right_lung

def get_orientation_training(seg):

    brain_center, heart_center, left_lung, right_lung = get_centers(seg)
    
    u = brain_center - heart_center
    v = right_lung - left_lung
    
    u /= np.linalg.norm(u)

    v -= np.dot(v,u)*u
    v /= np.linalg.norm(v)

    w = np.cross(u,v)
    w /= np.linalg.norm(w)

    return ( u.astype("float32"),
             v.astype("float32"),
             w.astype("float32") )

def get_orientation_training_jitter( brain_center,
                                     heart_center,
                                     left_lung,
                                     right_lung,
                                     n,
                                     brain_jitter=10,
                                     heart_jitter=5,
                                     lung_jitter=5 ):
    
    u = ( ( brain_center +
            np.random.randint(-brain_jitter, brain_jitter+1,
                              size=(n,3) )) -
          ( heart_center +
            np.random.randint(-heart_jitter, heart_jitter+1,
                              size=(n,3) )) )
    
    v = ( ( right_lung +
            np.random.randint(-lung_jitter, lung_jitter+1,
                              size=(n,3) )) -
          ( left_lung +
            np.random.randint(-lung_jitter, lung_jitter+1,
                              size=(n,3) )) )
    
    u /= np.linalg.norm( u, axis=1 )[...,np.newaxis]

    v -= (v*u).sum(axis=1)[...,np.newaxis]*u
    v /= np.linalg.norm( v, axis=1 )[...,np.newaxis]

    w = np.cross( u, v )
    w /= np.linalg.norm(w, axis=1)[...,np.newaxis]

    return ( u.astype("float32"),
             v.astype("float32"),
             w.astype("float32") )

def get_random_orientation(n):
    u = 2*np.random.rand( n, 3 ) - 1
    u /= np.linalg.norm( u, axis=1 )[...,np.newaxis]

    # np.random.rand() returns random floats in the interval [0;1)
    v = 2*np.random.rand( n, 3 ) - 1
    v -= (v*u).sum(axis=1)[...,np.newaxis]*u
    v /= np.linalg.norm( v, axis=1 )[...,np.newaxis]

    w = np.cross( u, v )
    w /= np.linalg.norm(w, axis=1)[...,np.newaxis]

    return ( u.astype("float32"),
             v.astype("float32"),
             w.astype("float32") )

def get_orientation_motion(seg):

    brain_center = np.array(nd.center_of_mass( (seg == 5).view(np.ndarray) ),
                            dtype='float32')

    heart_center = np.array(nd.center_of_mass( (seg == 3).view(np.ndarray) ),
                                                  dtype='float32')

    left_lung = np.array(nd.center_of_mass( (seg == 1).view(np.ndarray) ),
                                              dtype='float')
    right_lung = np.array(nd.center_of_mass( (seg == 2).view(np.ndarray) ),
                                               dtype='float')

    u = brain_center - heart_center
    v = right_lung - left_lung
    
    u /= np.linalg.norm(u)

    v -= np.dot(v,u)*u
    v /= np.linalg.norm(v)

    w = np.cross(u,v)
    w /= np.linalg.norm(w)

    return ( u.astype("float32"),
             v.astype("float32"),
             w.astype("float32") )

def Epanechnikov_kernel(r):
    """
    http://en.wikipedia.org/wiki/Kernel_(statistics)#Kernel_functions_in_common_use
    """
    kernel = np.zeros((2*r+1,2*r+1,2*r+1),dtype='float')
    for z in range(kernel.shape[0]):
        for y in range(kernel.shape[1]):
            for x in range(kernel.shape[2]):
                d = float((z-r)**2+(y-r)**2+(x-r)**2) / r**2
                if d <= 1:
                    kernel[z,y,x] = 3.0 / 4.0 * ( 1.0 - d )
    return kernel

def cos_kernel(r):
    kernel = np.zeros((2*r+1,2*r+1,2*r+1),dtype='float')
    kernel[r,r,r] = 1
    kernel = nd.distance_transform_edt(np.logical_not(kernel))
    mask = kernel>=r
    kernel[mask] = 0
    kernel /= kernel.max()
    kernel *= np.pi
    kernel = np.cos(kernel)
    kernel[mask] = 0
    kernel[kernel>0] /= np.sum(kernel[kernel>0])
    kernel[kernel<0] /= -np.sum(kernel[kernel<0])
    kernel *= 100
    return kernel[1:-1,1:-1,1:-1].copy()

def get_narrow_band( shape, center, r_min=0, r_max=None ):
    res = np.ones( shape, dtype=bool )
    coords = np.argwhere(res).astype('float64')
    coords -= np.array( center )
    R = np.linalg.norm(coords, axis=1)
    if r_min > 0:
        res.flat[R<r_min] = False
    if r_max is not None:
        res.flat[R>r_max] = False
    return res

def rescale( img, dtype='float' ):
    """ Stretch contrast."""
    img = img.astype(dtype)
    img -= img.min()
    m = img.max()
    if m != 0:
        img /= m
    return img, m

def shape_proba( heart, brain, left_lung, right_lung, liver, shape_model, verbose=False ):

    u = brain - heart
    v = right_lung - left_lung
    
    u /= np.linalg.norm(u)

    v -= np.dot(v,u)*u
    v /= np.linalg.norm(v)

    w = np.cross(u,v)
    w /= np.linalg.norm(w)

    if np.sum(np.isnan(w)) > 0:
        return 0

    u = u.astype("float32")
    v = v.astype("float32")
    w = w.astype("float32")

    M = np.array( [w,v,u], dtype='float32' ) # Change of basis matrix

    # centering and orient
    left_lung = np.dot( M, left_lung - heart)
    right_lung = np.dot( M, right_lung - heart)
    liver = np.dot( M, liver - heart)

    brain = np.dot( M, brain - heart)
    heart = np.dot( M, heart - heart)

    proba_left_lung = mgaussian( left_lung,
                                 shape_model['mean_left_lung'],
                                 shape_model['cov_left_lung'] )
    proba_right_lung = mgaussian( right_lung,
                                  shape_model['mean_right_lung'],
                                  shape_model['cov_right_lung'] )
    proba_liver = mgaussian( liver,
                             shape_model['mean_liver'],
                             shape_model['cov_liver'] )

    score = proba_left_lung + proba_right_lung + proba_liver

    if verbose:
        print "score =", score
        print 'heart = np.'+repr(heart)
        print 'brain = np.'+repr(brain)
        print 'left_lung = np.'+repr(left_lung), proba_left_lung
        print 'right_lung = np.'+repr(right_lung), proba_right_lung
        print 'liver = np.'+repr(liver), proba_liver
    
    return score

def shape_proba_brain( heart, brain, shape_model):
    d = np.linalg.norm(np.array(brain,dtype='float')-np.array(heart,dtype='float'))
    score = gaussian( d,
                      shape_model['mean_brain'],
                      shape_model['sigma_brain'] )
    return score
