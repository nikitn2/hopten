# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import jax.numpy as jnp
import jax.nn as jnn
import tensorflow as tf#; tf.config.set_visible_devices([], 'GPU');# tf.debugging.set_log_device_placement(True)
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import quimb.tensor as qtn
import quimb.utils as qu
import sys; sys.path.insert(1, '../tendeq/')
import tn4ml
import TnMachinery.MPSfunctions as mps

import sklearn.linear_model as sklm
import sklearn.model_selection as skms
import sklearn.metrics as skm
import sklearn.ensemble as ske
import sklearn.preprocessing as skpp
import sklearn.datasets as skds

def scatter_plot(x, y, labels_title = ["","",""]):
    
    plt.figure()
    x,y = x.flatten(), y.flatten()
    sns.scatterplot(x=x, y=y, marker=".")#; sns.kdeplot(x=x, y=y, levels=5);
    plt.xlabel(labels_title[0]); plt.ylabel(labels_title[1]); plt.title(labels_title[2])
    plt.tight_layout(); plt.yticks(plt.xticks()[0])
    plt.plot(plt.xlim(), plt.xlim(), ls='--', color='gray', alpha=0.2)

# losses: 'mean_squared_error', 'categorical_crossentropy'
def classical_logreg(X_train, X_test, y_train, y_test, poly = 1, alpha=1.0 , title = "classical logreg", penalty = "l2", solver='lbfgs', plot_print = False): 
    
    # Add polynomial features if neccessary
    poly_feats = skpp.PolynomialFeatures(degree = poly, include_bias = True)
    X_train = poly_feats.fit_transform(X_train.T)
    X_test  = poly_feats.transform(X_test.T)
    y_train = y_train.T
    y_test  = y_test.T
    
    # Convert from one-hot back to ordinal, if neccessary. 
    # The loss function now becomes categorical cross-entropy.
    if y_train.shape[1]>1:
        y_train=np.argmax(y_train,axis=1)
        y_test =np.argmax( y_test,axis=1)
    
    # Run logistic regression & return
    y_train = np.int32(y_train.squeeze()); y_test = np.int32(y_test.squeeze())
    model = sklm.LogisticRegression(C = alpha, fit_intercept = False, penalty=penalty, solver = solver) # fit_intercept should be False, given it's available already in X
    model.fit(X_train, y_train)
    y_pred_train, y_pred_test = model.predict(X_train), model.predict(X_test)
    acc_train = skm.accuracy_score(y_train, y_pred_train)
    acc_test  = skm.accuracy_score(y_test, y_pred_test)
     
    # Print report & visualise        
    if plot_print:
        print("Classical train / test accs = {:.3f}, {:.3f}".format(acc_train, acc_test))
        cm_train = skm.confusion_matrix(y_train,y_pred_train,normalize="true")
        cm_test  = skm.confusion_matrix(y_test,y_pred_test, normalize="true")
        for cm, acc in (cm_train, acc_train), (cm_test, acc_test): 
            skm.ConfusionMatrixDisplay(confusion_matrix=cm*100).plot(values_format=".0f")
            plt.title(title + " @ acc = {:.1f}%".format(100*acc))
        plt.show()
    
    return acc_test

def classical_linreg(X_train, X_test, y_train, y_test, poly = 1, alpha=0 , title = "classical polyreg", plot_print = False):     
    
    # Add polynomial features if neccessary
    poly_feats = skpp.PolynomialFeatures(degree = poly, include_bias = True)
    X_train = poly_feats.fit_transform(X_train)
    X_test  = poly_feats.transform(X_test)
    
    # Run linear regression & return
    model = sklm.Ridge(alpha = alpha)
    model.fit(X_train, y_train)
    r2 = skm.r2_score(y_test, model.predict(X_test))
    r2_train = skm.r2_score(y_train, model.predict(X_train))
    if plot_print:
        scatter_plot(y_train,model.predict(X_train), ["true", "pred",f"{title}@p={poly}; Train, r2={round(r2_train,2)}"] )
        scatter_plot(y_test,model.predict(X_test), ["true", "pred",f"{title}@p={poly}; Test, r2={round(r2,2)}"] )
        print("Train r2 = {:.3f}".format(r2_train), "\nTest r2 = {:.3f}".format(r2))
    
    return r2, model

def classical_rafreg(X_train, X_test, y_train, y_test, title = "classical random forests"): 
    
    model = ske.RandomForestRegressor()
    model.fit(X_train,y_train)
    r2 = skm.r2_score(y_test, model.predict(X_test))
    
    scatter_plot(y_test,model.predict(X_test), ["true", "pred",f"{title}@r2={round(r2,2)}"] )
    
    return r2, model

def test_wines(wines_split, wine_red_split, wine_white_split):
    
    datasets = (wines_split, wine_red_split, wine_white_split)
    titles = ("all", "red", "white")
    
    for i,dataset in enumerate(datasets):
        classical_linreg(*dataset, title = f"linreg/{titles[i]}")
        #classical_logreg(*dataset)
        classical_rafreg(*dataset, title =f"ranfor/{titles[i]}")
    
    return 0

def load_openml(name, y_dict = None):
    df = skds.fetch_openml(name=name,as_frame=True)
    X = df.data.to_numpy(dtype=np.float64)     
    y = df.target
    if y_dict is not None: y=y.astype("str").map(y_dict)
    y = y.to_numpy(dtype=np.float64)
    X_train, X_test, y_train, y_test = skms.train_test_split(X,y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

def load_houses():   return load_openml("house_16H")
def load_leukemia(): return load_openml("leukemia", y_dict = {"AML":0, "ALL":1})

def load_wines():
    
    wine_red  =pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",   sep=';')
    wine_red_split   = skms.train_test_split(wine_red.drop(  "quality",axis=1), wine_red[  "quality"], test_size=0.2, random_state=0)
    wine_white=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
    wine_white_split = skms.train_test_split(wine_white.drop("quality",axis=1), wine_white["quality"], test_size=0.2, random_state=0)
    
    wine_red[  "white"]=False
    wine_white["white"]=True
    wines = pd.concat([wine_red, wine_white], ignore_index=True)
    
    wines_split = skms.train_test_split(wines.drop("quality",axis=1), wines["quality"], test_size=0.2, random_state=0)
    
    return wines_split, wine_red_split, wine_white_split

def resize_images(images, sz = [28,28]): return tf.image.resize(images, sz, method=tf.image.ResizeMethod.AREA).numpy()
def load_mnist(sz = [28,28], propr = 0.01): 
    (train_images, y_train), (test_images, y_test) = tf.keras.datasets.mnist.load_data()
    
    train_sz = int(60000*propr)
    test_sz  = int(10000*propr)
        
    # Optionally reduce test/train sizes
    train_images = train_images[:train_sz,...]; y_train = y_train[...,:train_sz]
    test_images = test_images[:test_sz,...];    y_test = y_test[...,:test_sz]
        
    X_train= resize_images(train_images.reshape(-1,28,28,1),sz=sz).squeeze()/255.0
    X_test = resize_images(test_images.reshape( -1,28,28,1),sz=sz).squeeze()/255.0
    
    # One-hot encode y
    n_classes = 10
    y_train = np.uint8(tn4ml.util.integer_to_one_hot(y_train, n_classes))
    y_test  = np.uint8(tn4ml.util.integer_to_one_hot( y_test, n_classes))
    
    return (X_train, y_train), (X_test, y_test)

def print_vis(X, y, cutoff=1e-2):

    X_compr, fid_X_compr, CR_X_compr = mps.mpsGetFidelityCompRatio(X, split=True, chi=None, cutoff=cutoff); X_compr = X_compr[0]
    y_compr, fid_y_compr, CR_y_compr = mps.mpsGetFidelityCompRatio(y, split=True, chi=None, cutoff=cutoff); y_compr = y_compr[0]

    print("CRs (X & y) = ({:.3f},{:.3f}) and y-error = ({:.2e})".format(CR_X_compr,CR_y_compr,1-np.mean(y_compr.round(0)==y)) )
    hfont = {'fontname':'Courier New', 'fontsize': 15, 'fontweight': 'bold'}
    for i in range(10):
        if i == 1: plt.text(0.0,-6.0, f"@CR={round(CR_X_compr,1)}", **hfont)
        plt.subplot(2, 5, i + 1)
        # Select the first image of each digit
        digit_indices = np.where(np.argmax(y,0) == i)[0]
        plt.imshow(X_compr[...,digit_indices[0]], cmap='gray')
        plt.title(f'Label: {i}', **hfont)
        plt.axis('off')
    plt.tight_layout()

def mpsify(data, split, chi, cutoff, output = False, poly = 1):    
    
    # Add intercept
    if output == False:
        intercept = np.ones([1,data.shape[-1]], np.float64)
        data = np.concatenate((intercept, data.reshape(-1,data.shape[-1])),axis=0)
    
    # Regarless of what follows, compute data_compr; and data_mps, as it'll be useful if output==True or poly==1.
    data_mps = mps.mpsDecompFlowKD_uneven(data,split); data_mps.compress(max_bond=chi, cutoff=cutoff)
    data_compr = mps.mpsInvDecompFlowKD(data_mps, data.shape, split)
    
    # Transform into MPS with appropriate site and index ids
    if output==True or poly == 1:
        
        sti = data_mps.site_tag_id 
        sii = data_mps.site_ind_id
        
        if output == True:
            if data_mps.L>1:
                n_phys_inds = len(mps.primefact(data.shape[0]))
                phys_tags   = [sti.format(i) for i in range(n_phys_inds)]
                phys_inds   = [sii.format(i) for i in range(n_phys_inds)]
                data_mps.contract(phys_tags, inplace=True)
                data_mps[sti.format(0)].fuse({sii.format(0):phys_inds},inplace=True)
            
            n_contracs = len(phys_tags)-1
            data_mps = mps.shiftIndextagMps(data_mps,-1*n_contracs, n_contracs+1, tag_shift=True, N = data_mps.L)
            data_mps._L = len(data_mps.tensors)
            data_mps[sti.format(0)].drop_tags(phys_tags[1:])
    
    else:
        data_mpses = []
        for i in range(data.shape[-1]):
                        
            base_mps = mps.mpsDecompFlowKD_uneven(data[...,i],split); base_mps.compress(max_bond=chi, cutoff=cutoff)
            comb_mps = base_mps.copy()
            for p in range(poly-1): comb_mps = mps.mpsKron(base_mps,comb_mps)
            comb_mps.compress(max_bond=chi, cutoff=cutoff)
            data_mpses.append(comb_mps)
        data_mps = mps.mpsConcatenate(data_mpses, max_bond = chi, cutoff=cutoff)
        
    return data_mps, data_compr
    
def one_hot_to_logits(one_hot, epsilon=1e-3):
    clipped = jnp.clip(one_hot, epsilon, 1 - epsilon)/(1+(one_hot.shape[0]-2)*epsilon)

    return jnp.log(clipped / (1 - clipped))

def softmax(A, axis):
    A_max = jnp.max(A, axis=axis, keepdims=True)
    exp_A = jnp.exp(A - A_max)
    return exp_A / jnp.sum(exp_A, axis=axis, keepdims=True)

# def log_softmax(A, axis):
#     A_max = jnp.max(A, axis=axis, keepdims=True)
#     A_adj = A - A_max
#     return A_adj - jnp.log(jnp.sum(jnp.exp(A_adj), axis=axis, keepdims=True))   

def relu(A): return jnp.abs(A)

def sigmoid(A, axis): return jnp.exp(A)/(1+jnp.exp(A))

def classific_regr_score(mpo, in_out_mpses, classific = True, plot = False):
    
    # Initialise
    in_mps, out_mps = in_out_mpses
    sti = mpo.site_tag_id
    phys_tag_nums = [i for i in range(mpo.L)]
    phys_tags = [sti.format(num) for num in phys_tag_nums]
    sii = in_mps.site_ind_id
    site_data_ids = [sii.format(i) for i in range(len(phys_tags),in_mps.L)]
    
    # Contract ground truth
    y_true = out_mps.contract(all).fuse({"data" :site_data_ids}).data
    
    # Contract predictions
    in_mpo_tn = in_mps & mpo
    temp = in_mpo_tn.contract(all).fuse({"data" :site_data_ids}).data
    if classific: 
        if temp.shape[1]>1: y_pred = (temp == temp.max(axis=1, keepdims=True)).astype(jnp.uint8)
        else: 
            y_pred = (temp >0).astype(jnp.uint8)
            y_true = y_true.round().astype(jnp.uint8)
    else: y_pred = temp
    
    # Compute metric
    m = y_true.shape[0]
    if classific: 
        if y_pred.shape[1]>1: out= 1/m * np.sum(y_pred * y_true) # accuracy
        else: out = skm.accuracy_score(y_true, y_pred)
    else: out = 1 - np.sum( (y_true - y_pred)**2 )/np.sum( (y_true - np.mean(y_true))**2 ) # r2
    
    # Convert from one-hot back to ordinal, if neccessary before visualising
    if y_true.shape[1]>1: y_true=np.argmax(y_true,axis=1);  y_pred=np.argmax(y_pred,axis=1)
    
    # Visualise
    if plot:
        if classific == False and y_pred.shape[1]==1: scatter_plot(y_true,y_pred, ["true", "pred", f"mps linreg @ r2={out.round(2)}"] )
        if classific == True:
            cm = skm.confusion_matrix(y_true,y_pred,normalize="true")
            skm.ConfusionMatrixDisplay(confusion_matrix=cm*100).plot(values_format=".0f")
            plt.title("mps logreg @ acc = {:.1f}%".format(100*out))
            plt.show()
    return out

def classific_loss(mpo_var, in_out_mpses, loss_type = "cross_entropy", alpha_mpo = 0):
    
    # Initialise
    in_mps, out_mps = in_out_mpses
    sti = mpo_var.site_tag_id
    phys_tag_nums = [i for i in range(mpo_var.L)]
    phys_tags = [sti.format(num) for num in phys_tag_nums]
    
    # Connect input layer with mpo layer
    in_mpo_tn = in_mps & mpo_var
    
    # Contract-in all tensors
    in_mpo_tn = qtn.TensorNetwork([in_mpo_tn.contract(all)])
    
    # Compute and return loss: either cross entropy, or mean squared error
    loc_output_ind = in_mpo_tn[phys_tags[0]].inds.index("out")
    if loss_type == "cross_entropy": # Apply log after softmax/sigmoid and return cross entropy
        if in_mpo_tn["s0"].data.shape[loc_output_ind] > 1: # categorical cross entropy
            in_mpo_tn[phys_tags[0]].modify(apply = lambda x : jnn.log_softmax(x,axis=loc_output_ind))
            loss = - in_mpo_tn @ out_mps # - log(softmax(X)) . y
        else: # Binary cross-entropy
            pred_mps = in_mpo_tn.copy()
            pred_mps[phys_tags[0]].modify(apply = lambda x : jnn.log_sigmoid(x))
            one_minus_pred_mps = in_mpo_tn.copy()
            one_minus_pred_mps[phys_tags[0]].modify(apply = lambda x : jnn.log_sigmoid(-x))
            one_mps = in_mpo_tn.copy()
            one_mps[phys_tags[0]].modify(apply = lambda x : np.ones(x.shape,dtype=np.float64))
            
            loss = - ( pred_mps @ out_mps + one_minus_pred_mps @ one_mps - one_minus_pred_mps @ out_mps)
                
    else:  # Apply softmax and compute mse
        in_mpo_tn[phys_tags[0]].modify(apply = lambda x : softmax(x,axis=loc_output_ind))
        loss = in_mpo_tn @ in_mpo_tn - 2*in_mpo_tn @ out_mps
        num_samples = out_mps.outer_size()
        loss /= num_samples
    
    return loss + alpha_mpo*(mpo_var@mpo_var)

def build_Renv_regr_mse_loss(in_out_mpses, data_tag_nums): 
    
    # Initialise
    in_mps, out_mps = in_out_mpses
    in_mps.reindex_all(out_mps.site_ind_id,inplace=True)
    sti = in_mps.site_tag_id
    data_tags = [sti.format(num) for num in data_tag_nums]
            
    # Get physical-data connecting bonds for later before partitioning the physical & data parts of the networks
    in_conn_bond = next(iter(in_mps[sti.format(data_tag_nums[0]-1)].bonds(in_mps[data_tags[0]])))
    in_mps_L, in_mps_R = in_mps.partition( data_tags); in_mps_L._L  -= len(data_tag_nums)
    out_mps_L,out_mps_R= out_mps.partition(data_tags); out_mps_L._L -= len(data_tag_nums)
    
    # Connect input layer with output layer and itself
    in_in_R  = in_mps_R & in_mps_R
    in_out_R = in_mps_R & out_mps_R

    # Cut duplicate left bonds of in_in_R
    in_in_R.cut_bond(in_conn_bond, "free_bond_0", "free_bond_1")
    in_mps_L.reindex({in_conn_bond : "free_bond_0"}, inplace=True)
    in_out_R.reindex({in_conn_bond : "free_bond_0"}, inplace=True)
    
    # Pre-contract data tensors
    in_out_R = in_out_R.contract()
    in_in_R  = in_in_R.contract()
    
    # Clean tags and replace with "R"
    in_out_R.drop_tags(); in_in_R.drop_tags()
    in_out_R.tags.add("Right"); in_in_R.tags.add("Right")
    
    # Normalise in_in_R and in_out_R by sample size
    num_samples = out_mps.outer_size()
    in_in_R /= num_samples
    in_out_R /= num_samples
    
    return in_mps_L, out_mps_L, in_in_R, in_out_R

def regr_mse_loss(mpo_var, in_out_L_mpses, envs_R, alpha_mpo = 0):
        
    # Initialise
    in_mps_L, out_mps_L = in_out_L_mpses
    in_in_R, in_out_R = envs_R
    sti = in_mps_L.site_tag_id
    phys_tag_nums = [i for i in range(mpo_var.L)]
    phys_tags = [sti.format(num) for num in phys_tag_nums]
    
    # Connect input layer with mpo layer
    in_mpo_L = in_mps_L & mpo_var
    
    # Contract-in all physical tensors
    in_mpo_L = qtn.TensorNetwork([in_mpo_L.contract(phys_tags)])
        
    # Compute and return ~ squared-error loss
    loss = in_mpo_L @ in_mpo_L.reindex({"free_bond_0":"free_bond_1"}) @ in_in_R - 2*in_mpo_L @ out_mps_L @ in_out_R
    return loss  + alpha_mpo*(mpo_var@mpo_var)

def regr_solve_explicitCG(mpo_var, sites_to_opt, chi_max, in_out_L_mpses, envs_R, maxiter = 20, alpha_mpo = 0):
    
    # Initialise
    in_mps_L, out_mps_L = in_out_L_mpses
    in_in_R, in_out_R = envs_R
    sti = mpo_var.site_tag_id
    
    # Prune if sites_opt go above/below existing indices
    sites_to_opt = [site for site in sites_to_opt if site<mpo_var.L and site>=0]
    
    # Extract site to solve for
    mpo_without_site = mpo_var.copy()
    sites = [mpo_without_site.pop_tensor(sti.format(site_to_opt)) for site_to_opt in sites_to_opt]
    site  = qtn.TensorNetwork(sites)^...#; print(site)
    
    # Connect and contract input layer with mpo layers
    in_mpo_L = (in_mps_L & mpo_without_site)^...
    upper_mpo     = (in_mpo_L.reindex({"free_bond_0":"free_bond_1"}) @ in_in_R)

    # Define index orderings
    site_inds     = qu.oset(site.inds)
    upper_inds    = qu.oset(upper_mpo.inds)
    internal_inds = (upper_inds - site_inds)
    common_inds   = (site_inds & upper_inds)
    surplus_inds  = site_inds - upper_inds if mpo_var.L>1 else ["out"]
    
    # Define Hessian contraction with site function for CG update algorithm
    def H_contract_s(s):
        # Load, compute & return
        site.data[:] = s.reshape(site.shape)
        s_new = (upper_mpo @ (in_mpo_L @ site)).to_dense(site_inds)     
        
        s_new += alpha_mpo * s_new
        return s_new
    
    # Compute Jacobian
    j_ten = ((in_mpo_L & out_mps_L & in_out_R)^...)#
    j = j_ten.to_dense(site_inds)
    
    # Now run CG algorithm
    linop = sp.sparse.linalg.LinearOperator(
        shape=(site.size, site.size),
        matvec=H_contract_s,
        dtype=site.dtype)
        
    s_final, info = sp.sparse.linalg.cg(linop, j, rtol=0, maxiter = maxiter, x0 = site.to_dense(site_inds))
    
    if info>20: # Perform direct solve if CG not properly converged
        H = upper_mpo.to_dense(common_inds, internal_inds) @ in_mpo_L.to_dense(internal_inds, common_inds)
        j = j_ten.to_dense(common_inds, surplus_inds)
        
        # Regularise with penalty term (of |Hs-j|^2 loss)
        H += alpha_mpo*np.eye(*H.shape)
        
        s_final = np.linalg.lstsq(H,j, rcond=None)[0]
    
        #print("condNum=",np.linalg.cond(H),"i=", site_to_opt)#; mpo_var.show()
    
    # Put back into MPO, potentially split, and return
    site.data[:] = s_final.reshape(site.shape)
    
    if len(sites_to_opt)>1:
        site.drop_tags()
        sites_new = site.split(left_inds = qu.oset(sites[0].inds) & qu.oset(site_inds), ltags = sites[0].tags, rtags = sites[1].tags, 
                               max_bond= 100*chi_max, cutoff=1e-16, cutoff_mode = "rel") # <-- max_bond = 100*chimax is temp; but will be compressed down anyway in linear_sweeping
        mpo_without_site.add_tensor_network(sites_new)
        mpo_var = mpo_without_site    
    else: mpo_var[sites_to_opt[0]].data[:] = s_final.reshape(site.shape)
    
    return mpo_var

def linear_sweeping(mpo_init, chi_max, loss_fn, loss_kwargs, eps = 1e-16, max_sweeps = None, mode = "1site",
                     optimizer_opts = {"optimizer" : 'adam', "autodiff_backend" : 'jax'}, 
                     optimize_opts = {'jac': True, 'hessp':True,'ftol':1e-16, 'gtol': 1e-16, 'eps' : 1e-16, "n" : 50}):
    
    # Initialise
    patience = 5 if optimizer_opts["optimizer"] == "explicitCG" else 1 # how many “no-improve” steps to wait
    sti           = mpo_init.site_tag_id
    loss_new      = loss_fn(mpo_var = mpo_init, **loss_kwargs)
    loss_record   = np.inf
    chi_new       = mpo_init.max_bond()
    mpo_new       = mpo_init.copy()
    loss_prev     = np.inf
    if max_sweeps == None: max_sweeps = 1e6
    N = mpo_new.L
        
    # Perform DMRG until convergence
    sweeps= 0; sgn = 1; early_stop_counter = 0; success_counter = 0; n = optimize_opts["n"]
        
    while success_counter < patience and sweeps < max_sweeps and early_stop_counter < patience and not np.isnan(loss_new):
        
        # Update previous loss
        loss_prev = loss_new
        
        # Run dmrg on MPO tensors
        for i in range(sgn*min(0,(N-1)*sgn),sgn*max(0,N*sgn),sgn):
            
            # Prep before optimisation
            if N > 1 and sweeps > 0:
                mpo_new.canonize(i)
                #mpo_new.compress(form = i, cutoff_mode = "rel",cutoff=1e-16, max_bond = chi_max)
            
            # Optimise. For convergence reasons, explicitCG uses one-site, autodiff two-site.
            if optimizer_opts["optimizer"] == "explicitCG":
                sites_to_update = [i]
                mpo_new = regr_solve_explicitCG(mpo_new, sites_to_update, chi_max, maxiter = n, **loss_kwargs); print(f"CG-update at sites={str(sites_to_update):<10} gives loss: {loss_fn(mpo_var = mpo_new,**loss_kwargs):.6e}")
            else:
                sites_to_update = [j for j in range( i, min(i+2,N) )] if sgn==+1 else [j for j in range( max(0,i-1), i+1 )] 
                tagsToOpti = [sti.format(j) for j in sites_to_update]
                tnopt = qtn.TNOptimizer(mpo_new, tags = tagsToOpti, loss_fn=loss_fn, loss_constants=loss_kwargs,**optimizer_opts)
                mpo_new = tnopt.optimize(**optimize_opts)
        
        # Iterate
        sweeps+=1
        sgn *= -1
        
        # Increase bond-dimension if possible
        if chi_new != None and chi_new < chi_max and sweeps%2==0: 
            chi_new += 1
            if N>1: mpo_new.expand_bond_dimension(chi_new, rand_strength=1e-3, inplace=True)
        
        # Update new loss
        loss_new = loss_fn(mpo_var = mpo_new, **loss_kwargs)
        
        # Check if loss is still decreasing
        if loss_prev < loss_new: 
            early_stop_counter+=1
            if n>2: n//=2; print("Loss not decreasing; halving the number of local iterations down to n={}".format(n)) #<-- this seems to work well based on empiric observations
        else: early_stop_counter = 0
        
        if np.abs(loss_prev-loss_new)<=eps: success_counter += 1
        else: success_counter = 0
        
        # Check if new minimal loss reached
        if loss_new < loss_record:
            loss_record = loss_new
            mpo_record = mpo_new.copy()

        #print((loss_prev - loss_new), sweeps < max_sweeps, early_stop_counter < patience)

    
    # Compress 
    if mpo_record.L>1: mpo_record.compress(form=N-1,cutoff_mode = "rel",cutoff=1e-15)
    print("Linear sweeper done after {} sweeps; lowest loss seen is {:.6e}".format(sweeps, loss_record))
    return mpo_record

def learn_linreg(mpo, X_train_mps, y_train_mps, X_test_mps, y_test_mps, alpha_mpo = 0, explicitCG_prerun = True, eps= 1e-1, max_sweeps=[None,None], chi_max=15, n_local_its = 50, local_opt="L-BFGS-B", plot_print = False):
    
    # Init
    n_phys_inds = mpo.L
    n_data_inds = X_train_mps.L - mpo.L
    optimizer_opts = {"optimizer" : local_opt, "autodiff_backend" : 'jax'}
    optimize_opts = {'jac': True, 'hessp': True, 'xtol': 1e-8,'eps' : 1e-16, 'disp':False ,"n": n_local_its}#, 'ftol':1e-16, 'gtol': 1e-16}
        
    mpo_var=mpo#.astype(np.float64)
    in_mps_L, out_mps_L, in_in_R, in_out_R = build_Renv_regr_mse_loss(in_out_mpses = [X_train_mps, y_train_mps], data_tag_nums=[i for i in range(n_phys_inds,n_phys_inds+n_data_inds)])
    
    # Experimental prerun
    if local_opt != "explicitCG" and explicitCG_prerun == True:
        print("Prerun with explicit-CG @ sweeps = {} and n_local_its = {}".format(max_sweeps[0], n_local_its//10))
        optimizer_opts_prerun = optimizer_opts.copy(); optimizer_opts_prerun["optimizer"] = "explicitCG"
        optimize_opts_prerun = optimize_opts.copy(); optimize_opts_prerun["n"] = n_local_its//10
        mpo_var = linear_sweeping(mpo_var, eps=eps, max_sweeps=max_sweeps[0], chi_max=chi_max,
                                                     loss_fn= regr_mse_loss, loss_kwargs = {"in_out_L_mpses":[in_mps_L, out_mps_L],"envs_R": [in_in_R, in_out_R], "alpha_mpo": alpha_mpo},
                                                     optimizer_opts = optimizer_opts_prerun, optimize_opts=optimize_opts)
    # Solve
    mpo_var = linear_sweeping(mpo_var, eps=eps, max_sweeps=max_sweeps[-1], chi_max=chi_max,
                                                 loss_fn= regr_mse_loss, loss_kwargs = {"in_out_L_mpses":[in_mps_L, out_mps_L],"envs_R": [in_in_R, in_out_R], "alpha_mpo": alpha_mpo},
                                                 optimizer_opts = optimizer_opts, optimize_opts=optimize_opts)
    
    # Evaluate
    r2_test = classific_regr_score(mpo_var, in_out_mpses= [X_test_mps, y_test_mps], classific=False, plot=False)
    if plot_print:
        print("MPO-linreg train r2 = {:.3f}".format( classific_regr_score(mpo_var, in_out_mpses= [X_train_mps,y_train_mps],classific=False, plot=True)))
        print("MPO-linreg test r2  = {:.3f}".format( classific_regr_score(mpo_var, in_out_mpses= [X_test_mps, y_test_mps], classific=False, plot=False)))
    return mpo_var, r2_test

def learn_logreg(mpo, X_train_mps, y_train_mps, X_test_mps, y_test_mps, eps= 1e-3, max_sweeps = None, chi_max=15, n_local_its = 50,local_opt="L-BFGS-B", plot_print = False):
    
    optimizer_opts = {"optimizer" : local_opt, "autodiff_backend" : 'jax'}
    optimize_opts = {'jac': True, 'hessp': True, 'xtol': 1e-8,'eps' : 1e-16, 'disp':False ,"n": n_local_its}#, 'ftol':1e-16, 'gtol': 1e-16,}
    
    mpo_var=mpo.copy()
    X_train_mpsC  = X_train_mps.contract(["s{}".format(i) for i in range(mpo.L-1,X_train_mps.L)],inplace=False); X_train_mpsC._L = len(X_train_mpsC.tensors)
    y_train_ten = (y_train_mps.contract(["s0"] + ["s{}".format(i) for i in range(mpo.L,X_train_mps.L)],inplace=False))
         
    mpo_var = linear_sweeping(mpo_var, eps=eps, max_sweeps=max_sweeps, chi_max=chi_max, loss_fn= classific_loss, loss_kwargs = {"in_out_mpses":[X_train_mpsC, y_train_ten]},
                                                   optimizer_opts = optimizer_opts, optimize_opts=optimize_opts)
    acc_test = classific_regr_score(mpo_var, in_out_mpses= [X_test_mps, y_test_mps], classific=True, plot=False)
    if plot_print:
        print("MPO-logreg train acc ={:.3f}".format( classific_regr_score(mpo_var, in_out_mpses= [X_train_mps,y_train_mps],classific = True)) )
        print("MPO-logreg test acc = {:.3f}".format( classific_regr_score(mpo_var, in_out_mpses= [X_test_mps, y_test_mps], classific = True)) )
    return mpo_var, acc_test