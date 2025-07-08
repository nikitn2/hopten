import os; os.environ["KMP_WARNINGS"] = "0"; os.environ["JAX_PLATFORM_NAME"] = "cpu"; os.environ["NUMBA_DISABLE_JIT"] = "1"
#import requests, zipfile, io
import numpy as np
import argparse
import pickle
import quimb.tensor as qtn
import sys; sys.path.insert(1, '../tendeq/')
import Linear as ln
import TnMachinery.MPSfunctions as mps
import sklearn.model_selection as skms
import pandas as pd
import sklearn.preprocessing as skpp
import config_hopten as cfg

# Hack to deal with numpy mismatch between local and Oxford ARC clusters
import sys, types; sys.modules["numpy._core"] = types.ModuleType("numpy._core"); sys.modules["numpy._core.numeric"] = np.core.numeric; sys.modules["numpy._core.multiarray"] = np.core.multiarray

pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 10000)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 1000)

def parse_args():
    
    # Initialise
    parser = argparse.ArgumentParser(description="Required inputs: “--cutoff_x, --cutoff_y, --poly, --dataset. Optional: “--chi_mpo” (0 by default). Setting chi_mpo = 0 means no training is done.")
    def stringParser_setType_float(arg_string):
        if arg_string.find(" ") != -1: return [float(x) for x in arg_string.split()]
        else: return float(arg_string)
    def stringParser_setType_int(arg_string):
        if arg_string.find(" ") != -1: return [int(x) for x in arg_string.split()]
        else: return int(arg_string)
    def stringParser_delist(arg_list):
        if any(isinstance(x, list) for x in arg_list): return arg_list[0]
        else:                                          return arg_list
    
    # Parse
    parser.add_argument("--cutoff_x",  nargs = "+", type=stringParser_setType_float, required=True)
    parser.add_argument("--cutoff_y",  nargs = "+", type=stringParser_setType_float, required=False, default = [1e-16])
    parser.add_argument("--alpha",     nargs = "+", type=stringParser_setType_float, required=False, default = [1e-8])
    parser.add_argument("--chi_mpo",   nargs = "+", type=stringParser_setType_int,   required=False, default = [0])
    parser.add_argument("--eps_mpo",   nargs = "+", type=stringParser_setType_float, required=False, default = [1e-8])
    parser.add_argument("--poly",      nargs = "+", type=stringParser_setType_int,   required=True)
    parser.add_argument("--dataset",                type=str,                        required=True)
    args = parser.parse_args()
        
    return stringParser_delist(args.poly), {"X": stringParser_delist(args.cutoff_x), "y": stringParser_delist(args.cutoff_y)}, stringParser_delist(args.alpha), stringParser_delist(args.chi_mpo),stringParser_delist(args.eps_mpo), args.dataset

def load_proc(dataset, sort = True, scale = "quantile_uniform", transpose=True):
    
    if dataset == "wines":
        wines_split,_,_ = ln.load_wines()
        X_train, X_test, y_train,y_test = [df.to_numpy().astype(np.float64) for df in wines_split]
        classific = False
    elif dataset=="casp":
        df = pd.read_csv("https://archive.ics.uci.edu/static/public/265/physicochemical+properties+of+protein+tertiary+structure.zip")
        X_train, X_test, y_train, y_test = skms.train_test_split(df.drop(columns="RMSD").values, df["RMSD"].values, test_size=0.2, random_state=0)
        classific = False
    elif dataset=="houses":   
        X_train, X_test, y_train, y_test = ln.load_houses()
        classific = False
    elif dataset=="years":
        col_names = ["year"] + [f"feat_{i}" for i in range(90)]
        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip", names=col_names)
        X_train, X_test, y_train, y_test = skms.train_test_split(df.drop(columns="year").values, df["year"].values, test_size=0.2, random_state=0)
        classific = False
    # elif dataset=="electricity_time":
    #     r = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip") 
    #     z = zipfile.ZipFile(io.BytesIO(r.content))
    #     with z.open("LD2011_2014.txt") as f:
    #         df = pd.read_csv(f, sep=";", parse_dates=["Unnamed: 0"], index_col="Unnamed: 0")
    #         df.set_index(pd.to_datetime(df.index), inplace=True)
    #         df.sort_index(inplace=True)        
    # elif dataset=="mnist":
    #     n_pixels = 14
    #     (X_train, y_train), (X_test, y_test) = ln.load_mnist(propr = 1.0, sz = [n_pixels]*2) # 1% <=> 600 train/ 100 test samples
    #     X_train = X_train.reshape(-1,n_pixels**2)
    #     X_test  = X_test.reshape( -1,n_pixels**2)
    #     classific = True
    # elif dataset=="leukemia": 
    #     X_train, X_test, y_train, y_test = ln.load_leukemia()
    #     classific = True
    
    if sort == True:
        if len(y_train.shape)>1:
            sor_inds_train = np.argsort(np.argmax(y_train,axis=1))
            sor_inds_test  = np.argsort(np.argmax(y_test, axis=1))
        else: sor_inds_train = np.argsort(y_train); sor_inds_test = np.argsort(y_test)
        X_train = X_train[sor_inds_train]; X_test = X_test[sor_inds_test]
        y_train = y_train[sor_inds_train]; y_test = y_test[sor_inds_test]
        
    if len(y_train.shape)==1: 
        y_train = y_train[:,np.newaxis]
        y_test  = y_test[:,np.newaxis]
    
    # Quantile transform X if requested
    if scale in ["quantile_uniform", "quantile_uniform_tanh" "quantile_normal", "quantile_normal_tanh" ]:
        scaler_quant = skpp.QuantileTransformer(output_distribution=scale.split("_")[1],subsample=1_000_000,random_state=0)
        X_train = scaler_quant.fit_transform(X_train)
        X_test  = scaler_quant.transform(X_test)
    
    # Standardise X,y in any-case
    scaler_standard = skpp.StandardScaler()
    X_train = scaler_standard.fit_transform(X_train); X_test  = scaler_standard.transform(X_test)
    if classific == False: y_train = scaler_standard.fit_transform(y_train); y_test  = scaler_standard.transform(y_test)        
    
    # And force X towards the (-1,1) band
    scaler_maxabs = skpp.MaxAbsScaler()
    X_train = scaler_maxabs.fit_transform(X_train)
    X_test = scaler_maxabs.transform(X_test)

    # Transpose to put into right format for mpsify before finishing
    if transpose==True:
        X_train = X_train.T; y_train = y_train.T
        X_test  = X_test.T;  y_test  = y_test.T
    return X_train, y_train, X_test, y_test, classific

if __name__ == '__main__':
    
    #XXX TODO: Poly-4 casp & house left only; rest of it is done!! XXX
        
    # Main hyperparameters
    if len(sys.argv)>1: polys, cutoffs, alphas, chis_mpo, eps_MPOs, dataset = parse_args(); make_plots = False
    else:
        polys    = [4]
        cutoffs  = {"X": [1e-8], "y":[1e-16] }
        alphas   = [1e-8, 0.001, 0.01, 0.1, 1, 10, 100, 1000] # Tikhonov regularisation parameter
        chis_mpo = [6,8] # Must be >1 for ML code to run
        eps_MPOs = [1e-9]
        dataset  = "wines"
        
    # Further hyperparameters
    n_local_its = 50
    max_sweeps = 5000
    scale = "quantile_uniform"
    
    for poly in polys:
        for cutoff_y in cutoffs["y"]:
            for cutoff_X in cutoffs["X"]:
                
                # Either: load data or download & MPSify data & save
                mpses_file = os.path.join(cfg.dir_data, "mpses", f"{dataset}_poly{poly}_cutoffs_X{cutoff_X}_y{cutoff_y}_{scale}.pkl")
                if os.path.exists(mpses_file):
                    # load everything from one file
                    with open(mpses_file, 'rb') as f: data = pickle.load(f)
                    (_, _, X_train_mps, y_train_mps, X_train_compr, y_train_compr,
                     _,  _,  X_test_mps,  y_test_mps,  X_test_compr,  y_test_compr), classific = data
                else:
                    # Create
                    X_train, y_train, X_test, y_test, classific = load_proc(dataset,scale = scale)
                    X_train_mps, X_train_compr = ln.mpsify(           X_train, split=True, chi=None, cutoff=cutoff_X, poly = poly)
                    y_train_mps, y_train_compr = ln.mpsify(np.float64(y_train),split=True, chi=None, cutoff=cutoff_y, output=True)#; print(1-np.mean(y_train_compr.round(0)==y_train))
                    X_test_mps, X_test_compr   = ln.mpsify(           X_test,  split=True, chi=None, cutoff=1e-16,    poly = poly)
                    y_test_mps, y_test_compr   = ln.mpsify(np.float64(y_test), split=True, chi=None, cutoff=1e-16,    output = True)#; print(1-np.mean(y_test_compr.round(0)==y_test))
                    
                    # and save
                    data = (X_train, y_train, X_train_mps, y_train_mps, X_train_compr, y_train_compr, 
                            X_test,  y_test,  X_test_mps,  y_test_mps,  X_test_compr,  y_test_compr), classific
                    os.makedirs(os.path.dirname(mpses_file), exist_ok=True)
                    with open(mpses_file, 'wb') as f: pickle.dump(data, f)
                    
                if max(chis_mpo)>1: # Train on dataset                        

                    ## First run classic regression
                    classic_scores = []
                    for alpha in alphas:
                        if classific == False:
                            score_class, _ = ln.classical_linreg(X_train_compr.T, X_test_compr.T, y_train_compr.flatten(), y_test_compr.flatten(),poly=poly, alpha=alpha*poly)
                            classic_scores.append(score_class)
                        else:
                            #score_class = ln.classical_logreg(X_train_compr, X_test_compr, y_train_compr, y_test_compr, poly = poly, alpha=alpha, penalty="l2", solver= "liblinear")
                            #classic_scores.append(score_class)
                            sys.exit(1) # Classification currently not supported
                    
                    ## Now do MPO regression
                    for chi_mpo in chis_mpo:
                        for eps_MPO in eps_MPOs:
                            
                            # Get dim/feature nums
                            out_dim = y_test_compr.shape[0]
                            n_features = X_train_compr.shape[0] # Includes intercept
                            
                            # Prime-factorise to define physical index sizes
                            primes_base = mps.primefact(n_features)[::-1]
                            primes = primes_base*poly
                            n_phys_inds = len(primes)
                            
                            # Construct inital MPO; lrud are mpo bonds
                            ten_init = lambda *shape: 1e-3*np.float64(np.ones(shape))
                            if len(primes) == 1:   tensors = [ten_init(primes[0],out_dim)]
                            elif len(primes) == 2: tensors = [ten_init(2,primes[0],1), ten_init(2,primes[1],out_dim)]
                            else:                  tensors = [ten_init(2,primes[0],1)]+ [ten_init(2,2,prime,1) for prime in primes[1:-1]] + [ten_init(2,primes[-1],out_dim)]
                            mpo = qtn.MatrixProductOperator(tensors, site_tag_id="s{}"); mpo.reindex({"b{}".format(mpo.L-1):"out"},inplace=True)
                            if len(primes)>1: 
                                for ten in mpo: ten.squeeze(exclude="out", inplace=True)
                            
                            n_train_inds  = X_train_mps.L- n_phys_inds;  n_output_inds = y_train_mps.L - n_train_inds
                            y_train_mpsC =mps.shiftIndextagMps(y_train_mps, n_phys_inds - n_output_inds, n_output_inds, tag_shift=True); y_train_mpsC.reindex({"k0":"out"}, inplace=True)
                            y_test_mpsC = mps.shiftIndextagMps(y_test_mps,  n_phys_inds - n_output_inds, n_output_inds, tag_shift=True); y_test_mpsC.reindex( {"k0":"out"}, inplace=True)
                            
                            # Finally, begin training
                            mpo_scores = []
                            for alpha in alphas:
                                if classific == False: # Run linear regression
                                    _, score_mpo = ln.learn_linreg(mpo, X_train_mps, y_train_mpsC,X_test_mps, y_test_mpsC,eps=eps_MPO, chi_max = chi_mpo, n_local_its = n_local_its, max_sweeps = [max_sweeps],local_opt = "explicitCG", alpha_mpo = alpha*poly) # "newton-cg" for autdoiff, or "explicitCG" for explicit implimentation 
                                    mpo_scores.append(score_mpo)                                    
                                
                                else: # Run logistic regression
                                    #_, score_mpo = ln.learn_logreg(mpo, X_train_mps, y_train_mpsC, X_test_mps, y_test_mpsC, eps=eps_MPO, chi_max = chi_mpo, n_local_its = 20, max_sweeps=50, local_opt="newton-cg");    
                                    #mpo_scores.append(score_mpo)
                                    exit(1) # Classific currently not supported
                            
                                    
                            ## Now do processing
                            class_top_score = max(classic_scores); class_top_alpha = alphas[np.argmax(classic_scores)]
                            mpo_top_score = max(mpo_scores); mpo_top_alpha = alphas[np.argmax(mpo_scores)]
                            
                            # Save output
                            out_file = os.path.join(cfg.dir_data, "out", f"{dataset}_poly{poly}.pkl")
                            os.makedirs(os.path.dirname(out_file), exist_ok=True)
                            if os.path.isfile(out_file): df = pd.read_pickle(out_file) # load existing output files if exists 
                            else: # make an empty MultiIndex with the right names
                                idx = pd.MultiIndex.from_tuples([], names=["poly", "chi_mpo","eps_mpo","cutoff_y", "cutoff_X"])
                                df = pd.DataFrame(index=idx,columns=["dataset","alpha_class", "score_class", "alpha_mpo", "score_mpo", "CR_MPO","CR_X_train"])
                            
                            # If failed to converge, set to 0
                            class_top_score = 0 if class_top_score<0 else class_top_score; mpo_top_score = 0 if mpo_top_score<0 else mpo_top_score
                            
                            df.loc[(poly, chi_mpo, eps_MPO, cutoff_y, cutoff_X), :] = [dataset,class_top_alpha,class_top_score, mpo_top_alpha, mpo_top_score, 1/mps.mpsCompressionRatio(mpo), 1/mps.mpsCompressionRatio(X_train_mps)]
                            df = df.sort_index()
                            df.to_pickle(out_file)
                            
                            print(class_top_score, mpo_top_score )
                            