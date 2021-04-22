using BSON: @save, @load
using Statistics, DelimitedFiles, Printf
include("../MDToolbox.jl/src/MDToolbox.jl")
using .MDToolbox
using StatsBase 

function quate_dist(q1, q2)
    # return min(sum(q1 .+ q2) .^ 2, sum(q1 .- q2) .^ 2)
    # return acos(max(-1, min(1, 2.0 * sum(q1.*q2).^2 - 1.0)))
    return 1.0 - sum(q1.*q2)^2
end

function my_msmtransitionmatrix(C; TOLERANCE=10^(-8), verbose=false)
  c = Matrix{Float64}(C)
  nstate = size(c, 1)

  c_sym = c + c'
  x = c_sym

  c_rs = sum(c, dims=2)
  x_rs = sum(x, dims=2)

  delta = 10 * TOLERANCE
  logL_old = rand(nstate, nstate)
  logL = rand(nstate, nstate)
  count_iteration = 0

  x = zeros(Float64, nstate, nstate)
  x_new = zeros(Float64, nstate, nstate)

  while delta > TOLERANCE
    count_iteration = count_iteration + 1
    logL_old = deepcopy(logL)
  
    # fixed-point method
    for i = 1:nstate
      for j = 1:nstate
        denom = 0
        if !iszero(x_rs[i])
          denom += (c_rs[i]/x_rs[i]) 
        end
        if !iszero(x_rs[j])
          denom += (c_rs[j]/x_rs[j]) 
        end
        if !iszero(denom)
          x_new[i, j] = (c[i, j] + c[j, i]) / denom
        end
      end
    end
    
    # update
    x_rs .= sum(x_new, dims=2)
    x .= x_new
    for i = 1:nstate
      for j = 1:nstate
        if !iszero(x[i, j]) & !iszero(x_rs[i])
          logL[i, j] = c[i, j] * log(x[i, j] / x_rs[i]);
        end
      end
    end
        
    delta = sum(abs.(logL_old .- logL)) / (nstate ^ 2)
        
    if verbose & (mod(count_iteration, 10) == 0)
      Printf.@printf("%d iteration  LogLikelihood = %8.5e  delta = %8.5e  tolerance = %8.5e\n", count_iteration, sum(logL), delta, TOLERANCE)
    end
  end
  
  pi_i = x_rs ./ sum(x_rs)
  pi_i = pi_i[:]
  t = x ./ x_rs

  return t, pi_i
end

function my_msmbaumwelch(data_list, T0, pi_i0, emission0; TOLERANCE = 10.0^(-8), MAXITERATION=200)
    ## setup
    check_convergence = Inf64
    count_iteration = 0
    #if not isinstance(data_list, list):
    #    data_list = [data_list]
    ndata = length(data_list)
    logL_old = ones(Float64, ndata)
    nobs = length(emission0[1, :])
    nstate = length(T0[1, :])
    T = similar(T0)
    emission = similar(emission0)
    pi_i = similar(pi_i0)
    while (check_convergence > TOLERANCE) & (count_iteration <= MAXITERATION)
        ## E-step
        logL, alpha_list, factor_list = MDToolbox.msmforward(data_list, T0, pi_i0, emission0)
        #print("1"); println(logL)
        logL2, beta_list = MDToolbox.msmbackward(data_list, factor_list, T0, pi_i0, emission0)
        #print("2"); println(logL2)
        log_alpha_list = []
        for a in alpha_list
            push!(log_alpha_list, log.(a))
        end
        log_beta_list = []
        for b in beta_list
            push!(log_beta_list, log.(b))
        end
        log_T0 = log.(T0)
        log_emission0 = log.(emission0)

        ## M-step
        # pi
        # pi = np.zeros(nstate, dtype=np.float64)
        # log_gamma_list = []
        # for idata in range(ndata):
        #     log_gamma_list.append(log_alpha_list[idata] + log_beta_list[idata])
        #     pi = pi + np.exp(log_gamma_list[idata][0, :])
        # pi = pi/np.sum(pi)
        # pi_i = pi_i0
        # emission
        # emission = np.zeros((nstate, nobs), dtype=np.float64)
        # for idata in range(ndata):
        #     data = data_list[idata]
        #     for istate in range(nstate):
        #         for iobs in range(nobs):
        #             id = (data == iobs)
        #             if np.any(id):
        #                 emission[istate, iobs] = emission[istate, iobs] + np.sum(np.exp(log_gamma_list[idata][id, istate]))
        # emission[np.isnan(emission)] = 0.0
        # emission = emission / np.sum(emission, axis=1)[:, None]
        emission = emission0
        # T
        T = zeros(Float64, (nstate, nstate))
        for idata = 1:ndata
          data = data_list[idata]
          nframe = length(data)
          for iframe = 2:nframe
            #log_xi = bsxfun(@plus, log_alpha{idata}(iframe-1, :)', log_beta{idata}(iframe, :));
            log_xi = log_alpha_list[idata][iframe-1, :] .+ log_beta_list[idata][iframe, :]'
            #T = T .+ exp(bsxfun(@plus, log_xi, log_emission0(:, data(iframe))') + log_T0)./factor{idata}(iframe);
            T = T .+ exp.((log_xi .+ log_emission0[:, data[iframe]]') .+ log_T0) ./ factor_list[idata][iframe]
          end
        end
        #T[np.isnan(T)] = 0.0
        T = T ./ sum(T, dims=2)

        ## reversible T
        # T, pi_i = my_msmtransitionmatrix(T)

        ## Check convergence
        count_iteration += 1
        check_convergence = sum(abs.(logL_old .- logL)) / (nstate ^ 2)
        if mod(count_iteration, 100) == 0
            Printf.@printf("%d iteration: delta = %e  tolerance = %e\n" , count_iteration, check_convergence, TOLERANCE)
        end
        logL_old = logL
        pi_i0 = pi_i
        emission0 = emission
        T0 = T
    end
    T, pi_i, emission
end

function get_ids(emission_max, nmodel, nq)
    arr = reshape(deepcopy(emission_max), nmodel * nq)
    sort!(arr, rev=true)
    cou = 0
    border = 1
    for ele in arr
        if ele == 1
            continue
        end
        cou += 1
        if cou >= 30 || ele < 1e-300
            break
        end
        border = ele
    end
    
    ids = findall(x -> x >= border, emission_max)
    return ids
end

function make_extracted_baumwelch_data(ids, nid, qs, extracted_qs, nframe, test_radius, pred_radius; rotate_cutoff = false)
    init_T = rand(nid, nid)
    if rotate_cutoff
        for i in 1:nid
            for j in 1:nid
                if quate_dist(qs[ids[i][2], :], qs[ids[j][2], :]) > 0.1
                    init_T[i, j] = 0
                end
            end
        end
    end

    for i in 1:nid
        init_T[i, :] ./= sum(init_T[i, :])
    end
    init_P = ones(nid) ./ nid

    data_list = []
    emission = ones(nid, size(extracted_qs, 1) * nframe) .* 1e-5
    cou = 1
    for init_q in extracted_qs
        @load "data/01_result/test_radius_$(test_radius)_pred_radius_$(pred_radius)/iq_$(init_q)_nframe_$(nframe).bson" params r
        imodel_array_est = []
        for iframe = 1:nframe
            push!(imodel_array_est, cou)
            r.all[iframe] .-= maximum(r.all[iframe])
            r.all[iframe] = exp.(r.all[iframe])
            for (i, id) in enumerate(ids)
                for iparam in 1:size(r.all[iframe], 3)
                    emission[i, cou] += r.all[iframe][id, iparam]
                end
            end

            emission[:, cou] ./= sum(emission[:, cou])
            cou += 1
        end

        push!(data_list, imodel_array_est)
    end
    
    data_list, init_T, init_P, emission
end

function get_model_arr(models, qs, nframe, imodel_arr, iq_arr)
    models_arr = models[1, :]
    for iframe = 1:nframe
        imodel = imodel_arr[iframe]
        model = models[imodel, :]
        iq = iq_arr[iframe]
        q = qs[iq, :]
        MDToolbox.rotate!(model, q)
        models_arr = [models_arr; model]
    end
    models_arr = models_arr[2:end, :]
end

function calc_rmsd(models_true, models_est, rmsds, rmsds_a, rmsds_b, rmsds_fitted, rmsds_a_fitted, rmsds_b_fitted, iframe, i)
    d = compute_rmsd(models_true[iframe, :], models_est[iframe, :])
    rmsds[i, iframe] = d[1]
    d = compute_rmsd(models_true[iframe, "resid 1:512"], models_est[iframe, "resid 1:512"])
    rmsds_a[i, iframe] = d[1]
    d = compute_rmsd(models_true[iframe, "resid 513:1044"], models_est[iframe, "resid 513:1044"])
    rmsds_b[i, iframe] = d[1]
    
    ta = superimpose(models_true[iframe, :], models_est[iframe, :])
    d = compute_rmsd(models_true[iframe, :], ta)
    rmsds_fitted[i, iframe] = d[1]
    ta = superimpose(models_true[iframe, "resid 1:512"], models_est[iframe, "resid 1:512"])
    d = compute_rmsd(models_true[iframe, "resid 1:512"], ta)
    rmsds_a_fitted[i, iframe] = d[1]
    ta = superimpose(models_true[iframe, "resid 513:1044"], models_est[iframe, "resid 513:1044"])
    d = compute_rmsd(models_true[iframe, "resid 513:1044"], ta)
    rmsds_b_fitted[i, iframe] = d[1]
end

function test()
    qs = readdlm("data/quaternion/QUATERNION_LIST_576_Orient")
    models = readpdb("data/t1r/cluster.pdb")
    nmodel = size(models, 1)
    nframe = 100
    nq = size(qs, 1)
    test_radius = 20
    pred_radius = 25
    sigma_noise = 3.0
    extracted_qs = sample(1:576, 10, replace=false)

    emission_max = zeros(nmodel, nq)
    for init_q in extracted_qs
        @load "data/01_result/test_radius_$(test_radius)_pred_radius_$(pred_radius)/iq_$(init_q)_nframe_$(nframe).bson" params r
        for iframe = 1:nframe
            r.all[iframe] .-= maximum(r.all[iframe])
            r.all[iframe] = exp.(r.all[iframe])
            emission = reshape(sum(r.all[iframe], dims = 3), nmodel, nq)
            emission_max .= max.(emission_max, emission)
        end
    end

    ids = get_ids(emission_max, nmodel, nq)
    nid = size(ids, 1)

    data_list1, init_T1, init_P1, emission1 = make_extracted_baumwelch_data(ids, nid, qs, extracted_qs, nframe, test_radius, pred_radius)
    T1, pi_i1, emission1 = my_msmbaumwelch(data_list1, deepcopy(init_T1), init_P1, emission1)

    data_list2, init_T2, init_P2, emission2 = make_extracted_baumwelch_data(ids, nid, qs, extracted_qs, nframe, test_radius, pred_radius, rotate_cutoff = true)
    T2, pi_i2, emission2 = my_msmbaumwelch(data_list2, deepcopy(init_T2), init_P2, emission2)
    

    nextracted_qs = size(extracted_qs, 1)
    rmsds = zeros(nextracted_qs, nframe)
    rmsds_a = zeros(nextracted_qs, nframe)
    rmsds_b = zeros(nextracted_qs, nframe)
    rmsds_fitted = zeros(nextracted_qs, nframe)
    rmsds_a_fitted = zeros(nextracted_qs, nframe)
    rmsds_b_fitted = zeros(nextracted_qs, nframe)
    rmsds_original = zeros(nextracted_qs, nframe)
    rmsds_original_a = zeros(nextracted_qs, nframe)
    rmsds_original_b = zeros(nextracted_qs, nframe)
    rmsds_original_fitted = zeros(nextracted_qs, nframe)
    rmsds_original_a_fitted = zeros(nextracted_qs, nframe)
    rmsds_original_b_fitted = zeros(nextracted_qs, nframe)
    rmsds1 = zeros(nextracted_qs, nframe)
    rmsds1_a = zeros(nextracted_qs, nframe)
    rmsds1_b = zeros(nextracted_qs, nframe)
    rmsds1_fitted = zeros(nextracted_qs, nframe)
    rmsds1_a_fitted = zeros(nextracted_qs, nframe)
    rmsds1_b_fitted = zeros(nextracted_qs, nframe)
    rmsds2 = zeros(nextracted_qs, nframe)
    rmsds2_a = zeros(nextracted_qs, nframe)
    rmsds2_b = zeros(nextracted_qs, nframe)
    rmsds2_fitted = zeros(nextracted_qs, nframe)
    rmsds2_a_fitted = zeros(nextracted_qs, nframe)
    rmsds2_b_fitted = zeros(nextracted_qs, nframe)

    for (i, init_q) in zip(1:nextracted_qs, extracted_qs)
        @load "data/01_test_case/radius_$(test_radius)/iq_$(init_q)_noise_$(sigma_noise)_nframe_$(nframe).bson" models afms qs param imodel_array iq_array dxdy_array sigma_noise probe_radius T pi_i T_rot
        @load "data/01_result/test_radius_$(test_radius)_pred_radius_$(pred_radius)/iq_$(init_q)_nframe_$(nframe).bson" params r
        
        models_true = get_model_arr(models, qs, nframe, imodel_array, iq_array)  
        imodel_array_est = []
        iq_array_est = []
        for iframe = 1:length(afms)
            id = argmax(r.all[iframe])
            push!(imodel_array_est, id[1])
            push!(iq_array_est, id[2])
        end
        models_est = get_model_arr(models, qs, nframe, imodel_array_est, iq_array_est)
    
        T_ori = zeros(Float64, nid, nid)
        for i_id = 1:nid
            for j_id = 1:nid
                if quate_dist(qs[ids[i_id][2], :], qs[ids[j_id][2], :]) > 0.0001
                    continue
                end
                T_ori[i_id, j_id] = T[ids[i_id][1], ids[j_id][1]]
            end
            T_ori[i_id, :] .= T_ori[i_id, :] / sum(T_ori[i_id, :])
        end
        pi_ori = ones(nid) ./ nid
        state_estimated = msmviterbi(convert(Array{Int64,1}, data_list1[i]), T_ori, pi_ori, emission1)
        imodel_array_est = []
        iq_array_est = []
        for iframe in 1:nframe
            push!(imodel_array_est, ids[state_estimated[iframe]][1])
            push!(iq_array_est, ids[state_estimated[iframe]][2])
        end
        models_est_original = get_model_arr(models, qs, nframe, imodel_array_est, iq_array_est)   

        state_estimated = msmviterbi(convert(Array{Int64,1}, data_list1[i]), T1, pi_i1, emission1)
        imodel_array_est = []
        iq_array_est = []
        for iframe in 1:nframe
            push!(imodel_array_est, ids[state_estimated[iframe]][1])
            push!(iq_array_est, ids[state_estimated[iframe]][2])
        end
        models_est1 = get_model_arr(models, qs, nframe, imodel_array_est, iq_array_est)

        state_estimated = msmviterbi(convert(Array{Int64,1}, data_list2[i]), T2, pi_i2, emission2)
        imodel_array_est = []
        iq_array_est = []
        for iframe in 1:nframe
            push!(imodel_array_est, ids[state_estimated[iframe]][1])
            push!(iq_array_est, ids[state_estimated[iframe]][2])
        end
        models_est2 = get_model_arr(models, qs, nframe, imodel_array_est, iq_array_est)

        for iframe = 1:nframe
            calc_rmsd(models_true, models_est, rmsds, rmsds_a, rmsds_b, rmsds_fitted, rmsds_a_fitted, rmsds_b_fitted, iframe, i)
            calc_rmsd(models_true, models_est_original, rmsds_original, rmsds_original_a, rmsds_original_b, rmsds_original_fitted, rmsds_original_a_fitted, rmsds_original_b_fitted, iframe, i)
            calc_rmsd(models_true, models_est1, rmsds1, rmsds1_a, rmsds1_b, rmsds1_fitted, rmsds1_a_fitted, rmsds1_b_fitted, iframe, i)
            calc_rmsd(models_true, models_est2, rmsds2, rmsds2_a, rmsds2_b, rmsds2_fitted, rmsds2_a_fitted, rmsds2_b_fitted, iframe, i)
        end
    end

    test_num = 1
    @save "data/01_result/test_radius_$(test_radius)_pred_radius_$(pred_radius)/statistics_$(test_num)_ids.bson" ids extracted_qs 
    @save "data/01_result/test_radius_$(test_radius)_pred_radius_$(pred_radius)/statistics_$(test_num)_params.bson" init_T1 init_T2 T1 T2 pi_i1 pi_i2 
    @save "data/01_result/test_radius_$(test_radius)_pred_radius_$(pred_radius)/statistics_$(test_num)_rmsds.bson" rmsds rmsds_a rmsds_b rmsds_fitted rmsds_a_fitted rmsds_b_fitted 
    @save "data/01_result/test_radius_$(test_radius)_pred_radius_$(pred_radius)/statistics_$(test_num)_rmsds_original.bson" rmsds_original rmsds_original_a rmsds_original_b rmsds_original_fitted rmsds_original_a_fitted rmsds_original_b_fitted 
    @save "data/01_result/test_radius_$(test_radius)_pred_radius_$(pred_radius)/statistics_$(test_num)_rmsds1.bson" rmsds1 rmsds1_a rmsds1_b rmsds1_fitted rmsds1_a_fitted rmsds1_b_fitted 
    @save "data/01_result/test_radius_$(test_radius)_pred_radius_$(pred_radius)/statistics_$(test_num)_rmsds2.bson" rmsds2 rmsds2_a rmsds2_b rmsds2_fitted rmsds2_a_fitted rmsds2_b_fitted
end

test()