function my_logprob_eachmodel(afm_array, imodel::Int64, q_array::Matrix{Float64}, param_array)
    nafm = length(afm_array)
    nq = size(q_array, 1)
    nparam = length(param_array)
    logprob_array = zeros(eltype(afm_array[1]), nafm, nq, nparam)
    dxdy_array = Array{Tuple{Int,Int}}(undef, nafm, nq, nparam)
    dxdy = Array{Tuple{Int,Int}}(undef, nafm)
    x_center = ceil(Int64, (size(afm_array[1],2)/2)+1.0)
    y_center = ceil(Int64, (size(afm_array[1],1)/2)+1.0)
  
    ### loop over rotations
    for iq in 1:nq
        ### loop over afmize parameters
        for iparam in 1:nparam
            radius = Int(param_array[iparam].probeRadius)
            @load "data/calc_afms/radius_$(radius)/model_$(imodel)_iq_$(iq).bson" calculated
            for iafm in 1:nafm
                observed = afm_array[iafm]
                logprob = MDToolbox.calcLogProb(observed, calculated)
                logprob_array[iafm, iq, iparam] = logsumexp(logprob)
                dx = argmax(logprob)[2] - x_center
                dy = argmax(logprob)[1] - y_center
                dxdy_array[iafm, iq, iparam] = (dx, dy)
            end
        end
    end

    for iafm in 1:nafm
        ind = argmax(logprob_array[iafm, :, :])
        dxdy[iafm] = dxdy_array[iafm, :, :][ind]
    end

    return (logprob_array=logprob_array, dxdy=dxdy)
end


function getposterior_parallel_use_local_afm(models::TrjArray, afm_array, q_array::Matrix{Float64}, param_array)
    nmodel = models.nframe
    nafm = length(afm_array)
    nq = size(q_array, 1)
    nparam = length(param_array)

    p = pmap(x -> my_logprob_eachmodel(afm_array, x, q_array, param_array), 1:nmodel)

    logprob_all = []
    logprob_model = []
    logprob_q = []
    logprob_param = []
    dxdy_best = []
    for iafm = 1:nafm
        pp = zeros(Float64, nmodel, nq, nparam)
        for imodel = 1:nmodel
            for iq = 1:nq
                for iparam = 1:nparam
                    pp[imodel, iq, iparam] = p[imodel].logprob_array[iafm, iq, iparam]
                end
            end
        end

        push!(logprob_all, pp)

        t = zeros(Float64, nmodel)
        for imodel = 1:nmodel
            t[imodel] = logsumexp(pp[imodel, :, :][:])
        end
        push!(logprob_model, t)

        imax = argmax(t)
        push!(dxdy_best, p[imax].dxdy[iafm])

        t = zeros(Float64, nq)
        for iq = 1:nq
            t[iq] = logsumexp(pp[:, iq, :][:])
        end
        push!(logprob_q, t)

        t = zeros(Float64, nparam)
        for iparam = 1:nparam
            t[iparam] = logsumexp(pp[:, :, iparam][:])
        end
        push!(logprob_param, t)
    end

    return (all=logprob_all, 
            model=logprob_model, 
            q=logprob_q, 
            param=logprob_param,
            dxdy=dxdy_best)
end