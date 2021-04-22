using Printf, DelimitedFiles, Random, Statistics, ProgressMeter, Base.Threads
using MDToolbox
using BSON: @save, @load

function quate_dist(q1, q2)
    # return min(sum(q1 .+ q2) .^ 2, sum(q1 .- q2) .^ 2)
    # return acos(max(-1, min(1, 2.0 * sum(q1.*q2).^2 - 1.0)))
    return 1.0 - sum(q1.*q2)^2
end

qs = readdlm("data/quaternion/QUATERNION_LIST_576_Orient")
models = readpdb("data/t1r/cluster.pdb");
for iatom = 1:models.natom
    models.atomname[iatom] = models.resname[iatom]
end
MDToolbox.decenter!(models)

probe_radius = 20
param = AfmizeConfig(10.0 * (pi / 180),
    probe_radius, 
    MDToolbox.Point2D(-250, -200), 
    MDToolbox.Point2D(250, 200), 
    MDToolbox.Point2D(6.25, 6.25), 
    MDToolbox.defaultParameters())

myseed = 335
sigma_noise = 3.0
nq = size(qs, 1)
nframe = 100
nmodel = models.nframe

T_rot = zeros(Float64, nq, nq)
for i in 1:nq
    ncount = 0
    for j in 1:nq
        d = quate_dist(qs[i, :], qs[j, :])
        if d < 0.01
            T_rot[i, j] = 1.0
            ncount += 1
        end
    end
    #@show ncount
    T_rot[i, :] .= T_rot[i, :] ./ ncount
end

@load "data/t1r/t1r.bson" T pi_i

Threads.@threads for init_q in 1:nq
    @show init_q
    imodel_array = msmgenerate(nframe, T, pi_i)

    p0_q = zeros(Float64, nq)
    p0_q[init_q] = 1.0
    iq_array = msmgenerate(nframe, T_rot, p0_q);
    dxdy_array = zeros(Float64, nframe, 2);

    afms = []
    for iframe in 1:nframe
        imodel = imodel_array[iframe]
        model = models[imodel, :]
        iq = iq_array[iframe]
        q = qs[iq, :]
        model = MDToolbox.rotate(model, q)
        model.x .+=  dxdy_array[iframe, 1]
        model.y .+=  dxdy_array[iframe, 2]
        afm = MDToolbox.afmize(model, param)
        h, w = size(afm)
        afm .+= randn(h, w) * sigma_noise
        push!(afms, afm)
    end

    @save "data/01_test_case/radius_$(probe_radius)/iq_$(init_q)_noise_$(sigma_noise)_nframe_$(nframe).bson" models afms qs param imodel_array iq_array dxdy_array sigma_noise probe_radius T pi_i T_rot p0_q nframe nq nmodel
end